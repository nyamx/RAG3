"""Read managed originals by document identity; never accept a filesystem path."""

from __future__ import annotations

import base64
import hashlib
import io
import math
import re
import threading
from contextlib import closing
from pathlib import Path
from urllib.parse import quote

from qdrant_client import QdrantClient, models

from rag3.constants import DEFAULT_QDRANT_URL, ROOT
from rag3.source_identity import sha256_file

# A transport block, not a file-size limit. JSON/base64 plus SSE framing must
# fit the SDK client's default 1 MiB event limit as well as the MCP tunnel.
# Keep the block divisible by 3, so the UI can join its base64 without padding.
SOURCE_PART_BYTES = 384 * 1024
# The image is present both in model content and private preview metadata.
PAGE_IMAGE_BYTES = 256 * 1024
MIME_TYPES = {".pdf": "application/pdf", ".doc": "application/msword",
              ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
              ".md": "text/markdown"}
_PDF_LOCK = threading.Lock()  # PDFium is not thread-safe.


class SourceError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _resolve(collection: str, document_id: str, *, url: str,
             verify: bool = True) -> tuple[Path, str]:
    if (not re.fullmatch(r"[\w][\w.-]{0,127}", collection)
            or not re.fullmatch(r"[A-Za-z0-9_.-]{1,200}", document_id)
            or document_id in {".", ".."}):
        raise SourceError("INVALID_ARGUMENT", "Use a collection and document_id returned by RAG3.")
    client = QdrantClient(url=url)
    try:
        if not client.collection_exists(collection):
            raise SourceError("DOCUMENT_NOT_FOUND", "Collection does not exist.")
        records, _ = client.scroll(collection_name=collection,
            scroll_filter=models.Filter(must=[models.FieldCondition(
                key="meta.document_id", match=models.MatchValue(value=document_id))]),
            limit=1, with_payload=True, with_vectors=False)
    finally:
        client.close()
    if not records:
        raise SourceError("DOCUMENT_NOT_FOUND", "Document does not exist in this collection.")
    meta = (records[0].payload or {}).get("meta") or {}
    digest = meta.get("source_sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise SourceError("SOURCE_UNAVAILABLE", "Document has no verified original source identity.")
    source_uri = meta.get("source_uri")
    if not isinstance(source_uri, str) or not source_uri:
        raise SourceError("SOURCE_UNAVAILABLE", "Document has no managed original source.")
    try:
        managed_root = (ROOT / "data/sources").resolve()
        source = Path(source_uri).resolve(strict=True)
        expected_parent = managed_root / digest
        # Validate the resolved target too, so symlinks/junctions cannot escape.
        if (not source.is_relative_to(managed_root) or source.parent != expected_parent
                or not source.is_file()):
            raise SourceError("SOURCE_UNAVAILABLE", "Original is outside its managed source folder.")
        if source.suffix.lower() not in MIME_TYPES:
            raise SourceError("SOURCE_UNAVAILABLE", "Original source format is not supported.")
        if verify and sha256_file(source) != digest:
            raise SourceError("SOURCE_CHANGED", "Original source hash does not match the ingested document.")
    except (OSError, RuntimeError, ValueError) as exc:
        if isinstance(exc, SourceError):
            raise
        raise SourceError("SOURCE_UNAVAILABLE", "Managed original is missing or unreadable.") from None
    return source, digest


def source_metadata(collection: str, document_id: str, *, url: str = DEFAULT_QDRANT_URL) -> dict:
    source, digest = _resolve(collection, document_id, url=url)
    return {"collection": collection, "document_id": document_id, "filename": source.name,
            "mime_type": MIME_TYPES[source.suffix.lower()], "file_size": source.stat().st_size,
            "source_sha256": digest, "part_bytes": SOURCE_PART_BYTES,
            "first_part_uri": f"rag3://sources/{quote(collection, safe='')}/{quote(document_id, safe='')}/0"}


def source_part(collection: str, document_id: str, offset: int,
                *, url: str = DEFAULT_QDRANT_URL) -> dict:
    # The initial metadata call verifies the original; the receiver verifies
    # the final SHA256 again. Avoid hashing the entire file for every block.
    source, digest = _resolve(collection, document_id, url=url, verify=False)
    size = source.stat().st_size
    if isinstance(offset, bool) or offset < 0 or offset > size:
        raise SourceError("INVALID_ARGUMENT", "Source offset is outside the original file.")
    with source.open("rb") as stream:
        stream.seek(offset)
        content = stream.read(SOURCE_PART_BYTES)
    end = offset + len(content)
    if len(content) != min(SOURCE_PART_BYTES, size - offset):
        raise SourceError("SOURCE_CHANGED", "Original source changed during transfer; request it again.")
    return {"collection": collection, "document_id": document_id, "filename": source.name,
            "mime_type": MIME_TYPES[source.suffix.lower()], "file_size": size,
            "source_sha256": digest, "offset": offset, "end_offset": end,
            "next_offset": end if end < size else None,
            "part_sha256": hashlib.sha256(content).hexdigest(),
            "data_base64": base64.b64encode(content).decode("ascii")}


def render_page(collection: str, document_id: str, page: int, *, max_dimension: int = 2000,
                url: str = DEFAULT_QDRANT_URL) -> tuple[dict, bytes]:
    source, digest = _resolve(collection, document_id, url=url)
    if source.suffix.lower() != ".pdf":
        raise SourceError("INVALID_ARGUMENT", "Page images are available for original PDF files.")
    if isinstance(page, bool) or page < 1 or not 512 <= max_dimension <= 3000:
        raise SourceError("INVALID_ARGUMENT", "Use a physical PDF page number starting at 1.")
    import pypdfium2 as pdfium

    with _PDF_LOCK:
        try:
            with pdfium.PdfDocument(source) as pdf:
                count = len(pdf)
                if page > count:
                    raise SourceError("INVALID_ARGUMENT", f"PDF has {count} pages; requested page is out of range.")
                with closing(pdf[page - 1]) as pdf_page:
                    width, height = pdf_page.get_size()
                    longest = max(width, height)
                    if not math.isfinite(longest) or longest <= 0:
                        raise SourceError("SOURCE_UNAVAILABLE", "PDF page dimensions are invalid.")
                    with closing(pdf_page.render(scale=max_dimension / longest)) as bitmap:
                        image = bitmap.to_pil()
                        try:
                            with image.convert("RGB") as rgb:
                                for quality in (90, 80, 65):
                                    output = io.BytesIO()
                                    rgb.save(output, format="JPEG", quality=quality, optimize=True)
                                    content = output.getvalue()
                                    if len(content) <= PAGE_IMAGE_BYTES:
                                        break
                                else:
                                    while len(content) > PAGE_IMAGE_BYTES:
                                        limit = max(128, int(max(rgb.size) * 0.8))
                                        rgb.thumbnail((limit, limit))
                                        output = io.BytesIO()
                                        rgb.save(output, format="JPEG", quality=80, optimize=True)
                                        content = output.getvalue()
                                        if max(rgb.size) <= 128:
                                            break
                                dimensions = rgb.size
                        finally:
                            image.close()
        except SourceError:
            raise
        except Exception:
            raise SourceError("SOURCE_UNAVAILABLE", "Could not render this original PDF page.") from None
    return {"collection": collection, "document_id": document_id, "filename": source.name,
            "source_sha256": digest, "page": page, "page_count": count,
            "width": dimensions[0], "height": dimensions[1], "mime_type": "image/jpeg"}, content
