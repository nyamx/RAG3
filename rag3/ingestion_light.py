"""Docling-free source handling and identical plain text chunk normalization."""
from __future__ import annotations
import hashlib
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from haystack import Document
from haystack.components.preprocessors import DocumentSplitter
from rag3.source_identity import sha256_file
from rag3.constants import ROOT, DATA_ROOT
from rag3.ingestion_settings import DEFAULT_SPLIT_LENGTH, DEFAULT_SPLIT_OVERLAP

PLAIN_TEXT_SUFFIXES = frozenset({".txt", ".json", ".jsonl", ".tsv"})

def _utc_now():
    return datetime.now(timezone.utc).isoformat()

def preserve_source(source: str | Path, *, expected_sha256: str | None = None) -> dict[str, Any]:
    src = Path(source).expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(src)

    source_sha = expected_sha256 or sha256_file(src)
    if len(source_sha) != 64 or any(char not in "0123456789abcdef" for char in source_sha):
        raise ValueError("invalid source SHA-256")
    dest_dir = DATA_ROOT / "sources" / source_sha
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name

    if dest.exists():
        if sha256_file(dest) != source_sha:
            raise RuntimeError(f"Managed source hash mismatch: {dest}")
    else:
        shutil.copy2(src, dest)
        if sha256_file(dest) != source_sha:
            dest.unlink(missing_ok=True)
            raise RuntimeError(f"Managed source verification failed: {dest}")

    return {
        "source_sha256": source_sha,
        "document_id": source_sha,
        "title": src.stem,
        "source_uri": str(dest),
        "managed_source": dest,
    }


def split_plain_text_source(
    preserved: dict[str, Any], base_meta: dict[str, Any],
    *, split_length: int = DEFAULT_SPLIT_LENGTH,
    split_overlap: int = DEFAULT_SPLIT_OVERLAP,
) -> list[Document]:
    source = Path(preserved["managed_source"])
    if source.suffix.lower() not in PLAIN_TEXT_SUFFIXES:
        raise ValueError("Not a plain text source")
    content = source.read_text(encoding="utf-8-sig")
    if not content.strip():
        raise ValueError("text file must not be empty")
    splitter = DocumentSplitter(
        split_by="token", split_length=split_length,
        split_overlap=split_overlap, tokenizer_encoding="o200k_base",
    )
    return splitter.run(documents=[Document(content=content, meta=base_meta)])["documents"]



def _page_range_from_split_meta(
    meta: dict[str, Any],
    content_length: int,
) -> tuple[int | None, int | None]:
    """Resolve a split's physical page range from inherited document offsets."""

    def valid_page(value: Any) -> int | None:
        if isinstance(value, bool):
            return None
        try:
            page = int(value)
        except (TypeError, ValueError):
            return None
        return page if page >= 1 else None

    existing_start = valid_page(meta.get("page_start"))
    existing_end = valid_page(meta.get("page_end"))
    breaks = meta.get("_rag3_page_breaks")
    first_page = valid_page(meta.get("_rag3_first_page"))

    if not isinstance(breaks, list) or first_page is None:
        return existing_start, existing_end

    try:
        start = int(meta.get("split_idx_start", 0))
    except (TypeError, ValueError):
        return existing_start, existing_end
    if start < 0:
        return existing_start, existing_end

    end = start + max(0, int(content_length))
    page_start = first_page
    page_end = first_page

    for entry in breaks:
        if not isinstance(entry, dict):
            continue
        try:
            break_offset = int(entry["offset"])
        except (KeyError, TypeError, ValueError):
            continue
        next_page = valid_page(entry.get("next_page"))
        if break_offset < 0 or next_page is None:
            continue
        if break_offset <= start:
            page_start = next_page
        if break_offset < end:
            page_end = next_page

    return page_start, page_end


def normalize_chunk(doc: Document, idx: int) -> Document:
    """Assign the final stable ID and metadata before a vector is available."""
    content = doc.content or ""
    content_sha = hashlib.sha256(content.encode("utf-8")).hexdigest()
    meta = dict(doc.meta or {})
    page_start, page_end = _page_range_from_split_meta(meta, len(content))
    for temporary in ("_rag3_page_breaks", "_rag3_first_page", "_rag3_last_page", "section_path"):
        meta.pop(temporary, None)
    source_sha = str(meta.get("source_sha256") or "")
    chunk_index = int(meta.get("chunk_index", idx))
    chunk_id = hashlib.sha256(
        f"{source_sha}:{chunk_index}:{content_sha}".encode("utf-8")
    ).hexdigest()
    now = _utc_now()
    meta.update({
        "chunk_id": chunk_id,
        "document_id": meta.get("document_id") or source_sha,
        "chunk_index": chunk_index,
        "title": meta.get("title"),
        "source_uri": meta.get("source_uri"),
        "source_sha256": source_sha,
        "page_start": page_start,
        "page_end": page_end,
        "content_type": meta.get("content_type") or "prose",
        "document_version": meta.get("document_version") or "1",
        "content_sha256": content_sha,
        "created_at": meta.get("created_at") or now,
        "updated_at": now,
    })
    return Document(id=chunk_id, content=content, meta=meta)
