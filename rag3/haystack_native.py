from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from FlagEmbedding import BGEM3FlagModel
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter as DoclingDocumentConverter, PdfFormatOption
from docling_core.transforms.serializer.markdown import MarkdownDocSerializer, MarkdownParams
from haystack import Document, Pipeline, component
from haystack.dataclasses import SparseEmbedding
from haystack.document_stores.types import DuplicatePolicy
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack_integrations.components.converters.docling import DoclingConverter, MetaExtractor
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack_integrations.document_stores.qdrant.converters import (
    convert_haystack_documents_to_qdrant_points,
)
from qdrant_client import QdrantClient, models as qdrant_models

from rag3.source_identity import sha256_file
from rag3.constants import ROOT, DEFAULT_QDRANT_URL, DEFAULT_NATIVE_COLLECTION

DEFAULT_SPLIT_LENGTH = 1024
DEFAULT_SPLIT_OVERLAP = 128
DEFAULT_DOCLING_MODEL_DIR = ROOT / "data" / "models" / "docling"
BM25_VECTOR_NAME = "bm25"
DEFAULT_BM25_K = 1.2
DEFAULT_BM25_B = 0.75
DEFAULT_BM25_AVG_LEN = 256.0



def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def find_bge_m3() -> Path:
    roots = (ROOT / "data" / "models" / "huggingface", ROOT / "data" / "models")
    for base in roots:
        if not base.exists():
            continue
        candidates: list[Path] = []
        for cfg in base.rglob("config.json"):
            p = cfg.parent
            s = str(p).lower()
            if ("bge-m3" in s or "bge_m3" in s or "bge--m3" in s) and (
                (p / "tokenizer.json").exists() or (p / "tokenizer_config.json").exists()
            ):
                candidates.append(p)
        if candidates:
            return sorted(set(candidates), key=lambda p: (len(str(p)), str(p)))[0]
    raise RuntimeError("Local BGE-M3 model directory not found under data/models")


def preserve_source(source: str | Path) -> dict[str, Any]:
    src = Path(source).expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(src)

    source_sha = sha256_file(src)
    dest_dir = ROOT / "data" / "sources" / source_sha
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


def prepare_docling_source(preserved: dict[str, Any]) -> Path:
    """Convert legacy Word DOC to DOCX before the existing Docling pipeline.

    The original managed source and its SHA/document identity stay unchanged.
    The converted file is only a parser input, shared by UI and MCP workers.
    """
    source = Path(preserved["managed_source"])
    if source.suffix.lower() != ".doc":
        return source
    if os.name != "nt":
        raise RuntimeError("Legacy DOC ingestion requires Microsoft Word on Windows")
    target = source.with_suffix(".docx")

    def valid_docx(path: Path) -> bool:
        if not path.is_file():
            return False
        try:
            with zipfile.ZipFile(path) as archive:
                return "word/document.xml" in archive.namelist()
        except zipfile.BadZipFile:
            return False

    if valid_docx(target):
        return target

    temporary = target.with_name(target.stem + ".converting.docx")
    temporary.unlink(missing_ok=True)
    env = os.environ.copy()
    env["RAG3_DOC_INPUT"] = str(source)
    env["RAG3_DOC_OUTPUT"] = str(temporary)
    script = (
        "$ErrorActionPreference='Stop'; $word=$null; $doc=$null; "
        "try { $word=New-Object -ComObject Word.Application; "
        "$word.Visible=$false; $word.DisplayAlerts=0; $word.AutomationSecurity=3; "
        "$doc=$word.Documents.Open($env:RAG3_DOC_INPUT,$false,$true); "
        "$doc.SaveAs2($env:RAG3_DOC_OUTPUT,16) } "
        "finally { if ($doc) { $doc.Close($false) }; if ($word) { $word.Quit() } }"
    )
    try:
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
            cwd=ROOT, env=env, capture_output=True, text=True, timeout=120,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if result.returncode != 0 or not valid_docx(temporary):
            detail = (result.stderr or result.stdout).strip()[-500:]
            raise RuntimeError(f"Legacy DOC to DOCX conversion failed: {detail}")
        temporary.replace(target)
        return target
    finally:
        temporary.unlink(missing_ok=True)



def _bm25_metadata(k: float, b: float, avg_len: float) -> dict[str, Any]:
    return {
        "rag3": {
            "schema_version": 2,
            "bm25": {
                "vector_name": BM25_VECTOR_NAME,
                "model": "qdrant/bm25",
                "source": "content",
                "k": float(k),
                "b": float(b),
                "avg_len": float(avg_len),
                "tokenizer": "multilingual",
                "lowercase": True,
                "stemming": False,
                "stopwords": False,
            },
        }
    }


def _collection_bm25_settings(
    client: QdrantClient,
    collection: str,
) -> tuple[float, float, float]:
    info = client.get_collection(collection_name=collection).model_dump()
    sparse = info.get("config", {}).get("params", {}).get("sparse_vectors", {}) or {}
    if BM25_VECTOR_NAME not in sparse:
        raise RuntimeError(
            f"Collection '{collection}' is missing required sparse vector '{BM25_VECTOR_NAME}'"
        )

    metadata = info.get("config", {}).get("metadata") or {}
    bm25 = (metadata.get("rag3") or {}).get("bm25") or {}
    if not bm25:
        raise RuntimeError(f"Collection '{collection}' is missing RAG3 BM25 metadata")

    return (
        float(bm25.get("k", DEFAULT_BM25_K)),
        float(bm25.get("b", DEFAULT_BM25_B)),
        float(bm25.get("avg_len", DEFAULT_BM25_AVG_LEN)),
    )


def bm25_config(
    *,
    k: float = DEFAULT_BM25_K,
    b: float = DEFAULT_BM25_B,
    avg_len: float = DEFAULT_BM25_AVG_LEN,
) -> qdrant_models.Bm25Config:
    return qdrant_models.Bm25Config(
        k=float(k),
        b=float(b),
        avg_len=float(avg_len),
        tokenizer=qdrant_models.TokenizerType.MULTILINGUAL,
        lowercase=True,
        stopwords=None,
        stemmer=qdrant_models.DisabledStemmerParams(type=qdrant_models.NoStemmer.NONE),
    )


def collection_bm25_config(
    collection: str,
    *,
    url: str = DEFAULT_QDRANT_URL,
) -> qdrant_models.Bm25Config:
    client = QdrantClient(url=url)
    k, b, avg_len = _collection_bm25_settings(client, collection)
    return bm25_config(k=k, b=b, avg_len=avg_len)


@component
class QdrantBM25Writer:
    """Add Qdrant-native BM25 vectors after Haystack has written the points."""

    def __init__(self, *, collection: str, url: str = DEFAULT_QDRANT_URL) -> None:
        self.collection = collection
        self.url = url

    @component.output_types(bm25_vectors_written=int)
    def run(
        self,
        documents: list[Document],
        documents_written: int,
    ) -> dict[str, int]:
        if documents_written < 0:
            raise ValueError("documents_written must be >= 0")
        if not documents:
            return {"bm25_vectors_written": 0}

        config = collection_bm25_config(self.collection, url=self.url)
        qdrant_points = convert_haystack_documents_to_qdrant_points(
            documents,
            use_sparse_embeddings=True,
        )
        updates = [
            qdrant_models.PointVectors(
                id=point.id,
                vector={
                    BM25_VECTOR_NAME: qdrant_models.Document(
                        text=doc.content or "",
                        model="qdrant/bm25",
                        options=config,
                    )
                },
            )
            for point, doc in zip(qdrant_points, documents, strict=True)
        ]
        client = QdrantClient(url=self.url)
        client.update_vectors(
            collection_name=self.collection,
            points=updates,
            wait=True,
        )
        return {"bm25_vectors_written": len(updates)}



class PageProvenanceMetaExtractor(MetaExtractor):
    """Preserve physical PDF page transitions without changing Markdown content."""

    def extract_dl_doc_meta(self, dl_doc: Any) -> dict[str, Any]:
        meta = super().extract_dl_doc_meta(dl_doc=dl_doc)

        serializer = MarkdownDocSerializer(
            doc=dl_doc,
            params=MarkdownParams(
                image_placeholder="",
                page_break_placeholder="\f",
            ),
        )
        parts = serializer.get_parts()

        clean_parts: list[str] = []
        page_breaks: list[dict[str, int]] = []
        pending_breaks: list[tuple[int, int]] = []
        clean_len = 0

        for part in parts:
            part_text = part.text or ""
            matches = list(serializer._get_page_breaks(part_text))
            if len(matches) == 1 and part_text == matches[0][0]:
                _, prev_page, next_page = matches[0]
                pending_breaks.append((int(prev_page), int(next_page)))
                continue
            if not part_text:
                continue

            if clean_parts:
                clean_len += 2

            if pending_breaks:
                for prev_page, next_page in pending_breaks:
                    page_breaks.append(
                        {
                            "offset": clean_len,
                            "prev_page": prev_page,
                            "next_page": next_page,
                        }
                    )
                pending_breaks.clear()

            clean_parts.append(part_text)
            clean_len += len(part_text)

        clean_markdown = "\n\n".join(clean_parts)
        expected_markdown = dl_doc.export_to_markdown(image_placeholder="")
        if clean_markdown != expected_markdown:
            raise RuntimeError(
                "Docling page-provenance serialization no longer matches normal Markdown export"
            )

        page_numbers: list[int] = []
        for item, _ in dl_doc.iterate_items(traverse_pictures=False):
            for prov in (getattr(item, "prov", None) or []):
                page_no = getattr(prov, "page_no", None)
                if isinstance(page_no, int) and not isinstance(page_no, bool) and page_no >= 1:
                    page_numbers.append(page_no)

        if page_numbers:
            meta["_rag3_first_page"] = page_numbers[0]
            meta["_rag3_last_page"] = page_numbers[-1]
        elif page_breaks:
            meta["_rag3_first_page"] = page_breaks[0]["prev_page"]
            meta["_rag3_last_page"] = page_breaks[-1]["next_page"]

        meta["_rag3_page_breaks"] = page_breaks
        return meta


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


@component
class BGEM3OnePassEmbedder:
    """Haystack component that emits BGE-M3 dense + lexical sparse vectors in one model encode call."""

    def __init__(
        self,
        model_path: str | None = None,
        batch_size: int = 8,
        max_length: int = 8192,
        device: str | None = None,
    ) -> None:
        self.model_path = model_path or str(find_bge_m3())
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self.device = device
        self._model: BGEM3FlagModel | None = None

    def warm_up(self) -> None:
        if self._model is not None:
            return
        device = self.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self._model = BGEM3FlagModel(
            self.model_path,
            use_fp16=device.startswith("cuda"),
            pooling_method="cls",
            devices=device,
        )

    def _encode(self, texts: list[str], *, query: bool) -> tuple[list[list[float]], list[SparseEmbedding]]:
        self.warm_up()
        if self._model is None:
            raise RuntimeError("BGE-M3 model did not initialize")

        kwargs = dict(
            batch_size=self.batch_size,
            max_length=self.max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        if query and hasattr(self._model, "encode_queries"):
            out = self._model.encode_queries(texts, **kwargs)
        elif hasattr(self._model, "encode_corpus"):
            out = self._model.encode_corpus(texts, **kwargs)
        else:
            out = self._model.encode(texts, **kwargs)

        dense_arr = np.asarray(out["dense_vecs"], dtype=np.float32)
        if dense_arr.ndim == 1:
            dense_arr = dense_arr.reshape(1, -1)
        norms = np.linalg.norm(dense_arr, axis=1, keepdims=True)
        dense_arr = dense_arr / np.where(norms == 0.0, 1.0, norms)

        sparse_out: list[SparseEmbedding] = []
        for lexical in out["lexical_weights"]:
            pairs = sorted((int(k), float(v)) for k, v in lexical.items() if float(v) > 0.0)
            sparse_out.append(
                SparseEmbedding(
                    indices=[i for i, _ in pairs],
                    values=[v for _, v in pairs],
                )
            )
        return dense_arr.tolist(), sparse_out

    @component.output_types(
        documents=list[Document],
        embedding=list[float],
        sparse_embedding=SparseEmbedding,
    )
    def run(
        self,
        documents: list[Document] | None = None,
        text: str | None = None,
    ) -> dict[str, Any]:
        if (documents is None) == (text is None):
            raise ValueError("Provide exactly one of documents or text")

        if text is not None:
            dense, sparse = self._encode([text], query=True)
            return {
                "documents": [],
                "embedding": dense[0],
                "sparse_embedding": sparse[0],
            }

        assert documents is not None
        if not documents:
            return {
                "documents": [],
                "embedding": [],
                "sparse_embedding": SparseEmbedding(indices=[], values=[]),
            }

        texts = [(doc.content or "") for doc in documents]
        dense, sparse = self._encode(texts, query=False)
        now = _utc_now()
        embedded: list[Document] = []

        for idx, (doc, dvec, svec) in enumerate(zip(documents, dense, sparse, strict=True)):
            content = doc.content or ""
            content_sha = hashlib.sha256(content.encode("utf-8")).hexdigest()
            meta = dict(doc.meta or {})
            page_start, page_end = _page_range_from_split_meta(meta, len(content))
            meta.pop("_rag3_page_breaks", None)
            meta.pop("_rag3_first_page", None)
            meta.pop("_rag3_last_page", None)
            source_sha = str(meta.get("source_sha256") or "")
            chunk_index = int(meta.get("chunk_index", idx))
            chunk_id = hashlib.sha256(
                f"{source_sha}:{chunk_index}:{content_sha}".encode("utf-8")
            ).hexdigest()

            meta.update(
                {
                    "chunk_id": chunk_id,
                    "document_id": meta.get("document_id") or source_sha,
                    "chunk_index": chunk_index,
                    "title": meta.get("title"),
                    "source_uri": meta.get("source_uri"),
                    "source_sha256": source_sha,
                    "page_start": page_start,
                    "page_end": page_end,
                    "section_path": meta.get("section_path") or [],
                    "content_type": meta.get("content_type") or "prose",
                    "document_version": meta.get("document_version") or "1",
                    "content_sha256": content_sha,
                    "created_at": meta.get("created_at") or now,
                    "updated_at": now,
                }
            )
            embedded.append(
                Document(
                    id=chunk_id,
                    content=content,
                    meta=meta,
                    embedding=dvec,
                    sparse_embedding=svec,
                )
            )

        return {
            "documents": embedded,
            "embedding": [],
            "sparse_embedding": SparseEmbedding(indices=[], values=[]),
        }


def create_document_store(
    *,
    collection: str = DEFAULT_NATIVE_COLLECTION,
    url: str = DEFAULT_QDRANT_URL,
    recreate_index: bool = False,
    bm25_k: float = DEFAULT_BM25_K,
    bm25_b: float = DEFAULT_BM25_B,
    bm25_avg_len: float = DEFAULT_BM25_AVG_LEN,
) -> QdrantDocumentStore:
    client = QdrantClient(url=url)

    if recreate_index and client.collection_exists(collection):
        client.delete_collection(collection_name=collection)

    if not client.collection_exists(collection):
        client.create_collection(
            collection_name=collection,
            vectors_config={
                "text-dense": qdrant_models.VectorParams(
                    size=1024,
                    distance=qdrant_models.Distance.COSINE,
                )
            },
            sparse_vectors_config={
                "text-sparse": qdrant_models.SparseVectorParams(),
                BM25_VECTOR_NAME: qdrant_models.SparseVectorParams(
                    modifier=qdrant_models.Modifier.IDF
                ),
            },
            metadata=_bm25_metadata(bm25_k, bm25_b, bm25_avg_len),
        )
    else:
        _collection_bm25_settings(client, collection)

    return QdrantDocumentStore(
        url=url,
        index=collection,
        embedding_dim=1024,
        use_sparse_embeddings=True,
        similarity="cosine",
        recreate_index=False,
        return_embedding=False,
        progress_bar=False,
    )


def build_ingestion_pipeline(
    *,
    document_store: QdrantDocumentStore,
    model_path: str | None = None,
    split_length: int = DEFAULT_SPLIT_LENGTH,
    split_overlap: int = DEFAULT_SPLIT_OVERLAP,
) -> Pipeline:
    pdf_options = PdfPipelineOptions()
    if hasattr(pdf_options, "do_ocr"):
        pdf_options.do_ocr = True
    if hasattr(pdf_options, "do_table_structure"):
        pdf_options.do_table_structure = True
    if hasattr(pdf_options, "generate_picture_images"):
        pdf_options.generate_picture_images = True
    if hasattr(pdf_options, "generate_page_images"):
        pdf_options.generate_page_images = False
    if hasattr(pdf_options, "images_scale"):
        pdf_options.images_scale = 2.0
    if hasattr(pdf_options, "artifacts_path"):
        pdf_options.artifacts_path = DEFAULT_DOCLING_MODEL_DIR

    docling_converter = DoclingDocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options)
        }
    )

    pipe = Pipeline()
    pipe.add_component(
        "converter",
        DoclingConverter(
            converter=docling_converter,
            meta_extractor=PageProvenanceMetaExtractor(),
        ),
    )
    pipe.add_component(
        "splitter",
        DocumentSplitter(
            split_by="token",
            split_length=split_length,
            split_overlap=split_overlap,
            tokenizer_encoding="o200k_base",
        ),
    )
    pipe.add_component("embedder", BGEM3OnePassEmbedder(model_path=model_path))
    pipe.add_component(
        "writer",
        DocumentWriter(
            document_store=document_store,
            policy=DuplicatePolicy.OVERWRITE,
        ),
    )
    pipe.add_component(
        "bm25_writer",
        QdrantBM25Writer(collection=document_store.index),
    )
    pipe.connect("converter.documents", "splitter.documents")
    pipe.connect("splitter.documents", "embedder.documents")
    pipe.connect("embedder.documents", "writer.documents")
    pipe.connect("embedder.documents", "bm25_writer.documents")
    pipe.connect("writer.documents_written", "bm25_writer.documents_written")
    return pipe


def ingest_source(
    source: str | Path,
    *,
    collection: str = DEFAULT_NATIVE_COLLECTION,
    qdrant_url: str = DEFAULT_QDRANT_URL,
    recreate_index: bool = False,
    model_path: str | None = None,
) -> dict[str, Any]:
    preserved = preserve_source(source)
    docling_source = prepare_docling_source(preserved)
    store = create_document_store(
        collection=collection,
        url=qdrant_url,
        recreate_index=recreate_index,
    )
    pipe = build_ingestion_pipeline(document_store=store, model_path=model_path)

    base_meta = {
        "document_id": preserved["document_id"],
        "title": preserved["title"],
        "source_uri": preserved["source_uri"],
        "source_sha256": preserved["source_sha256"],
        "page_start": None,
        "page_end": None,
        "section_path": [],
        "content_type": "prose",
        "document_version": "1",
    }

    before = store.count_documents()
    result = pipe.run(
        {
            "converter": {
                "sources": [str(docling_source)],
                "meta": base_meta,
            }
        },
        include_outputs_from={"writer", "bm25_writer"},
    )
    after = store.count_documents()

    return {
        "source_sha256": preserved["source_sha256"],
        "managed_source": str(preserved["managed_source"]),
        "collection": collection,
        "count_before": before,
        "count_after": after,
        "documents_written": result.get("writer", {}).get("documents_written"),
        "bm25_vectors_written": result.get("bm25_writer", {}).get("bm25_vectors_written"),
    }
