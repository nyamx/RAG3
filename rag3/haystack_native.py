from __future__ import annotations

import hashlib
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from FlagEmbedding import BGEM3FlagModel
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter as DoclingDocumentConverter, PdfFormatOption
from haystack import Document, Pipeline, component
from haystack.dataclasses import SparseEmbedding
from haystack.document_stores.types import DuplicatePolicy
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack_integrations.components.converters.docling import DoclingConverter
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

from rag3.source_identity import sha256_file

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QDRANT_URL = "http://127.0.0.1:16333"
DEFAULT_NATIVE_COLLECTION = "rag3_haystack"
DEFAULT_SPLIT_LENGTH = 1024
DEFAULT_SPLIT_OVERLAP = 128
DEFAULT_DOCLING_MODEL_DIR = ROOT / "data" / "models" / "docling"



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
                    "page_start": meta.get("page_start"),
                    "page_end": meta.get("page_end"),
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
) -> QdrantDocumentStore:
    return QdrantDocumentStore(
        url=url,
        index=collection,
        embedding_dim=1024,
        use_sparse_embeddings=True,
        similarity="cosine",
        recreate_index=recreate_index,
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
    pipe.add_component("converter", DoclingConverter(converter=docling_converter))
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
    pipe.connect("converter.documents", "splitter.documents")
    pipe.connect("splitter.documents", "embedder.documents")
    pipe.connect("embedder.documents", "writer.documents")
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
                "sources": [str(preserved["managed_source"])],
                "meta": base_meta,
            }
        }
    )
    after = store.count_documents()

    return {
        "source_sha256": preserved["source_sha256"],
        "managed_source": str(preserved["managed_source"]),
        "collection": collection,
        "count_before": before,
        "count_after": after,
        "documents_written": result.get("writer", {}).get("documents_written"),
    }
