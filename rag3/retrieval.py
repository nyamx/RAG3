from __future__ import annotations

from typing import Literal

from haystack import Pipeline
from haystack_integrations.components.retrievers.qdrant import (
    QdrantEmbeddingRetriever,
    QdrantHybridRetriever,
    QdrantSparseEmbeddingRetriever,
)

from rag3.haystack_native import (
    BGEM3OnePassEmbedder,
    DEFAULT_NATIVE_COLLECTION,
    DEFAULT_QDRANT_URL,
    create_document_store,
)

Mode = Literal["dense", "sparse", "hybrid"]


def build_retrieval_pipeline(
    *,
    mode: Mode = "hybrid",
    collection: str = DEFAULT_NATIVE_COLLECTION,
    qdrant_url: str = DEFAULT_QDRANT_URL,
    model_path: str | None = None,
    top_k: int = 5,
) -> Pipeline:
    store = create_document_store(
        collection=collection,
        url=qdrant_url,
        recreate_index=False,
    )

    pipe = Pipeline()
    pipe.add_component("query_embedder", BGEM3OnePassEmbedder(model_path=model_path))

    if mode == "dense":
        retriever = QdrantEmbeddingRetriever(
            document_store=store,
            top_k=top_k,
            return_embedding=False,
        )
        pipe.add_component("retriever", retriever)
        pipe.connect("query_embedder.embedding", "retriever.query_embedding")
    elif mode == "sparse":
        retriever = QdrantSparseEmbeddingRetriever(
            document_store=store,
            top_k=top_k,
            return_embedding=False,
        )
        pipe.add_component("retriever", retriever)
        pipe.connect("query_embedder.sparse_embedding", "retriever.query_sparse_embedding")
    elif mode == "hybrid":
        retriever = QdrantHybridRetriever(
            document_store=store,
            top_k=top_k,
            return_embedding=False,
        )
        pipe.add_component("retriever", retriever)
        pipe.connect("query_embedder.embedding", "retriever.query_embedding")
        pipe.connect("query_embedder.sparse_embedding", "retriever.query_sparse_embedding")
    else:
        raise ValueError(f"Unsupported retrieval mode: {mode}")

    return pipe


def _document_to_hit(rank: int, doc) -> dict:
    meta = dict(doc.meta or {})
    return {
        "rank": rank,
        "score": float(doc.score) if doc.score is not None else None,
        "chunk_id": meta.get("chunk_id") or doc.id,
        "document_id": meta.get("document_id"),
        "title": meta.get("title"),
        "page_start": meta.get("page_start"),
        "page_end": meta.get("page_end"),
        "section_path": meta.get("section_path") or [],
        "content_type": meta.get("content_type"),
        "source_uri": meta.get("source_uri"),
        "content": doc.content or "",
    }


def search(
    query: str,
    *,
    mode: Mode = "hybrid",
    top_k: int = 5,
    candidate_k: int | None = None,
    collection: str = DEFAULT_NATIVE_COLLECTION,
    qdrant_url: str = DEFAULT_QDRANT_URL,
    model_path: str | None = None,
) -> dict:
    if not query or not query.strip():
        raise ValueError("query must not be empty")
    if top_k < 1:
        raise ValueError("top_k must be >= 1")

    # candidate_k is accepted only for backward call compatibility.
    # Native QdrantHybridRetriever performs its own dense+sparse candidate retrieval and RRF fusion.
    pipe = build_retrieval_pipeline(
        mode=mode,
        collection=collection,
        qdrant_url=qdrant_url,
        model_path=model_path,
        top_k=top_k,
    )
    result = pipe.run({"query_embedder": {"text": query}})
    docs = result.get("retriever", {}).get("documents", [])

    return {
        "query": query,
        "mode": mode,
        "collection": collection,
        "top_k": top_k,
        "candidate_k": candidate_k,
        "hits": [_document_to_hit(i + 1, doc) for i, doc in enumerate(docs)],
    }
