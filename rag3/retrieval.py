from __future__ import annotations

import threading
from typing import Literal

from haystack import Pipeline
from haystack_integrations.components.retrievers.qdrant import (
    QdrantEmbeddingRetriever,
    QdrantHybridRetriever,
    QdrantSparseEmbeddingRetriever,
)
from haystack_integrations.document_stores.qdrant.converters import (
    DENSE_VECTORS_NAME,
    SPARSE_VECTORS_NAME,
)
from qdrant_client import QdrantClient, models as qdrant_models

from rag3.haystack_native import (
    BGEM3OnePassEmbedder,
    BM25_VECTOR_NAME,
    DEFAULT_NATIVE_COLLECTION,
    DEFAULT_QDRANT_URL,
    collection_bm25_config,
    create_document_store,
)

Mode = Literal["dense", "sparse", "hybrid"]

SEMANTIC = "bge_m3_semantic"
BGE_LEXICAL = "bge_m3_lexical"
BM25_LEXICAL = "bm25"
_QUERY_EMBEDDERS: dict[str | None, BGEM3OnePassEmbedder] = {}
_QUERY_EMBEDDER_LOCK = threading.RLock()


def _embed_query(query: str, model_path: str | None) -> dict:
    """Reuse the core BGE-M3 embedder across searches in a long-lived process."""
    with _QUERY_EMBEDDER_LOCK:
        embedder = _QUERY_EMBEDDERS.get(model_path)
        if embedder is None:
            embedder = BGEM3OnePassEmbedder(model_path=model_path)
            _QUERY_EMBEDDERS[model_path] = embedder
        return embedder.run(text=query)


def build_retrieval_pipeline(
    *,
    mode: Mode = "hybrid",
    collection: str = DEFAULT_NATIVE_COLLECTION,
    qdrant_url: str = DEFAULT_QDRANT_URL,
    model_path: str | None = None,
    top_k: int = 10,
) -> Pipeline:
    " ""Compatibility pipeline for the original dense/sparse/hybrid Haystack path." ""
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


def _point_to_hit(rank: int, point, retriever: str) -> dict:
    payload = dict(point.payload or {})
    meta = dict(payload.get("meta") or {})
    return {
        "rank": rank,
        "retriever": retriever,
        "score": float(point.score) if point.score is not None else None,
        "point_id": str(point.id),
        "chunk_id": meta.get("chunk_id") or payload.get("id") or str(point.id),
        "document_id": meta.get("document_id"),
        "chunk_index": meta.get("chunk_index"),
        "document_version": meta.get("document_version"),
        "title": meta.get("title"),
        "page_start": meta.get("page_start"),
        "page_end": meta.get("page_end"),
        "section_path": meta.get("section_path") or [],
        "content_type": meta.get("content_type"),
        "source_uri": meta.get("source_uri"),
        "content": payload.get("content") or "",
    }


def _selected_methods(
    semantic: bool,
    bge_lexical: bool,
    bm25: bool,
) -> list[str]:
    methods: list[str] = []
    if semantic:
        methods.append(SEMANTIC)
    if bge_lexical:
        methods.append(BGE_LEXICAL)
    if bm25:
        methods.append(BM25_LEXICAL)
    return methods


def search(
    query: str,
    *,
    semantic: bool = True,
    bge_lexical: bool = True,
    bm25: bool = True,
    rank_fusion: bool = True,
    top_k: int = 10,
    candidate_k: int | None = None,
    collection: str = DEFAULT_NATIVE_COLLECTION,
    qdrant_url: str = DEFAULT_QDRANT_URL,
    model_path: str | None = None,
    mode: Mode | None = None,
) -> dict:
    " ""Search selected representations; optionally fuse them with Qdrant-native RRF." ""
    if not query or not query.strip():
        raise ValueError("query must not be empty")
    if not 1 <= top_k <= 100:
        raise ValueError("top_k must be between 1 and 100")

    if mode is not None:
        if mode == "dense":
            semantic, bge_lexical, bm25, rank_fusion = True, False, False, False
        elif mode == "sparse":
            semantic, bge_lexical, bm25, rank_fusion = False, True, False, False
        elif mode == "hybrid":
            semantic, bge_lexical, bm25, rank_fusion = True, True, False, True
        else:
            raise ValueError(f"Unsupported retrieval mode: {mode}")

    methods = _selected_methods(semantic, bge_lexical, bm25)
    if not methods:
        raise ValueError("select at least one retrieval method")
    if rank_fusion and len(methods) < 2:
        raise ValueError("rank_fusion requires at least two retrieval methods")

    client = QdrantClient(url=qdrant_url)
    if not client.collection_exists(collection):
        raise ValueError(f"collection does not exist: {collection}")

    dense_query: list[float] | None = None
    sparse_query: qdrant_models.SparseVector | None = None
    if semantic or bge_lexical:
        embedded = _embed_query(query, model_path)
        dense_query = list(embedded["embedding"])
        sparse = embedded["sparse_embedding"]
        sparse_query = qdrant_models.SparseVector(
            indices=list(sparse.indices),
            values=list(sparse.values),
        )

    bm25_query = None
    if bm25:
        bm25_query = qdrant_models.Document(
            text=query,
            model="qdrant/bm25",
            options=collection_bm25_config(collection, url=qdrant_url),
        )

    query_by_method = {
        SEMANTIC: (dense_query, DENSE_VECTORS_NAME),
        BGE_LEXICAL: (sparse_query, SPARSE_VECTORS_NAME),
        BM25_LEXICAL: (bm25_query, BM25_VECTOR_NAME),
    }

    if rank_fusion:
        effective_candidate_k = candidate_k if candidate_k is not None else top_k * 3
        if effective_candidate_k < top_k:
            raise ValueError("candidate_k must be >= top_k")

        prefetch = [
            qdrant_models.Prefetch(
                query=query_by_method[method][0],
                using=query_by_method[method][1],
                limit=effective_candidate_k,
            )
            for method in methods
        ]
        response = client.query_points(
            collection_name=collection,
            prefetch=prefetch,
            query=qdrant_models.FusionQuery(fusion=qdrant_models.Fusion.RRF),
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )
        hits = [
            _point_to_hit(i + 1, point, "rrf")
            for i, point in enumerate(response.points)
        ]
        return {
            "query": query,
            "collection": collection,
            "methods": methods,
            "rank_fusion": True,
            "top_k": top_k,
            "candidate_k": effective_candidate_k,
            "hits": hits,
        }

    hits: list[dict] = []
    for method in methods:
        query_value, vector_name = query_by_method[method]
        response = client.query_points(
            collection_name=collection,
            query=query_value,
            using=vector_name,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )
        hits.extend(
            _point_to_hit(i + 1, point, method)
            for i, point in enumerate(response.points)
        )

    return {
        "query": query,
        "collection": collection,
        "methods": methods,
        "rank_fusion": False,
        "top_k": top_k,
        "candidate_k": None,
        "hits": hits,
    }
