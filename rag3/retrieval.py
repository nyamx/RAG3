from __future__ import annotations

import threading
from typing import Callable, Literal

from haystack import Pipeline
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever
from haystack_integrations.document_stores.qdrant.converters import DENSE_VECTORS_NAME
from qdrant_client import QdrantClient, models as qdrant_models

from rag3.constants import DEFAULT_NATIVE_COLLECTION, DEFAULT_QDRANT_URL
from rag3.retrieval_embedding import BGEM3OnePassEmbedder
from rag3.qdrant_bm25 import collection_bm25_config
from rag3.qdrant_clients import get_client
BM25_VECTOR_NAME = 'bm25'

Mode = Literal["dense", "sparse", "hybrid"]

SEMANTIC = "bge_m3_semantic"
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
    mode: Mode = "dense",
    collection: str = DEFAULT_NATIVE_COLLECTION,
    qdrant_url: str = DEFAULT_QDRANT_URL,
    model_path: str | None = None,
    top_k: int = 10,
) -> Pipeline:
    """Build the Haystack dense path; use search() for BM25 or fused retrieval."""
    if mode != "dense":
        raise ValueError("Use search() for BM25 or semantic+BM25 retrieval")
    from rag3.qdrant_store import create_document_store
    store = create_document_store(
        collection=collection,
        url=qdrant_url,
        recreate_index=False,
    )

    pipe = Pipeline()
    pipe.add_component("query_embedder", BGEM3OnePassEmbedder(model_path=model_path))

    retriever = QdrantEmbeddingRetriever(
        document_store=store,
        top_k=top_k,
        return_embedding=False,
    )
    pipe.add_component("retriever", retriever)
    pipe.connect("query_embedder.embedding", "retriever.query_embedding")

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
        "section_paths": meta.get("section_paths") or [],
        "content_type": meta.get("content_type"),
        "source_uri": meta.get("source_uri"),
        "content": payload.get("content") or "",
    }


def _selected_methods(
    semantic: bool,
    bm25: bool,
) -> list[str]:
    methods: list[str] = []
    if semantic:
        methods.append(SEMANTIC)
    if bm25:
        methods.append(BM25_LEXICAL)
    return methods


def search(
    query: str,
    *,
    semantic: bool = True,
    bm25: bool = True,
    rank_fusion: bool = True,
    top_k: int = 10,
    candidate_k: int | None = None,
    metadata_filter: dict | qdrant_models.Filter | None = None,
    collection: str = DEFAULT_NATIVE_COLLECTION,
    qdrant_url: str = DEFAULT_QDRANT_URL,
    model_path: str | None = None,
    mode: Mode | None = None,
    dense_encoder: Callable[[str], list[float]] | None = None,
) -> dict:
    " ""Search selected representations; optionally fuse them with Qdrant-native RRF." ""
    if not query or not query.strip():
        raise ValueError("query must not be empty")
    if not 1 <= top_k <= 100:
        raise ValueError("top_k must be between 1 and 100")

    if mode is not None:
        if mode == "dense":
            semantic, bm25, rank_fusion = True, False, False
        elif mode == "sparse":
            semantic, bm25, rank_fusion = False, True, False
        elif mode == "hybrid":
            semantic, bm25, rank_fusion = True, True, True
        else:
            raise ValueError(f"Unsupported retrieval mode: {mode}")

    methods = _selected_methods(semantic, bm25)
    if not methods:
        raise ValueError("select at least one retrieval method")
    if rank_fusion and len(methods) < 2:
        raise ValueError("rank_fusion requires at least two retrieval methods")

    query_filter = None
    if metadata_filter is not None:
        if isinstance(metadata_filter, qdrant_models.Filter):
            query_filter = metadata_filter
        elif isinstance(metadata_filter, dict) and metadata_filter:
            if not set(metadata_filter) <= {"must", "should", "must_not", "min_should"}:
                raise ValueError("metadata_filter must use Qdrant filter clauses")
            query_filter = qdrant_models.Filter.model_validate(metadata_filter)
        else:
            raise ValueError("metadata_filter must be a non-empty Qdrant filter object")
        if not any((query_filter.must, query_filter.should, query_filter.must_not, query_filter.min_should)):
            raise ValueError("metadata_filter must contain at least one condition")

    client = get_client(url=qdrant_url)
    from qdrant_client.http.exceptions import UnexpectedResponse
    try:
        info = client.get_collection(collection_name=collection)
    except UnexpectedResponse as exc:
        if exc.status_code == 404:
            raise ValueError(f"collection does not exist: {collection}") from exc
        raise

    dense_query: list[float] | None = None
    if semantic:
        if dense_encoder is None:
            embedded = _embed_query(query, model_path)
            dense_query = list(embedded["embedding"])
        else:
            dense_query = list(dense_encoder(query))

    bm25_query = None
    if bm25:
        bm25_query = qdrant_models.Document(
            text=query,
            model="qdrant/bm25",
            options=collection_bm25_config(collection, url=qdrant_url, client=client, collection_info=info),
        )

    query_by_method = {
        SEMANTIC: (dense_query, DENSE_VECTORS_NAME),
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
                filter=query_filter,
                limit=effective_candidate_k,
            )
            for method in methods
        ]
        # Native RRF may return equal-score points in a different order across
        # otherwise identical requests. Retrieve the complete fusion candidate
        # pool, break ties by stable Qdrant point ID, then apply final top_k.
        # Sorting only the first top_k server results leaves cut-line ties unstable.
        response = client.query_points(
            collection_name=collection,
            prefetch=prefetch,
            query=qdrant_models.FusionQuery(fusion=qdrant_models.Fusion.RRF),
            limit=effective_candidate_k * len(methods),
            with_payload=True,
            with_vectors=False,
        )
        ordered_points = sorted(
            response.points,
            key=lambda point: (
                -(float(point.score) if point.score is not None else float('-inf')),
                str(point.id),
            ),
        )[:top_k]
        hits = [
            _point_to_hit(i + 1, point, "rrf")
            for i, point in enumerate(ordered_points)
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

    # Stage D: both UI and MCP consume the same first-occurrence union.
    # Preserve individual method score/rank for presentation and diagnostics.
    hits: list[dict] = []
    by_point: dict[str, dict] = {}
    for method in methods:
        query_value, vector_name = query_by_method[method]
        response = client.query_points(
            collection_name=collection,
            query=query_value,
            using=vector_name,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )
        for position, point in enumerate(response.points, 1):
            key = str(point.id)
            detail = {"retriever": method, "rank": position,
                      "score": float(point.score) if point.score is not None else None}
            existing = by_point.get(key)
            if existing is not None:
                existing["method_details"].append(detail)
                continue
            hit = _point_to_hit(len(hits) + 1, point, method)
            hit["method_details"] = [detail]
            by_point[key] = hit
            hits.append(hit)

    return {
        "query": query,
        "collection": collection,
        "methods": methods,
        "rank_fusion": False,
        "top_k": top_k,
        "candidate_k": None,
        "hits": hits,
    }
