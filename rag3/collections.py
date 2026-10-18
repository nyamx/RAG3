"""Collection and document operations shared by the UI worker and MCP adapter."""

from __future__ import annotations

from qdrant_client import QdrantClient, models

from rag3.constants import DEFAULT_QDRANT_URL


def list_collections(*, url: str = DEFAULT_QDRANT_URL) -> list[str]:
    return sorted(item.name for item in QdrantClient(url=url).get_collections().collections)


def create_collection(name: str, *, url: str = DEFAULT_QDRANT_URL,
                      bm25_k: float = 1.2, bm25_b: float = 0.75,
                      bm25_avg_len: float = 256.0,
                      reuse_existing: bool = False) -> dict:
    from rag3.haystack_native import create_document_store

    if QdrantClient(url=url).collection_exists(name) and not reuse_existing:
        raise ValueError(f"collection already exists: {name}")
    store = create_document_store(collection=name, url=url, recreate_index=False,
                                  bm25_k=bm25_k, bm25_b=bm25_b,
                                  bm25_avg_len=bm25_avg_len)
    return {"collection": name, "count": store.count_documents(),
            "bm25": {"k": bm25_k, "b": bm25_b, "avg_len": bm25_avg_len}}


def delete_collections(names: list[str], *, url: str = DEFAULT_QDRANT_URL) -> dict:
    if not names:
        raise ValueError("at least one collection is required")
    client = QdrantClient(url=url)
    existing = set(list_collections(url=url))
    missing = [name for name in names if name not in existing]
    if missing:
        raise ValueError(f"collection(s) do not exist: {', '.join(missing)}")
    for name in names:
        client.delete_collection(collection_name=name)
    return {"deleted": names}


def collection_contents(collection: str, *, url: str = DEFAULT_QDRANT_URL,
                        create_if_missing: bool = False) -> dict:
    """Use the same Haystack document view for the UI and MCP."""
    from rag3.haystack_native import create_document_store

    if not create_if_missing and not QdrantClient(url=url).collection_exists(collection):
        raise ValueError(f"collection does not exist: {collection}")
    chunks = create_document_store(collection=collection, url=url,
                                   recreate_index=False).filter_documents()
    grouped: dict[str, dict] = {}
    for chunk in chunks:
        meta = dict(chunk.meta or {})
        document_id = str(meta.get("document_id") or meta.get("source_sha256")
                          or meta.get("source_uri") or chunk.id)
        item = grouped.setdefault(document_id, {
            "document_id": document_id, "title": meta.get("title"),
            "source_uri": meta.get("source_uri"),
            "source_sha256": meta.get("source_sha256"),
            "chunk_count": 0, "_page_starts": [], "_page_ends": [],
        })
        item["chunk_count"] += 1
        for key, bucket in (("page_start", "_page_starts"),
                            ("page_end", "_page_ends")):
            value = meta.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                item[bucket].append(value)
        for key in ("title", "source_uri", "source_sha256"):
            if not item.get(key) and meta.get(key):
                item[key] = meta[key]
    documents = []
    for item in grouped.values():
        starts = item.pop("_page_starts")
        ends = item.pop("_page_ends")
        item["page_start"] = min(starts) if starts else None
        item["page_end"] = max(ends) if ends else None
        documents.append(item)
    documents.sort(key=lambda item: (
        str(item.get("title") or "").casefold(),
        str(item.get("source_uri") or "").casefold(),
        item["document_id"],
    ))
    return {"collection": collection, "document_count": len(documents),
            "chunk_count": len(chunks), "documents": documents}


def get_document(collection: str, document_id: str, *, url: str = DEFAULT_QDRANT_URL,
                 chunk_offset: int = 0, chunk_limit: int = 1) -> dict:
    result = collection_contents(collection, url=url)
    for item in result["documents"]:
        if item["document_id"] == document_id:
            client = QdrantClient(url=url)
            condition = models.Filter(must=[models.FieldCondition(
                key="meta.document_id", match=models.MatchValue(value=document_id))])
            points = []
            cursor = None
            while True:
                batch, cursor = client.scroll(collection_name=collection,
                    scroll_filter=condition, limit=256, offset=cursor,
                    with_payload=True, with_vectors=False)
                points.extend(batch)
                if cursor is None:
                    break
            points.sort(key=lambda point: (
                (point.payload or {}).get("meta", {}).get("chunk_index", 0),
                str(point.id),
            ))
            selected = points[chunk_offset:chunk_offset + chunk_limit]
            chunks = [{"point_id": str(point.id),
                       "chunk_index": (point.payload or {}).get("meta", {}).get("chunk_index"),
                       "page_start": (point.payload or {}).get("meta", {}).get("page_start"),
                       "page_end": (point.payload or {}).get("meta", {}).get("page_end"),
                       "content": (point.payload or {}).get("content") or ""}
                      for point in selected]
            return {"collection": collection, **item, "chunks": chunks,
                    "chunk_offset": chunk_offset,
                    "next_chunk_offset": chunk_offset + len(chunks)
                    if chunk_offset + len(chunks) < len(points) else None}
    raise ValueError(f"document does not exist: {document_id}")


def delete_document(collection: str, document_id: str, *, url: str = DEFAULT_QDRANT_URL) -> dict:
    # Resolve first so a typo never becomes a silent successful deletion.
    item = get_document(collection, document_id, url=url)
    client = QdrantClient(url=url)
    condition = models.Filter(must=[models.FieldCondition(
        key="meta.document_id", match=models.MatchValue(value=document_id))])
    client.delete(collection_name=collection,
                  points_selector=models.FilterSelector(filter=condition), wait=True)
    return {"collection": collection, "document_id": document_id,
            "chunks_deleted": item["chunk_count"]}
