"""Collection and document operations shared by the UI worker and MCP adapter."""

from __future__ import annotations

import re
from typing import Literal
from uuid import UUID

from qdrant_client import QdrantClient, models

from rag3.constants import DEFAULT_QDRANT_URL
from rag3.ingestion_jobs import content_scope, mark_change


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
        mark_change('collections')
        mark_change(content_scope(name))
    return {"deleted": names}


def collection_contents(collection: str, *, url: str = DEFAULT_QDRANT_URL,
                        create_if_missing: bool = False) -> dict:
    """Read only the payload fields needed for the UI and MCP document list."""
    client = QdrantClient(url=url)
    if not client.collection_exists(collection):
        if not create_if_missing:
            raise ValueError(f"collection does not exist: {collection}")
        from rag3.haystack_native import create_document_store
        create_document_store(collection=collection, url=url, recreate_index=False)
    grouped: dict[str, dict] = {}
    chunk_count = 0
    cursor = None
    while True:
        points, cursor = client.scroll(
            collection_name=collection, limit=256, offset=cursor,
            with_payload=models.PayloadSelectorInclude(include=["meta", "id"]),
            with_vectors=False,
        )
        chunk_count += len(points)
        for point in points:
            payload = point.payload or {}
            meta = dict(payload.get("meta") or {})
            document_id = str(meta.get("document_id") or meta.get("source_sha256")
                              or meta.get("source_uri") or payload.get("id") or point.id)
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
        if cursor is None:
            break
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
            "chunk_count": chunk_count, "documents": documents}


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
                       "section_paths": (point.payload or {}).get("meta", {}).get("section_paths") or [],
                       "content": (point.payload or {}).get("content") or ""}
                      for point in selected]
            return {"collection": collection, **item, "chunks": chunks,
                    "chunk_offset": chunk_offset,
                    "next_chunk_offset": chunk_offset + len(chunks)
                    if chunk_offset + len(chunks) < len(points) else None}
    raise ValueError(f"document does not exist: {document_id}")


_SECTION_NUMBER = re.compile(r"(?:0|[1-9][0-9]*)(?:\.(?:0|[1-9][0-9]*))*\Z")


def _point_id(value: object) -> int | str:
    if not isinstance(value, str) or not value:
        raise ValueError("point_id must be a Qdrant point ID from rag3_get_document")
    if value.isascii() and value.isdecimal():
        return int(value)
    try:
        parsed = str(UUID(value))
    except ValueError as exc:
        raise ValueError("point_id must be a Qdrant point ID from rag3_get_document") from exc
    if parsed != value.lower():
        raise ValueError("point_id must be a Qdrant point ID from rag3_get_document")
    return parsed


def _section_key(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split("."))


def _section_paths(value: object) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(path, str) or not _SECTION_NUMBER.fullmatch(path)
        for path in value
    ):
        raise ValueError("section_paths must be numeric dotted strings, such as 2.3 or 2.3.1")
    return sorted(set(value), key=_section_key)


def _section_scopes(paths: list[str]) -> list[str]:
    scopes = {".".join(path.split(".")[:depth])
              for path in paths for depth in range(1, len(path.split(".")) + 1)}
    return sorted(scopes, key=_section_key)


def update_chunk_sections(collection: str, document_id: str, updates: list[dict], *,
                          mode: Literal["add", "replace"] = "add", dry_run: bool = True,
                          url: str = DEFAULT_QDRANT_URL) -> dict:
    """Annotate existing points without changing content, vectors, or other metadata."""
    if not isinstance(document_id, str) or not document_id:
        raise ValueError("document_id is required")
    if mode not in ("add", "replace"):
        raise ValueError("mode must be add or replace")
    if not isinstance(dry_run, bool):
        raise ValueError("dry_run must be a boolean")
    if not isinstance(updates, list) or not 1 <= len(updates) <= 100:
        raise ValueError("updates must contain 1 to 100 chunks")
    requested: dict[int | str, list[str]] = {}
    for update in updates:
        if not isinstance(update, dict) or set(update) != {"point_id", "section_paths"}:
            raise ValueError("each update requires only point_id and section_paths")
        point_id = _point_id(update["point_id"])
        if point_id in requested:
            raise ValueError(f"duplicate point_id: {point_id}")
        paths = _section_paths(update["section_paths"])
        if mode == "add" and not paths:
            raise ValueError("add mode requires at least one section path per chunk")
        requested[point_id] = paths

    client = QdrantClient(url=url)
    records = client.retrieve(collection_name=collection, ids=list(requested),
                              with_payload=True, with_vectors=False)
    by_id = {str(record.id): record for record in records}
    if set(by_id) != {str(point_id) for point_id in requested}:
        raise ValueError("one or more point IDs do not exist in this collection")

    changes = []
    for point_id, paths in requested.items():
        record = by_id[str(point_id)]
        meta = (record.payload or {}).get("meta")
        if not isinstance(meta, dict) or meta.get("document_id") != document_id:
            raise ValueError(f"point {point_id} does not belong to document {document_id}")
        previous = _section_paths(meta.get("section_paths") or [])
        revised = _section_paths(previous + paths if mode == "add" else paths)
        scopes = _section_scopes(revised)
        changed = previous != revised or meta.get("section_scopes") != scopes
        changes.append({"point_id": str(point_id), "chunk_index": meta.get("chunk_index"),
                        "previous_section_paths": previous, "section_paths": revised,
                        "section_scopes": scopes, "changed": changed})

    if not dry_run:
        for change in changes:
            if change["changed"]:
                client.set_payload(collection_name=collection,
                                   payload={"section_paths": change["section_paths"],
                                            "section_scopes": change["section_scopes"]},
                                   points=[_point_id(change["point_id"])], key="meta", wait=True)
        if any(change["changed"] for change in changes):
            mark_change(content_scope(collection))
    return {"collection": collection, "document_id": document_id, "mode": mode,
            "dry_run": dry_run, "changed_count": sum(change["changed"] for change in changes),
            "chunks": changes}


def delete_document(collection: str, document_id: str, *, url: str = DEFAULT_QDRANT_URL) -> dict:
    # Resolve first so a typo never becomes a silent successful deletion.
    item = get_document(collection, document_id, url=url)
    client = QdrantClient(url=url)
    condition = models.Filter(must=[models.FieldCondition(
        key="meta.document_id", match=models.MatchValue(value=document_id))])
    client.delete(collection_name=collection,
                  points_selector=models.FilterSelector(filter=condition), wait=True)
    mark_change(content_scope(collection))
    return {"collection": collection, "document_id": document_id,
            "chunks_deleted": item["chunk_count"]}
