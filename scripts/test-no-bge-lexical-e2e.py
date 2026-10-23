"""Live Qdrant smoke test for dense + BM25 indexing and retrieval."""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

from qdrant_client import QdrantClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag3.collections import delete_collections
from rag3.constants import DEFAULT_QDRANT_URL
from rag3.haystack_native import ingest_source
from rag3.retrieval import search


def main() -> None:
    collection = "__rag3_dense_bm25_" + uuid.uuid4().hex[:10]
    client = QdrantClient(url=DEFAULT_QDRANT_URL)
    try:
        result = ingest_source(
            ROOT / "artifacts" / "e2e-formats" / "e2e.txt",
            collection=collection,
        )
        assert result["documents_written"] == result["bm25_vectors_written"] == 1
        info = client.get_collection(collection_name=collection)
        assert set(info.config.params.vectors) == {"text-dense"}
        assert set(info.config.params.sparse_vectors) == {"bm25"}
        points, _ = client.scroll(
            collection_name=collection,
            limit=2,
            with_vectors=True,
        )
        assert len(points) == 1
        assert set(points[0].vector) == {"text-dense", "bm25"}
        hits = search("cobalt river alpha", collection=collection, top_k=1)
        assert hits["methods"] == ["bge_m3_semantic", "bm25"]
        assert hits["hits"] and hits["hits"][0]["title"] == "e2e"
        print("LIVE_DENSE_BM25_OK")
    finally:
        if client.collection_exists(collection):
            delete_collections([collection])


if __name__ == "__main__":
    main()
