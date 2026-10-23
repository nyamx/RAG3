"""Focused checks for dense + BM25 ingestion and retrieval configuration."""

from __future__ import annotations

import sys
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from haystack import Document
from haystack_integrations.document_stores.qdrant.converters import (
    convert_haystack_documents_to_qdrant_points,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag3 import haystack_native, retrieval


def test_dense_only_embedding() -> None:
    model = Mock()
    model.encode_corpus.return_value = {"dense_vecs": [[3.0, 4.0]]}
    model.encode_queries.return_value = {"dense_vecs": [[3.0, 4.0]]}
    embedder = haystack_native.BGEM3OnePassEmbedder(model_path="unused")
    embedder._model = model

    doc = embedder.run(documents=[Document(content="example")])["documents"][0]
    query = embedder.run(text="example")
    assert all(math.isclose(actual, expected, rel_tol=1e-6)
               for actual, expected in zip(doc.embedding, [0.6, 0.8], strict=True))
    assert doc.sparse_embedding is None
    point = convert_haystack_documents_to_qdrant_points(
        [doc], use_sparse_embeddings=True
    )[0]
    assert set(point.vector) == {"text-dense"}
    assert "sparse_embedding" not in query
    assert model.encode_corpus.call_args.kwargs["return_sparse"] is False
    assert model.encode_queries.call_args.kwargs["return_sparse"] is False


def test_new_collection_schema() -> None:
    client = Mock()
    client.collection_exists.return_value = False
    with patch.object(haystack_native, "QdrantClient", return_value=client), patch.object(
        haystack_native, "QdrantDocumentStore"
    ) as store:
        haystack_native.create_document_store(collection="test", url="http://unused")
    config = client.create_collection.call_args.kwargs
    assert set(config["vectors_config"]) == {"text-dense"}
    assert set(config["sparse_vectors_config"]) == {"bm25"}
    # Haystack requires this flag for its named dense vector, even without BGE sparse vectors.
    assert store.call_args.kwargs["use_sparse_embeddings"] is True


def test_retrieval_methods() -> None:
    client = Mock()
    client.collection_exists.return_value = True
    client.query_points.return_value = SimpleNamespace(points=[])
    with patch.object(retrieval, "QdrantClient", return_value=client), patch.object(
        retrieval, "_embed_query", return_value={"embedding": [1.0, 0.0]}
    ) as embed, patch.object(
        retrieval, "collection_bm25_config", return_value=haystack_native.bm25_config()
    ):
        fused = retrieval.search("example", collection="test")
        assert fused["methods"] == ["bge_m3_semantic", "bm25"]
        prefetch = client.query_points.call_args.kwargs["prefetch"]
        assert [item.using for item in prefetch] == ["text-dense", "bm25"]

        embed.reset_mock()
        bm25 = retrieval.search("example", semantic=False, bm25=True,
                                rank_fusion=False, collection="test")
        assert bm25["methods"] == ["bm25"]
        embed.assert_not_called()


if __name__ == "__main__":
    test_dense_only_embedding()
    test_new_collection_schema()
    test_retrieval_methods()
    print("dense + BM25 checks passed")
