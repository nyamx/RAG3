"""Docling-free Qdrant/Haystack collection initialization."""
from __future__ import annotations
from qdrant_client import models as qdrant_models
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from rag3.constants import DEFAULT_NATIVE_COLLECTION, DEFAULT_QDRANT_URL
from rag3.qdrant_clients import get_client
from rag3.qdrant_bm25 import from_collection_info
BM25_VECTOR_NAME='bm25'
DEFAULT_BM25_K=1.2
DEFAULT_BM25_B=.75
DEFAULT_BM25_AVG_LEN=256.0

def _bm25_metadata(k,b,avg_len):
    return {'rag3':{'schema_version':3,'bm25':{'vector_name':'bm25','model':'qdrant/bm25',
        'source':'content','k':float(k),'b':float(b),'avg_len':float(avg_len),
        'tokenizer':'multilingual','lowercase':True,'stemming':False,'stopwords':False}}}

def _collection_bm25_settings(client,collection):
    cfg = from_collection_info(client.get_collection(collection_name=collection))
    return cfg.k, cfg.b, cfg.avg_len

def create_document_store(
    *,
    collection: str = DEFAULT_NATIVE_COLLECTION,
    url: str = DEFAULT_QDRANT_URL,
    recreate_index: bool = False,
    bm25_k: float = DEFAULT_BM25_K,
    bm25_b: float = DEFAULT_BM25_B,
    bm25_avg_len: float = DEFAULT_BM25_AVG_LEN,
) -> QdrantDocumentStore:
    from rag3.ingestion_jobs import mark_change

    client = get_client(url=url)

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
                BM25_VECTOR_NAME: qdrant_models.SparseVectorParams(
                    modifier=qdrant_models.Modifier.IDF
                ),
            },
            metadata=_bm25_metadata(bm25_k, bm25_b, bm25_avg_len),
        )
        mark_change('collections')
    else:
        _collection_bm25_settings(client, collection)

    return QdrantDocumentStore(
        url=url,
        index=collection,
        embedding_dim=1024,
        # Haystack uses this flag to address the named "text-dense" vector.
        # Documents carry no BGE sparse_embedding, so none is written.
        use_sparse_embeddings=True,
        similarity="cosine",
        recreate_index=False,
        return_embedding=False,
        progress_bar=False,
    )

