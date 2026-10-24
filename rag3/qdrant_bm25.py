"""Docling-free metadata and native BM25 configuration."""
from __future__ import annotations
from qdrant_client import models
from rag3.qdrant_clients import get_client


def from_collection_info(info):
    raw = info.model_dump() if hasattr(info, 'model_dump') else info
    sparse = raw.get('config', {}).get('params', {}).get('sparse_vectors') or {}
    if 'bm25' not in sparse:
        raise RuntimeError("Collection is missing required sparse vector 'bm25'")
    cfg = (raw.get('config', {}).get('metadata') or {}).get('rag3', {}).get('bm25') or {}
    if not cfg:
        raise RuntimeError('Collection is missing RAG3 BM25 metadata')
    return models.Bm25Config(
        k=float(cfg.get('k',1.2)), b=float(cfg.get('b',.75)),
        avg_len=float(cfg.get('avg_len',256)),
        tokenizer=models.TokenizerType.MULTILINGUAL,
        lowercase=True, stopwords=None,
        stemmer=models.DisabledStemmerParams(type=models.NoStemmer.NONE))


def collection_bm25_config(collection, *, url, client=None, collection_info=None):
    client = client if client is not None else get_client(url=url)
    info = collection_info if collection_info is not None else client.get_collection(collection_name=collection)
    return from_collection_info(info)
