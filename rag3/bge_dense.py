"""Shared BGE-M3 dense encoding semantics, with heavyweight imports deferred."""
from __future__ import annotations


def encode_dense(model, texts: list[str], *, query: bool, batch_size: int = 8,
                 max_length: int = 8192) -> list[list[float]]:
    # Keep this identical for the legacy Haystack component and Stage B GPU owner.
    import numpy as np

    kwargs = dict(batch_size=batch_size, max_length=max_length, return_dense=True,
                  return_sparse=False, return_colbert_vecs=False)
    if query and hasattr(model, 'encode_queries'):
        output = model.encode_queries(texts, **kwargs)
    elif hasattr(model, 'encode_corpus'):
        output = model.encode_corpus(texts, **kwargs)
    else:
        output = model.encode(texts, **kwargs)
    vectors = np.asarray(output['dense_vecs'], dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    if vectors.ndim != 2 or vectors.shape[0] != len(texts) or vectors.shape[1] == 0:
        raise ValueError('BGE-M3 returned unexpected dense vector shape')
    norm = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.where(norm == 0, 1.0, norm)
    if not np.isfinite(vectors).all():
        raise ValueError('BGE-M3 returned nonfinite dense vectors')
    return vectors.tolist()
