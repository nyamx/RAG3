"""Lightweight BGE query/document component without Docling imports."""
from __future__ import annotations
from typing import Any
from haystack import Document, component
from rag3.ingestion_settings import BGE_BATCH_SIZE, find_bge_m3

@component
class BGEM3OnePassEmbedder:
    """Haystack component that emits only BGE-M3 dense vectors."""

    def __init__(
        self,
        model_path: str | None = None,
        batch_size: int = BGE_BATCH_SIZE,
        max_length: int = 8192,
        device: str | None = None,
    ) -> None:
        self.model_path = model_path or str(find_bge_m3())
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self.device = device
        self._model: BGEM3FlagModel | None = None

    def warm_up(self) -> None:
        if self._model is not None:
            return
        import torch
        from FlagEmbedding import BGEM3FlagModel

        device = self.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self._model = BGEM3FlagModel(
            self.model_path,
            use_fp16=device.startswith("cuda"),
            pooling_method="cls",
            devices=device,
        )

    def _encode(self, texts: list[str], *, query: bool) -> list[list[float]]:
        self.warm_up()
        if self._model is None:
            raise RuntimeError("BGE-M3 model did not initialize")
        from rag3.bge_dense import encode_dense
        return encode_dense(self._model, texts, query=query,
                            batch_size=self.batch_size, max_length=self.max_length)

    @component.output_types(
        documents=list[Document],
        embedding=list[float],
    )
    def run(
        self,
        documents: list[Document] | None = None,
        text: str | None = None,
    ) -> dict[str, Any]:
        if (documents is None) == (text is None):
            raise ValueError("Provide exactly one of documents or text")

        if text is not None:
            dense = self._encode([text], query=True)
            return {
                "documents": [],
                "embedding": dense[0],
            }

        assert documents is not None
        if not documents:
            return {
                "documents": [],
                "embedding": [],
            }

        texts = [(doc.content or "") for doc in documents]
        dense = self._encode(texts, query=False)
        embedded: list[Document] = []

        for idx, (doc, dvec) in enumerate(zip(documents, dense, strict=True)):
            from rag3.ingestion_light import normalize_chunk
            normalized = normalize_chunk(doc, idx)
            embedded.append(
                Document(
                    id=normalized.id,
                    content=normalized.content,
                    meta=normalized.meta,
                    embedding=dvec,
                )
            )

        return {
            "documents": embedded,
            "embedding": [],
        }

