"""Lightweight settings shared by the ingestion manager and phase workers."""

from __future__ import annotations

from pathlib import Path

from rag3.constants import ROOT

PIPELINE_SCHEMA_VERSION = "qdrant-full-gpu-v3-ocr-tablefast-no-pictures"
DEFAULT_SPLIT_LENGTH = 1024
DEFAULT_SPLIT_OVERLAP = 128
BGE_BATCH_SIZE = 8
CHUNK_GROUP_MULTIPLIER = 4
CHUNK_GROUP_SIZE = BGE_BATCH_SIZE * CHUNK_GROUP_MULTIPLIER


def find_bge_m3() -> Path:
    roots = (ROOT / "data" / "models" / "huggingface", ROOT / "data" / "models")
    for base in roots:
        if not base.exists():
            continue
        candidates: list[Path] = []
        for cfg in base.rglob("config.json"):
            parent = cfg.parent
            name = str(parent).lower()
            if ("bge-m3" in name or "bge_m3" in name or "bge--m3" in name) and (
                (parent / "tokenizer.json").exists()
                or (parent / "tokenizer_config.json").exists()
            ):
                candidates.append(parent)
        if candidates:
            return sorted(set(candidates), key=lambda p: (len(str(p)), str(p)))[0]
    raise RuntimeError("Local BGE-M3 model directory not found under data/models")


def progress_identity(source_sha256: str, collection: str) -> dict[str, str | int]:
    return {
        "source_sha256": source_sha256,
        "collection": collection,
        "pipeline_schema_version": PIPELINE_SCHEMA_VERSION,
        "split_length": DEFAULT_SPLIT_LENGTH,
        "split_overlap": DEFAULT_SPLIT_OVERLAP,
        "embedding_model": str(find_bge_m3()),
    }
