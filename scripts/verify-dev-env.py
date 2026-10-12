from __future__ import annotations
import importlib.metadata as md
import os
from pathlib import Path
import torch
from sentence_transformers import SentenceTransformer
from docling.document_converter import DocumentConverter
from haystack_integrations.components.converters.docling import DoclingConverter
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack_integrations.components.embedders.sentence_transformers import SentenceTransformersTextEmbedder

ROOT = Path(__file__).resolve().parents[1]
for pkg in ("haystack-ai","docling","docling-haystack","sentence-transformers",
            "sentence-transformers-haystack","qdrant-haystack","rapidocr","torch"):
    print(f"{pkg}={md.version(pkg)}")
assert torch.cuda.is_available()
print("GPU_NAME=" + torch.cuda.get_device_name(0))
cache = ROOT / "data" / "models" / "huggingface"
m = SentenceTransformer("BAAI/bge-m3", device="cpu", cache_folder=str(cache), local_files_only=True)
v = m.encode(["RAG environment verification"], show_progress_bar=False)
assert v.shape[1] == 1024
print("BGE_M3_DIM=1024")
_ = DocumentConverter, DoclingConverter, QdrantDocumentStore, SentenceTransformersTextEmbedder
print("DEV_ENV_CORE_VERIFICATION=OK")
