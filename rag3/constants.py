"""Lightweight paths and defaults shared by RAG3 entry points."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QDRANT_URL = "http://127.0.0.1:16333"
DEFAULT_NATIVE_COLLECTION = "rag3_haystack"
UTF8_TEXT_SUFFIXES = frozenset({".txt", ".json", ".jsonl", ".csv", ".tsv"})
MCP_FILE_SUFFIXES = frozenset({".pdf", ".doc", ".docx", ".md"}) | UTF8_TEXT_SUFFIXES
