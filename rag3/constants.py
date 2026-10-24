"""Lightweight paths and defaults shared by RAG3 entry points."""

from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]

# Explicitly named, isolated backend tests may redirect *all* mutable job and
# original-source storage. Never route production data through this override.
_test_data = os.environ.get('RAG3_TEST_DATA_ROOT')
if _test_data:
    if not os.environ.get('RAG3_BACKEND_TEST_INSTANCE'):
        raise RuntimeError('RAG3_TEST_DATA_ROOT requires RAG3_BACKEND_TEST_INSTANCE')
    DATA_ROOT = Path(_test_data).expanduser().resolve()
    if DATA_ROOT == (ROOT / 'data').resolve() or DATA_ROOT.is_relative_to((ROOT / 'data').resolve()):
        raise RuntimeError('Test data root must be outside the production data directory')
else:
    DATA_ROOT = ROOT / 'data'
DEFAULT_QDRANT_URL = "http://127.0.0.1:16333"
DEFAULT_NATIVE_COLLECTION = "rag3_haystack"
UTF8_TEXT_SUFFIXES = frozenset({".txt", ".json", ".jsonl", ".csv", ".tsv"})
MCP_FILE_SUFFIXES = frozenset({".pdf", ".doc", ".docx", ".md"}) | UTF8_TEXT_SUFFIXES
