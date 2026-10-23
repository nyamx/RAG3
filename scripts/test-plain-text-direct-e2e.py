"""Check that direct text ingestion needs no Markdown parser copy."""

from __future__ import annotations

import json
import sys
import tempfile
import uuid
from pathlib import Path

from qdrant_client import QdrantClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag3.constants import DEFAULT_QDRANT_URL
from rag3.haystack_native import ingest_source
from rag3.source_identity import sha256_file


def main() -> None:
    collection = "__rag3_plain_text_" + uuid.uuid4().hex[:10]
    client = QdrantClient(url=DEFAULT_QDRANT_URL)
    with tempfile.TemporaryDirectory(prefix="rag3-plain-") as folder:
        source = Path(folder) / "source.jsonl"
        source.write_text('{"evidence":"sapphire orchard", "id":1}\n', encoding="utf-8")
        sha = sha256_file(source)
        managed = ROOT / "data" / "sources" / sha
        try:
            result = ingest_source(source, collection=collection)
            assert result["documents_written"] == result["bm25_vectors_written"] == 1
            assert client.count(collection_name=collection, exact=True).count == 1
            assert not (managed / "source.jsonl.rag3.md").exists()
            print(json.dumps({"direct_text": True, "parser_copy": False,
                              "points": result["documents_written"]}))
        finally:
            if client.collection_exists(collection):
                client.delete_collection(collection)
            (managed / source.name).unlink(missing_ok=True)
            if managed.is_dir() and not any(managed.iterdir()):
                managed.rmdir()


if __name__ == "__main__":
    main()
