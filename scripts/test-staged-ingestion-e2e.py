"""Exercise durable Qdrant staging and resume with the real phase workers."""

from __future__ import annotations

import json
import importlib.util
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qdrant_client import QdrantClient, models

from rag3.constants import DEFAULT_QDRANT_URL
from rag3.haystack_native import BGEM3OnePassEmbedder, create_document_store
from rag3.ingestion_jobs import (clear_ingestion_progress, ingestion_progress,
                                 set_ingestion_progress)
from rag3.ingestion_settings import CHUNK_GROUP_SIZE, progress_identity
from rag3.source_identity import sha256_file


def run_worker(source: Path, collection: str, pause_on_qdrant: bool) -> tuple[int, list[dict]]:
    proc = subprocess.Popen(
        [sys.executable, str(ROOT / "scripts" / "ingest-worker.py"), str(source),
         "--collection", collection, "--mode", "run" if pause_on_qdrant else "resume"],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
    )
    events = []
    assert proc.stdout is not None
    for raw in proc.stdout:
        try:
            event = json.loads(raw)
        except ValueError:
            continue
        events.append(event)
        if (pause_on_qdrant and event.get("type") == "log"
                and event.get("message") == "STAGE qdrant PASS"):
            marker = ROOT / "data" / "checkpoints" / sha256_file(source) / "pause.request"
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text("pause\n", encoding="utf-8")
    return proc.wait(timeout=120), events


def main() -> None:
    collection = "__rag3_staged_resume_" + uuid.uuid4().hex[:10]
    client = QdrantClient(url=DEFAULT_QDRANT_URL)
    with tempfile.TemporaryDirectory(prefix="rag3-staged-") as directory:
        source = Path(directory) / "resume.txt"
        source.write_text("RAG3 staged resume " + uuid.uuid4().hex + "\n", encoding="utf-8")
        sha = sha256_file(source)
        try:
            create_document_store(collection=collection)
            code, events = run_worker(source, collection, True)
            assert code == 75, (code, events[-3:])
            assert any(e.get("type") == "result" and
                       (e.get("result") or {}).get("status") == "PAUSED" for e in events)
            progress = ingestion_progress(collection, sha)
            assert progress and progress["phase"] == "CHUNKS_STORED", progress
            assert not (ROOT / "data" / "checkpoints" / sha / "snapshot.json").exists()

            code, events = run_worker(source, collection, False)
            assert code == 0, (code, events[-3:])
            assert not any(e.get("message") == "STAGE parse RUN" for e in events)
            result = next(e["result"] for e in events if e.get("type") == "result")
            assert result["status"] == "COMPLETE" and result["source_point_count"] == 1
            assert result["source_dense_count"] == result["source_bm25_count"] == 1
            test_bounded_groups(client, collection, Path(directory))
            print(json.dumps({"pause_resume": True, "snapshot_created": False,
                              "parse_repeated": False, "verified_points": 1,
                              "bounded_groups": [CHUNK_GROUP_SIZE, 1]}))
        finally:
            clear_ingestion_progress(collection, sha)
            (ROOT / "data" / "checkpoints" / sha / "pause.request").unlink(missing_ok=True)
            if client.collection_exists(collection):
                client.delete_collection(collection)
            managed = ROOT / "data" / "sources" / sha
            for name in ("resume.txt", "resume.txt.rag3.md"):
                (managed / name).unlink(missing_ok=True)
            if managed.is_dir() and not any(managed.iterdir()):
                managed.rmdir()


def test_bounded_groups(client: QdrantClient, collection: str, folder: Path) -> None:
    """Prove that a 33-point source enters BGE in 32- and 1-point calls."""
    source = folder / "groups.txt"
    source.write_text("bounded groups " + uuid.uuid4().hex, encoding="utf-8")
    sha = sha256_file(source)
    count = CHUNK_GROUP_SIZE + 1
    points = [models.PointStruct(
        id=str(uuid.uuid4()), vector={},
        payload={"content": f"short chunk {idx}",
                 "meta": {"source_sha256": sha, "_rag3_ingest_state": "parsed"}},
    ) for idx in range(count)]
    client.upsert(collection_name=collection, points=points, wait=True)
    set_ingestion_progress(collection, sha, progress_identity(sha, collection),
                           "CHUNKS_STORED", count)
    spec = importlib.util.spec_from_file_location("rag3_staged_worker", ROOT / "scripts" / "ingest-worker.py")
    assert spec and spec.loader
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    calls = []

    def fake_encode(self, texts, *, query):
        assert not query
        calls.append(len(texts))
        return [[1.0] + [0.0] * 1023 for _ in texts]

    try:
        with patch.object(BGEM3OnePassEmbedder, "_encode", fake_encode):
            result = worker.embed_source(source, collection)
        assert calls == [CHUNK_GROUP_SIZE, 1], calls
        assert result["source_point_count"] == count
        assert result["source_dense_count"] == result["source_bm25_count"] == count
    finally:
        clear_ingestion_progress(collection, sha)


if __name__ == "__main__":
    main()
