"""Exercise whole-document parsing followed by GPU embedding on a disposable PDF."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

import psutil
from qdrant_client import QdrantClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag3.constants import DEFAULT_QDRANT_URL
from rag3.haystack_native import create_document_store
from rag3.ingestion_jobs import clear_ingestion_progress
from rag3.source_identity import sha256_file


def main() -> None:
    spec = importlib.util.spec_from_file_location(
        "rag3_page_range_test", ROOT / "scripts" / "test-docling-page-range.py"
    )
    assert spec and spec.loader
    page_test = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(page_test)

    collection = "__rag3_full_pdf_" + uuid.uuid4().hex[:10]
    client = QdrantClient(url=DEFAULT_QDRANT_URL)
    with tempfile.TemporaryDirectory(prefix="rag3-full-pdf-") as directory:
        source = Path(directory) / "nine.pdf"
        page_test.make_pdf(source, page_count=9)
        sha = sha256_file(source)
        try:
            create_document_store(collection=collection)
            proc = subprocess.Popen(
                [sys.executable, str(ROOT / "scripts" / "ingest-worker.py"),
                 str(source), "--collection", collection],
                cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1,
            )
            events = []

            def read_events() -> None:
                assert proc.stdout is not None
                for line in proc.stdout:
                    try:
                        events.append(json.loads(line))
                    except ValueError:
                        pass

            reader = threading.Thread(target=read_events, daemon=True)
            reader.start()
            started = time.monotonic()
            peak_rss = 0
            owner = psutil.Process(proc.pid)
            while proc.poll() is None:
                try:
                    processes = [owner, *owner.children(recursive=True)]
                    rss = sum(p.memory_info().rss for p in processes if p.is_running())
                    peak_rss = max(peak_rss, rss)
                except psutil.Error:
                    pass
                time.sleep(.1)
            reader.join(timeout=10)
            assert proc.returncode == 0, (proc.returncode, events[-5:])
            logs = [e.get("message") for e in events if e.get("type") == "log"]
            assert logs.index("STAGE parse PASS") < logs.index("STAGE embedding RUN"), logs
            result = next(e["result"] for e in events if e.get("type") == "result")
            total = result["source_point_count"]
            assert total >= 1 and total == result["source_dense_count"] == result["source_bm25_count"]
            print(json.dumps({"full_document_gpu": True, "pages": 9, "chunks": total,
                              "seconds": round(time.monotonic() - started, 1),
                              "peak_worker_rss_mib": round(peak_rss / 2**20, 1)}))
        finally:
            clear_ingestion_progress(collection, sha)
            if client.collection_exists(collection):
                client.delete_collection(collection)
            managed = ROOT / "data" / "sources" / sha
            (managed / source.name).unlink(missing_ok=True)
            if managed.is_dir() and not any(managed.iterdir()):
                managed.rmdir()


if __name__ == "__main__":
    main()
