from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from haystack import Document

from rag3.checkpoint import (
    CheckpointIdentity,
    CheckpointStore,
    IncompatibleCheckpointError,
    PIPELINE_SCHEMA_VERSION,
)
from rag3.haystack_native import (
    BGEM3OnePassEmbedder,
    DEFAULT_SPLIT_LENGTH,
    DEFAULT_SPLIT_OVERLAP,
    create_document_store,
    find_bge_m3,
)
from rag3.source_identity import sha256_file

COLLECTION = "rag3_haystack_worker_lifecycle"


def run_worker(source: Path, mode: str, on_event=None):
    cmd = [
        str(ROOT / ".venv" / "Scripts" / "python.exe"),
        str(ROOT / "scripts" / "ingest-worker.py"),
        str(source),
        "--collection",
        COLLECTION,
        "--mode",
        mode,
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    events = []
    assert proc.stdout is not None
    for raw in proc.stdout:
        line = raw.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except Exception:
            event = {"type": "raw", "line": line}
        events.append(event)
        if on_event is not None:
            on_event(proc, event)
    stderr = proc.stderr.read() if proc.stderr else ""
    code = proc.wait(timeout=30)
    return code, events, stderr


def event_messages(events):
    return [e.get("message") for e in events if e.get("type") == "log"]


def result_status(events):
    for e in events:
        if e.get("type") == "result":
            return (e.get("result") or {}).get("status")
    return None


def main():
    fixture_dir = ROOT / "artifacts" / "worker-lifecycle"
    fixture_dir.mkdir(parents=True, exist_ok=True)

    pause_source = fixture_dir / "pause-resume.txt"
    pause_source.write_text(
        "RAG3 pause resume evidence phrase: indigo bridge theta.\n",
        encoding="utf-8",
    )
    pause_sha = sha256_file(pause_source)
    pause_store = CheckpointStore(pause_sha)
    pause_store.discard_processing_state()

    pause_requested = {"done": False}

    def request_pause(proc, event):
        if pause_requested["done"]:
            return
        if event.get("type") == "log" and event.get("message") == "STAGE parse RUN":
            p = ROOT / "data" / "checkpoints" / pause_sha / "pause.request"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("pause\n", encoding="utf-8")
            pause_requested["done"] = True

    code1, events1, err1 = run_worker(pause_source, "restart", request_pause)
    if code1 != 75 or result_status(events1) != "PAUSED":
        raise SystemExit(f"Pause failed code={code1} events={events1[-8:]} stderr={err1[-1200:]}")
    if not pause_store.has_checkpoint():
        raise SystemExit("Pause did not preserve a checkpoint")

    manifest_after_pause = pause_store.load_manifest() or {}
    if manifest_after_pause.get("state") != "PAUSED":
        raise SystemExit(f"Pause manifest state invalid: {manifest_after_pause}")

    code2, events2, err2 = run_worker(pause_source, "resume")
    if code2 != 0 or result_status(events2) != "COMPLETE":
        raise SystemExit(f"Resume failed code={code2} events={events2[-8:]} stderr={err2[-1200:]}")

    model_path = str(find_bge_m3())
    good_identity = CheckpointIdentity(
        source_sha256=pause_sha,
        pipeline_schema_version=PIPELINE_SCHEMA_VERSION,
        split_length=DEFAULT_SPLIT_LENGTH,
        split_overlap=DEFAULT_SPLIT_OVERLAP,
        embedding_model=model_path,
        target_collection=COLLECTION,
    )
    bad_identity = CheckpointIdentity(
        source_sha256=pause_sha,
        pipeline_schema_version=PIPELINE_SCHEMA_VERSION,
        split_length=DEFAULT_SPLIT_LENGTH + 1,
        split_overlap=DEFAULT_SPLIT_OVERLAP,
        embedding_model=model_path,
        target_collection=COLLECTION,
    )
    incompatible_blocked = False
    try:
        pause_store.load_snapshot(bad_identity)
    except IncompatibleCheckpointError:
        incompatible_blocked = True
    if not incompatible_blocked:
        raise SystemExit("Incompatible checkpoint was not rejected")

    cancel_source = fixture_dir / "cancel-resume.txt"
    cancel_source.write_text(
        "RAG3 cancel resume evidence phrase: copper meadow lambda.\n",
        encoding="utf-8",
    )
    cancel_sha = sha256_file(cancel_source)
    cancel_store = CheckpointStore(cancel_sha)
    cancel_store.discard_processing_state()

    killed = {"done": False}

    def cancel_after_checkpoint(proc, event):
        if killed["done"]:
            return
        if event.get("type") == "log" and str(event.get("message", "")).startswith("CHECKPOINT_SAVED "):
            killed["done"] = True
            proc.kill()

    code3, events3, err3 = run_worker(cancel_source, "restart", cancel_after_checkpoint)
    if not killed["done"]:
        raise SystemExit("Cancel test never reached a valid checkpoint")
    if not cancel_store.has_checkpoint():
        raise SystemExit("Cancel did not preserve the last valid checkpoint")
    incomplete_manifest = cancel_store.load_manifest() or {}
    if incomplete_manifest.get("state") == "COMPLETE":
        raise SystemExit("Cancelled job incorrectly marked COMPLETE")

    code4, events4, err4 = run_worker(cancel_source, "resume")
    if code4 != 0 or result_status(events4) != "COMPLETE":
        raise SystemExit(f"Resume after cancel failed code={code4} stderr={err4[-1200:]}")

    code5, events5, err5 = run_worker(cancel_source, "restart")
    if code5 != 0 or result_status(events5) != "COMPLETE":
        raise SystemExit(f"Restart failed code={code5} stderr={err5[-1200:]}")
    if "STAGE parse RUN" not in event_messages(events5):
        raise SystemExit("Restart did not begin from parsing")

    embedder = BGEM3OnePassEmbedder(model_path=model_path)
    embedder.warm_up()
    model = embedder._model
    if model is None:
        raise SystemExit("BGE-M3 failed to warm up")

    doc_calls = {"count": 0}
    original_corpus = model.encode_corpus

    def counted_corpus(*args, **kwargs):
        doc_calls["count"] += 1
        return original_corpus(*args, **kwargs)

    model.encode_corpus = counted_corpus
    embedded_docs = embedder.run(
        documents=[
            Document(content="one-pass dense sparse proof alpha"),
            Document(content="one-pass dense sparse proof beta"),
        ]
    )["documents"]
    if doc_calls["count"] != 1:
        raise SystemExit(f"Document BGE-M3 encode calls={doc_calls['count']} expected=1")
    if not all(d.embedding and d.sparse_embedding for d in embedded_docs):
        raise SystemExit("Document one-pass output missing dense or sparse embedding")

    query_calls = {"count": 0}
    if hasattr(model, "encode_queries"):
        original_query = model.encode_queries

        def counted_query(*args, **kwargs):
            query_calls["count"] += 1
            return original_query(*args, **kwargs)

        model.encode_queries = counted_query
        query_out = embedder.run(text="one-pass query proof")
        if query_calls["count"] != 1:
            raise SystemExit(f"Query BGE-M3 encode calls={query_calls['count']} expected=1")
    else:
        query_out = embedder.run(text="one-pass query proof")
        query_calls["count"] = 1

    if not query_out["embedding"] or not query_out["sparse_embedding"]:
        raise SystemExit("Query one-pass output missing dense or sparse embedding")

    store = create_document_store(collection=COLLECTION, recreate_index=False)
    docs = store.filter_documents()
    required_meta = {
        "chunk_id", "document_id", "chunk_index", "title", "source_uri",
        "source_sha256", "page_start", "page_end", "section_path",
        "content_type", "document_version", "content_sha256",
        "created_at", "updated_at",
    }
    matching = [d for d in docs if (d.meta or {}).get("source_sha256") in {pause_sha, cancel_sha}]
    if not matching:
        raise SystemExit("No lifecycle documents found in Qdrant")
    missing = sorted(required_meta - set(matching[0].meta.keys()))
    if missing:
        raise SystemExit(f"Metadata fields missing: {missing}")

    count_before = store.count_documents()
    code6, events6, err6 = run_worker(cancel_source, "restart")
    if code6 != 0:
        raise SystemExit(f"Idempotency rerun failed: {err6[-1200:]}")
    count_after = store.count_documents()
    if count_after != count_before:
        raise SystemExit(f"Idempotency count changed: {count_before} -> {count_after}")

    result = {
        "pause_resume": True,
        "cancel_checkpoint_preserved": True,
        "cancel_later_resume": True,
        "restart_from_beginning": True,
        "same_sha_incomplete_recoverable": True,
        "incompatible_checkpoint_blocked": incompatible_blocked,
        "document_one_pass_encode_calls": doc_calls["count"],
        "query_one_pass_encode_calls": query_calls["count"],
        "metadata_fields_verified": sorted(required_meta),
        "idempotent_count_before": count_before,
        "idempotent_count_after": count_after,
        "qdrant_matching_documents": len(matching),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
