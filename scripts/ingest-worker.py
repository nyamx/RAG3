from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from haystack.dataclasses import Breakpoint
from haystack.core.errors import BreakpointException
from qdrant_client import QdrantClient, models as qdrant_models

from rag3.checkpoint import CheckpointIdentity, CheckpointStore, PIPELINE_SCHEMA_VERSION
from rag3.haystack_native import (
    DEFAULT_NATIVE_COLLECTION,
    DEFAULT_SPLIT_LENGTH,
    DEFAULT_SPLIT_OVERLAP,
    build_ingestion_pipeline,
    create_document_store,
    find_bge_m3,
    preserve_source,
    prepare_docling_source,
)


def emit(kind: str, **payload) -> None:
    print(json.dumps({"type": kind, **payload}, ensure_ascii=False), flush=True)


def pause_request_path(source_sha: str) -> Path:
    return ROOT / "data" / "checkpoints" / source_sha / "pause.request"


def checkpoint_identity(source_sha: str, model_path: str, collection: str) -> CheckpointIdentity:
    return CheckpointIdentity(
        source_sha256=source_sha,
        pipeline_schema_version=PIPELINE_SCHEMA_VERSION,
        split_length=DEFAULT_SPLIT_LENGTH,
        split_overlap=DEFAULT_SPLIT_OVERLAP,
        embedding_model=model_path,
        target_collection=collection,
    )


def checkpoint_callback(store: CheckpointStore, identity: CheckpointIdentity):
    def _save(snapshot):
        saved = store.save_snapshot(snapshot, identity=identity, state="RUNNING")
        emit("log", message=f"CHECKPOINT_SAVED {snapshot.break_point.component_name}")
        return saved
    return _save


def should_pause(source_sha: str) -> bool:
    return pause_request_path(source_sha).exists()


def clear_pause(source_sha: str) -> None:
    pause_request_path(source_sha).unlink(missing_ok=True)


def pause_if_requested(store: CheckpointStore, identity: CheckpointIdentity, source_sha: str) -> bool:
    if not should_pause(source_sha):
        return False
    clear_pause(source_sha)
    store.set_state("PAUSED", identity=identity)
    emit("paused", source_sha256=source_sha)
    return True


def run_to_breakpoint(pipe, *, data, snapshot, component: str, callback):
    try:
        if snapshot is None:
            pipe.run(
                data,
                break_point=Breakpoint(component_name=component),
                snapshot_callback=callback,
            )
        else:
            pipe.run(
                {},
                pipeline_snapshot=snapshot,
                break_point=Breakpoint(component_name=component),
                snapshot_callback=callback,
            )
    except BreakpointException:
        return
    raise RuntimeError(f"Expected breakpoint before {component} was not reached")



def verify_source_in_qdrant(
    *,
    collection: str,
    source_sha256: str,
    url: str = "http://127.0.0.1:16333",
) -> tuple[int, int]:
    client = QdrantClient(url=url)
    source_filter = qdrant_models.Filter(
        must=[
            qdrant_models.FieldCondition(
                key="meta.source_sha256",
                match=qdrant_models.MatchValue(value=source_sha256),
            )
        ]
    )

    point_count = 0
    bm25_count = 0
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            scroll_filter=source_filter,
            limit=256,
            offset=offset,
            with_payload=False,
            with_vectors=["bm25"],
        )
        for point in points:
            point_count += 1
            vectors = point.vector or {}
            if isinstance(vectors, dict) and vectors.get("bm25") is not None:
                bm25_count += 1
        if offset is None:
            break

    return point_count, bm25_count


def run_job(source: Path, collection: str, mode: str) -> dict:
    preserved = preserve_source(source)
    docling_source = prepare_docling_source(preserved)
    source_sha = preserved["source_sha256"]
    model_path = str(find_bge_m3())
    identity = checkpoint_identity(source_sha, model_path, collection)
    checkpoints = CheckpointStore(source_sha)
    clear_pause(source_sha)

    if mode == "restart":
        checkpoints.discard_processing_state()

    store = create_document_store(collection=collection, recreate_index=False)
    pipe = build_ingestion_pipeline(document_store=store, model_path=model_path)
    cb = checkpoint_callback(checkpoints, identity)

    base_meta = {
        "document_id": preserved["document_id"],
        "title": preserved["title"],
        "source_uri": preserved["source_uri"],
        "source_sha256": source_sha,
        "page_start": None,
        "page_end": None,
        "section_path": [],
        "content_type": "prose",
        "document_version": "1",
    }
    data = {
        "converter": {
            "sources": [str(docling_source)],
            "meta": base_meta,
        }
    }

    if mode == "resume":
        emit("log", message="STAGE resume RUN")
        snapshot = checkpoints.load_snapshot(identity)
    else:
        emit("log", message="STAGE parse RUN")
        run_to_breakpoint(pipe, data=data, snapshot=None, component="embedder", callback=cb)
        emit("log", message="STAGE parse PASS")
        emit("log", message="STAGE chunk PASS")
        if pause_if_requested(checkpoints, identity, source_sha):
            return {"status": "PAUSED", "source_sha256": source_sha}
        snapshot = checkpoints.load_snapshot(identity)

    if snapshot.break_point.component_name == "embedder":
        emit("log", message="STAGE embedding RUN")
        run_to_breakpoint(pipe, data={}, snapshot=snapshot, component="writer", callback=cb)
        emit("log", message="STAGE embedding PASS")
        if pause_if_requested(checkpoints, identity, source_sha):
            return {"status": "PAUSED", "source_sha256": source_sha}
        snapshot = checkpoints.load_snapshot(identity)

    emit("log", message="STAGE qdrant RUN")
    result = pipe.run(
        {},
        include_outputs_from={"writer", "bm25_writer"},
        pipeline_snapshot=snapshot,
        snapshot_callback=cb,
    )
    count_after = store.count_documents()
    written = result.get("writer", {}).get("documents_written")
    bm25_written = result.get("bm25_writer", {}).get("bm25_vectors_written")

    source_point_count, source_bm25_count = verify_source_in_qdrant(
        collection=collection,
        source_sha256=source_sha,
    )
    if source_point_count < 1:
        raise RuntimeError(
            f"Final Qdrant verification failed: no points found for source {source_sha}"
        )
    if source_bm25_count != source_point_count:
        raise RuntimeError(
            "Final BM25 verification failed: "
            f"source_points={source_point_count}, bm25_vectors={source_bm25_count}"
        )

    if written is not None and bm25_written is not None and bm25_written != written:
        raise RuntimeError(
            f"BM25 vector count mismatch: documents={written}, bm25={bm25_written}"
        )

    checkpoints.set_state("COMPLETE", identity=identity)
    emit("log", message="STAGE qdrant PASS")
    return {
        "status": "COMPLETE",
        "source_sha256": source_sha,
        "managed_source": str(preserved["managed_source"]),
        "collection": collection,
        "documents_written": written,
        "bm25_vectors_written": bm25_written,
        "source_point_count": source_point_count,
        "source_bm25_count": source_bm25_count,
        "count_after": count_after,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("document")
    ap.add_argument("--collection", default=DEFAULT_NATIVE_COLLECTION)
    ap.add_argument("--mode", choices=("run", "resume", "restart"), default="run")
    args = ap.parse_args()

    source = Path(args.document).expanduser().resolve()
    try:
        result = run_job(source, args.collection, args.mode)
        emit("result", result=result)
        if result.get("status") == "PAUSED":
            raise SystemExit(75)
    except Exception as exc:
        emit(
            "error",
            message=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(limit=16),
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
