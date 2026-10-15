from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from haystack.dataclasses import Breakpoint

from rag3.checkpoint import (
    CheckpointIdentity,
    CheckpointStore,
    PIPELINE_SCHEMA_VERSION,
)
from rag3.haystack_native import (
    DEFAULT_SPLIT_LENGTH,
    DEFAULT_SPLIT_OVERLAP,
    build_ingestion_pipeline,
    create_document_store,
    find_bge_m3,
    preserve_source,
)

COLLECTION = "rag3_haystack_checkpoint_smoke"


def main() -> None:
    fixture_dir = ROOT / "artifacts" / "haystack-checkpoint-smoke"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    source = fixture_dir / "checkpoint-smoke.txt"
    source.write_text(
        "RAG3 checkpoint smoke test.\n\n"
        "This document proves that Haystack can stop before the BGE-M3 embedder, "
        "persist PipelineSnapshot state, and resume without rerunning completed components.\n",
        encoding="utf-8",
    )

    preserved = preserve_source(source)
    model_path = str(find_bge_m3())
    identity = CheckpointIdentity(
        source_sha256=preserved["source_sha256"],
        pipeline_schema_version=PIPELINE_SCHEMA_VERSION,
        split_length=DEFAULT_SPLIT_LENGTH,
        split_overlap=DEFAULT_SPLIT_OVERLAP,
        embedding_model=model_path,
        target_collection=COLLECTION,
    )
    checkpoints = CheckpointStore(preserved["source_sha256"])
    checkpoints.discard_processing_state()

    store = create_document_store(collection=COLLECTION, recreate_index=True)
    pipe = build_ingestion_pipeline(document_store=store, model_path=model_path)

    base_meta = {
        "document_id": preserved["document_id"],
        "title": preserved["title"],
        "source_uri": preserved["source_uri"],
        "source_sha256": preserved["source_sha256"],
        "page_start": None,
        "page_end": None,
        "section_path": [],
        "content_type": "prose",
        "document_version": "1",
    }
    data = {
        "converter": {
            "sources": [str(preserved["managed_source"])],
            "meta": base_meta,
        }
    }

    interrupted = False
    try:
        pipe.run(
            data,
            break_point=Breakpoint(component_name="embedder"),
            snapshot_callback=checkpoints.callback(identity=identity, state="PAUSED"),
        )
    except Exception as exc:
        if exc.__class__.__name__ != "BreakpointException":
            raise
        interrupted = True
        print(f"BREAKPOINT_EXCEPTION={exc.__class__.__module__}.{exc.__class__.__name__}")

    if not interrupted:
        raise SystemExit("Pipeline did not stop at breakpoint")
    if not checkpoints.has_checkpoint():
        raise SystemExit("Snapshot callback did not persist checkpoint")
    if store.count_documents() != 0:
        raise SystemExit("Writer ran before the embedder breakpoint")

    snapshot = checkpoints.load_snapshot(identity)
    result = pipe.run(
        {},
        pipeline_snapshot=snapshot,
        snapshot_callback=checkpoints.callback(identity=identity, state="PAUSED"),
    )
    count_after = store.count_documents()
    checkpoints.set_state("COMPLETE", identity=identity)

    manifest = checkpoints.load_manifest()
    out = {
        "interrupted": interrupted,
        "checkpoint_exists": checkpoints.has_checkpoint(),
        "count_after_resume": count_after,
        "documents_written": result.get("writer", {}).get("documents_written"),
        "manifest_state": (manifest or {}).get("state"),
        "snapshot_breakpoint": snapshot.break_point.component_name,
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))

    if count_after < 1:
        raise SystemExit("Resume did not reach DocumentWriter")
    if out["manifest_state"] != "COMPLETE":
        raise SystemExit("Checkpoint manifest did not reach COMPLETE")


if __name__ == "__main__":
    main()
