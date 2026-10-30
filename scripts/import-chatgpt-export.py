"""Import ChatGPT export JSON files as RAG3 documents through the shared queue."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag3.chatgpt_export import MAX_MESSAGE_CHARS, iter_shards, normalize_conversation
from rag3.constants import DATA_ROOT, DEFAULT_QDRANT_URL


def _prepare_collection(name: str, url: str):
    from qdrant_client import models
    from rag3.collections import create_collection
    from rag3.qdrant_clients import get_client

    create_collection(name, url=url, reuse_existing=True)
    client = get_client(url=url)
    info = client.get_collection(name)
    vectors = info.config.params.vectors
    dense = vectors.get("text-dense") if isinstance(vectors, dict) else None
    if dense is None or dense.size != 1024 or dense.distance != models.Distance.COSINE:
        raise ValueError("collection has incompatible dense vector settings")
    indexed = info.payload_schema or {}
    fields = {
        "meta.source_sha256": models.PayloadSchemaType.KEYWORD,
        "meta.document_id": models.PayloadSchemaType.KEYWORD,
        "meta._rag3_ingest_state": models.PayloadSchemaType.KEYWORD,
        "meta.source_kind": models.PayloadSchemaType.KEYWORD,
        "meta.chatgpt_export_file": models.PayloadSchemaType.KEYWORD,
        "meta.chatgpt_conversation_id": models.PayloadSchemaType.KEYWORD,
        "meta.chatgpt_role": models.PayloadSchemaType.KEYWORD,
        "meta.chatgpt_message_created_at_ms": models.PayloadSchemaType.INTEGER,
        "meta.chatgpt_conversation_created_at_ms": models.PayloadSchemaType.INTEGER,
        "meta.chatgpt_conversation_updated_at_ms": models.PayloadSchemaType.INTEGER,
    }
    for field, schema in fields.items():
        if field not in indexed:
            client.create_payload_index(collection_name=name, field_name=field,
                                        field_schema=schema, wait=True)
    return client, models


def _write_source(path: Path, raw: bytes, sha: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
            raise RuntimeError("prepared source hash mismatch")
        return
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        temporary.write_bytes(raw)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export", help="ChatGPT export ZIP, extracted folder, or one conversation JSON file")
    parser.add_argument("--collection", default="chatgpt_history_files")
    parser.add_argument("--qdrant-url", default=DEFAULT_QDRANT_URL)
    parser.add_argument("--dry-run", action="store_true", help="validate and count without writing or submitting")
    parser.add_argument("--limit", type=int, default=0, help="stop after this many submitted JSON files")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be non-negative")

    if not args.dry_run:
        _prepare_collection(args.collection, args.qdrant_url)
        from rag3.ingestion_manager import client as job_client

    counts = {"files": 0, "conversations": 0, "submitted": 0,
              "chunks_written": 0, "files_without_new_conversations": 0}
    if args.dry_run:
        counts.update(messages=0, chunks=0, empty=0, unsupported=0)
    for filename, raw, conversations in iter_shards(args.export):
        counts["files"] += 1
        counts["conversations"] += len(conversations)
        if args.dry_run:
            for conversation in conversations:
                chat = normalize_conversation(conversation)
                counts["messages"] += len(chat["messages"])
                counts["empty"] += chat["skipped_empty"]
                counts["unsupported"] += chat["skipped_unsupported"]
                counts["chunks"] += sum((len(message["text"]) + MAX_MESSAGE_CHARS - 1) // MAX_MESSAGE_CHARS
                                        for message in chat["messages"])
            continue

        sha = hashlib.sha256(raw).hexdigest()
        source = DATA_ROOT / "sources" / sha / filename
        _write_source(source, raw, sha)
        job = job_client("submit", origin="ChatGPT export", collection=args.collection,
                         path=str(source), requested=False)
        job_client("action", job_id=job["job_id"], operation="run")
        counts["submitted"] += 1
        print(f"Submitted {counts['submitted']}: {filename}, job {job['job_id']}", flush=True)
        while True:
            status = job_client("status", job_id=job["job_id"])
            if status["status"] in {"COMPLETE", "FAILED", "CANCELLED", "PAUSED"}:
                if status["status"] != "COMPLETE":
                    raise RuntimeError(f"job {job['job_id']} ended as {status['status']}: {status.get('error') or ''}")
                written = int((status.get("result") or {}).get("documents_written") or 0)
                counts["chunks_written"] += written
                if written == 0:
                    counts["files_without_new_conversations"] += 1
                break
            time.sleep(2)
        if args.limit and counts["submitted"] >= args.limit:
            break
    print(json.dumps(counts, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
