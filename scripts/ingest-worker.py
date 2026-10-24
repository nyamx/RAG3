"""Legacy document worker; the shared backend uses its CPU parse phase."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag3.constants import DEFAULT_NATIVE_COLLECTION, DEFAULT_QDRANT_URL, DATA_ROOT
from rag3.ingestion_settings import CHUNK_GROUP_SIZE, progress_identity
from rag3.source_identity import sha256_file


def emit(kind: str, **payload) -> None:
    print(json.dumps({"type": kind, **payload}, ensure_ascii=True), flush=True)


def pause_request_path(source_sha: str) -> Path:
    return DATA_ROOT / "checkpoints" / source_sha / "pause.request"


def pause_if_requested(source_sha: str) -> bool:
    marker = pause_request_path(source_sha)
    if not marker.exists():
        return False
    emit("paused", source_sha256=source_sha)
    return True


def source_filter(source_sha: str, *, phase: str | None = None):
    from qdrant_client import models

    conditions = [models.FieldCondition(
        key="meta.source_sha256",
        match=models.MatchValue(value=source_sha),
    )]
    if phase is not None:
        conditions.append(models.FieldCondition(
            key="meta._rag3_ingest_state",
            match=models.MatchValue(value=phase),
        ))
    return models.Filter(must=conditions)


def parse_source(source: Path, collection: str, source_sha: str | None = None) -> dict:
    """Convert the whole source, then stage its chunks in bounded Qdrant writes."""
    from haystack_integrations.document_stores.qdrant.converters import (
        convert_haystack_documents_to_qdrant_points,
    )
    from qdrant_client import QdrantClient, models

    from rag3.haystack_native import (
        PLAIN_TEXT_SUFFIXES, build_parsing_pipeline, normalize_chunk,
        prepare_docling_source, preserve_source, split_plain_text_source,
    )
    from rag3.ingestion_jobs import ingestion_progress, set_ingestion_progress

    preserved = preserve_source(source, expected_sha256=source_sha)
    source_sha = preserved["source_sha256"]
    identity = progress_identity(source_sha, collection)
    client = QdrantClient(url=os.environ.get("RAG3_QDRANT_URL", DEFAULT_QDRANT_URL))
    progress = ingestion_progress(collection, source_sha)
    if progress is None:
        raise ValueError("Parsing requires an initialized matching ingestion progress; refuse unscoped reset")
    if progress is None or progress["identity"] != identity:
        raise ValueError("No compatible Qdrant ingestion progress for this source")
    if progress["phase"] in {"CHUNKS_STORED", "EMBEDDING"}:
        return {"status": "PARSED", "source_sha256": source_sha,
                "total_chunks": progress["total_chunks"]}

    base_meta = {
        "document_id": preserved["document_id"],
        "title": preserved["title"],
        "source_uri": preserved["source_uri"],
        "source_sha256": source_sha,
        "page_start": None,
        "page_end": None,
        "content_type": "prose",
        "document_version": "1",
    }
    emit("log", message="STAGE parse RUN")
    emit("log", message="STAGE qdrant RUN")

    def stage(documents: list, first_index: int) -> bool:
        for start in range(0, len(documents), CHUNK_GROUP_SIZE):
            staged = []
            for idx in range(start, min(start + CHUNK_GROUP_SIZE, len(documents))):
                chunk = normalize_chunk(documents[idx], first_index + idx)
                chunk.meta["_rag3_ingest_state"] = "parsed"
                staged.append(chunk)
            client.upsert(
                collection_name=collection,
                points=convert_haystack_documents_to_qdrant_points(
                    staged, use_sparse_embeddings=True,
                ),
                wait=True,
            )
            emit("log", message=f"STAGE qdrant STORED {first_index + start + len(staged)}")
            del staged
            if pause_if_requested(source_sha):
                return False
        return True

    if pause_if_requested(source_sha):
        return {"status": "PAUSED", "source_sha256": source_sha}
    if source.suffix.lower() in PLAIN_TEXT_SUFFIXES:
        chunks = split_plain_text_source(preserved, base_meta)
    else:
        docling_source = prepare_docling_source(preserved)
        # Child parser is CPU-only; it must never initialize CUDA/OCR on GPU.
        pipeline = build_parsing_pipeline(pdf_cpu=True)
        parsed = pipeline.run({"converter": {
            "sources": [str(docling_source)], "meta": base_meta,
        }})
        chunks = parsed["splitter"]["documents"]
    total = len(chunks)
    if not stage(chunks, 0):
        return {"status": "PAUSED", "source_sha256": source_sha}
    if total < 1:
        raise ValueError("Document produced no searchable chunks")
    set_ingestion_progress(collection, source_sha, identity, "CHUNKS_STORED", total)
    emit("log", message="STAGE parse PASS")
    emit("log", message="STAGE chunk PASS")
    emit("log", message="STAGE qdrant PASS")
    return {"status": "PARSED", "source_sha256": source_sha,
            "total_chunks": total}


def verify_source(client, collection: str, source_sha: str) -> tuple[int, int, int, int]:
    from qdrant_client import models

    base = source_filter(source_sha)

    def exact(extra=None) -> int:
        conditions = list(base.must or [])
        if extra is not None:
            conditions.append(extra)
        return client.count(
            collection_name=collection,
            count_filter=models.Filter(must=conditions),
            exact=True,
        ).count

    return (
        exact(),
        exact(models.HasVectorCondition(has_vector="text-dense")),
        exact(models.HasVectorCondition(has_vector="bm25")),
        exact(models.FieldCondition(
            key="meta._rag3_ingest_state",
            match=models.MatchValue(value="ready"),
        )),
    )


def embed_source(source: Path, collection: str, source_sha: str | None = None) -> dict:
    """Read at most four BGE batches from Qdrant on each loop iteration."""
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(name, os.environ.get("RAG3_WORKER_CPU_THREADS", "2"))
    import torch
    torch.set_num_threads(max(1, int(os.environ.get("RAG3_WORKER_CPU_THREADS", "2"))))
    from qdrant_client import QdrantClient, models

    from rag3.haystack_native import BGEM3OnePassEmbedder, collection_bm25_config
    from rag3.ingestion_jobs import ingestion_progress

    source_sha = source_sha or sha256_file(source)
    progress = ingestion_progress(collection, source_sha)
    identity = progress_identity(source_sha, collection)
    if progress is None or progress["identity"] != identity:
        raise ValueError("No compatible Qdrant ingestion progress for this source")
    total = progress["total_chunks"]

    url = os.environ.get("RAG3_QDRANT_URL", DEFAULT_QDRANT_URL)
    client = QdrantClient(url=url)
    bm25_config = collection_bm25_config(collection, url=url)
    embedder = BGEM3OnePassEmbedder(model_path=identity["embedding_model"])
    emit("log", message="STAGE embedding RUN")
    completed = 0
    while True:
        if pause_if_requested(source_sha):
            return {"status": "PAUSED", "source_sha256": source_sha}
        points, _ = client.scroll(
            collection_name=collection,
            scroll_filter=source_filter(source_sha, phase="parsed"),
            limit=CHUNK_GROUP_SIZE, with_payload=True, with_vectors=False,
        )
        if not points:
            progress = ingestion_progress(collection, source_sha)
            if progress is None or progress["identity"] != identity:
                raise RuntimeError("Ingestion progress disappeared while embedding")
            total = progress["total_chunks"]
            if progress["phase"] in {"CHUNKS_STORED", "EMBEDDING"}:
                break
            time.sleep(.25)
            continue
        texts = [str((point.payload or {}).get("content") or "") for point in points]
        vectors = embedder._encode(texts, query=False)
        updates = []
        for point, text, dense in zip(points, texts, vectors, strict=True):
            updates.append(models.PointVectors(
                id=point.id,
                vector={
                    "text-dense": dense,
                    "bm25": models.Document(
                        text=text, model="qdrant/bm25", options=bm25_config,
                    ),
                },
            ))
        client.update_vectors(collection_name=collection, points=updates, wait=True)
        client.set_payload(
            collection_name=collection,
            payload={"_rag3_ingest_state": "ready"},
            points=[point.id for point in points],
            key="meta",
            wait=True,
        )
        completed += len(points)
        emit("log", message=f"STAGE embedding {completed} written")
        del points, texts, vectors, updates

    source_count, dense_count, bm25_count, ready_count = verify_source(
        client, collection, source_sha,
    )
    if (source_count != total or dense_count != total or bm25_count != total
            or ready_count != total):
        raise RuntimeError(
            "Final Qdrant verification failed: "
            f"expected={total}, points={source_count}, dense={dense_count}, "
            f"bm25={bm25_count}, ready={ready_count}"
        )
    emit("log", message="STAGE embedding PASS")
    emit("log", message="STAGE qdrant PASS")
    return {
        "status": "COMPLETE", "source_sha256": source_sha,
        "managed_source": str(DATA_ROOT / "sources" / source_sha / source.name),
        "collection": collection, "documents_written": total,
        "bm25_vectors_written": bm25_count,
        "source_point_count": source_count,
        "source_dense_count": dense_count,
        "source_bm25_count": bm25_count,
        "count_after": client.count(collection_name=collection, exact=True).count,
    }


def run_phase(source: Path, collection: str, phase: str, source_sha: str) -> dict:
    command = [sys.executable, str(Path(__file__).resolve()), str(source),
               "--collection", collection, "--phase", phase,
               "--source-sha", source_sha]
    proc = subprocess.Popen(
        command, cwd=ROOT, env=os.environ.copy(), stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0,
    )
    result = None
    error = None
    assert proc.stdout is not None
    for raw in proc.stdout:
        try:
            event = json.loads(raw)
        except ValueError:
            continue
        if event.get("type") == "result":
            result = event.get("result")
        elif event.get("type") == "error":
            error = str(event.get("message") or "phase worker failed")
            emit("error", message=error)
        elif event.get("type") in {"log", "paused"}:
            emit(event["type"], **{k: v for k, v in event.items() if k != "type"})
    code = proc.wait()
    if code == 75 and isinstance(result, dict) and result.get("status") == "PAUSED":
        return result
    if code != 0 or not isinstance(result, dict):
        raise RuntimeError(error or f"{phase} worker exited with code {code}")
    return result



def run_job(source: Path, collection: str, mode: str,
            expected_sha: str | None = None) -> dict:
    from qdrant_client import QdrantClient, models
    from rag3.ingestion_jobs import ingestion_progress, set_ingestion_progress

    source_sha = sha256_file(source)
    if expected_sha is not None and source_sha != expected_sha:
        raise ValueError("original file hash changed")
    progress = ingestion_progress(collection, source_sha)
    identity = progress_identity(source_sha, collection)
    if mode == "resume":
        if progress is None or progress["identity"] != identity:
            raise ValueError("No compatible Qdrant ingestion progress for resume")
    reset = mode != "resume" or (progress is not None and
                                  progress["phase"] == "PARSING")
    if reset:
        client = QdrantClient(url=os.environ.get("RAG3_QDRANT_URL", DEFAULT_QDRANT_URL))
        client.delete(collection_name=collection,
                      points_selector=models.FilterSelector(filter=source_filter(source_sha)),
                      wait=True)
        set_ingestion_progress(collection, source_sha, identity, "PARSING")
        progress = ingestion_progress(collection, source_sha)
    pause_request_path(source_sha).unlink(missing_ok=True)
    if progress is None or progress["phase"] == "PARSING":
        parsed = run_phase(source, collection, "parse", source_sha)
        if parsed.get("status") == "PAUSED":
            return parsed
    # run_phase waits for the parser process to exit and release GPU memory.
    return run_phase(source, collection, "embed", source_sha)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("document")
    parser.add_argument("--collection", default=DEFAULT_NATIVE_COLLECTION)
    parser.add_argument("--mode", choices=("run", "resume", "restart"), default="run")
    parser.add_argument("--phase", choices=("coordinate", "parse", "embed"),
                        default="coordinate")
    parser.add_argument("--source-sha", default=None)
    args = parser.parse_args()
    source = Path(args.document).expanduser().resolve()
    try:
        if args.phase == "parse" and os.environ.get("RAG3_PARSE_PARENT_GATE") == "1":
            # The backend first attaches this process to a kill-on-close
            # Windows Job Object. EOF means the parent died before arming it.
            if sys.stdin.read(1) != "G":
                raise RuntimeError("CPU parser parent gate closed")
        if args.phase == "parse":
            result = parse_source(source, args.collection, args.source_sha)
        elif args.phase == "embed":
            result = embed_source(source, args.collection, args.source_sha)
        else:
            result = run_job(source, args.collection, args.mode, args.source_sha)
        emit("result", result=result)
        if result.get("status") == "PAUSED":
            raise SystemExit(75)
    except Exception as exc:
        emit("error", message=f"{type(exc).__name__}: {exc}",
             traceback=traceback.format_exc(limit=16))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
