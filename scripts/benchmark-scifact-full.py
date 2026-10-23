"""Run the official BEIR SciFact test split against the current RAG3 retrieval path.

The harness batches indexing for speed, but uses RAG3's splitter, BGE-M3 embedder,
Qdrant collection schema, BM25 configuration, and search() implementation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import statistics
import sys
import time
import uuid
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from haystack import Document
from haystack.components.preprocessors import DocumentSplitter
from qdrant_client import QdrantClient, models

from rag3.constants import DEFAULT_QDRANT_URL
from rag3.haystack_native import (
    BGEM3OnePassEmbedder,
    bm25_config,
    create_document_store,
)
from rag3.ingestion_settings import DEFAULT_SPLIT_LENGTH, DEFAULT_SPLIT_OVERLAP
from rag3.retrieval import search

DATASET = ROOT / "artifacts" / "analysis" / "scifact.zip"
OUTPUT = ROOT / "artifacts" / "analysis" / "scifact_full_20261022"
COLLECTION = "rag3_beir_scifact_full_20261022"
BM25_COLLECTION = "rag3_beir_scifact_bm25_full_20261022"
EXPECTED_MD5 = "5f7d1de60b170fc8027bb7898e2efca1"


def load_dataset():
    if hashlib.md5(DATASET.read_bytes()).hexdigest() != EXPECTED_MD5:
        raise ValueError("SciFact ZIP MD5 differs from the official BEIR checksum")
    with zipfile.ZipFile(DATASET) as archive:
        def records(name):
            return [json.loads(line) for line in archive.read(f"scifact/{name}.jsonl").splitlines()]

        corpus = records("corpus")
        queries = {str(row["_id"]): row["text"] for row in records("queries")}
        qrels = defaultdict(dict)
        reader = csv.DictReader(io.StringIO(archive.read("scifact/qrels/test.tsv").decode()), delimiter="\t")
        for row in reader:
            if int(row["score"]) > 0:
                qrels[row["query-id"]][row["corpus-id"]] = int(row["score"])
    if len(corpus) != 5183 or len(qrels) != 300 or not set(qrels) <= set(queries):
        raise ValueError("Unexpected SciFact corpus/query/qrels counts")
    return corpus, queries, dict(qrels)


def checkpoint_path():
    return OUTPUT / "ingest-checkpoint.json"


def ingest(corpus):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    create_document_store(collection=COLLECTION)
    client = QdrantClient(url=DEFAULT_QDRANT_URL, timeout=120)
    start = 0
    chunk_count = 0
    if checkpoint_path().exists():
        state = json.loads(checkpoint_path().read_text(encoding="utf-8"))
        start = state["next_document"]
        chunk_count = state["chunk_count"]
    splitter = DocumentSplitter(
        split_by="token", split_length=DEFAULT_SPLIT_LENGTH,
        split_overlap=DEFAULT_SPLIT_OVERLAP, tokenizer_encoding="o200k_base",
    )
    embedder = BGEM3OnePassEmbedder()
    config = bm25_config()
    started = time.perf_counter()
    for offset in range(start, len(corpus), 32):
        source_rows = corpus[offset:offset + 32]
        chunks = []
        for row in source_rows:
            doc_id = str(row["_id"])
            content = f"# {row['title']}\n\n{row['text']}"
            parts = splitter.run(documents=[Document(content=content)])["documents"]
            for idx, part in enumerate(parts):
                if not (part.content or "").strip():
                    continue
                chunks.append((doc_id, idx, part.content))
        points = []
        for suboffset in range(0, len(chunks), 32):
            group = chunks[suboffset:suboffset + 32]
            vectors = embedder._encode([text for _, _, text in group], query=False)
            for (doc_id, idx, text), dense in zip(group, vectors, strict=True):
                point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"beir:scifact:{doc_id}:{idx}"))
                points.append(models.PointStruct(
                    id=point_id,
                    vector={"text-dense": dense, "bm25": models.Document(
                        text=text, model="qdrant/bm25", options=config)},
                    payload={"id": point_id, "content": text, "meta": {
                        "chunk_id": point_id, "document_id": doc_id,
                        "chunk_index": idx, "title": f"BEIR SciFact {doc_id}",
                        "content_type": "prose", "document_version": "1",
                    }},
                ))
        client.upsert(collection_name=COLLECTION, points=points, wait=True)
        chunk_count += len(points)
        next_document = offset + len(source_rows)
        checkpoint_path().write_text(json.dumps({
            "next_document": next_document, "chunk_count": chunk_count,
            "collection": COLLECTION, "dataset_md5": EXPECTED_MD5,
        }, indent=2) + "\n", encoding="utf-8")
        if next_document % 320 < 32 or next_document == len(corpus):
            print(f"ingest {next_document}/{len(corpus)} documents, {chunk_count} chunks, "
                  f"{time.perf_counter() - started:.1f}s this run", flush=True)
    actual = client.count(collection_name=COLLECTION, exact=True).count
    if actual != chunk_count:
        raise RuntimeError(f"Expected {chunk_count} chunks, found {actual}")
    print(f"ingest complete: {len(corpus)} documents, {actual} chunks", flush=True)


def ingest_bm25(corpus):
    """Quick full-corpus lexical baseline while dense vectors are being built."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    create_document_store(collection=BM25_COLLECTION)
    client = QdrantClient(url=DEFAULT_QDRANT_URL, timeout=120)
    checkpoint = OUTPUT / "bm25-ingest-checkpoint.json"
    state = json.loads(checkpoint.read_text(encoding="utf-8")) if checkpoint.exists() else {}
    start = state.get("next_document", 0)
    chunk_count = state.get("chunk_count", 0)
    splitter = DocumentSplitter(
        split_by="token", split_length=DEFAULT_SPLIT_LENGTH,
        split_overlap=DEFAULT_SPLIT_OVERLAP, tokenizer_encoding="o200k_base",
    )
    config = bm25_config()
    for offset in range(start, len(corpus), 128):
        points = []
        rows = corpus[offset:offset + 128]
        for row in rows:
            doc_id = str(row["_id"])
            content = f"# {row['title']}\n\n{row['text']}"
            parts = splitter.run(documents=[Document(content=content)])["documents"]
            for idx, part in enumerate(parts):
                text = part.content or ""
                if not text.strip():
                    continue
                point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"beir:scifact:{doc_id}:{idx}"))
                points.append(models.PointStruct(
                    id=point_id,
                    vector={"bm25": models.Document(
                        text=text, model="qdrant/bm25", options=config)},
                    payload={"id": point_id, "content": text, "meta": {
                        "chunk_id": point_id, "document_id": doc_id,
                        "chunk_index": idx, "title": f"BEIR SciFact {doc_id}",
                        "content_type": "prose", "document_version": "1",
                    }},
                ))
        client.upsert(collection_name=BM25_COLLECTION, points=points, wait=True)
        chunk_count += len(points)
        next_document = offset + len(rows)
        checkpoint.write_text(json.dumps({"next_document": next_document,
            "chunk_count": chunk_count, "dataset_md5": EXPECTED_MD5}, indent=2) + "\n",
            encoding="utf-8")
        if next_document % 512 < 128 or next_document == len(corpus):
            print(f"bm25 ingest {next_document}/{len(corpus)} documents, "
                  f"{chunk_count} chunks", flush=True)
    actual = client.count(collection_name=BM25_COLLECTION, exact=True).count
    if actual != chunk_count:
        raise RuntimeError(f"Expected {chunk_count} BM25 chunks, found {actual}")
    print(f"bm25 ingest complete: {actual} chunks", flush=True)


def evaluate_bm25(queries, qrels):
    checkpoint = json.loads((OUTPUT / "bm25-ingest-checkpoint.json").read_text(encoding="utf-8"))
    client = QdrantClient(url=DEFAULT_QDRANT_URL)
    count = client.count(collection_name=BM25_COLLECTION, exact=True).count
    if checkpoint["next_document"] != 5183 or count != checkpoint["chunk_count"]:
        raise RuntimeError("BM25 full corpus is incomplete")
    ranked = {}
    durations = []
    duplicates = 0
    for index, qid in enumerate(qrels, 1):
        tic = time.perf_counter()
        response = search(queries[qid], collection=BM25_COLLECTION, top_k=10,
                          semantic=False, bm25=True, rank_fusion=False)
        durations.append(time.perf_counter() - tic)
        ids = [str(hit["document_id"]) for hit in response["hits"]]
        duplicates += len(ids) - len(set(ids))
        ranked[qid] = list(dict.fromkeys(ids))
        if index % 25 == 0 or index == len(qrels):
            print(f"bm25 evaluate {index}/{len(qrels)}", flush=True)
    scored = score(ranked, qrels)
    times = sorted(durations)
    result = {"dataset": "BEIR SciFact test", "dataset_md5": EXPECTED_MD5,
        "collection": BM25_COLLECTION, "documents": 5183, "chunks": count,
        "queries": len(qrels), "metrics": {
            key: value for key, value in scored.items() if key != "by_query"},
        "latency_median_s": statistics.median(times),
        "latency_p95_s": times[math.ceil(.95 * len(times)) - 1],
        "duplicate_chunks_top10": duplicates,
        "missing_relevant_top10": sum(1 for row in scored["by_query"].values()
                                      if row["recall@10"] == 0)}
    (OUTPUT / "bm25-run.json").write_text(json.dumps(ranked, indent=2) + "\n", encoding="utf-8")
    (OUTPUT / "bm25-results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


def score(ranked, qrels, top_k=10):
    rows = {}
    for qid, relevant in qrels.items():
        ids = ranked[qid][:top_k]
        gains = [1 if doc_id in relevant else 0 for doc_id in ids]
        hit_ranks = [idx + 1 for idx, gain in enumerate(gains) if gain]
        ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(len(relevant), top_k) + 1))
        rows[qid] = {
            f"ndcg@{top_k}": sum(gain / math.log2(idx + 2) for idx, gain in enumerate(gains)) / ideal,
            f"recall@{top_k}": len(hit_ranks) / len(relevant),
            f"map@{top_k}": sum(n / rank for n, rank in enumerate(hit_ranks, 1)) / len(relevant),
            f"mrr@{top_k}": 1 / hit_ranks[0] if hit_ranks else 0,
            "hit_ranks": hit_ranks,
        }
    return {metric: statistics.mean(row[metric] for row in rows.values())
            for metric in (f"ndcg@{top_k}", f"recall@{top_k}", f"map@{top_k}", f"mrr@{top_k}")} | {"by_query": rows}


def evaluate(queries, qrels, top_k=10):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(url=DEFAULT_QDRANT_URL)
    count = client.count(collection_name=COLLECTION, exact=True).count
    state = json.loads(checkpoint_path().read_text(encoding="utf-8"))
    if state["next_document"] != 5183 or count != state["chunk_count"]:
        raise RuntimeError("Full-corpus ingestion has not been verified")
    modes = {
        "dense": dict(semantic=True, bm25=False, rank_fusion=False),
        "bm25": dict(semantic=False, bm25=True, rank_fusion=False),
        "hybrid": dict(semantic=True, bm25=True, rank_fusion=True, candidate_k=top_k * 3),
    }
    run = {mode: {} for mode in modes}
    latencies = {mode: [] for mode in modes}
    duplicate_counts = Counter()
    started = time.perf_counter()
    for index, qid in enumerate(qrels, 1):
        query = queries[qid]
        for mode, options in modes.items():
            tic = time.perf_counter()
            response = search(query, collection=COLLECTION, top_k=top_k, **options)
            latencies[mode].append(time.perf_counter() - tic)
            ids = [str(hit["document_id"]) for hit in response["hits"]]
            duplicate_counts[mode] += len(ids) - len(set(ids))
            run[mode][qid] = list(dict.fromkeys(ids))
        if index % 25 == 0 or index == len(qrels):
            print(f"evaluate {index}/{len(qrels)} queries, "
                  f"{time.perf_counter() - started:.1f}s", flush=True)
            partial_name = "run.partial.json" if top_k == 10 else f"run.k{top_k}.partial.json"
            (OUTPUT / partial_name).write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
    results = {}
    for mode in modes:
        times = sorted(latencies[mode])
        metrics = score(run[mode], qrels, top_k)
        results[mode] = {
            **{key: value for key, value in metrics.items() if key != "by_query"},
            "latency_median_s": statistics.median(times),
            "latency_p95_s": times[math.ceil(.95 * len(times)) - 1],
            f"duplicate_chunks_top{top_k}": duplicate_counts[mode],
            f"missing_relevant_top{top_k}": sum(1 for row in metrics["by_query"].values()
                                                if row[f"recall@{top_k}"] == 0),
        }
    manifest = {
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "BEIR SciFact test", "dataset_md5": EXPECTED_MD5,
        "collection": COLLECTION, "documents": 5183, "chunks": count,
        "queries": len(qrels), "top_k": top_k,
        "hybrid_candidate_k": top_k * 3, "results": results,
    }
    run_name = "run.json" if top_k == 10 else f"run.k{top_k}.json"
    results_name = "results.json" if top_k == 10 else f"results.k{top_k}.json"
    (OUTPUT / run_name).write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
    (OUTPUT / results_name).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("ingest", "evaluate", "all", "bm25-all"))
    parser.add_argument("--top-k", type=int, choices=(5, 10, 20, 30, 100), default=10)
    args = parser.parse_args()
    phase = args.phase
    corpus, queries, qrels = load_dataset()
    if phase in ("ingest", "all"):
        ingest(corpus)
    if phase in ("evaluate", "all"):
        evaluate(queries, qrels, args.top_k)
    if phase == "bm25-all":
        ingest_bm25(corpus)
        evaluate_bm25(queries, qrels)


if __name__ == "__main__":
    main()
