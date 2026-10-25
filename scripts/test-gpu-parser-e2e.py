"""Isolated live-Qdrant PDF ingest through CUDA parsing and BGE reload."""
from __future__ import annotations

import json
import os
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
test_id = uuid.uuid4().hex[:12]
test_root = ROOT / 'artifacts' / f'gpu-parser-e2e-{test_id}'
test_root.mkdir(parents=True, exist_ok=False)
os.environ['RAG3_BACKEND_TEST_INSTANCE'] = '1'
os.environ['RAG3_TEST_DATA_ROOT'] = str(test_root)
os.environ['RAG3_CUDA_PARSE_MIN_PAGES'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'

from qdrant_client import QdrantClient
from rag3.collections import create_collection
from rag3.gpu_model_service import GPUModelService
from rag3.ingestion_backend import BackendScheduler
from rag3.ingestion_jobs import one

collection = f'__rag3_gpu_parser_e2e_{test_id}'
source = ROOT / 'artifacts' / 'e2e-formats' / 'e2e.pdf'
client = QdrantClient(url='http://127.0.0.1:16333')
gpu = GPUModelService()
scheduler = BackendScheduler(gpu)
started = time.perf_counter()
job_id = None
try:
    create_collection(collection)
    assert len(gpu.submit_query('BGE loaded before PDF parsing').result(timeout=180)) == 1024
    scheduler.boot()
    submitted = scheduler.submit(origin='ui', collection=collection,
                                 path=str(source), requested=True)
    job_id = submitted['job_id']
    deadline = time.monotonic() + 360
    previous_stage = None
    while time.monotonic() < deadline:
        job = one(job_id, private=True)
        if job['stage'] != previous_stage:
            print(json.dumps({'status': job['status'], 'stage': job['stage']}), flush=True)
            previous_stage = job['stage']
        if job['status'] in {'COMPLETE', 'FAILED', 'PAUSED', 'CANCELLED'}:
            break
        time.sleep(.5)
    else:
        raise TimeoutError('isolated GPU parsing job did not finish')
    info = client.get_collection(collection_name=collection)
    result = {
        'status': job['status'], 'error': job.get('error'),
        'documents_written': (job.get('result') or {}).get('documents_written'),
        'timings': (job.get('result') or {}).get('timings'),
        'point_count': info.points_count,
        'bge_load_count': gpu.snapshot()['model_load_count'],
        'elapsed_seconds': round(time.perf_counter() - started, 2),
        'test_root': str(test_root),
    }
    (test_root / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)
    assert result['status'] == 'COMPLETE' and result['documents_written'] >= 1
    assert result['bge_load_count'] == 2
    assert result['timings']['conversion_seconds'] >= 0
    assert result['timings']['embedding_phase_seconds'] >= 0
finally:
    try:
        scheduler.stop()
        gpu.close()
    finally:
        if client.collection_exists(collection):
            client.delete_collection(collection_name=collection)
        client.close()
