"""One-command, isolated live PDF -> CUDA Docling -> BGE -> Qdrant check.

Examples:
  .venv/Scripts/python.exe scripts/verify-rag3-pdf-gpu.py
  .venv/Scripts/python.exe scripts/verify-rag3-pdf-gpu.py --pdf book.pdf --pages 24
  .venv/Scripts/python.exe scripts/verify-rag3-pdf-gpu.py --pdf book.pdf --pages 0

Test results and managed source copies go under artifacts/verify-pdf-gpu-*.
The source PDF is read-only. A local Qdrant server is started if needed, and
the uniquely named test collection is removed after the run.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pdf', action='append', type=Path, default=[],
                        help='PDF to verify; repeat for multiple documents')
    parser.add_argument('--pages', type=int, default=24,
                        help='Pages per PDF from --start-page; 0 means the full PDF (default: 24)')
    parser.add_argument('--start-page', type=int, default=1, help='1-based first page (default: 1)')
    parser.add_argument('--timeout-minutes', type=float, default=30,
                        help='Deadline for each ingestion job (default: 30)')
    parser.add_argument('--min-free-ram-gib', type=float, default=4,
                        help='Cancel if system available RAM drops below this (default: 4)')
    parser.add_argument('--qdrant-url', default='http://127.0.0.1:16333')
    args = parser.parse_args()
    if args.pages < 0 or args.start_page < 1 or args.timeout_minutes <= 0 or args.min_free_ram_gib < 0:
        parser.error('--pages >= 0, --start-page >= 1, --timeout-minutes > 0 and --min-free-ram-gib >= 0 are required')
    for path in args.pdf:
        if not path.is_file() or path.suffix.lower() != '.pdf':
            parser.error(f'PDF does not exist: {path}')
    return args


def ensure_qdrant(url: str) -> str:
    try:
        with urlopen(url.rstrip('/') + '/collections', timeout=3) as response:
            if response.status == 200:
                return 'already_running'
    except Exception:
        pass
    if os.name != 'nt' or url != 'http://127.0.0.1:16333':
        raise RuntimeError(f'Qdrant is unavailable at {url}')
    script = ROOT / 'scripts' / 'ensure-infra.ps1'
    result = subprocess.run(
        ['powershell.exe', '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass',
         '-File', str(script)], cwd=ROOT, capture_output=True, text=True, timeout=120,
        encoding='utf-8', errors='replace',
    )
    if result.returncode:
        raise RuntimeError('Qdrant bootstrap failed: ' + (result.stderr or result.stdout)[-800:])
    with urlopen(url.rstrip('/') + '/collections', timeout=5) as response:
        if response.status != 200:
            raise RuntimeError(f'Qdrant returned HTTP {response.status}')
    return 'bootstrapped'


def prepare_pdf(source: Path | None, folder: Path, pages: int, start_page: int):
    from pypdfium2 import PdfDocument

    if source is None:
        spec = importlib.util.spec_from_file_location(
            'rag3_page_fixture', ROOT / 'scripts' / 'test-docling-page-range.py')
        assert spec and spec.loader
        fixture = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixture)
        source = folder / 'generated-nine-pages.pdf'
        fixture.make_pdf(source, page_count=9)
    source = source.resolve()
    pdf = PdfDocument(str(source))
    try:
        total_pages = len(pdf)
        if start_page > total_pages:
            raise ValueError(f'{source.name}: start page {start_page} exceeds {total_pages}')
        end_page = total_pages if pages == 0 else min(total_pages, start_page + pages - 1)
        if start_page == 1 and end_page == total_pages:
            selected = source
        else:
            selected = folder / 'selected-pages.pdf'
            subset = PdfDocument.new()
            try:
                subset.import_pages(pdf, list(range(start_page - 1, end_page)))
                subset.save(str(selected))
            finally:
                subset.close()
    finally:
        pdf.close()
    return selected, {'source_name': source.name, 'source_bytes': source.stat().st_size,
                      'source_pages': total_pages, 'selected_pages': end_page - start_page + 1,
                      'selected_bytes': selected.stat().st_size, 'first_page': start_page}


class ResourceSampler:
    """Device-wide nvidia-smi readings, plus this test process and its children."""

    def __init__(self, min_free_ram_gib: float):
        self._stop = threading.Event()
        self._thread = None
        self._min_free_ram_mib = min_free_ram_gib * 1024
        self.abort_reason = None
        self.current = None
        self.results = {}

    def start(self):
        self._thread = threading.Thread(target=self._run, name='rag3-resource-sampler', daemon=True)
        self._thread.start()

    def select(self, key):
        self.current = key
        self.abort_reason = None
        self.results[key] = {'samples': 0, 'gpu_used_mib_peak': None,
                             'gpu_free_mib_min': None, 'gpu_util_percent_peak': None,
                             'process_tree_rss_mib_peak': 0, 'system_available_mib_min': None}

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self):
        import psutil
        parent = psutil.Process()
        while not self._stop.is_set():
            key = self.current
            if key is not None:
                sample = self.results[key]
                try:
                    processes = [parent, *parent.children(recursive=True)]
                    rss = sum(p.memory_info().rss for p in processes if p.is_running()) / 2**20
                    sample['process_tree_rss_mib_peak'] = round(max(
                        sample['process_tree_rss_mib_peak'], rss), 1)
                except psutil.Error:
                    pass
                available = psutil.virtual_memory().available / 2**20
                prior = sample['system_available_mib_min']
                sample['system_available_mib_min'] = round(
                    available if prior is None else min(prior, available), 1)
                if available < self._min_free_ram_mib:
                    self.abort_reason = f'system available RAM fell to {available:.0f} MiB'
                try:
                    proc = subprocess.run(
                        ['nvidia-smi', '-i', '0',
                         '--query-gpu=memory.used,memory.free,utilization.gpu',
                         '--format=csv,noheader,nounits'], capture_output=True, text=True,
                        timeout=4, check=True)
                    used, free, util = (int(x.strip()) for x in proc.stdout.splitlines()[0].split(','))
                    sample['gpu_used_mib_peak'] = max(sample['gpu_used_mib_peak'] or 0, used)
                    prior_free = sample['gpu_free_mib_min']
                    sample['gpu_free_mib_min'] = free if prior_free is None else min(prior_free, free)
                    sample['gpu_util_percent_peak'] = max(sample['gpu_util_percent_peak'] or 0, util)
                except (OSError, ValueError, subprocess.SubprocessError, IndexError):
                    sample['gpu_sampling_error'] = True
                sample['samples'] += 1
            self._stop.wait(.5)


def verify_search(collection, qdrant_url, gpu, client, source_sha):
    from rag3.retrieval import search
    from qdrant_client import models

    points, _ = client.scroll(
        collection_name=collection, limit=1, with_payload=True, with_vectors=False,
        scroll_filter=models.Filter(must=[models.FieldCondition(
            key='meta.source_sha256', match=models.MatchValue(value=source_sha))]))
    if not points:
        raise AssertionError('No stored chunk to use as a search probe')
    content = (points[0].payload or {}).get('content') or ''
    query = re.sub(r'\s+', ' ', content).strip()[:200]
    if not query:
        raise AssertionError('Stored chunk has no searchable text')
    checks = {}
    for mode in ('dense', 'sparse', 'hybrid'):
        response = search(query, mode=mode, collection=collection, qdrant_url=qdrant_url,
                          top_k=3, dense_encoder=lambda text: gpu.submit_query(text).result(timeout=180))
        hits = response['hits']
        if not hits or not any(hit.get('document_id') == source_sha for hit in hits):
            raise AssertionError(f'{mode} search did not return the ingested PDF')
        checks[mode] = {'hits': len(hits), 'source_found': True}
    return checks


def main():
    args = parse_args()
    run_id = uuid.uuid4().hex[:12]
    run_root = ROOT / 'artifacts' / f'verify-pdf-gpu-{run_id}'
    run_root.mkdir(parents=True, exist_ok=False)
    os.environ.update(RAG3_BACKEND_TEST_INSTANCE='1', RAG3_TEST_DATA_ROOT=str(run_root),
                      RAG3_QDRANT_URL=args.qdrant_url, HF_HUB_OFFLINE='1',
                      TRANSFORMERS_OFFLINE='1', RAG3_CUDA_PARSE_MIN_PAGES='1',
                      OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    report = {'status': 'RUNNING', 'run_root': str(run_root),
              'scope': 'isolated collection; forced CUDA parsing; BGE reload; dense/BM25/hybrid search',
              'device_memory_scope': 'whole GPU, not per process', 'documents': []}
    sampler = ResourceSampler(args.min_free_ram_gib)
    gpu = scheduler = client = None
    collections = []
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable')
        report['gpu_name'] = torch.cuda.get_device_name(0)
        report['qdrant'] = ensure_qdrant(args.qdrant_url)
        from qdrant_client import QdrantClient
        from rag3.collections import create_collection
        from rag3.gpu_model_service import GPUModelService
        from rag3.ingestion_backend import BackendScheduler
        from rag3.ingestion_jobs import one

        client = QdrantClient(url=args.qdrant_url, timeout=120)
        gpu = GPUModelService()
        scheduler = BackendScheduler(gpu)
        scheduler.boot()
        sampler.start()
        sources = args.pdf or [None]
        for index, source in enumerate(sources, 1):
            folder = run_root / f'document-{index}'
            folder.mkdir()
            selected, source_info = prepare_pdf(source, folder, args.pages, args.start_page)
            collection = f'__rag3_verify_pdf_gpu_{run_id}_{index}'
            entry = {**source_info, 'collection': collection, 'status': 'RUNNING'}
            report['documents'].append(entry)
            sampler.select(index)
            start = time.monotonic()
            before_loads = gpu.snapshot()['model_load_count']
            if index == 1:
                probe = gpu.submit_query('RAG3 GPU model prewarm').result(timeout=180)
                if len(probe) != 1024:
                    raise AssertionError('BGE prewarm vector is not 1024-dimensional')
                before_loads = gpu.snapshot()['model_load_count']
            create_collection(collection, url=args.qdrant_url)
            collections.append(collection)
            submitted = scheduler.submit(origin='ui', collection=collection,
                                         path=str(selected), requested=True)
            job_id = submitted['job_id']
            deadline = time.monotonic() + args.timeout_minutes * 60
            last_stage = None
            while time.monotonic() < deadline:
                if sampler.abort_reason:
                    scheduler.action(job_id, 'cancel')
                    raise MemoryError(sampler.abort_reason)
                job = one(job_id, private=True)
                if job.get('stage') != last_stage:
                    print(json.dumps({'document': index, 'status': job['status'],
                                      'stage': job.get('stage')}, ensure_ascii=False), flush=True)
                    last_stage = job.get('stage')
                if job['status'] in {'COMPLETE', 'FAILED', 'PAUSED', 'CANCELLED'}:
                    break
                time.sleep(.5)
            else:
                scheduler.action(job_id, 'cancel')
                raise TimeoutError(f'PDF {index} did not complete within {args.timeout_minutes} minutes')
            entry.update(status=job['status'], error=job.get('error'),
                         timings=(job.get('result') or {}).get('timings'),
                         seconds=round(time.monotonic() - start, 1),
                         bge_loads_before_parse=before_loads,
                         bge_loads_after_ingest=gpu.snapshot()['model_load_count'])
            if job['status'] != 'COMPLETE':
                raise AssertionError(f'PDF {index} ingestion ended {job["status"]}: {job.get("error")}')
            result = job['result']
            counts = [result[k] for k in ('source_point_count', 'source_dense_count',
                                          'source_bm25_count', 'documents_written')]
            if not counts[0] or len(set(counts)) != 1:
                raise AssertionError(f'PDF {index} Qdrant counts disagree: {counts}')
            entry['chunks'] = counts[0]
            entry['bge_reloaded_after_parse'] = entry['bge_loads_after_ingest'] == before_loads + 1
            if not entry['bge_reloaded_after_parse']:
                raise AssertionError(f'PDF {index} did not release and reload BGE exactly once')
            entry['search'] = verify_search(collection, args.qdrant_url, gpu, client,
                                            result['source_sha256'])
            entry['resources'] = dict(sampler.results[index])
            print(json.dumps({'document': index, 'status': 'PASS', 'pages': entry['selected_pages'],
                              'chunks': entry['chunks'], 'resources': entry['resources']}), flush=True)
        report['status'] = 'PASS'
    except BaseException as exc:
        report['status'] = 'FAIL'
        report['error'] = f'{type(exc).__name__}: {exc}'[:1000]
        print(report['error'], file=sys.stderr, flush=True)
    finally:
        sampler.stop()
        for index, entry in enumerate(report['documents'], 1):
            if index in sampler.results:
                entry['resources'] = dict(sampler.results[index])
        if scheduler is not None:
            try:
                scheduler.stop()
            except Exception as exc:
                report.setdefault('cleanup_errors', []).append(f'scheduler: {exc}')
        if gpu is not None:
            try:
                gpu.close()
            except Exception as exc:
                report.setdefault('cleanup_errors', []).append(f'gpu: {exc}')
        if client is not None:
            for collection in collections:
                try:
                    if client.collection_exists(collection):
                        client.delete_collection(collection_name=collection)
                except Exception as exc:
                    report.setdefault('cleanup_errors', []).append(f'{collection}: {exc}')
            client.close()
        if report.get('cleanup_errors'):
            report['status'] = 'FAIL'
        path = run_root / 'report.json'
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps({'status': report['status'], 'report': str(path)}, ensure_ascii=False), flush=True)
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
