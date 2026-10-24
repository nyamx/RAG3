"""Stage D live-backend crash/outage/GPU-failure acceptance in an isolated fixture.

The real HTTP/MCP backend entry point runs in subprocesses. Only its ingestion
worker and CUDA model loader are replaced with deterministic blocking/failing
test hooks. The journal, source, and Qdrant endpoint are test-specific; this
test must never stop the user's Qdrant, backend, WSL, or SQLite database.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import runpy
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PYTHON = ROOT / '.venv' / 'Scripts' / 'python.exe'
COLLECTION = '__rag3_stage_d_recovery_fixture'
POINT_ID = 'c91dbf4b-f16f-4c37-9a37-66dfd707c0d8'


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


class FakeQdrant(BaseHTTPRequestHandler):
    """Minimal wire-compatible fixture for checkpoint checks and BM25 search."""

    def reply(self, result):
        body = json.dumps({'result': result, 'status': 'ok', 'time': 0.001}).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == '/collections':
            self.reply({'collections': [{'name': COLLECTION}]})
        elif path == f'/collections/{COLLECTION}/exists':
            self.reply({'exists': True})
        elif path == f'/collections/{COLLECTION}':
            self.reply({'status': 'green', 'optimizer_status': 'ok', 'segments_count': 1,
                        'config': {'params': {
                            'vectors': {'text-dense': {'size': 1024, 'distance': 'Cosine'}},
                            'sparse_vectors': {'bm25': {'modifier': 'idf'}}},
                            'hnsw_config': {'m': 16, 'ef_construct': 100,
                                            'full_scan_threshold': 10000},
                            'optimizer_config': {'default_segment_number': 1,
                                                 'flush_interval_sec': 5},
                            'metadata': {'rag3': {'bm25': {'k': 1.2, 'b': 0.75,
                                                          'avg_len': 256}}}},
                        'payload_schema': {}})
        else:
            self.send_error(404)

    def do_POST(self):
        path = urlsplit(self.path).path
        try:
            payload = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))))
        except (ValueError, TypeError):
            self.send_error(400)
            return
        if path == f'/collections/{COLLECTION}/points/count':
            # The PARSING checkpoint legitimately has no committed points.
            self.reply({'count': 0})
        elif path == f'/collections/{COLLECTION}/points/query' and payload.get('using') == 'bm25':
            self.reply({'points': [{'id': POINT_ID, 'version': 1, 'score': 0.8,
                                   'payload': {'content': 'Isolated BM25 result after CUDA failure',
                                               'meta': {'document_id': 'fixture-document',
                                                        'title': 'Fixture', 'chunk_index': 0,
                                                        'page_start': 1, 'page_end': 1,
                                                        '_rag3_ingest_state': 'ready'}}}]})
        else:
            self.send_error(404)

    def log_message(self, *_args):
        pass


class ReusableServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def start_qdrant(port: int):
    server = ReusableServer(('127.0.0.1', port), FakeQdrant)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def stop_qdrant(server):
    if server is not None:
        server.shutdown()
        server.server_close()


def command(port: int, token: str, action: str, params=None, *, timeout=15):
    request = urllib.request.Request(
        f'http://127.0.0.1:{port}/api/v1/command',
        data=json.dumps({'action': action, 'params': params or {}}).encode(),
        headers={'Content-Type': 'application/json', 'X-RAG3-Token': token},
        method='POST')
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def health(port: int):
    with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=3) as response:
        return json.loads(response.read())


def wait_until(predicate, description: str, timeout: float = 35):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        value = predicate()
        if value:
            return value
        time.sleep(0.1)
    raise AssertionError(f'Timed out waiting for {description}')


def child_backend():
    """Run the actual backend entry point with only heavy work test-hooked."""
    sys.path.insert(0, str(ROOT))
    from rag3 import ingestion_backend
    from rag3.gpu_model_service import GPUModelService, GPUServiceError
    from rag3.constants import DATA_ROOT
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    (DATA_ROOT / 'child-phase.marker').write_text('imports-done', encoding='ascii')

    def blocked_execute(self, job):
        from rag3.constants import DATA_ROOT
        (DATA_ROOT / 'worker-running.marker').write_text(str(os.getpid()), encoding='ascii')
        while True:
            time.sleep(0.2)

    def unavailable_cuda(_self):
        raise GPUServiceError('CUDA_UNAVAILABLE', 'injected isolated CUDA init failure')

    ingestion_backend.BackendScheduler._execute = blocked_execute
    GPUModelService._load_cuda_model = unavailable_cuda
    (DATA_ROOT / 'child-phase.marker').write_text('runpy-entrypoint', encoding='ascii')
    runpy.run_path(str(ROOT / 'scripts' / 'rag3-mcp-server.py'), run_name='__main__')


def seed_job(data_root: pathlib.Path):
    from rag3 import ingestion_jobs as jobs
    from rag3.ingestion_settings import progress_identity
    assert jobs.DB_PATH.resolve().is_relative_to(data_root.resolve())
    assert not jobs.DB_PATH.resolve().is_relative_to((ROOT / 'data').resolve())
    original = data_root / 'original' / 'fixture.txt'
    original.parent.mkdir(parents=True)
    original.write_text('Isolated original source; preserve exactly on crash.\n', encoding='utf-8')
    digest = hashlib.sha256(original.read_bytes()).hexdigest()
    job_id = uuid.uuid4().hex
    with jobs.connect() as db:
        db.execute('''INSERT INTO jobs (job_id,origin,collection,filename,source,
                   source_sha256,status,stage,created_at,updated_at,requested)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)''',
                   (job_id, 'UI', COLLECTION, original.name, str(original), digest,
                    'QUEUED', 'Ready', jobs.now(), jobs.now(), 0))
    jobs.set_ingestion_progress(COLLECTION, digest,
                                progress_identity(digest, COLLECTION), 'PARSING')
    return jobs, job_id, original, digest


def spawn_backend(env, log_path):
    output = log_path.open('ab')
    try:
        proc = subprocess.Popen([str(PYTHON), '-B', str(pathlib.Path(__file__).resolve()),
                                 '--backend-child'], cwd=ROOT, env=env,
                                stdin=subprocess.DEVNULL, stdout=output,
                                stderr=subprocess.STDOUT,
                                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    finally:
        output.close()
    return proc


def wait_backend(proc, port, log_path):
    last_error = None
    def ready():
        nonlocal last_error
        if proc.poll() is not None:
            raise AssertionError('Backend exited: ' + log_path.read_text('utf-8', 'replace')[-1200:])
        try:
            state = health(port)
            if state.get('backend') == 'ready':
                return state
        except (OSError, ValueError) as error:
            last_error = repr(error)
        return None
    try:
        return wait_until(ready, 'backend health', timeout=35)
    except AssertionError as error:
        tail = log_path.read_text('utf-8', 'replace')[-2400:] if log_path.exists() else '(no log)'
        phase_file = pathlib.Path(os.environ['RAG3_TEST_DATA_ROOT']) / 'child-phase.marker'
        phase = phase_file.read_text('ascii') if phase_file.exists() else '(none)'
        raise AssertionError(f'{error}; child phase={phase}; last health error={last_error}; '
                             f'backend log tail: {tail}') from error


def kill_test_process(proc):
    if proc is None or proc.poll() is not None:
        return
    if os.name == 'nt':
        result = subprocess.run(['taskkill', '/F', '/T', '/PID', str(proc.pid)],
                                capture_output=True, timeout=18,
                                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        if result.returncode and proc.poll() is None:
            proc.kill()
    else:
        proc.kill()
    proc.wait(timeout=15)


def main():
    if not PYTHON.is_file():
        raise RuntimeError('RAG3 venv Python is required')
    if os.name != 'nt':
        raise RuntimeError('This process-tree crash test targets the Windows deployment')
    with tempfile.TemporaryDirectory(prefix='rag3-stage-d-backend-recovery-') as temp:
        data_root = pathlib.Path(temp) / 'isolated-data'
        port = free_port()
        qdrant_port = free_port()
        while port in {8000, 16333, 18747, qdrant_port} or qdrant_port in {16333, 18747}:
            port, qdrant_port = free_port(), free_port()
        token_file = ROOT / 'data' / f'rag3-backend-ui-{port}.token'
        lock_file = ROOT / 'data' / f'backend-test-{port}.lock'
        assert not token_file.exists() and not lock_file.exists(), 'Port-scoped backend state already exists'
        env = os.environ.copy()
        env.update(RAG3_TEST_DATA_ROOT=str(data_root),
                   RAG3_BACKEND_TEST_INSTANCE=str(port),
                   RAG3_QDRANT_URL=f'http://127.0.0.1:{qdrant_port}',
                   RAG3_MCP_PORT=str(port), RAG3_MCP_TRANSPORT='streamable-http',
                   RAG3_MCP_WRITE_ENABLED='0', RAG3_MCP_DELETE_ENABLED='0',
                   PYTHONUNBUFFERED='1')
        os.environ.update({key: env[key] for key in
                           ('RAG3_TEST_DATA_ROOT', 'RAG3_BACKEND_TEST_INSTANCE', 'RAG3_QDRANT_URL')})
        jobs, job_id, original, digest = seed_job(data_root)
        progress_before = jobs.ingestion_progress(COLLECTION, digest)
        log_path = pathlib.Path(temp) / 'backend.log'
        qdrant = start_qdrant(qdrant_port)
        proc = None
        results = {}
        try:
            proc = spawn_backend(env, log_path)
            first = wait_backend(proc, port, log_path)
            assert first['qdrant_transport'] == 'reachable', first
            token = token_file.read_text('ascii')
            status, response = command(port, token, 'jobs.action',
                                       {'job_id': job_id, 'operation': 'run'})
            assert status == 200 and response['ok'], (status, response)

            def running():
                status, result = command(port, token, 'jobs.status', {'job_id': job_id})
                if status == 200 and result['result']['status'] == 'RUNNING' and \
                        (data_root / 'worker-running.marker').exists():
                    return result['result']
                return None
            wait_until(running, 'RUNNING journal + worker marker')
            kill_test_process(proc)
            assert jobs.one(job_id, private=True)['status'] == 'RUNNING'
            assert hashlib.sha256(original.read_bytes()).hexdigest() == digest
            assert jobs.ingestion_progress(COLLECTION, digest) == progress_before
            results['os_kill_running_backend_preserved_checkpoint_and_original'] = True

            stop_qdrant(qdrant)
            qdrant = None
            (data_root / 'worker-running.marker').unlink()
            proc = spawn_backend(env, log_path)
            restarted = wait_backend(proc, port, log_path)
            assert restarted['pid'] != first['pid'], (first, restarted)
            assert restarted['qdrant_transport'] == 'unreachable', restarted
            token2 = token_file.read_text('ascii')
            assert token2 != token
            status, response = command(port, token2, 'jobs.status', {'job_id': job_id})
            assert status == 200 and response['result']['status'] == 'PAUSED', (status, response)
            assert response['result']['checkpoint_available']
            assert 'deferred' in response['result']['stage']
            status, failure = command(port, token2, 'jobs.action',
                                      {'job_id': job_id, 'operation': 'resume'})
            assert status != 200 and not failure['ok'], (status, failure)
            assert jobs.one(job_id, private=True)['status'] == 'PAUSED'
            assert jobs.ingestion_progress(COLLECTION, digest) == progress_before
            results['qdrant_outage_defers_verification_and_rejects_resume'] = True

            qdrant = start_qdrant(qdrant_port)
            wait_until(lambda: health(port)['qdrant_transport'] == 'reachable',
                       'Qdrant transport recovery')
            status, resumed = command(port, token2, 'jobs.action',
                                      {'job_id': job_id, 'operation': 'resume'})
            assert status == 200 and resumed['ok'], (status, resumed)
            assert resumed['result']['job_id'] == job_id
            wait_until(lambda: jobs.one(job_id, private=True)['status'] == 'RESUMING',
                       'manual resume started')
            assert hashlib.sha256(original.read_bytes()).hexdigest() == digest
            assert jobs.ingestion_progress(COLLECTION, digest) == progress_before
            results['manual_resume_after_qdrant_recovery'] = True

            status, semantic = command(port, token2, 'search',
                {'query': 'fixture', 'collection': COLLECTION,
                 'options': {'semantic': True, 'bm25': False,
                             'rank_fusion': False, 'top_k': 1}}, timeout=100)
            assert status == 503 and semantic['error']['code'] == 'CUDA_UNAVAILABLE', \
                (status, semantic)
            failed_gpu = health(port)['gpu_model_service']
            assert failed_gpu['state'] == 'failed' and failed_gpu['model_load_count'] == 0
            status, bm25 = command(port, token2, 'search',
                {'query': 'fixture', 'collection': COLLECTION,
                 'options': {'semantic': False, 'bm25': True,
                             'rank_fusion': False, 'top_k': 1}}, timeout=100)
            if status != 200 or not bm25.get('ok'):
                # Only for a failing fixture: expose the local Qdrant schema/
                # response mismatch that the public backend correctly masks.
                import traceback
                from rag3.retrieval import search
                try:
                    search('fixture', collection=COLLECTION,
                           qdrant_url=f'http://127.0.0.1:{qdrant_port}',
                           semantic=False, bm25=True, rank_fusion=False, top_k=1)
                except Exception:
                    raise AssertionError(f'BM25 fixture failed: {status} {bm25}; '
                                         f'local trace: {traceback.format_exc()}')
            assert status == 200 and bm25['ok'], (status, bm25)
            assert bm25['result']['ui']['hits'][0]['point_id'] == POINT_ID
            assert health(port)['gpu_model_service']['model_load_count'] == 0
            results['cuda_init_failure_semantic_error_bm25_still_works'] = True

            with jobs.connect() as db:
                assert db.execute('PRAGMA quick_check').fetchone()[0] == 'ok'
                assert db.execute('SELECT COUNT(*) FROM jobs').fetchone()[0] == 1
            print(json.dumps({'status': 'PASS', 'checks': results,
                              'isolation': 'temporary SQLite/source + fake Qdrant; '
                                           'only test-owned backend processes killed',
                              'limits': 'ingestion worker and CUDA loader were injected; '
                                        'no real Docling/CUDA/Qdrant write'},
                             sort_keys=True))
        finally:
            kill_test_process(proc)
            stop_qdrant(qdrant)
            token_file.unlink(missing_ok=True)
            lock_file.unlink(missing_ok=True)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--backend-child':
        child_backend()
    else:
        main()
