"""Offscreen Qt -> real HTTP backend job controls, with isolated SQLite/Qdrant.

Only the ingestion *work* is replaced by a deterministic wait, so this checks
Stage D UI wiring and the real backend scheduler without loading BGE/Docling.
No request is sent to the operational Qdrant port or job database.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import socket
import sqlite3
from contextlib import closing
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.parse import unquote, urlsplit
from urllib.request import Request, urlopen
from uuid import uuid4

import psutil


ROOT = Path(__file__).resolve().parents[1]
BASE_COLLECTION = '__rag3_stage_d_controls_' + uuid4().hex[:10]


class FakeQdrant(BaseHTTPRequestHandler):
    collections = {BASE_COLLECTION}
    lock = threading.Lock()
    deleted: list[str] = []

    def respond(self, value, status=200):
        raw = json.dumps({'result': value, 'status': 'ok', 'time': 0.001}).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def collection(self):
        pieces = urlsplit(self.path).path.split('/')
        return unquote(pieces[2]) if len(pieces) > 2 else ''

    def do_GET(self):
        path = urlsplit(self.path).path
        with self.lock:
            names = sorted(self.collections)
        if path == '/collections':
            self.respond({'collections': [{'name': name} for name in names]})
        elif path.endswith('/exists'):
            self.respond({'exists': self.collection() in names})
        elif path == f'/collections/{self.collection()}' and self.collection() in names:
            self.respond({'status': 'green', 'optimizer_status': 'ok',
                          'segments_count': 1, 'points_count': 0,
                          'config': {'params': {
                              'vectors': {'text-dense': {'size': 1024, 'distance': 'Cosine'}},
                              'sparse_vectors': {'bm25': {'modifier': 'idf'}}},
                              'metadata': {'rag3': {'bm25': {'k': 1.2, 'b': 0.75,
                                                               'avg_len': 256}}}},
                          'payload_schema': {}})
        else:
            self.send_error(404)

    def do_POST(self):
        path = urlsplit(self.path).path
        if path.endswith('/points/count'):
            self.respond({'count': 0})
        elif path.endswith('/points/scroll'):
            self.respond({'points': [], 'next_page_offset': None})
        else:
            self.send_error(404)

    def do_PUT(self):
        name = self.collection()
        if not name or urlsplit(self.path).path != f'/collections/{name}':
            self.send_error(404)
            return
        with self.lock:
            self.collections.add(name)
        self.respond(True)

    def do_DELETE(self):
        name = self.collection()
        with self.lock:
            if name not in self.collections:
                self.send_error(404)
                return
            self.collections.remove(name)
            self.deleted.append(name)
        self.respond(True)

    def log_message(self, *_args):
        pass


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def request(base: str, path: str, token: str | None = None, body: dict | None = None):
    headers = {'Content-Type': 'application/json'}
    if token:
        headers['X-RAG3-Token'] = token
    payload = None if body is None else json.dumps(body).encode()
    req = Request(base + path, data=payload, headers=headers,
                  method='GET' if body is None else 'POST')
    try:
        with urlopen(req, timeout=12) as response:
            return response.status, json.loads(response.read())
    except HTTPError as exc:
        return exc.code, json.loads(exc.read())


def command(base: str, token: str, action: str, **params):
    status, data = request(base, '/api/v1/command', token,
                           {'action': action, 'params': params})
    assert status == 200 and data.get('ok'), (action, status, data)
    return data['result']


BOOTSTRAP = r'''
import os, runpy, time
import rag3.ingestion_backend as backend
import rag3.ingestion_manager as manager
from rag3.ingestion_jobs import ingestion_progress
from pathlib import Path

# The scheduler, journal, HTTP routes, and UI controls remain real. Only the
# parsing/embedding workload is replaced so a T1000 or production Qdrant is
# never used during this Stage D UI control check.
backend._client = lambda _url: object()
backend._source_counts = lambda *_args, **_kwargs: (0, 0, 0, 0)
backend._compatible = lambda job, **_kwargs: ingestion_progress(
    job['collection'], job['source_sha256'])
backend._stage_plain = lambda *_args, **_kwargs: None
def wait_for_control(_job, _gpu, should_stop, _emit, _url, **_kwargs):
    while not should_stop():
        time.sleep(.025)
    return {'status': 'PAUSED'}
backend._embed_source = wait_for_control
undelayed_client = manager.client
def delayed_client(action, **params):
    if action == 'submit' and (Path(os.environ['RAG3_TEST_DATA_ROOT']) / 'delay-submit').exists():
        time.sleep(.6)
    return undelayed_client(action, **params)
manager.client = delayed_client
runpy.run_path(os.environ['RAG3_STAGE_D_SERVER'], run_name='__main__')
'''


def wait_for(app, predicate, label, *, seconds=8):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return
        time.sleep(.025)
    raise AssertionError(f'{label} timed out')


class OneStaleRead:
    """Reproduce a token-file read that races with a genuine backend restart."""
    def __init__(self, path: Path, previous_token: str):
        self.path = path
        self.previous_token = previous_token
        self.reads = 0

    def read_text(self, *, encoding: str):
        self.reads += 1
        if self.reads == 1:
            return self.previous_token
        return self.path.read_text(encoding=encoding)


def _same_process(pid: int, created: float) -> psutil.Process | None:
    try:
        process = psutil.Process(pid)
        if abs(process.create_time() - created) < .01:
            return process
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    return None


def stop_test_process_tree(proc: subprocess.Popen, created: float, port: int):
    """Stop only PIDs descended from this exact test launch; fail if still bound."""
    root = _same_process(proc.pid, created)
    if root is not None:
        descendants = []
        for child in root.children(recursive=True):
            try:
                descendants.append((child.pid, child.create_time()))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        # Children first, then the venv redirector or direct interpreter.
        for pid, started in reversed(descendants):
            child = _same_process(pid, started)
            if child is not None:
                try:
                    child.terminate()
                except psutil.NoSuchProcess:
                    pass
        for pid, started in reversed(descendants):
            child = _same_process(pid, started)
            if child is not None:
                try:
                    child.wait(timeout=5)
                except psutil.TimeoutExpired:
                    same = _same_process(pid, started)
                    if same is not None:
                        same.kill()
        root = _same_process(proc.pid, created)
        if root is not None:
            try:
                root.terminate()
                root.wait(timeout=5)
            except psutil.TimeoutExpired:
                same = _same_process(proc.pid, created)
                if same is not None:
                    same.kill()
            except psutil.NoSuchProcess:
                pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        raise AssertionError('test backend parent is still running')
    # A surviving test child can keep SQLite open even when Popen has exited.
    for _ in range(80):
        with socket.socket() as sock:
            sock.settimeout(.1)
            if sock.connect_ex(('127.0.0.1', port)) != 0:
                return
        time.sleep(.05)
    raise AssertionError('test backend port is still open; token/lock retained')


def main():
    if sys.executable.lower().endswith('pythonw.exe'):
        raise RuntimeError('Run this regression with .venv/Scripts/python.exe')
    qdrant = ThreadingHTTPServer(('127.0.0.1', 0), FakeQdrant)
    qthread = threading.Thread(target=qdrant.serve_forever, daemon=True)
    qthread.start()
    port = free_port()
    scope = 'stage_d_ui_' + uuid4().hex[:12]
    token_file = ROOT / 'data' / f'rag3-backend-ui-{port}.token'
    lock_file = ROOT / 'data' / f'backend-test-{scope}.lock'
    assert port != 8000 and qdrant.server_port != 16333
    assert not token_file.exists() and not lock_file.exists()
    proc = None
    proc_created = None
    with tempfile.TemporaryDirectory(prefix='rag3-stage-d-ui-controls-') as temp:
        isolated = (Path(temp) / 'data').resolve()
        assert not isolated.is_relative_to((ROOT / 'data').resolve())
        env = os.environ.copy()
        env.update(RAG3_TEST_DATA_ROOT=str(isolated),
                   RAG3_BACKEND_TEST_INSTANCE=scope,
                   RAG3_QDRANT_URL=f'http://127.0.0.1:{qdrant.server_port}',
                   RAG3_MCP_PORT=str(port), RAG3_MCP_TRANSPORT='streamable-http',
                   RAG3_STAGE_D_SERVER=str(ROOT / 'scripts' / 'rag3-mcp-server.py'),
                   QT_QPA_PLATFORM='offscreen', RAG3_MCP_WRITE_ENABLED='1',
                   RAG3_MCP_DELETE_ENABLED='1')
        log = Path(temp) / 'backend.log'
        try:
            with log.open('wb') as output:
                proc = subprocess.Popen([sys.executable, '-B', '-c', BOOTSTRAP],
                                        cwd=ROOT, env=env, stdout=output,
                                        stderr=subprocess.STDOUT,
                                        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                proc_created = psutil.Process(proc.pid).create_time()
                base = f'http://127.0.0.1:{port}'
                for _ in range(1200):
                    assert proc.poll() is None, log.read_text('utf-8', 'replace')[-2500:]
                    try:
                        status, health = request(base, '/health')
                        if status == 200:
                            break
                    except (OSError, ValueError):
                        pass
                    time.sleep(.1)
                else:
                    raise AssertionError('isolated backend did not become healthy: ' +
                                         log.read_text('utf-8', 'replace')[-2500:])
                assert token_file.is_file()
                owner_ids = {proc.pid}
                owner_ids.update(child.pid for child in
                                 psutil.Process(proc.pid).children(recursive=True))
                assert health['pid'] in owner_ids, 'health belongs to another process'
                token = token_file.read_text(encoding='ascii').strip()
                assert health['gpu_model'] == 'not_initialized'
                status, _ = request(base, '/api/v1/command', body={
                    'action': 'jobs.list', 'params': {}})
                assert status == 401

                os.environ.update({key: env[key] for key in (
                    'RAG3_TEST_DATA_ROOT', 'RAG3_BACKEND_TEST_INSTANCE',
                    'RAG3_QDRANT_URL', 'RAG3_MCP_PORT', 'QT_QPA_PLATFORM')})
                from PySide6.QtWidgets import QApplication
                spec = importlib.util.spec_from_file_location(
                    'stage_d_ui_controls', ROOT / 'scripts' / 'ingest-ui.py')
                ui = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = ui
                spec.loader.exec_module(ui)
                app = QApplication.instance() or QApplication([])
                window = ui.IngestionWindow()
                wait_for(app, lambda: window.collection.findText(BASE_COLLECTION) >= 0,
                         'UI collection list')
                window.collection.setText(BASE_COLLECTION)

                def state(job_id, expected):
                    return command(base, token, 'jobs.status', job_id=job_id)['status'] == expected

                def ui_state(job_id, expected):
                    window.load_jobs()
                    wait_for(app, lambda: any(job['job_id'] == job_id and
                        job['status'] == expected for job in window.jobs),
                        f'UI {job_id} {expected}')
                    row = next(i for i, job in enumerate(window.jobs)
                               if job['job_id'] == job_id)
                    window.table.selectRow(row)

                original = isolated / 'original' / 'ui-one.txt'
                original.parent.mkdir(parents=True, exist_ok=True)
                original.write_text('An isolated UI control source.\n', encoding='utf-8')
                window.add_path(str(original))
                wait_for(app, lambda: len(window.jobs) == 1, 'UI job submission')
                job_id = window.jobs[0]['job_id']
                ui_state(job_id, 'QUEUED')
                with closing(sqlite3.connect(isolated / 'ingestion_jobs.sqlite')) as db:
                    requested, origin = db.execute(
                        'SELECT requested, origin FROM jobs WHERE job_id=?',
                        (job_id,)).fetchone()
                assert (requested, origin) == (0, 'UI')
                assert command(base, token, 'jobs.local_source', job_id=job_id)['source'] == str(original)
                print('UI_SUBMIT_QUEUED_AND_LOCAL_SOURCE PASS', flush=True)

                window.buttons['run'].click()
                wait_for(app, lambda: state(job_id, 'RUNNING'), 'Run -> RUNNING')
                ui_state(job_id, 'RUNNING')
                window.buttons['pause'].click()
                wait_for(app, lambda: state(job_id, 'PAUSED'), 'Pause -> PAUSED')
                ui_state(job_id, 'PAUSED')
                window.buttons['resume'].click()
                try:
                    wait_for(app, lambda: state(job_id, 'RESUMING'), 'Resume -> RESUMING')
                except AssertionError:
                    print('RESUME_DIAG', command(base, token, 'jobs.status', job_id=job_id),
                          window.errors.toPlainText(), flush=True)
                    raise
                ui_state(job_id, 'RESUMING')
                window.buttons['cancel'].click()
                wait_for(app, lambda: state(job_id, 'CANCELLED'), 'Cancel -> CANCELLED')
                ui_state(job_id, 'CANCELLED')
                window.buttons['retry'].click()
                wait_for(app, lambda: state(job_id, 'RESUMING'), 'Retry -> RESUMING')
                ui_state(job_id, 'RESUMING')
                window.buttons['cancel'].click()
                wait_for(app, lambda: state(job_id, 'CANCELLED'), 'second Cancel')
                ui_state(job_id, 'CANCELLED')
                window.buttons['delete'].click()
                wait_for(app, lambda: all(j['job_id'] != job_id for j in window.jobs),
                         'UI Hide')
                assert any(j['job_id'] == job_id for j in
                           command(base, token, 'jobs.list')['jobs'])
                assert not any(j['job_id'] == job_id for j in
                               command(base, token, 'jobs.list', include_hidden=False)['jobs'])
                print('UI_RUN_PAUSE_RESUME_CANCEL_RETRY_HIDE PASS', flush=True)

                # The real backend accepts the request only after the Qt
                # transport has timed out. The UI must reconcile the journal
                # change and never replay the submission.
                delayed = isolated / 'original' / 'late-submit.txt'
                delayed.write_text('A separate delayed UI submission.\n', encoding='utf-8')
                marker = isolated / 'delay-submit'
                marker.write_text('1', encoding='ascii')
                window.backend.request_timeout_ms = 150
                window.add_path(str(delayed))
                wait_for(app, lambda: 'completion unknown' in window.errors.toPlainText(),
                         'UI write timeout')
                assert not window._job_write_pending
                wait_for(app, lambda: any(j['filename'] == delayed.name for j in window.jobs),
                         'late submission reconciled in UI', seconds=12)
                late_jobs = [j for j in command(base, token, 'jobs.list')['jobs']
                             if j['filename'] == delayed.name]
                assert len(late_jobs) == 1, late_jobs
                late_id = late_jobs[0]['job_id']
                assert late_jobs[0]['status'] == 'QUEUED'
                with closing(sqlite3.connect(isolated / 'ingestion_jobs.sqlite')) as db:
                    count = db.execute('SELECT COUNT(*) FROM jobs WHERE job_id=?',
                                       (late_id,)).fetchone()[0]
                assert count == 1
                window.backend.request_timeout_ms = 97000
                marker.unlink()
                ui_state(late_id, 'QUEUED')
                window.buttons['cancel'].click()
                wait_for(app, lambda: state(late_id, 'CANCELLED'),
                         'cancel test-only late submission')
                print('UI_LATE_WRITE_UNCERTAIN_THEN_RECONCILED_ONCE PASS', flush=True)

                created = BASE_COLLECTION + '_created'
                assert window._start_collection_worker('create', created)
                try:
                    wait_for(app, lambda: created in FakeQdrant.collections and
                             not window._collection_pending, 'UI Create collection',
                             seconds=80)
                except AssertionError:
                    print('CREATE_DIAG', window.statusBar().currentMessage(),
                          sorted(FakeQdrant.collections),
                          log.read_text('utf-8', 'replace')[-2000:], flush=True)
                    raise
                names = command(base, token, 'collections.list')['collections']
                assert set(names) == {BASE_COLLECTION, created}, names
                assert window._start_collection_worker('delete_many',
                                                        collections=[BASE_COLLECTION, created])
                wait_for(app, lambda: not FakeQdrant.collections and
                         not window._collection_pending, 'UI multi-delete')
                assert set(FakeQdrant.deleted) == {BASE_COLLECTION, created}
                assert command(base, token, 'collections.list')['collections'] == []
                print('UI_CREATE_AND_CONFIRMED_MULTI_DELETE PASS', flush=True)

                window.close()
                assert request(base, '/health')[1]['pid'] == health['pid']
                assert request(base, '/health')[1]['gpu_model'] == 'not_initialized'
                assert (isolated / 'ingestion_jobs.sqlite').is_file()
                print('STAGE_D_UI_CONTROLS_ISOLATED PASS', flush=True)

                # Restart the *same isolated backend* to rotate its real token.
                # Force exactly one stale read to model an in-flight 401 race.
                old_pid, old_token = health['pid'], token
                stop_test_process_tree(proc, proc_created, port)
                proc, proc_created = None, None
                proc = subprocess.Popen([sys.executable, '-B', '-c', BOOTSTRAP],
                                        cwd=ROOT, env=env, stdout=output,
                                        stderr=subprocess.STDOUT,
                                        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                proc_created = psutil.Process(proc.pid).create_time()
                for _ in range(300):
                    assert proc.poll() is None, log.read_text('utf-8', 'replace')[-2500:]
                    try:
                        status, restarted = request(base, '/health')
                        if status == 200:
                            break
                    except (OSError, ValueError):
                        pass
                    time.sleep(.1)
                else:
                    raise AssertionError('isolated backend restart timed out')
                new_token = token_file.read_text(encoding='ascii').strip()
                assert new_token != old_token and restarted['pid'] != old_pid
                restarted_owners = {proc.pid}
                restarted_owners.update(child.pid for child in
                                        psutil.Process(proc.pid).children(recursive=True))
                assert restarted['pid'] in restarted_owners
                status, _ = request(base, '/api/v1/command', old_token,
                                    {'action': 'ui.revisions', 'params': {}})
                assert status == 401
                from rag3.ui_backend_client import UIBackendClient
                ui_client = UIBackendClient(port=port)
                stale_file = OneStaleRead(token_file, old_token)
                ui_client.token_file = stale_file
                read_result = []
                ui_client.command('ui.revisions', {},
                                  lambda result, error: read_result.append((result, error)))
                wait_for(app, lambda: bool(read_result), 'read retry after token rotation')
                assert stale_file.reads == 2, stale_file.reads
                assert read_result[0][1] is None and 'versions' in read_result[0][0], read_result
                ui_client.deleteLater()
                app.processEvents()
                print('UI_REAL_TOKEN_ROTATION_401_READ_RETRY PASS', flush=True)
        finally:
            try:
                if proc is not None and proc_created is not None:
                    stop_test_process_tree(proc, proc_created, port)
                else:
                    with socket.socket() as sock:
                        sock.settimeout(.1)
                        assert sock.connect_ex(('127.0.0.1', port)) != 0
                # Exact, newly chosen test scope only; only after port closure.
                token_file.unlink(missing_ok=True)
                lock_file.unlink(missing_ok=True)
            finally:
                qdrant.shutdown()
                qdrant.server_close()


if __name__ == '__main__':
    main()
