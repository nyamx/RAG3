"""Run a live login-entry smoke with fake local Qdrant and isolated job/source DB.

Does NOT register Task Scheduler, connect a tunnel or modify production Qdrant.
Only two unique test-scoped files (lock/token) are created under production data
because the existing HTTP backend's lock and token paths are port-scoped there.
"""
from __future__ import annotations
import json
import os
import pathlib
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = pathlib.Path(__file__).resolve().parents[1]
ENTRY = ROOT / 'scripts' / 'rag3-backend-login.py'
PREPARER = ROOT / 'scripts' / 'prepare-rag3-direct-pythonw.py'
VENV_PYTHON = ROOT / '.venv' / 'Scripts' / 'python.exe'
DIRECT_PYTHONW = ROOT / '.venv' / 'Scripts' / 'rag3-pythonw-direct.exe'

class FakeQdrant(BaseHTTPRequestHandler):
    def do_GET(self):
        from urllib.parse import urlsplit
        if urlsplit(self.path).path == '/collections':
            body = b'{"result":{"collections":[]},"status":"ok","time":0.001}'
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)
    def log_message(self, *args):
        pass

def free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]

def request(url, token=None, action=None, timeout=2):
    headers = {'Content-Type':'application/json'}
    body = None
    if token is not None:
        headers['X-RAG3-Token'] = token
    if action is not None:
        body = json.dumps({'action':action,'params':{}}).encode()
    req = urllib.request.Request(url, data=body, headers=headers,
                                 method='POST' if action else 'GET')
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read(16384))
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read(16384))

def assert_backend_owned_by_launcher(server_pid, launcher_pid):
    """A direct base Pythonw copy must host the backend in its own process."""
    assert server_pid == launcher_pid, (launcher_pid, server_pid)


def await_port_closed(port, timeout=12):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        try:
            with socket.create_connection(('127.0.0.1',port),timeout=.3):
                time.sleep(.1)
        except OSError:
            return
    raise AssertionError('Test backend port remained open after launcher tree stopped')


def await_health(proc, port, logfile, timeout=105):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if proc.poll() is not None:
            text = logfile.read_text(encoding='utf-8', errors='replace')[-900:] if logfile.exists() else '(none)'
            raise AssertionError('login process died; '+text)
        try:
            status, data = request(f'http://127.0.0.1:{port}/health', timeout=1.2)
            if status == 200 and data.get('backend') == 'ready':
                return data
        except (OSError, ValueError):
            pass
        time.sleep(.25)
    raise AssertionError('login health timeout; logfile_tail='+(
        logfile.read_text(encoding='utf-8', errors='replace')[-900:] if logfile.exists() else '(none)'))

def stop(proc):
    if proc is None or proc.poll() is not None:
        return
    result = subprocess.run(['taskkill','/PID',str(proc.pid),'/T','/F'],
        stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
        timeout=18,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    if result.returncode and proc.poll() is None:
        # Only the process that this test created may be terminated.
        proc.terminate()
    try:
        proc.wait(timeout=12)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=12)

def main():
    if os.name != 'nt':
        raise RuntimeError('Windows-only live login test')
    subprocess.run([str(VENV_PYTHON), str(PREPARER)], cwd=ROOT,
                   stdin=subprocess.DEVNULL, check=True,
                   creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    q = ThreadingHTTPServer(('127.0.0.1', 0), FakeQdrant)
    qt = threading.Thread(target=q.serve_forever, daemon=True)
    qt.start()
    port = free_port()
    while port in (8000, 16333, q.server_port):
        port = free_port()
    token_file = ROOT / 'data' / f'rag3-backend-ui-{port}.token'
    lock_file = ROOT / 'data' / f'backend-test-{port}.lock'
    assert not token_file.exists() and not lock_file.exists(), 'Refusing to modify existing scoped backend state'
    proc = None
    with tempfile.TemporaryDirectory(prefix='rag3-stage-d-login-live-') as d:
        temp = pathlib.Path(d)
        data_root = temp / 'job-source-db'
        logfile = data_root / 'logs' / 'shared-backend-login.log'
        cfg = {'port':port, 'qdrant_url':f'http://127.0.0.1:{q.server_port}',
               'write_enabled':False, 'delete_enabled':False,
               'bootstrap_infra':False, 'tunnel_autostart':False}
        cfg_path = temp / 'isolated-login-config.json'
        cfg_path.write_text(json.dumps(cfg), encoding='utf-8')
        env = os.environ.copy()
        env.update({'RAG3_TEST_DATA_ROOT':str(data_root),
                    'RAG3_BACKEND_TEST_INSTANCE':str(port),
                    'PYTHONUNBUFFERED':'1'})
        production_db = ROOT / 'data' / 'ingestion_jobs.sqlite'
        db_before = (production_db.stat().st_size, production_db.stat().st_mtime_ns) if production_db.exists() else None
        try:
            proc = subprocess.Popen([str(DIRECT_PYTHONW),str(ENTRY),'--run','--config',str(cfg_path)],
                cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            health = await_health(proc,port,logfile)
            assert_backend_owned_by_launcher(health['pid'], proc.pid)
            assert health.get('gpu_model') == 'not_initialized', health.get('gpu_model')
            assert health.get('qdrant_transport') == 'reachable', health.get('qdrant_transport')
            assert token_file.exists(), 'No test-scoped UI API token generated'
            token = token_file.read_text(encoding='ascii').strip()
            assert len(token) == 64
            base = f'http://127.0.0.1:{port}/api/v1/command'
            noauth, _ = request(base,action='jobs.health')
            assert noauth == 401, noauth
            status, result = request(base,token,'jobs.health')
            assert status == 200 and result.get('ok') and result['result']['alive'], (status,result)
            print('LOGIN_LIVE_HTTP_PASS pid_matches=true gpu_cold=true scoped_auth=true',flush=True)
            assert (data_root/'ingestion_jobs.sqlite').exists(), 'Test DB not created'
            assert logfile.exists(), 'No isolated login logfile'
            stop(proc)
            assert proc.poll() is not None
            await_port_closed(port)
            # Reboot same config: same port, fresh local token, same test-only DB.
            proc = subprocess.Popen([str(DIRECT_PYTHONW),str(ENTRY),'--run','--config',str(cfg_path)],
                cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            restarted = await_health(proc,port,logfile)
            assert_backend_owned_by_launcher(restarted['pid'], proc.pid)
            token2 = token_file.read_text(encoding='ascii').strip()
            assert token2 != token, 'Token not rotated after login restart'
            assert restarted['gpu_model']=='not_initialized'
            print('LOGIN_RESTART_PASS token_rotated=true same_test_db=true',flush=True)
            db_after = (production_db.stat().st_size, production_db.stat().st_mtime_ns) if production_db.exists() else None
            assert db_before == db_after, 'Production DB changed during isolated test; investigate'
            print('LOGIN_ISOLATION_PASS production_db_unchanged=true tunnel_not_started=true task_not_registered=true',flush=True)
        finally:
            stop(proc)
            await_port_closed(port)
            q.shutdown(); q.server_close()
            # These were absent before launch, and were created only for this test port.
            token_file.unlink(missing_ok=True)
            lock_file.unlink(missing_ok=True)
    print('STAGE_D_LOGIN_LIVE_RESULT=PASS',flush=True)

if __name__ == '__main__':
    main()
