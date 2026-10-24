"""RAG3 login task's single long-lived Python process.

Task Scheduler executes this Python program directly. It validates a non-secret
configuration, prepares Qdrant once, and executes the MCP backend *in this PID*.
The tunnel reconnector is a short-lived helper, not a resident tray process.
"""
from __future__ import annotations

import argparse
import json
import os
import runpy
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'data' / 'backend-login-config.json'
SERVER = ROOT / 'scripts' / 'rag3-mcp-server.py'
INFRA = ROOT / 'scripts' / 'ensure-infra.ps1'
TUNNEL = ROOT / 'scripts' / 'rag3-tunnel-autostart.py'
LOG = ROOT / 'data' / 'logs' / 'shared-backend-login.log'
REQUIRED = frozenset({'port', 'qdrant_url', 'write_enabled', 'delete_enabled',
                      'bootstrap_infra', 'tunnel_autostart'})


def validate_config(value: object) -> dict:
    if not isinstance(value, dict) or set(value) != REQUIRED:
        raise ValueError('Expected exactly the documented backend-login fields')
    port = value['port']
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError('Invalid port')
    if not isinstance(value['qdrant_url'], str):
        raise ValueError('Invalid Qdrant URL')
    url = urlsplit(value['qdrant_url'])
    if (url.scheme not in ('http', 'https') or not url.hostname or url.username or
            url.password or url.query or url.fragment or url.path not in ('', '/') or
            url.port is None):
        raise ValueError('Qdrant URL must be a no-credential host:port endpoint')
    for name in ('write_enabled', 'delete_enabled', 'bootstrap_infra', 'tunnel_autostart'):
        if type(value[name]) is not bool:
            raise ValueError(f'{name} must be a boolean')
    # The bundled WSL Docker compose starts only the default local Qdrant service.
    if value['bootstrap_infra'] and (url.scheme, url.hostname, url.port) != (
            'http', '127.0.0.1', 16333):
        raise ValueError('Bundled infrastructure bootstrap supports only local 127.0.0.1:16333')
    return value


def read_config(path: Path = CONFIG) -> dict:
    return validate_config(json.loads(path.read_text('utf-8-sig')))


def health_matches_this_process(port: int) -> bool:
    url = f'http://127.0.0.1:{port}/health'
    try:
        with urllib.request.urlopen(url, timeout=1.5) as resp:
            result = json.loads(resp.read(1024 * 4))
        return result.get('backend') == 'ready' and result.get('pid') == os.getpid()
    except (OSError, ValueError, TypeError, urllib.error.URLError):
        return False


def one_shot_tunnel(port: int) -> None:
    """Connect saved tunnel only after this backend's health responds."""
    for _ in range(120):
        if health_matches_this_process(port):
            try:
                pythonw = Path(sys.executable).with_name('pythonw.exe')
                executor = pythonw if pythonw.is_file() else Path(sys.executable)
                env = os.environ.copy()
                env['RAG3_MCP_PORT'] = str(port)
                subprocess.Popen([str(executor), str(TUNNEL)], cwd=ROOT, env=env,
                                 stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL,
                                 creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            except OSError:
                print('TUNNEL_HELPER_LAUNCH_FAILED', flush=True)
            return
        time.sleep(1)
    print('TUNNEL_HELPER_BACKEND_READY_TIMEOUT', flush=True)


def bootstrap_qdrant(cfg: dict) -> None:
    if not cfg['bootstrap_infra']:
        return
    ps = Path(os.environ.get('SystemRoot', r'C:\Windows')) / 'System32' / 'WindowsPowerShell' / 'v1.0' / 'powershell.exe'
    result = subprocess.run([str(ps), '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass',
                             '-File', str(INFRA), '-QdrantUrl', cfg['qdrant_url']],
                            cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, timeout=110,
                            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    if result.returncode:
        # Never print raw stderr, which could include environment details.
        raise RuntimeError(f'Qdrant bootstrap failed (exit {result.returncode})')
    print('QDRANT_BOOTSTRAP_OK', flush=True)


def run_backend(cfg: dict) -> None:
    if os.name != 'nt':
        raise RuntimeError('Login task requires Windows')
    os.environ.update({
        'RAG3_MCP_TRANSPORT': 'streamable-http',
        'RAG3_MCP_PORT': str(cfg['port']),
        'RAG3_QDRANT_URL': cfg['qdrant_url'],
        'RAG3_MCP_WRITE_ENABLED': '1' if cfg['write_enabled'] else '0',
        'RAG3_MCP_DELETE_ENABLED': '1' if cfg['delete_enabled'] else '0',
        'RAG3_PROJECT_ROOT': str(ROOT),
    })
    # Test instances must keep logs inside isolated data roots.
    test_root = os.environ.get('RAG3_TEST_DATA_ROOT')
    log_path = (Path(test_root) / 'logs' / 'shared-backend-login.log'
                if test_root and os.environ.get('RAG3_BACKEND_TEST_INSTANCE') else LOG)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('a', encoding='utf-8', buffering=1) as log:
        sys.stdout = log
        sys.stderr = log
        print('BACKEND_STARTUP_BEGIN', flush=True)
        bootstrap_qdrant(cfg)
        if cfg['tunnel_autostart'] and TUNNEL.is_file():
            threading.Thread(target=one_shot_tunnel, args=(cfg['port'],),
                             name='rag3-tunnel-reconnect-once', daemon=True).start()
        if not SERVER.is_file():
            raise FileNotFoundError('Backend server entry not found')
        sys.argv = [str(SERVER)]
        # No backend child is spawned; Task Scheduler supervises this PID.
        runpy.run_path(str(SERVER), run_name='__main__')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--validate', action='store_true')
    group.add_argument('--run', action='store_true')
    parser.add_argument('--config', type=Path, default=CONFIG)
    args = parser.parse_args(argv)
    try:
        cfg = read_config(args.config)
    except (OSError, ValueError, TypeError) as exc:
        print('BACKEND_CONFIG_INVALID', type(exc).__name__, file=sys.stderr)
        return 2
    if args.validate:
        print('BACKEND_LOGIN_CONFIG_VALID port=' + str(cfg['port']) +
              ' qdrant=' + cfg['qdrant_url'] +
              ' write=' + str(int(cfg['write_enabled'])) +
              ' delete=' + str(int(cfg['delete_enabled'])))
        return 0
    try:
        run_backend(cfg)
    except Exception as exc:
        print('BACKEND_LOGIN_FAILURE', type(exc).__name__, file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
