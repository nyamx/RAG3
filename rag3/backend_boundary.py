"""A-stage MCP session isolation, authenticated local UI gateway and health.

The UI endpoint deliberately exposes only named operations; no file uploads or
model loading occur on service startup or a health request.
"""
from __future__ import annotations

import asyncio
import hmac
import json
import os
import secrets
import socket
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlsplit

from starlette.requests import Request
from starlette.responses import JSONResponse


class SessionCollections:
    """Selections never become process-wide defaults after a client changes one."""
    def __init__(self, preferred: str | None, *, limit: int = 512, ttl: float = 4 * 3600):
        self.preferred = preferred
        self._selection: dict[str, tuple[str | None, float]] = {}
        self._lock = threading.RLock()
        self.limit, self.ttl = limit, ttl

    def _purge(self, now: float) -> None:
        old = [key for key, (_, seen) in self._selection.items() if now - seen > self.ttl]
        for key in old:
            self._selection.pop(key, None)
        while len(self._selection) >= self.limit:
            least_recent = min(self._selection, key=lambda k: self._selection[k][1])
            del self._selection[least_recent]

    def get(self, key: str, available: list[str]) -> str | None:
        now = time.monotonic()
        with self._lock:
            if key not in self._selection:
                self._purge(now)
                selected = (self.preferred if self.preferred in available else
                            available[0] if len(available) == 1 else None)
            else:
                selected = self._selection[key][0]
                if selected not in available:
                    selected = None
            self._selection[key] = (selected, now)
            return selected

    def select(self, key: str, name: str) -> None:
        with self._lock:
            self._purge(time.monotonic())
            self._selection[key] = (name, time.monotonic())


def session_identifier(context) -> str:
    """Return a validated, stable MCP connection key.

    MCP SDK v2 creates a fresh ServerSession proxy for each request. Only the
    transport's connection/session ID (exposed on Context) persists. Stateful
    HTTP session headers are validated by the MCP transport before tools run.
    Stdio has one logical session per dedicated server process.
    """
    session_id = getattr(context, "session_id", None)
    if isinstance(session_id, str) and session_id:
        return "mcp:" + session_id
    headers = getattr(context, "headers", None)
    if headers is not None:
        session_id = headers.get("mcp-session-id")
        if isinstance(session_id, str) and session_id:
            return "mcp:" + session_id
        # Never share a process-global selection across independent HTTP
        # clients just because a session identifier is absent.
        raise RuntimeError("Stateful HTTP MCP session ID is unavailable")
    return "stdio:single-client"

@contextmanager
def single_instance(root: Path, port: int):
    """Hold an OS lock for this local listen endpoint for the server lifetime."""
    path = root / 'data' / f'backend-{port}.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as handle:
        handle.seek(0)
        if handle.read(1) != b'1':
            handle.seek(0)
            handle.write(b'1')
            handle.flush()
        handle.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(f'Another RAG3 backend already owns local port {port}') from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == 'nt':
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _token_file(root: Path, port: int) -> Path:
    return root / 'data' / f'rag3-backend-ui-{port}.token'


def issue_ui_token(root: Path, port: int) -> str:
    """Rotate token at backend start; never return it through an HTTP response."""
    token_file = _token_file(root, port)
    token_file.parent.mkdir(parents=True, exist_ok=True)
    token = secrets.token_hex(32)
    tmp = token_file.with_suffix('.token.tmp')
    tmp.write_text(token, encoding='ascii')
    if os.name != 'nt':
        tmp.chmod(0o600)
    os.replace(tmp, token_file)
    return token


def install_routes(mcp, *, root: Path, qdrant_url: str, ui_commands, ui_token: str):
    """Add HTTP routes to the *same* MCP Starlette server instance.

    Custom routes bypass MCP SDK authorization, so the UI endpoint has its own
    bearer-equivalent header token. Only the loopback binding is supported.
    """
    @mcp.custom_route('/health', methods=['GET'], include_in_schema=False)
    async def health(_request: Request):
        def qdrant_transport():
            url = urlsplit(qdrant_url)
            host = url.hostname or '127.0.0.1'
            port = url.port or (443 if url.scheme == 'https' else 6333)
            try:
                with socket.create_connection((host, port), timeout=.3):
                    return 'reachable'
            except OSError:
                return 'unreachable'
        status = await asyncio.to_thread(qdrant_transport)
        return JSONResponse({'backend': 'ready', 'pid': os.getpid(),
                             'qdrant_transport': status,
                             'gpu_model': 'not_initialized' if 'torch' not in sys.modules
                             else 'module_imported_state_unverified'})

    @mcp.custom_route('/api/v1/command', methods=['POST'], include_in_schema=False)
    async def command(request: Request):
        supplied = request.headers.get('X-RAG3-Token', '')
        if not supplied or not hmac.compare_digest(supplied, ui_token):
            return JSONResponse({'ok': False, 'error': {'code': 'UNAUTHORIZED'}}, status_code=401)
        try:
            if int(request.headers.get('content-length', '0')) > 262144:
                return JSONResponse({'ok': False, 'error': {'code': 'PAYLOAD_TOO_LARGE'}}, status_code=413)
            raw = await request.body()
            if len(raw) > 262144:
                return JSONResponse({'ok': False, 'error': {'code': 'PAYLOAD_TOO_LARGE'}}, status_code=413)
            payload = json.loads(raw)
            if not isinstance(payload, dict) or not isinstance(payload.get('action'), str):
                raise ValueError('action must be a string')
            params = payload.get('params', {})
            if not isinstance(params, dict):
                raise ValueError('params must be an object')
        except (ValueError, UnicodeDecodeError, TypeError):
            return JSONResponse({'ok': False, 'error': {'code': 'INVALID_ARGUMENT'}}, status_code=400)
        try:
            # Synchronous Qdrant and legacy job RPC must not block the MCP event loop.
            result = await asyncio.wait_for(asyncio.to_thread(ui_commands, payload['action'], params), timeout=90)
            return JSONResponse({'ok': True, 'result': result})
        except ValueError as exc:
            return JSONResponse({'ok': False, 'error': {'code': 'INVALID_ARGUMENT',
                                                       'message': str(exc)[:250]}}, status_code=400)
        except TimeoutError:
            return JSONResponse({'ok': False, 'error': {'code': 'TIMEOUT'}}, status_code=504)
        except Exception:
            return JSONResponse({'ok': False, 'error': {'code': 'BACKEND_ERROR'}}, status_code=503)
