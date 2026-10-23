"""A-stage MCP session isolation, authenticated local UI gateway and health.

The UI endpoint deliberately exposes only named operations; no file uploads or
model loading occur on service startup or a health request.
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import re
import secrets
import socket
import stat
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
        if limit < 1 or ttl <= 0:
            raise ValueError('Session collection limit and TTL must be positive')
        self.preferred = preferred
        self._selection: dict[str, tuple[str | None, float]] = {}
        self._lock = threading.RLock()
        self.limit, self.ttl = limit, ttl
        # A fixed-size Bloom filter remembers *every* ID seen by this process.
        # It has no false negatives: after TTL/LRU eviction, an old live MCP
        # session must select explicitly instead of inheriting the default.
        # A rare false positive merely asks a new session to select explicitly.
        self._seen = bytearray(1 << 20)

    def _seen_positions(self, key: str):
        digest = hashlib.blake2b(key.encode('utf-8'), digest_size=16).digest()
        first = int.from_bytes(digest[:8], 'little')
        step = int.from_bytes(digest[8:], 'little') | 1
        size = len(self._seen) * 8
        return ((first + i * step) % size for i in range(4))

    def _was_seen(self, key: str) -> bool:
        return all(self._seen[pos >> 3] & (1 << (pos & 7)) for pos in self._seen_positions(key))

    def _mark_seen(self, key: str) -> None:
        for pos in self._seen_positions(key):
            self._seen[pos >> 3] |= 1 << (pos & 7)

    def _purge(self, now: float) -> None:
        old = [key for key, (_, seen) in self._selection.items() if now - seen > self.ttl]
        for key in old:
            self._selection.pop(key, None)

    def _reserve(self) -> None:
        while len(self._selection) >= self.limit:
            least_recent = min(self._selection, key=lambda k: self._selection[k][1])
            del self._selection[least_recent]

    def get(self, key: str, available: list[str]) -> str | None:
        now = time.monotonic()
        with self._lock:
            self._purge(now)
            if key not in self._selection:
                was_seen = self._was_seen(key)
                self._mark_seen(key)
                self._reserve()
                selected = (None if was_seen else
                            self.preferred if self.preferred in available else
                            available[0] if len(available) == 1 else None)
            else:
                selected = self._selection[key][0]
                if selected not in available:
                    selected = None
            self._selection[key] = (selected, now)
            return selected

    def select(self, key: str, name: str) -> None:
        with self._lock:
            now = time.monotonic()
            self._purge(now)
            if key not in self._selection:
                self._reserve()
            self._mark_seen(key)
            self._selection[key] = (name, now)


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
    """Hold one workspace-wide OS lock for the server lifetime.

    A deliberately isolated fake-Qdrant test can opt into a separate scope;
    changing the production listen port cannot start a second backend.
    """
    test_scope = os.environ.get('RAG3_BACKEND_TEST_INSTANCE')
    if test_scope is None:
        lock_name = 'backend.lock'
    else:
        from .constants import DEFAULT_QDRANT_URL
        qdrant_url = urlsplit(os.environ.get('RAG3_QDRANT_URL', ''))
        if (not re.fullmatch(r'[A-Za-z0-9_-]{1,48}', test_scope) or
                qdrant_url.hostname not in {'127.0.0.1', 'localhost', '::1'} or
                qdrant_url.port is None or
                qdrant_url.port == urlsplit(DEFAULT_QDRANT_URL).port or
                port == 8000):
            raise RuntimeError('Test backend isolation requires a named, nondefault loopback test endpoint')
        lock_name = f'backend-test-{test_scope}.lock'
    path = root / 'data' / lock_name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as handle:
        handle.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError('Another RAG3 backend already owns this workspace') from exc
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


def _windows_user_sid() -> str:
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
    advapi32 = ctypes.WinDLL('advapi32', use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    advapi32.OpenProcessToken.argtypes = [wintypes.HANDLE, wintypes.DWORD,
                                          ctypes.POINTER(wintypes.HANDLE)]
    advapi32.GetTokenInformation.argtypes = [wintypes.HANDLE, ctypes.c_int,
                                             ctypes.c_void_p, wintypes.DWORD,
                                             ctypes.POINTER(wintypes.DWORD)]
    advapi32.ConvertSidToStringSidW.argtypes = [ctypes.c_void_p,
                                                ctypes.POINTER(wintypes.LPWSTR)]
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    token = wintypes.HANDLE()
    if not advapi32.OpenProcessToken(kernel32.GetCurrentProcess(), 0x0008, ctypes.byref(token)):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        length = wintypes.DWORD()
        advapi32.GetTokenInformation(token, 1, None, 0, ctypes.byref(length))
        if not length.value:
            raise ctypes.WinError(ctypes.get_last_error())
        info = ctypes.create_string_buffer(length.value)
        if not advapi32.GetTokenInformation(token, 1, info, length,
                                            ctypes.byref(length)):
            raise ctypes.WinError(ctypes.get_last_error())

        class SidAndAttributes(ctypes.Structure):
            _fields_ = [('sid', ctypes.c_void_p), ('attributes', wintypes.DWORD)]

        sid = ctypes.cast(info, ctypes.POINTER(SidAndAttributes)).contents.sid
        sid_text = wintypes.LPWSTR()
        if not advapi32.ConvertSidToStringSidW(sid, ctypes.byref(sid_text)):
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            return sid_text.value
        finally:
            kernel32.LocalFree(ctypes.cast(sid_text, ctypes.c_void_p))
    finally:
        kernel32.CloseHandle(token)


def _windows_write_private_file(path: Path, data: bytes, user_sid: str) -> None:
    """Create with a protected DACL before any token bytes reach the file."""
    import ctypes
    from ctypes import wintypes

    class SecurityAttributes(ctypes.Structure):
        _fields_ = [('length', wintypes.DWORD), ('descriptor', ctypes.c_void_p),
                    ('inherit_handle', wintypes.BOOL)]

    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
    advapi32 = ctypes.WinDLL('advapi32', use_last_error=True)
    advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW.argtypes = [
        wintypes.LPCWSTR, wintypes.DWORD, ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(wintypes.DWORD)]
    kernel32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                    ctypes.POINTER(SecurityAttributes), wintypes.DWORD,
                                    wintypes.DWORD, wintypes.HANDLE]
    kernel32.CreateFileW.restype = wintypes.HANDLE
    kernel32.WriteFile.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD,
                                  ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p]
    kernel32.FlushFileBuffers.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    sddl = f'D:P(A;;FA;;;{user_sid})(A;;FA;;;SY)(A;;FA;;;BA)'
    descriptor = ctypes.c_void_p()
    if not advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW(
            sddl, 1, ctypes.byref(descriptor), None):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        attrs = SecurityAttributes(ctypes.sizeof(SecurityAttributes), descriptor, False)
        handle = kernel32.CreateFileW(str(path), 0x40000000, 0, ctypes.byref(attrs),
                                      1, 0x80, None)  # GENERIC_WRITE, CREATE_NEW
        if handle == ctypes.c_void_p(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            buffer = ctypes.create_string_buffer(data)
            written = wintypes.DWORD()
            if not kernel32.WriteFile(handle, buffer, len(data), ctypes.byref(written), None):
                raise ctypes.WinError(ctypes.get_last_error())
            if written.value != len(data):
                raise OSError('Incomplete UI token write')
            if not kernel32.FlushFileBuffers(handle):
                raise ctypes.WinError(ctypes.get_last_error())
        finally:
            kernel32.CloseHandle(handle)
    finally:
        kernel32.LocalFree(descriptor)


def _verify_windows_private_file(path: Path, user_sid: str) -> None:
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
    advapi32 = ctypes.WinDLL('advapi32', use_last_error=True)
    advapi32.GetNamedSecurityInfoW.argtypes = [
        wintypes.LPWSTR, wintypes.DWORD, wintypes.DWORD,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p)]
    advapi32.GetNamedSecurityInfoW.restype = wintypes.DWORD
    advapi32.ConvertSecurityDescriptorToStringSecurityDescriptorW.argtypes = [
        ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD,
        ctypes.POINTER(wintypes.LPWSTR), ctypes.POINTER(wintypes.DWORD)]
    kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    descriptor = ctypes.c_void_p()
    status = advapi32.GetNamedSecurityInfoW(str(path), 1, 4, None, None, None, None,
                                             ctypes.byref(descriptor))
    if status:
        raise OSError(status, 'Cannot inspect UI token ACL')
    try:
        sddl = wintypes.LPWSTR()
        if not advapi32.ConvertSecurityDescriptorToStringSecurityDescriptorW(
                descriptor, 1, 4, ctypes.byref(sddl), None):
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            actual = sddl.value
        finally:
            kernel32.LocalFree(ctypes.cast(sddl, ctypes.c_void_p))
    finally:
        kernel32.LocalFree(descriptor)
    if not actual.startswith('D:P'):
        raise PermissionError('UI token file DACL is not protected')
    ace_text = actual[3:]
    aces = re.findall(r'\([^()]*\)', ace_text)
    expected = {f'(A;;FA;;;{principal})' for principal in (user_sid, 'SY', 'BA')}
    if ''.join(aces) != ace_text or len(aces) != len(expected) or set(aces) != expected:
        raise PermissionError('UI token file ACL grants unexpected access')


def issue_ui_token(root: Path, port: int) -> str:
    """Rotate token at backend start; never return it through an HTTP response."""
    token_file = _token_file(root, port)
    token_file.parent.mkdir(parents=True, exist_ok=True)
    token = secrets.token_hex(32)
    tmp = token_file.with_name(token_file.name + '.' + secrets.token_hex(8) + '.tmp')
    # Remove a possibly insecure token left by an older backend before writing
    # the replacement. The caller holds the backend's exclusive process lock.
    token_file.unlink(missing_ok=True)
    try:
        if os.name == 'nt':
            user_sid = _windows_user_sid()
            _windows_write_private_file(tmp, token.encode('ascii'), user_sid)
            _verify_windows_private_file(tmp, user_sid)
        else:
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, 'wb') as stream:
                os.fchmod(stream.fileno(), 0o600)
                stream.write(token.encode('ascii'))
                stream.flush()
                os.fsync(stream.fileno())
        os.replace(tmp, token_file)
        if os.name == 'nt':
            _verify_windows_private_file(token_file, user_sid)
        elif stat.S_IMODE(token_file.stat().st_mode) != 0o600:
            raise PermissionError('UI token file mode is not 0600')
    except BaseException:
        token_file.unlink(missing_ok=True)
        raise
    finally:
        tmp.unlink(missing_ok=True)
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
            raw = bytearray()
            async for chunk in request.stream():
                raw.extend(chunk)
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
