"""Read-only source handle that refuses concurrent Windows write/delete sharing."""
from __future__ import annotations
import hashlib
import os
from pathlib import Path


class OriginalSourceGuard:
    def __init__(self, source, expected_sha):
        self.source = Path(source).resolve()
        self.expected_sha = expected_sha
        self._file = None
        self.strong = os.name == 'nt'
        try:
            if self.strong:
                import ctypes
                import msvcrt
                from ctypes import wintypes
                k32 = ctypes.WinDLL('kernel32', use_last_error=True)
                create = k32.CreateFileW
                create.argtypes = [wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,
                                   ctypes.c_void_p,wintypes.DWORD,wintypes.DWORD,wintypes.HANDLE]
                create.restype = wintypes.HANDLE
                # GENERIC_READ; FILE_SHARE_READ only; OPEN_EXISTING; FILE_ATTRIBUTE_NORMAL
                h = create(str(self.source), 0x80000000, 0x00000001, None, 3, 0x80, None)
                if h == wintypes.HANDLE(-1).value or h is None:
                    raise OSError(ctypes.get_last_error(), 'Cannot lock original source against modification')
                try:
                    fd = msvcrt.open_osfhandle(h, os.O_RDONLY | os.O_BINARY)
                except BaseException:
                    k32.CloseHandle(h)
                    raise
                self._file = os.fdopen(fd, 'rb')
            else:
                self._file = self.source.open('rb')
            sha = hashlib.sha256()
            for block in iter(lambda:self._file.read(1024 * 1024),b''):
                sha.update(block)
            if sha.hexdigest() != expected_sha:
                raise ValueError('Original SHA-256 mismatch')
            self._file.seek(0)
        except BaseException:
            self.close()
            raise

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self): return self
    def __exit__(self,*_): self.close()
