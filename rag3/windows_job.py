"""Own a Windows parser process tree for the backend's lifetime.

The job handle is deliberately not inheritable.  Closing the last handle kills
all assigned processes, including ordinary descendants, even if Python is
terminated without running a ``finally`` block.  A caller must assign its own
newly-created parser before allowing that parser to do useful work; if
assignment fails, it must also stop and reap the subprocess.
"""
from __future__ import annotations

import os
import threading


if os.name == "nt":
    import ctypes
    from ctypes import wintypes

    _PROCESS_TERMINATE = 0x0001
    _PROCESS_SET_QUOTA = 0x0100
    _HANDLE_FLAG_INHERIT = 0x0001
    _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
    _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000


    class _BasicLimitInformation(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]


    class _IoCounters(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_uint64),
            ("WriteOperationCount", ctypes.c_uint64),
            ("OtherOperationCount", ctypes.c_uint64),
            ("ReadTransferCount", ctypes.c_uint64),
            ("WriteTransferCount", ctypes.c_uint64),
            ("OtherTransferCount", ctypes.c_uint64),
        ]


    class _ExtendedLimitInformation(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _BasicLimitInformation),
            ("IoInfo", _IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]


    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _kernel32.CreateJobObjectW.argtypes = (ctypes.c_void_p, wintypes.LPCWSTR)
    _kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    _kernel32.SetInformationJobObject.argtypes = (
        wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD)
    _kernel32.SetInformationJobObject.restype = wintypes.BOOL
    _kernel32.SetHandleInformation.argtypes = (
        wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD)
    _kernel32.SetHandleInformation.restype = wintypes.BOOL
    _kernel32.OpenProcess.argtypes = (
        wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    _kernel32.OpenProcess.restype = wintypes.HANDLE
    _kernel32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
    _kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    _kernel32.TerminateJobObject.argtypes = (wintypes.HANDLE, wintypes.UINT)
    _kernel32.TerminateJobObject.restype = wintypes.BOOL
    _kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    _kernel32.CloseHandle.restype = wintypes.BOOL


    def _winerror(operation: str) -> OSError:
        return OSError(ctypes.get_last_error(), f"{operation} failed")


class WindowsParserJob:
    """Kill assigned parser processes on explicit close or backend exit.

    ``assign`` accepts only the PID of a subprocess owned by this backend.
    The caller must not let that subprocess parse or write before assignment.
    """

    def __init__(self) -> None:
        if os.name != "nt":
            raise OSError("Windows Job Objects are available only on Windows")
        self._lock = threading.Lock()
        self._handle = None
        handle = _kernel32.CreateJobObjectW(None, None)
        if not handle:
            raise _winerror("CreateJobObjectW")
        try:
            if not _kernel32.SetHandleInformation(handle, _HANDLE_FLAG_INHERIT, 0):
                raise _winerror("SetHandleInformation")
            limits = _ExtendedLimitInformation()
            limits.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            if not _kernel32.SetInformationJobObject(
                    handle, _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
                    ctypes.byref(limits), ctypes.sizeof(limits)):
                raise _winerror("SetInformationJobObject")
        except BaseException:
            _kernel32.CloseHandle(handle)
            raise
        self._handle = handle

    def assign(self, pid: int) -> None:
        if not isinstance(pid, int) or pid <= 0:
            raise ValueError("pid must be a positive integer")
        with self._lock:
            if self._handle is None:
                raise OSError("parser job is closed")
            process = _kernel32.OpenProcess(
                _PROCESS_SET_QUOTA | _PROCESS_TERMINATE, False, pid)
            if not process:
                raise _winerror("OpenProcess")
            try:
                if not _kernel32.AssignProcessToJobObject(self._handle, process):
                    raise _winerror("AssignProcessToJobObject")
            finally:
                _kernel32.CloseHandle(process)

    def terminate(self) -> None:
        """Stop every assigned process while retaining the job handle."""
        with self._lock:
            if self._handle is not None and not _kernel32.TerminateJobObject(
                    self._handle, 1):
                raise _winerror("TerminateJobObject")

    def close(self) -> None:
        """Close the sole job handle; this terminates all assigned processes."""
        with self._lock:
            if self._handle is not None:
                if not _kernel32.CloseHandle(self._handle):
                    raise _winerror("CloseHandle")
                self._handle = None

    def __enter__(self) -> "WindowsParserJob":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except BaseException:
            pass
