"""Launch the built EXE, close its UI, and verify the MCP tray survives."""

from __future__ import annotations

import ctypes
import os
import socket
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / "RAG3.exe"
TRAY = ROOT / "scripts" / "rag3-mcp-tray.py"


def ui_window() -> int:
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    find = user32.FindWindowW
    find.argtypes = (ctypes.c_wchar_p, ctypes.c_wchar_p)
    find.restype = ctypes.c_void_p
    return find(None, "RAG3 Haystack") or 0


def port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.2):
            return True
    except OSError:
        return False


def wait_for(predicate, seconds: float, description: str) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.25)
    raise AssertionError(f"Timed out waiting for {description}")


def close_ui() -> None:
    hwnd = ui_window()
    if hwnd:
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        post = user32.PostMessageW
        post.argtypes = (ctypes.c_void_p, ctypes.c_uint, ctypes.c_size_t, ctypes.c_ssize_t)
        post.restype = ctypes.c_int
        if not post(ctypes.c_void_p(hwnd), 0x0010, 0, 0):  # WM_CLOSE
            raise OSError(ctypes.get_last_error(), "Could not close RAG3 UI")


def main() -> None:
    if os.name != "nt":
        raise SystemExit("Windows test only")
    if not EXE.is_file():
        raise SystemExit("Build RAG3.exe before running this test")
    if ui_window():
        raise SystemExit("Close the existing RAG3 UI before running this test")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    env = os.environ.copy()
    env["RAG3_MCP_PORT"] = str(port)
    env.pop("RAG3_MCP_WRITE_ENABLED", None)
    env.pop("RAG3_MCP_DELETE_ENABLED", None)
    launcher = subprocess.Popen([str(EXE)], cwd=ROOT, env=env,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    try:
        wait_for(lambda: bool(ui_window()), 120, "RAG3 UI window")
        launcher.wait(timeout=20)
        assert launcher.returncode == 0, f"launcher exited {launcher.returncode}"
        second = subprocess.run([str(EXE)], cwd=ROOT, env=env, timeout=15,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        assert second.returncode == 0, f"second launch exited {second.returncode}"
        wait_for(lambda: port_open(port), 120, "MCP server")
        close_ui()
        wait_for(lambda: not ui_window(), 20, "UI shutdown")
        assert port_open(port), "MCP server stopped when the UI closed"
    finally:
        if launcher.poll() is None:
            launcher.terminate()
            launcher.wait(timeout=10)
        close_ui()
        stop = subprocess.run([sys.executable, str(TRAY), "--exit"], cwd=ROOT,
                              env=env, timeout=15, capture_output=True)
        if stop.returncode == 0:
            wait_for(lambda: not port_open(port), 10, "MCP tray shutdown")
    print(f"RAG3_EXE_LAUNCH_OK port={port}")


if __name__ == "__main__":
    main()
