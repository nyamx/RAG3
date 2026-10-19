"""Reconnect the existing RAG3 Secure MCP Tunnel using saved DPAPI credentials.

Launched once after the tray-managed MCP server is listening. No UI or secrets in logs.
The tunnel-client CLI owns the persistent runtime process after `runtimes connect`.
"""
from __future__ import annotations
import ctypes
import json
import os
from pathlib import Path
import runpy
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
CLIENT_SCRIPT = ROOT / "data" / "tools" / "tunnel-client" / "user-connect.pyw"
LOG = ROOT / "data" / "logs" / "tunnel-autostart.log"


def report(state: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(datetime.now(timezone.utc).isoformat(timespec="seconds") + " " + state + "\n")


def autostart() -> None:
    if os.name != "nt":
        report("not_windows")
        return
    if not CLIENT_SCRIPT.is_file():
        report("connector_not_found")
        return
    # Prevent concurrent connector commands, including after a tray restart.
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateMutexW.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_wchar_p)
    kernel.CreateMutexW.restype = ctypes.c_void_p
    kernel.CloseHandle.argtypes = (ctypes.c_void_p,)
    handle = kernel.CreateMutexW(None, False, r"Local\RAG3_TUNNEL_AUTOSTART")
    if not handle:
        report("mutex_failed")
        return
    try:
        if ctypes.get_last_error() == 183:
            report("already_connecting")
            return
        # Reuse the existing DPAPI decryption, redaction, CLI and status logic.
        mod = runpy.run_path(str(CLIENT_SCRIPT), run_name="rag3_tunnel_autostart")
        saved = mod["SAVED_KEY_FILE"]
        if not saved.is_file():
            report("no_saved_key")
            return
        if not mod["EXE"].is_file():
            report("client_not_found")
            return
        try:
            key = mod["dpapi"](saved.read_bytes(), decrypt=True).decode("utf-8")
        except Exception:
            report("saved_key_unavailable")
            return
        if not key or any(c.isspace() for c in key):
            report("saved_key_invalid")
            return
        env = mod["child_environment"](key)
        try:
            run = mod["run_client"]
            alias = mod["ALIAS"]
            code, out, _ = run(["runtimes", "status", alias, "--json"], env, key, 25)
            if code == 0:
                try:
                    state = json.loads(out)
                    if mod["find_bool"](state, "process_running") is True:
                        report("already_running")
                        return
                except (TypeError, ValueError):
                    pass
            port = int(os.environ.get("RAG3_MCP_PORT", "8000"))
            args = [
                "runtimes", "connect", "--alias", alias,
                "--profile", "rag3-web", "--profile-dir", str(mod["PROFILE_DIR"]),
                "--tunnel-id", mod["TUNNEL_ID"],
                "--mcp-server-url", f"http://127.0.0.1:{port}/mcp",
                "--runtime-api-key", "env:CONTROL_PLANE_API_KEY", "--json",
            ]
            code, _, _ = run(args, env, key, 120)
            if code != 0:
                report("connect_failed")
                return
            for attempt in range(5):
                code, out, _ = run(["runtimes", "status", alias, "--json"], env, key, 25)
                if code == 0:
                    try:
                        state = json.loads(out)
                        if mod["find_bool"](state, "process_running") is True:
                            report("connected")
                            return
                    except (TypeError, ValueError):
                        pass
                if attempt < 4:
                    time.sleep(2)
            report("connect_status_unconfirmed")
        finally:
            env.pop("CONTROL_PLANE_API_KEY", None)
            key = ""
    except Exception:
        # Do not log native exception text, key material, CLI output or URLs.
        report("unexpected_failure")
    finally:
        kernel.CloseHandle(handle)


if __name__ == "__main__":
    autostart()
