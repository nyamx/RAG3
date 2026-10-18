"""Verify that the UI starts a tray server that outlives the UI process."""

from __future__ import annotations

import asyncio
import os
import socket
import subprocess
import sys
from pathlib import Path

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


ROOT = Path(__file__).resolve().parents[1]
TRAY = ROOT / "scripts" / "rag3-mcp-tray.py"
UI = ROOT / "scripts" / "ingest-ui.py"


def _port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


async def _wait_for_server(port: int, ui_process: subprocess.Popen | None = None) -> None:
    async with httpx.AsyncClient(timeout=2) as client:
        for _ in range(120):
            if ui_process is not None and ui_process.poll() is not None:
                raise AssertionError(f"UI exited before MCP server started: {ui_process.returncode}")
            try:
                response = await client.get(f"http://127.0.0.1:{port}/")
                if response.status_code in {200, 404}:
                    return
            except httpx.RequestError:
                pass
            await asyncio.sleep(1)
    raise AssertionError("UI-launched MCP server did not start")


async def main() -> None:
    port = _port()
    env = os.environ.copy()
    env.update(QT_QPA_PLATFORM="offscreen", RAG3_MCP_PORT=str(port))
    env.pop("RAG3_MCP_WRITE_ENABLED", None)
    env.pop("RAG3_MCP_DELETE_ENABLED", None)
    launch = (
        "import runpy; "
        f"ok, message = runpy.run_path({str(UI)!r})['launch_mcp_tray'](); "
        "print(message); raise SystemExit(0 if ok else 1)"
    )
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    ui_process = subprocess.Popen([sys.executable, str(UI)], cwd=ROOT, env=env,
                                  stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL, creationflags=flags)
    try:
        await _wait_for_server(port, ui_process)
        ui_process.terminate()
        ui_process.wait(timeout=10)
        # The UI process has exited; the independent tray must still serve MCP.
        async with streamable_http_client(f"http://127.0.0.1:{port}/mcp") as streams:
            async with ClientSession(streams[0], streams[1]) as session:
                await session.initialize()
                tools = await session.list_tools()
                assert len(tools.tools) == 16
        second = subprocess.run([sys.executable, "-c", launch], cwd=ROOT, env=env,
                                capture_output=True, text=True, timeout=30,
                                creationflags=flags)
        assert second.returncode == 0, second.stdout + second.stderr
        await _wait_for_server(port)
    finally:
        if ui_process.poll() is None:
            ui_process.terminate()
            ui_process.wait(timeout=10)
        stop = subprocess.run([sys.executable, str(TRAY), "--exit"], cwd=ROOT, env=env,
                              capture_output=True, text=True, timeout=15,
                              creationflags=flags)
        if stop.returncode != 0:
            print("MCP tray shutdown request could not be delivered", file=sys.stderr)
        else:
            for _ in range(30):
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                        pass
                except OSError:
                    break
                await asyncio.sleep(0.2)
            else:
                raise AssertionError("MCP server stayed on after tray shutdown")
    print(f"MCP_TRAY_LIFECYCLE_OK tools=16 port={port}")


if __name__ == "__main__":
    asyncio.run(main())
