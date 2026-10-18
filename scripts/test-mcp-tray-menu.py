"""Exercise the tray menu's stop and restart actions against a live MCP server."""

from __future__ import annotations

import os
import runpy
import socket
import sys
import time
from pathlib import Path

from PySide6.QtWidgets import QApplication


ROOT = Path(__file__).resolve().parents[1]
with socket.socket() as probe:
    probe.bind(("127.0.0.1", 0))
    PORT = probe.getsockname()[1]
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["RAG3_MCP_PORT"] = str(PORT)
os.environ.pop("RAG3_MCP_WRITE_ENABLED", None)
os.environ.pop("RAG3_MCP_DELETE_ENABLED", None)
tray_module = runpy.run_path(str(ROOT / "scripts" / "rag3-mcp-tray.py"))


def port_open() -> bool:
    try:
        with socket.create_connection(("127.0.0.1", PORT), timeout=0.2):
            return True
    except OSError:
        return False


app = QApplication(sys.argv[:1])
controller = tray_module["TrayController"](app)


def pump_until(predicate, timeout: float = 90) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return
        time.sleep(0.1)
    raise AssertionError(f"Tray state did not change within {timeout}s: {controller.status_action.text()}")


try:
    pump_until(port_open)
    controller.stop_action.trigger()
    pump_until(lambda: not port_open(), timeout=10)
    assert controller.process is None
    controller.restart_action.trigger()
    pump_until(port_open)
    controller.exit_action.trigger()
    pump_until(lambda: not port_open(), timeout=10)
    print(f"MCP_TRAY_MENU_OK port={PORT}")
finally:
    controller.exit()
