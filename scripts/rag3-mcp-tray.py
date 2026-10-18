"""Keep the local RAG3 MCP server running independently of the UI window."""

from __future__ import annotations

import ctypes
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

from PySide6.QtCore import QObject, QTimer, Qt
from PySide6.QtGui import QAction, QColor, QFont, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import QApplication, QMenu, QSystemTrayIcon


ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "scripts" / "rag3-mcp-server.py"
UI = ROOT / "scripts" / "ingest-ui.py"


def _configured_port() -> int:
    port = int(os.environ.get("RAG3_MCP_PORT", "8000"))
    if not 1 <= port <= 65535:
        raise ValueError("RAG3_MCP_PORT must be between 1 and 65535")
    return port


PORT = _configured_port()
LOG = ROOT / "data" / "logs" / ("mcp-server.log" if PORT == 8000 else f"mcp-server-{PORT}.log")
MUTEX_NAME = f"Local\\RAG3_MCP_TRAY_{PORT}"
EXIT_EVENT_NAME = f"Local\\RAG3_MCP_TRAY_EXIT_{PORT}"


def _single_instance_handle() -> int | None:
    """A Windows mutex remains owned while this process is alive."""
    if os.name != "nt":
        raise RuntimeError("The RAG3 tray controller requires Windows")
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create = kernel32.CreateMutexW
    create.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_wchar_p)
    create.restype = ctypes.c_void_p
    handle = create(None, 0, MUTEX_NAME)
    if not handle:
        raise OSError(ctypes.get_last_error(), "CreateMutexW failed")
    if ctypes.get_last_error() == 183:  # ERROR_ALREADY_EXISTS
        kernel32.CloseHandle(ctypes.c_void_p(handle))
        return None
    return handle


def _request_exit() -> bool:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    open_event = kernel32.OpenEventW
    open_event.argtypes = (ctypes.c_uint, ctypes.c_int, ctypes.c_wchar_p)
    open_event.restype = ctypes.c_void_p
    handle = open_event(0x0002, 0, EXIT_EVENT_NAME)  # EVENT_MODIFY_STATE
    if not handle:
        return False
    try:
        return bool(kernel32.SetEvent(ctypes.c_void_p(handle)))
    finally:
        kernel32.CloseHandle(ctypes.c_void_p(handle))


def _create_exit_event() -> int:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create = kernel32.CreateEventW
    create.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_wchar_p)
    create.restype = ctypes.c_void_p
    handle = create(None, 1, 0, EXIT_EVENT_NAME)
    if not handle:
        raise OSError(ctypes.get_last_error(), "CreateEventW failed")
    return handle


def _icon(color: str) -> QIcon:
    pixmap = QPixmap(64, 64)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor(color))
    painter.drawEllipse(2, 2, 60, 60)
    painter.setPen(QColor("white"))
    font = QFont("Segoe UI", 22)
    font.setBold(True)
    painter.setFont(font)
    painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, "R3")
    painter.end()
    return QIcon(pixmap)


def _port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.2):
            return True
    except OSError:
        return False


def _gui_python() -> str:
    executable = Path(sys.executable)
    windowless = executable.with_name("pythonw.exe")
    return str(windowless if windowless.is_file() else executable)


def _console_python() -> str:
    executable = Path(sys.executable)
    console = executable.with_name("python.exe")
    return str(console if console.is_file() else executable)


class TrayController(QObject):
    def __init__(self, app: QApplication) -> None:
        super().__init__()
        self.app = app
        self.process: subprocess.Popen | None = None
        self.ui_process: subprocess.Popen | None = None
        self.port = 0
        self.started_at = 0.0
        self.crashes: list[float] = []
        self.retry_pending = False
        self.stopping = False

        self.tray = QSystemTrayIcon(_icon("#8b8f98"), self)
        self.tray.setToolTip("RAG3 MCP: starting")
        self.menu = QMenu()
        self.open_action = QAction("Open RAG3 UI", self.menu)
        self.open_action.triggered.connect(self.open_ui)
        self.menu.addAction(self.open_action)
        self.menu.addSeparator()
        self.status_action = QAction("MCP: starting", self.menu)
        self.status_action.setEnabled(False)
        self.menu.addAction(self.status_action)
        self.restart_action = QAction("Restart MCP server", self.menu)
        self.restart_action.triggered.connect(self.restart_server)
        self.menu.addAction(self.restart_action)
        self.stop_action = QAction("Stop MCP server", self.menu)
        self.stop_action.triggered.connect(self.stop_server)
        self.menu.addAction(self.stop_action)
        self.menu.addSeparator()
        self.exit_action = QAction("Exit RAG3 background", self.menu)
        self.exit_action.triggered.connect(self.exit)
        self.menu.addAction(self.exit_action)
        self.tray.setContextMenu(self.menu)
        self.tray.activated.connect(self._activated)
        self.tray.show()

        self.timer = QTimer(self)
        self.timer.setInterval(500)
        self.timer.timeout.connect(self._check_server)
        self.timer.start()
        QTimer.singleShot(0, self.start_server)

    def _status(self, label: str, color: str) -> None:
        self.status_action.setText(f"MCP: {label}")
        self.tray.setToolTip(f"RAG3 MCP: {label}")
        self.tray.setIcon(_icon(color))
        self.stop_action.setEnabled(self.process is not None)

    def start_server(self) -> None:
        self.retry_pending = False
        if self.stopping or self.process is not None:
            return
        try:
            self.port = PORT
            if _port_open(PORT):
                self._status(f"port {PORT} is already in use", "#d98b19")
                return
            LOG.parent.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            env["RAG3_MCP_TRANSPORT"] = "streamable-http"
            with LOG.open("a", encoding="utf-8") as log:
                self.process = subprocess.Popen(
                    [_console_python(), str(SERVER)], cwd=ROOT, env=env,
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            self.started_at = time.monotonic()
            self._status("starting", "#8b8f98")
        except Exception as exc:
            self.process = None
            self._status(f"start failed: {exc}", "#c0392b")

    def _check_server(self) -> None:
        if self.process is None:
            return
        code = self.process.poll()
        if code is not None:
            self.process = None
            self._status(f"stopped (exit {code})", "#c0392b")
            if not self.stopping:
                now = time.monotonic()
                self.crashes = [t for t in self.crashes if now - t < 60]
                self.crashes.append(now)
                if len(self.crashes) <= 3 and not self.retry_pending:
                    self.retry_pending = True
                    QTimer.singleShot(2000, self.start_server)
            return
        if _port_open(self.port):
            if "running" not in self.status_action.text():
                self._status(f"running at 127.0.0.1:{self.port}", "#238b45")
        elif time.monotonic() - self.started_at > 90:
            self._status("startup timed out; see log", "#c0392b")

    def stop_server(self) -> None:
        self.stopping = True
        self.retry_pending = False
        if self.process is not None:
            proc = self.process
            self.process = None
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
        self._status("stopped", "#8b8f98")

    def restart_server(self) -> None:
        self.stop_server()
        self.stopping = False
        self.crashes.clear()
        self.start_server()

    def _activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason == QSystemTrayIcon.ActivationReason.DoubleClick:
            self.open_ui()

    def open_ui(self) -> None:
        # Reuse the existing window when the UI was launched independently.
        if os.name == "nt":
            user32 = ctypes.windll.user32
            find_window = user32.FindWindowW
            find_window.argtypes = (ctypes.c_wchar_p, ctypes.c_wchar_p)
            find_window.restype = ctypes.c_void_p
            hwnd = find_window(None, "RAG3 Haystack")
            if hwnd:
                user32.ShowWindow(ctypes.c_void_p(hwnd), 9)  # SW_RESTORE
                user32.SetForegroundWindow(ctypes.c_void_p(hwnd))
                return
        if self.ui_process is not None and self.ui_process.poll() is None:
            return
        try:
            self.ui_process = subprocess.Popen(
                [_gui_python(), str(UI)], cwd=ROOT,
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except Exception as exc:
            self.tray.showMessage("RAG3 UI", str(exc), QSystemTrayIcon.MessageIcon.Warning)

    def exit(self) -> None:
        self.stop_server()
        self.tray.hide()
        self.app.quit()


def main() -> int:
    if "--exit" in sys.argv[1:]:
        return 0 if _request_exit() else 1
    handle = _single_instance_handle()
    if handle is None:
        return 0
    exit_event = _create_exit_event()
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    controller = TrayController(app)
    app.aboutToQuit.connect(controller.stop_server)
    exit_timer = QTimer()
    exit_timer.setInterval(500)
    exit_timer.timeout.connect(
        lambda: controller.exit() if ctypes.windll.kernel32.WaitForSingleObject(
            ctypes.c_void_p(exit_event), 0) == 0 else None
    )
    exit_timer.start()
    try:
        return app.exec()
    finally:
        controller.stop_server()
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CloseHandle(ctypes.c_void_p(exit_event))
        kernel32.CloseHandle(ctypes.c_void_p(handle))


if __name__ == "__main__":
    raise SystemExit(main())
