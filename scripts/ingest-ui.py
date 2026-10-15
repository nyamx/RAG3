from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PySide6.QtCore import QProcess, Qt, QUrl
from PySide6.QtGui import QAction, QColor, QDesktopServices
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
    QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton,
    QSplitter, QTableWidget, QTableWidgetItem, QTabWidget, QToolBar,
    QVBoxLayout, QWidget,
)

from rag3.checkpoint import CheckpointStore
from rag3.source_identity import sha256_file

DEFAULT_NATIVE_COLLECTION = "rag3_haystack"
from rag3.ui_state import JobState, allowed_actions

SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".pptx", ".xlsx", ".html", ".htm", ".md", ".txt",
    ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp",
}

COL_FILE, COL_TYPE, COL_PARSER, COL_STAGE, COL_STATUS = range(5)


def display_detection(path: Path) -> tuple[str, str]:
    ext = path.suffix.lower()
    if ext in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
        return ext.lstrip(".").upper(), "Docling OCR"
    return (ext.lstrip(".").upper() or "FILE"), "DoclingConverter"


class IngestionWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RAG3 Haystack Ingestion")
        self.resize(1280, 780)
        self.jobs: list[dict] = []
        self.current_row: int | None = None
        self.process: QProcess | None = None
        self.stdout_buffer = ""
        self.requested_terminal: JobState | None = None

        toolbar = QToolBar("Ingestion")
        self.addToolBar(toolbar)
        for label, handler in (
            ("Add Files", self.add_files),
            ("Add Folder", self.add_folder),
        ):
            action = QAction(label, self)
            action.triggered.connect(handler)
            toolbar.addAction(action)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        top = QHBoxLayout()
        top.addWidget(QLabel("Qdrant collection"))
        self.collection = QLineEdit(DEFAULT_NATIVE_COLLECTION)
        self.collection.setMaximumWidth(260)
        top.addWidget(self.collection)

        self.buttons = {}
        for key, label, handler in (
            ("run", "Run", self.run_selected),
            ("run_all", "Run All", self.run_all),
            ("pause", "Pause", self.pause_current),
            ("resume", "Resume", self.resume_selected),
            ("cancel", "Cancel", self.cancel_current),
            ("retry", "Retry", self.retry_selected),
            ("restart", "Restart", self.restart_selected),
            ("delete", "Delete", self.remove_selected),
            ("open_source", "Open Source", self.open_source),
            ("open_artifacts", "Open Artifact Folder", self.open_artifacts),
        ):
            btn = QPushButton(label)
            btn.clicked.connect(handler)
            self.buttons[key] = btn
            top.addWidget(btn)
        top.addStretch(1)
        root.addLayout(top)

        splitter = QSplitter(Qt.Orientation.Vertical)
        root.addWidget(splitter, 1)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["File", "Type", "Parser", "Current stage", "Status"])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(COL_FILE, QHeaderView.ResizeMode.Stretch)
        for c in range(1, 5):
            header.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        self.table.itemSelectionChanged.connect(self.selection_changed)
        splitter.addWidget(self.table)

        self.tabs = QTabWidget()
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.inspector = QPlainTextEdit()
        self.inspector.setReadOnly(True)
        self.errors = QPlainTextEdit()
        self.errors.setReadOnly(True)
        self.tabs.addTab(self.log, "Log")
        self.tabs.addTab(self.inspector, "Inspector")
        self.tabs.addTab(self.errors, "Detailed Error")
        splitter.addWidget(self.tabs)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        root.addWidget(self.progress)
        self.statusBar().showMessage("Ready")
        self.apply_action_policy()

    def selected_row(self) -> int | None:
        rows = sorted({x.row() for x in self.table.selectedIndexes()})
        return rows[0] if rows else None

    def state_for_row(self, row: int) -> JobState:
        return JobState(self.jobs[row]["state"])

    def set_job_state(self, row: int, state: JobState, stage: str | None = None) -> None:
        self.jobs[row]["state"] = state.value
        if stage is not None:
            self.jobs[row]["stage"] = stage
            self.table.item(row, COL_STAGE).setText(stage)
        item = self.table.item(row, COL_STATUS)
        item.setText(state.value)
        color = {
            JobState.RUNNING: "#fff2b2",
            JobState.RESUMING: "#fff2b2",
            JobState.PAUSING: "#fff2b2",
            JobState.COMPLETE: "#d7f5dd",
            JobState.FAILED: "#ffd6d6",
            JobState.CANCELLED: "#e5e5e5",
            JobState.PAUSED: "#e3e8ff",
        }.get(state)
        if color:
            item.setBackground(QColor(color))
        self.apply_action_policy()

    def apply_action_policy(self) -> None:
        row = self.selected_row()
        if row is None:
            allowed = frozenset({"run_all"})
        else:
            allowed = allowed_actions(self.state_for_row(row))
        process_busy = self.process is not None and self.process.state() != QProcess.ProcessState.NotRunning
        for key, btn in self.buttons.items():
            enabled = key in allowed
            if process_busy and key not in {"pause", "cancel", "open_source", "open_artifacts"}:
                enabled = False
            btn.setEnabled(enabled)

    def add_path(self, path: str) -> None:
        p = Path(path).expanduser().resolve()
        if not p.is_file() or p.suffix.lower() not in SUPPORTED_EXTENSIONS:
            return
        if any(Path(j["path"]).resolve() == p for j in self.jobs):
            return

        source_sha = sha256_file(p)
        checkpoints = CheckpointStore(source_sha)
        manifest = checkpoints.load_manifest()
        mode = "run"
        state = JobState.QUEUED

        if checkpoints.has_checkpoint() and manifest and manifest.get("state") != "COMPLETE":
            box = QMessageBox(self)
            box.setWindowTitle("Interrupted ingestion found")
            box.setText("Interrupted ingestion found.")
            resume_btn = box.addButton("Resume from checkpoint", QMessageBox.ButtonRole.AcceptRole)
            restart_btn = box.addButton("Restart from beginning", QMessageBox.ButtonRole.DestructiveRole)
            box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
            box.exec()
            if box.clickedButton() is resume_btn:
                mode = "resume"
                state = JobState.PAUSED
            elif box.clickedButton() is restart_btn:
                mode = "restart"
                state = JobState.QUEUED
            else:
                return
        elif manifest and manifest.get("state") == "COMPLETE":
            box = QMessageBox(self)
            box.setWindowTitle("Already indexed")
            box.setText("This source SHA is already COMPLETE.")
            skip_btn = box.addButton("Skip", QMessageBox.ButtonRole.AcceptRole)
            reindex_btn = box.addButton("Re-index", QMessageBox.ButtonRole.DestructiveRole)
            box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
            box.exec()
            if box.clickedButton() is skip_btn:
                state = JobState.COMPLETE
                mode = "run"
            elif box.clickedButton() is reindex_btn:
                state = JobState.QUEUED
                mode = "restart"
            else:
                return

        source_type, parser = display_detection(p)
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.jobs.append({
            "path": str(p),
            "source_sha256": source_sha,
            "state": state.value,
            "stage": "Ready",
            "mode": mode,
            "result": None,
        })
        values = [p.name, source_type, parser, "Ready", state.value]
        for col, value in enumerate(values):
            self.table.setItem(row, col, QTableWidgetItem(value))
        self.table.item(row, COL_FILE).setToolTip(str(p))
        self.table.selectRow(row)
        self.apply_action_policy()

    def add_files(self) -> None:
        pattern = " ".join(f"*{x}" for x in sorted(SUPPORTED_EXTENSIONS))
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Add RAG3 documents", "", f"Supported documents ({pattern});;All files (*)"
        )
        for p in paths:
            self.add_path(p)

    def add_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Add documents from folder")
        if folder:
            for p in sorted(Path(folder).iterdir()):
                if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
                    self.add_path(str(p))

    def selection_changed(self) -> None:
        row = self.selected_row()
        if row is not None:
            self.inspector.setPlainText(json.dumps(self.jobs[row], ensure_ascii=False, indent=2))
        self.apply_action_policy()

    def run_selected(self) -> None:
        row = self.selected_row()
        if row is not None:
            self.start_row(row, self.jobs[row].get("mode", "run"))

    def run_all(self) -> None:
        if self.process and self.process.state() != QProcess.ProcessState.NotRunning:
            return
        for i, job in enumerate(self.jobs):
            if JobState(job["state"]) in {JobState.QUEUED, JobState.IDLE}:
                self.start_row(i, job.get("mode", "run"))
                return
        self.statusBar().showMessage("No queued documents.")

    def start_row(self, row: int, mode: str) -> None:
        if self.process and self.process.state() != QProcess.ProcessState.NotRunning:
            return
        self.current_row = row
        self.requested_terminal = None
        self.jobs[row]["mode"] = mode
        state = JobState.RESUMING if mode == "resume" else JobState.RUNNING
        self.set_job_state(row, state, "Starting")
        self.progress.setValue(5)
        self.log.appendPlainText(f"=== {Path(self.jobs[row]['path']).name} [{mode}] ===")

        self.process = QProcess(self)
        self.process.setProgram(str(ROOT / ".venv" / "Scripts" / "python.exe"))
        self.process.setArguments([
            str(ROOT / "scripts" / "ingest-worker.py"),
            self.jobs[row]["path"],
            "--collection", self.collection.text().strip() or DEFAULT_NATIVE_COLLECTION,
            "--mode", mode,
        ])
        self.process.setWorkingDirectory(str(ROOT))
        self.process.readyReadStandardOutput.connect(self.read_stdout)
        self.process.readyReadStandardError.connect(self.read_stderr)
        self.process.finished.connect(self.finished)
        self.process.start()
        self.apply_action_policy()

    def read_stdout(self) -> None:
        if not self.process:
            return
        text = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        self.stdout_buffer += text
        while "\n" in self.stdout_buffer:
            line, self.stdout_buffer = self.stdout_buffer.split("\n", 1)
            if line.strip():
                self.handle_line(line.strip())

    def read_stderr(self) -> None:
        if not self.process:
            return
        text = bytes(self.process.readAllStandardError()).decode("utf-8", errors="replace").strip()
        if text:
            self.errors.appendPlainText(text[-8000:])

    def handle_line(self, line: str) -> None:
        try:
            event = json.loads(line)
        except Exception:
            self.log.appendPlainText(line)
            return

        etype = event.get("type")
        if etype == "log":
            msg = str(event.get("message", ""))
            self.log.appendPlainText(msg)
            if self.current_row is not None and msg.startswith("STAGE "):
                parts = msg.split()
                if len(parts) >= 3:
                    self.jobs[self.current_row]["stage"] = " ".join(parts[1:])
                    self.table.item(self.current_row, COL_STAGE).setText(" ".join(parts[1:]))
        elif etype == "paused" and self.current_row is not None:
            self.set_job_state(self.current_row, JobState.PAUSED, "Paused at checkpoint")
        elif etype == "result" and self.current_row is not None:
            result = event.get("result") or {}
            self.jobs[self.current_row]["result"] = result
            self.inspector.setPlainText(json.dumps(result, ensure_ascii=False, indent=2))
        elif etype == "error":
            self.errors.appendPlainText(event.get("message", "Worker error"))
            if event.get("traceback"):
                self.errors.appendPlainText(event["traceback"])

    def finished(self, exit_code: int, _exit_status) -> None:
        row = self.current_row
        if row is None:
            return
        if self.stdout_buffer.strip():
            self.handle_line(self.stdout_buffer.strip())
        self.stdout_buffer = ""

        if self.requested_terminal == JobState.CANCELLED:
            self.set_job_state(row, JobState.CANCELLED, "Cancelled; checkpoint preserved")
        elif exit_code == 75:
            self.set_job_state(row, JobState.PAUSED, "Paused at checkpoint")
        elif exit_code == 0:
            self.set_job_state(row, JobState.COMPLETE, "Complete")
            self.progress.setValue(100)
        else:
            self.set_job_state(row, JobState.FAILED, "Failed")

        self.process = None
        self.current_row = None
        self.requested_terminal = None
        self.apply_action_policy()

        for i, job in enumerate(self.jobs):
            if JobState(job["state"]) == JobState.QUEUED:
                self.start_row(i, job.get("mode", "run"))
                return
        self.statusBar().showMessage("Queue finished.")

    def pause_current(self) -> None:
        if self.current_row is None:
            return
        row = self.current_row
        sha = self.jobs[row]["source_sha256"]
        p = ROOT / "data" / "checkpoints" / sha / "pause.request"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("pause\n", encoding="utf-8")
        self.set_job_state(row, JobState.PAUSING, "Pause requested")

    def resume_selected(self) -> None:
        row = self.selected_row()
        if row is not None:
            self.start_row(row, "resume")

    def retry_selected(self) -> None:
        row = self.selected_row()
        if row is None:
            return
        sha = self.jobs[row]["source_sha256"]
        mode = "resume" if CheckpointStore(sha).has_checkpoint() else "run"
        self.start_row(row, mode)

    def restart_selected(self) -> None:
        row = self.selected_row()
        if row is not None:
            self.set_job_state(row, JobState.RESETTING, "Resetting")
            self.start_row(row, "restart")

    def cancel_current(self) -> None:
        if not self.process or self.current_row is None:
            return
        self.requested_terminal = JobState.CANCELLED
        self.set_job_state(self.current_row, JobState.CANCELLING, "Cancelling")
        self.process.kill()

    def remove_selected(self) -> None:
        if self.process and self.process.state() != QProcess.ProcessState.NotRunning:
            return
        for row in sorted({x.row() for x in self.table.selectedIndexes()}, reverse=True):
            self.table.removeRow(row)
            self.jobs.pop(row)
        self.apply_action_policy()

    def open_source(self) -> None:
        row = self.selected_row()
        if row is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.jobs[row]["path"]))

    def open_artifacts(self) -> None:
        row = self.selected_row()
        if row is None:
            return
        folder = ROOT / "data" / "checkpoints" / self.jobs[row]["source_sha256"]
        folder.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

def main() -> None:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--smoke-test", action="store_true")
    args, _ = ap.parse_known_args()
    if args.smoke_test:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication(sys.argv[:1])
    win = IngestionWindow()
    if args.smoke_test:
        print("PYSIDE6_UI_SMOKE_OK")
        return
    win.show()
    raise SystemExit(app.exec())

if __name__ == "__main__":
    main()
