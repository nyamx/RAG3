from __future__ import annotations

import argparse
import json
import os
import sys
import subprocess
from enum import Enum
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PySide6.QtCore import QProcess, Qt, QUrl
from PySide6.QtGui import QAction, QColor, QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QListWidget,
)

from rag3.checkpoint import CheckpointStore
from rag3.source_identity import sha256_file
from rag3.ui_state import JobState, allowed_actions

DEFAULT_NATIVE_COLLECTION = "rag3_haystack"
UI_STATE_PATH = ROOT / "data" / "ui-state.json"

SUPPORTED_EXTENSIONS = {
    ".pdf", ".doc", ".docx", ".pptx", ".xlsx", ".html", ".htm", ".md", ".txt",
    ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp",
}

COL_FILE, COL_TYPE, COL_PARSER, COL_STAGE, COL_STATUS = range(5)


class UiState(str, Enum):
    READY = "READY"
    COLLECTION_BUSY = "COLLECTION_BUSY"
    INGEST_RUNNING = "INGEST_RUNNING"
    INGEST_PAUSED = "INGEST_PAUSED"
    RETRIEVAL_RUNNING = "RETRIEVAL_RUNNING"


class TextComboBox(QComboBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setEditable(True)

    def text(self) -> str:
        return self.currentText()

    def setText(self, value: str) -> None:
        value = str(value)
        idx = self.findText(value)
        if idx >= 0:
            self.setCurrentIndex(idx)
        else:
            self.setEditText(value)


def display_detection(path: Path) -> tuple[str, str]:
    ext = path.suffix.lower()
    if ext in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
        return ext.lstrip(".").upper(), "Docling OCR"
    return (ext.lstrip(".").upper() or "FILE"), "DoclingConverter"


class IngestionWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RAG3 Haystack")
        self.resize(1280, 820)

        self.jobs: list[dict] = []
        self.current_row: int | None = None
        self.process: QProcess | None = None
        self.stdout_buffer = ""
        self.requested_terminal: JobState | None = None

        self.retrieval_process: QProcess | None = None
        self.retrieval_stdout_buffer = ""

        # A single worker serializes collection list/create/content requests.
        # Keeping this boundary here prevents the UI from owning document-store logic.
        self.collection_process: QProcess | None = None
        self.collection_stdout_buffer = ""
        self.collection_stderr_buffer = ""
        self.collection_action: str | None = None
        self.ui_state = UiState.READY
        self.collection_busy_message = ""

        toolbar = QToolBar("Documents")
        self.addToolBar(toolbar)
        self.document_actions: dict[str, QAction] = {}
        for key, label, handler, tip in (
            ("add_files", "Add Files", self.add_files, "Add one or more supported files to the ingestion queue."),
            ("add_folder", "Add Folder", self.add_folder, "Add supported files from one folder to the ingestion queue."),
        ):
            action = QAction(label, self)
            action.setToolTip(tip)
            action.triggered.connect(handler)
            self.document_actions[key] = action
            toolbar.addAction(action)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        collection_bar = QHBoxLayout()
        collection_bar.addWidget(QLabel("Qdrant collection"))
        self.collection = TextComboBox()
        self.collection.setMinimumWidth(280)
        self.collection.setToolTip("Collection used by both ingestion and retrieval.")
        collection_bar.addWidget(self.collection)

        self.refresh_collections_btn = QPushButton("Refresh")
        self.refresh_collections_btn.setToolTip("Reload collection names from the running Qdrant server.")
        self.refresh_collections_btn.clicked.connect(self.refresh_collections)
        collection_bar.addWidget(self.refresh_collections_btn)

        self.create_collection_btn = QPushButton("Create")
        self.create_collection_btn.setToolTip(
            "Create the typed collection with the same 1024D dense + sparse schema used by RAG3."
        )
        self.create_collection_btn.clicked.connect(self.create_collection)
        collection_bar.addWidget(self.create_collection_btn)

        self.manage_collections_btn = QPushButton("Manage...")
        self.manage_collections_btn.setToolTip(
            "Select and delete one or more Qdrant collections."
        )
        self.manage_collections_btn.clicked.connect(self.manage_collections)
        collection_bar.addWidget(self.manage_collections_btn)

        self.collection_busy_label = QLabel("")
        self.collection_busy_label.setVisible(False)
        collection_bar.addWidget(self.collection_busy_label)

        self.collection_progress = QProgressBar()
        self.collection_progress.setRange(0, 0)
        self.collection_progress.setFixedWidth(120)
        self.collection_progress.setTextVisible(False)
        self.collection_progress.setVisible(False)
        collection_bar.addWidget(self.collection_progress)

        collection_bar.addStretch(1)
        root.addLayout(collection_bar)

        self.main_tabs = QTabWidget()
        root.addWidget(self.main_tabs, 1)

        self.ingestion_tab = QWidget()
        self.main_tabs.addTab(self.ingestion_tab, "Ingestion")
        ingest_root = QVBoxLayout(self.ingestion_tab)

        button_row = QHBoxLayout()
        self.buttons = {}
        button_specs = (
            ("run", "Run", self.run_selected, "Run the selected queued document."),
            ("run_all", "Run All", self.run_all, "Run all queued documents sequentially."),
            ("pause", "Pause", self.pause_current, "Request a safe pause after the current pipeline component; save checkpoint."),
            ("resume", "Resume", self.resume_selected, "Resume the selected document from its saved checkpoint."),
            ("cancel", "Cancel", self.cancel_current, "Stop the active worker; the last valid checkpoint is preserved."),
            ("retry", "Retry", self.retry_selected, "Retry a failed/cancelled job; resume if a checkpoint exists."),
            ("restart", "Restart", self.restart_selected, "Discard processing state/checkpoint and process the source again from the beginning."),
            ("delete", "Delete", self.remove_selected, "Remove selected rows from this UI queue only; source/Qdrant data are not deleted."),
            ("open_source", "Open Source", self.open_source, "Open the original source file."),
            ("open_artifacts", "Open Artifact Folder", self.open_artifacts, "Open this source's checkpoint/artifact folder."),
        )
        for key, label, handler, tip in button_specs:
            btn = QPushButton(label)
            btn.setToolTip(tip)
            btn.clicked.connect(handler)
            self.buttons[key] = btn
            button_row.addWidget(btn)
        button_row.addStretch(1)
        ingest_root.addLayout(button_row)

        splitter = QSplitter(Qt.Orientation.Vertical)
        ingest_root.addWidget(splitter, 1)

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
        ingest_root.addWidget(self.progress)

        self.retrieval_tab = QWidget()
        self.main_tabs.addTab(self.retrieval_tab, "Retrieval")
        retrieve_root = QVBoxLayout(self.retrieval_tab)

        query_row = QHBoxLayout()
        query_row.addWidget(QLabel("Query"))
        self.query = QLineEdit()
        self.query.setPlaceholderText("Enter a retrieval query")
        self.query.returnPressed.connect(self.run_retrieval)
        query_row.addWidget(self.query, 1)

        query_row.addWidget(QLabel("K"))
        self.retrieval_top_k = QSpinBox()
        self.retrieval_top_k.setRange(1, 100)
        self.retrieval_top_k.setValue(10)
        self.retrieval_top_k.setToolTip(
            "Final return count. Rank Fusion retrieves 3 x K candidates per selected method."
        )
        query_row.addWidget(self.retrieval_top_k)

        self.retrieval_semantic = QCheckBox("BGE-M3 (Semantic)")
        self.retrieval_bge_lexical = QCheckBox("BGE-M3 (Lexical)")
        self.retrieval_bm25 = QCheckBox("BM25 (Lexical)")
        self.retrieval_rank_fusion = QCheckBox("Rank Fusion (RRF)")
        for checkbox in (
            self.retrieval_semantic,
            self.retrieval_bge_lexical,
            self.retrieval_bm25,
            self.retrieval_rank_fusion,
        ):
            checkbox.setChecked(True)
            checkbox.toggled.connect(lambda _checked: self.update_controls())
            query_row.addWidget(checkbox)

        self.search_btn = QPushButton("Search")
        self.search_btn.setToolTip("Search the selected Qdrant collection and return Top-K chunks.")
        self.search_btn.clicked.connect(self.run_retrieval)
        query_row.addWidget(self.search_btn)
        retrieve_root.addLayout(query_row)

        self.retrieval_status = QLabel("Ready")
        retrieve_root.addWidget(self.retrieval_status)

        self.retrieval_results = QPlainTextEdit()
        self.retrieval_results.setReadOnly(True)
        self.retrieval_results.setPlaceholderText("Top-K retrieval results will appear here.")
        retrieve_root.addWidget(self.retrieval_results, 1)

        # Read-only browser for source documents already stored as chunks.
        self.contents_tab = QWidget()
        self.main_tabs.addTab(self.contents_tab, "Collection Contents")
        contents_root = QVBoxLayout(self.contents_tab)

        contents_actions = QHBoxLayout()
        self.refresh_contents_btn = QPushButton("Refresh Contents")
        self.refresh_contents_btn.setToolTip(
            "Load a document-level summary for the selected Qdrant collection."
        )
        self.refresh_contents_btn.clicked.connect(self.refresh_contents)
        contents_actions.addWidget(self.refresh_contents_btn)
        contents_actions.addStretch(1)
        contents_root.addLayout(contents_actions)

        self.contents_status = QLabel(
            "Press Refresh Contents to load the selected collection."
        )
        contents_root.addWidget(self.contents_status)

        self.contents_table = QTableWidget(0, 5)
        self.contents_table.setHorizontalHeaderLabels(
            ["Title", "Chunks", "Pages", "Source", "Document ID"]
        )
        contents_header = self.contents_table.horizontalHeader()
        contents_header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        contents_header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        contents_header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        contents_header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        contents_header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self.contents_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.contents_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        contents_root.addWidget(self.contents_table, 1)

        # Changing collections invalidates rows from the previous selection.
        self.collection.currentTextChanged.connect(self.collection_changed)

        self.load_ui_state()
        self.statusBar().showMessage("Ready")
        self.update_controls()
        self.refresh_collections()

    def current_collection(self) -> str:
        return self.collection.text().strip() or DEFAULT_NATIVE_COLLECTION

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
        self.update_controls()

    def set_ui_state(self, state: UiState, message: str = "") -> None:
        """Set the global UI operation state and re-apply the control policy."""
        self.ui_state = state
        self.collection_busy_message = message if state == UiState.COLLECTION_BUSY else ""
        self.update_controls()

    def update_controls(self) -> None:
        """Apply all main-window enable/disable rules from one place."""
        state = self.ui_state
        row = self.selected_row()
        selected_allowed = (
            allowed_actions(self.state_for_row(row))
            if row is not None
            else frozenset()
        )
        runnable_count = sum(
            1
            for job in self.jobs
            if JobState(job["state"]) in {JobState.QUEUED, JobState.IDLE}
        )

        collection_ready = state == UiState.READY
        self.collection.setEnabled(collection_ready)
        self.refresh_collections_btn.setEnabled(collection_ready)
        self.create_collection_btn.setEnabled(collection_ready)
        self.manage_collections_btn.setEnabled(collection_ready)

        collection_busy = state == UiState.COLLECTION_BUSY
        self.collection_busy_label.setText(self.collection_busy_message)
        self.collection_busy_label.setVisible(collection_busy)
        self.collection_progress.setVisible(collection_busy)

        for action in self.document_actions.values():
            action.setEnabled(state == UiState.READY)

        retrieval_ready = state == UiState.READY
        selected_retrievers = sum(
            checkbox.isChecked()
            for checkbox in (
                self.retrieval_semantic,
                self.retrieval_bge_lexical,
                self.retrieval_bm25,
            )
        )
        self.query.setEnabled(retrieval_ready)
        self.retrieval_top_k.setEnabled(retrieval_ready)
        self.retrieval_semantic.setEnabled(retrieval_ready)
        self.retrieval_bge_lexical.setEnabled(retrieval_ready)
        self.retrieval_bm25.setEnabled(retrieval_ready)
        self.retrieval_rank_fusion.setEnabled(
            retrieval_ready and selected_retrievers >= 2
        )
        self.search_btn.setEnabled(
            retrieval_ready
            and selected_retrievers >= 1
            and bool(self.current_collection())
        )
        self.refresh_contents_btn.setEnabled(retrieval_ready)

        for button in self.buttons.values():
            button.setEnabled(False)

        if state == UiState.READY:
            for key, button in self.buttons.items():
                if key == "run_all":
                    button.setEnabled(runnable_count > 0)
                else:
                    button.setEnabled(key in selected_allowed)

        elif state == UiState.INGEST_RUNNING:
            active_state = (
                self.state_for_row(self.current_row)
                if self.current_row is not None
                else None
            )
            self.buttons["pause"].setEnabled(
                active_state in {JobState.RUNNING, JobState.RESUMING}
            )
            self.buttons["cancel"].setEnabled(self.current_row is not None)
            if row is not None:
                self.buttons["open_source"].setEnabled("open_source" in selected_allowed)
                self.buttons["open_artifacts"].setEnabled("open_artifacts" in selected_allowed)

        elif state == UiState.INGEST_PAUSED and row is not None:
            for key in ("resume", "restart", "delete", "open_source", "open_artifacts"):
                self.buttons[key].setEnabled(key in selected_allowed)

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
        self.update_controls()

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
        self.update_controls()

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
        self.set_ui_state(UiState.INGEST_RUNNING)
        state = JobState.RESUMING if mode == "resume" else JobState.RUNNING
        self.set_job_state(row, state, "Starting")
        self.progress.setValue(5)
        self.log.appendPlainText(f"=== {Path(self.jobs[row]['path']).name} [{mode}] ===")

        self.process = QProcess(self)
        self.process.setProgram(str(ROOT / ".venv" / "Scripts" / "python.exe"))
        self.process.setArguments([
            str(ROOT / "scripts" / "ingest-worker.py"),
            self.jobs[row]["path"],
            "--collection", self.current_collection(),
            "--mode", mode,
        ])
        self.process.setWorkingDirectory(str(ROOT))
        self.process.readyReadStandardOutput.connect(self.read_stdout)
        self.process.readyReadStandardError.connect(self.read_stderr)
        self.process.finished.connect(self.finished)
        self.process.start()
        self.update_controls()

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
            self.process = None
            self.set_ui_state(UiState.READY)
            return

        if self.stdout_buffer.strip():
            self.handle_line(self.stdout_buffer.strip())
        self.stdout_buffer = ""

        if self.requested_terminal == JobState.CANCELLED:
            final_state = JobState.CANCELLED
            self.set_job_state(row, final_state, "Cancelled; checkpoint preserved")
        elif exit_code == 75:
            final_state = JobState.PAUSED
            self.set_job_state(row, final_state, "Paused at checkpoint")
        elif exit_code == 0:
            final_state = JobState.COMPLETE
            self.set_job_state(row, final_state, "Complete")
            self.progress.setValue(100)
        else:
            final_state = JobState.FAILED
            self.set_job_state(row, final_state, "Failed")

        self.process = None
        self.current_row = None
        self.requested_terminal = None

        for i, job in enumerate(self.jobs):
            if JobState(job["state"]) == JobState.QUEUED:
                self.start_row(i, job.get("mode", "run"))
                return

        if final_state == JobState.PAUSED:
            self.set_ui_state(UiState.INGEST_PAUSED)
        else:
            self.set_ui_state(UiState.READY)
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
        rows = sorted({item.row() for item in self.table.selectedIndexes()}, reverse=True)
        for row in rows:
            self.table.removeRow(row)
            self.jobs.pop(row)

        if self.ui_state == UiState.INGEST_PAUSED and not any(
            JobState(job["state"]) == JobState.PAUSED for job in self.jobs
        ):
            self.set_ui_state(UiState.READY)
        else:
            self.update_controls()

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

    def _start_collection_worker(
        self,
        action: str,
        collection: str | None = None,
        collections: list[str] | None = None,
        bm25_settings: tuple[float, float, float] | None = None,
    ) -> bool:
        if (
            self.collection_process
            and self.collection_process.state() != QProcess.ProcessState.NotRunning
        ):
            self.statusBar().showMessage(
                "A collection operation is already running. Please wait."
            )
            return False

        busy_message = {
            "list": "Loading collections…",
            "create": "Creating collection…",
            "delete_many": "Deleting collections…",
            "contents": "Loading collection contents…",
        }.get(action, "Working with collections…")

        self.collection_action = action
        self.collection_stdout_buffer = ""
        self.collection_stderr_buffer = ""
        self.set_ui_state(UiState.COLLECTION_BUSY, busy_message)
        self.statusBar().showMessage(busy_message)

        self.collection_process = QProcess(self)
        self.collection_process.setProgram(str(ROOT / ".venv" / "Scripts" / "python.exe"))
        args = [str(ROOT / "scripts" / "collection-worker.py"), "--action", action]
        if collection:
            args += ["--collection", collection]
        if collections:
            args += ["--collections-json", json.dumps(collections, ensure_ascii=False)]
        if bm25_settings is not None:
            bm25_k, bm25_b, bm25_avg_len = bm25_settings
            args += [
                "--bm25-k", str(bm25_k),
                "--bm25-b", str(bm25_b),
                "--bm25-avg-len", str(bm25_avg_len),
            ]
        self.collection_process.setArguments(args)
        self.collection_process.setWorkingDirectory(str(ROOT))
        self.collection_process.readyReadStandardOutput.connect(self.read_collection_stdout)
        self.collection_process.readyReadStandardError.connect(self.read_collection_stderr)
        self.collection_process.finished.connect(self.collection_finished)
        self.collection_process.start()
        return True

    def refresh_collections(self) -> None:
        self._start_collection_worker("list")

    def create_collection(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("Create Collection")
        form = QFormLayout(dialog)

        name_edit = QLineEdit(self.current_collection())
        form.addRow("Collection", name_edit)

        bm25_k = QDoubleSpinBox()
        bm25_k.setRange(0.01, 100.0)
        bm25_k.setDecimals(3)
        bm25_k.setValue(1.2)
        form.addRow("BM25 k", bm25_k)

        bm25_b = QDoubleSpinBox()
        bm25_b.setRange(0.0, 1.0)
        bm25_b.setDecimals(3)
        bm25_b.setValue(0.75)
        form.addRow("BM25 b", bm25_b)

        bm25_avg_len = QDoubleSpinBox()
        bm25_avg_len.setRange(1.0, 100000.0)
        bm25_avg_len.setDecimals(1)
        bm25_avg_len.setValue(256.0)
        form.addRow("BM25 avg_len", bm25_avg_len)

        form.addRow(
            "BM25 policy",
            QLabel("multilingual / lowercase / no stemming / no stopwords"),
        )

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        name = name_edit.text().strip()
        if not name:
            self.statusBar().showMessage("Collection name is required.")
            return

        self.collection.setText(name)
        self._start_collection_worker(
            "create",
            name,
            bm25_settings=(
                bm25_k.value(),
                bm25_b.value(),
                bm25_avg_len.value(),
            ),
        )

    def manage_collections(self) -> None:
        """Open a multi-select collection manager for destructive operations."""
        if self.ui_state != UiState.READY:
            self.statusBar().showMessage("Another operation is still running. Please wait.")
            return

        names = [self.collection.itemText(i) for i in range(self.collection.count())]
        names = [name for name in names if name]
        if not names:
            QMessageBox.information(self, "Manage Collections", "No collections are available.")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Manage Qdrant Collections")
        dialog.resize(520, 420)
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel("Select one or more collections. Use Ctrl/Shift, or Select All."))

        collection_list = QListWidget()
        collection_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        collection_list.addItems(names)
        layout.addWidget(collection_list, 1)

        current = self.current_collection()
        matches = collection_list.findItems(current, Qt.MatchFlag.MatchExactly)
        if matches:
            matches[0].setSelected(True)
            collection_list.setCurrentItem(matches[0])

        buttons = QHBoxLayout()
        select_all_btn = QPushButton("Select All")
        clear_btn = QPushButton("Deselect All")
        delete_btn = QPushButton("Delete Selected")
        close_btn = QPushButton("Close")
        buttons.addWidget(select_all_btn)
        buttons.addWidget(clear_btn)
        buttons.addStretch(1)
        buttons.addWidget(delete_btn)
        buttons.addWidget(close_btn)
        layout.addLayout(buttons)

        select_all_btn.clicked.connect(collection_list.selectAll)
        clear_btn.clicked.connect(collection_list.clearSelection)
        close_btn.clicked.connect(dialog.reject)

        def delete_selected() -> None:
            selected = [item.text() for item in collection_list.selectedItems()]
            if not selected:
                QMessageBox.information(dialog, "Delete Collections", "Select at least one collection.")
                return
            preview = "\n".join(f"• {name}" for name in selected)
            answer = QMessageBox.warning(
                dialog,
                "Delete Collections",
                (
                    f"Delete {len(selected)} collection(s) permanently?\n\n"
                    f"{preview}\n\n"
                    "All chunks and metadata in these collections will be removed. "
                    "This cannot be undone."
                ),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
            if self._start_collection_worker("delete_many", collections=selected):
                dialog.accept()

        delete_btn.clicked.connect(delete_selected)
        dialog.exec()

    def collection_changed(self, _text: str) -> None:
        """Clear stale rows whenever the shared collection selector changes."""
        self.contents_table.setRowCount(0)
        self.contents_status.setText(
            f"Collection selected: {self.current_collection()} — press Refresh Contents."
        )

    def refresh_contents(self) -> None:
        """Request a fresh document-level view from the collection worker."""
        if self.ui_state != UiState.READY:
            self.contents_status.setText("Another operation is still running.")
            return
        self.contents_table.setRowCount(0)
        self.contents_status.setText(f"Loading {self.current_collection()}...")
        self._start_collection_worker("contents", self.current_collection())

    def _render_collection_contents(self, payload: dict) -> None:
        """Render worker output without reimplementing storage/grouping logic."""
        documents = payload.get("documents") or []
        self.contents_table.setRowCount(len(documents))

        for row, document in enumerate(documents):
            page_start = document.get("page_start")
            page_end = document.get("page_end")
            if page_start is None and page_end is None:
                pages = "—"
            elif page_end is None or page_start == page_end:
                pages = str(page_start)
            elif page_start is None:
                pages = str(page_end)
            else:
                pages = f"{page_start}–{page_end}"

            values = (
                document.get("title") or "—",
                str(document.get("chunk_count") or 0),
                pages,
                document.get("source_uri") or "—",
                document.get("document_id") or "—",
            )
            for column, value in enumerate(values):
                self.contents_table.setItem(
                    row, column, QTableWidgetItem(str(value))
                )

        self.contents_status.setText(
            f"{payload.get('document_count', len(documents))} document(s), "
            f"{payload.get('chunk_count', 0)} chunk(s) | "
            f"collection={payload.get('collection') or self.current_collection()}"
        )

    def read_collection_stdout(self) -> None:
        if not self.collection_process:
            return
        self.collection_stdout_buffer += bytes(
            self.collection_process.readAllStandardOutput()
        ).decode("utf-8", errors="replace")

    def read_collection_stderr(self) -> None:
        if not self.collection_process:
            return
        text = bytes(
            self.collection_process.readAllStandardError()
        ).decode("utf-8", errors="replace")
        if text:
            self.collection_stderr_buffer += text
            self.statusBar().showMessage(text.strip()[-300:])

    def collection_finished(self, exit_code: int, _exit_status) -> None:
        text = self.collection_stdout_buffer.strip()
        error_text = self.collection_stderr_buffer.strip()
        action = self.collection_action
        self.collection_stdout_buffer = ""
        self.collection_stderr_buffer = ""
        self.collection_process = None
        self.collection_action = None

        if exit_code != 0:
            message = "Qdrant collection operation failed."
            try:
                payload = json.loads(text.splitlines()[-1]) if text else {}
                detail = str(payload.get("error") or "").strip()
            except Exception:
                detail = ""
            if not detail and error_text:
                detail = error_text.splitlines()[-1].strip()
            if detail:
                message = f"{message} {detail}"
            self.set_ui_state(UiState.READY)
            self.statusBar().showMessage(message)
            if action == "delete_many":
                QMessageBox.critical(self, "Delete Collections Failed", message)
            if action == "contents":
                self.contents_status.setText("Failed to load collection contents.")
            return

        try:
            payload = json.loads(text.splitlines()[-1]) if text else {}
        except Exception:
            self.set_ui_state(UiState.READY)
            self.statusBar().showMessage("Invalid collection worker response.")
            if action == "contents":
                self.contents_status.setText("Invalid collection contents response.")
            return

        if action == "list":
            current = self.current_collection()
            names = payload.get("collections") or []
            self.collection.blockSignals(True)
            self.collection.clear()
            self.collection.addItems(names)
            if current in names:
                self.collection.setText(current)
            elif names:
                self.collection.setText(names[0])
            else:
                self.collection.setText("")
            self.collection.blockSignals(False)
            if self.current_collection() != current:
                self.collection_changed(self.current_collection())
            self.set_ui_state(UiState.READY)
            self.statusBar().showMessage(f"Loaded {len(names)} Qdrant collections.")

        elif action == "create":
            name = str(payload.get("collection") or self.current_collection())
            self.collection.setText(name)
            self.statusBar().showMessage(f"Collection ready: {name}")
            self.refresh_collections()

        elif action == "delete_many":
            deleted = payload.get("deleted") or []
            deleted = deleted if isinstance(deleted, list) else []
            deleted_set = set(deleted)
            count = len(deleted)
            remaining = [
                self.collection.itemText(i)
                for i in range(self.collection.count())
                if self.collection.itemText(i) not in deleted_set
            ]
            self.collection.blockSignals(True)
            self.collection.clear()
            self.collection.addItems(remaining)
            self.collection.setText(remaining[0] if remaining else "")
            self.collection.blockSignals(False)
            self.contents_table.setRowCount(0)
            self.collection_changed(self.current_collection())
            self.contents_status.setText(f"Deleted {count} collection(s). Refreshing list…")
            self.statusBar().showMessage(f"Deleted {count} collection(s). Refreshing list…")
            self.refresh_collections()

        elif action == "contents":
            self._render_collection_contents(payload)
            self.set_ui_state(UiState.READY)
            self.statusBar().showMessage(
                f"Loaded contents for {payload.get('collection') or self.current_collection()}."
            )
        else:
            self.set_ui_state(UiState.READY)

    def run_retrieval(self) -> None:
        if self.ui_state != UiState.READY:
            return
        if self.retrieval_process and self.retrieval_process.state() != QProcess.ProcessState.NotRunning:
            return
        query = self.query.text().strip()
        if not query:
            self.retrieval_status.setText("Enter a query.")
            return

        self.retrieval_results.clear()
        self.retrieval_status.setText("Searching...")
        self.retrieval_stdout_buffer = ""
        self.set_ui_state(UiState.RETRIEVAL_RUNNING)

        methods = []
        if self.retrieval_semantic.isChecked():
            methods.append("--semantic")
        if self.retrieval_bge_lexical.isChecked():
            methods.append("--bge-lexical")
        if self.retrieval_bm25.isChecked():
            methods.append("--bm25")
        if not methods:
            self.retrieval_status.setText("Select at least one retrieval method.")
            self.set_ui_state(UiState.READY)
            return

        use_fusion = self.retrieval_rank_fusion.isChecked() and len(methods) >= 2

        self.retrieval_process = QProcess(self)
        self.retrieval_process.setProgram(str(ROOT / ".venv" / "Scripts" / "python.exe"))
        args = [
            str(ROOT / "scripts" / "retrieval-worker.py"),
            "--query", query,
            "--collection", self.current_collection(),
            "--top-k", str(self.retrieval_top_k.value()),
            *methods,
        ]
        if use_fusion:
            args.append("--rank-fusion")
        self.retrieval_process.setArguments(args)
        self.retrieval_process.setWorkingDirectory(str(ROOT))
        self.retrieval_process.readyReadStandardOutput.connect(self.read_retrieval_stdout)
        self.retrieval_process.readyReadStandardError.connect(self.read_retrieval_stderr)
        self.retrieval_process.finished.connect(self.retrieval_finished)
        self.retrieval_process.start()

    def read_retrieval_stdout(self) -> None:
        if not self.retrieval_process:
            return
        self.retrieval_stdout_buffer += bytes(
            self.retrieval_process.readAllStandardOutput()
        ).decode("utf-8", errors="replace")

    def read_retrieval_stderr(self) -> None:
        if not self.retrieval_process:
            return
        text = bytes(self.retrieval_process.readAllStandardError()).decode("utf-8", errors="replace").strip()
        if text:
            self.retrieval_results.appendPlainText(text[-8000:])

    def retrieval_finished(self, exit_code: int, _exit_status) -> None:
        text = self.retrieval_stdout_buffer.strip()
        self.retrieval_stdout_buffer = ""
        self.retrieval_process = None
        self.set_ui_state(UiState.READY)

        if exit_code != 0:
            self.retrieval_status.setText("Search failed.")
            return
        try:
            payload = json.loads(text.splitlines()[-1])
            result = payload.get("result") or {}
            hits = result.get("hits") or []
        except Exception as exc:
            self.retrieval_status.setText(f"Invalid search response: {exc}")
            return

        methods = result.get("methods") or []
        fusion_label = "RRF" if result.get("rank_fusion") else "independent"
        self.retrieval_status.setText(
            f"{len(hits)} hit(s) | collection={result.get('collection')} | "
            f"methods={','.join(methods)} | {fusion_label}"
        )
        chunks = []
        for hit in hits:
            chunks.append(
                "\n".join([
                    f"#{hit.get('rank')}  [{hit.get('retriever')}]  score={hit.get('score')}",
                    f"title: {hit.get('title')}",
                    f"page: {hit.get('page_start')} - {hit.get('page_end')}",
                    f"section: {' > '.join(hit.get('section_path') or [])}",
                    f"source: {hit.get('source_uri')}",
                    "",
                    str(hit.get("content") or ""),
                ])
            )
        self.retrieval_results.setPlainText(("\n\n" + ("-" * 80) + "\n\n").join(chunks))

    def load_ui_state(self) -> None:
        try:
            state = json.loads(UI_STATE_PATH.read_text(encoding="utf-8"))
        except Exception:
            state = {}

        self.collection.setText(str(state.get("collection") or DEFAULT_NATIVE_COLLECTION))
        self.query.setText(str(state.get("retrieval_query") or ""))

        self.retrieval_semantic.setChecked(
            bool(state.get("retrieval_semantic", True))
        )
        self.retrieval_bge_lexical.setChecked(
            bool(state.get("retrieval_bge_lexical", True))
        )
        self.retrieval_bm25.setChecked(
            bool(state.get("retrieval_bm25", True))
        )
        self.retrieval_rank_fusion.setChecked(
            bool(state.get("retrieval_rank_fusion", True))
        )

        try:
            self.retrieval_top_k.setValue(int(state.get("retrieval_top_k", 10)))
        except Exception:
            self.retrieval_top_k.setValue(10)

        try:
            self.main_tabs.setCurrentIndex(int(state.get("main_tab", 0)))
        except Exception:
            pass

        width = state.get("window_width")
        height = state.get("window_height")
        if isinstance(width, int) and isinstance(height, int) and width > 500 and height > 400:
            self.resize(width, height)

    def save_ui_state(self) -> None:
        UI_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "collection": self.current_collection(),
            "retrieval_query": self.query.text(),
            "retrieval_semantic": self.retrieval_semantic.isChecked(),
            "retrieval_bge_lexical": self.retrieval_bge_lexical.isChecked(),
            "retrieval_bm25": self.retrieval_bm25.isChecked(),
            "retrieval_rank_fusion": self.retrieval_rank_fusion.isChecked(),
            "retrieval_top_k": self.retrieval_top_k.value(),
            "main_tab": self.main_tabs.currentIndex(),
            "window_width": self.width(),
            "window_height": self.height(),
        }
        tmp = UI_STATE_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(UI_STATE_PATH)

    def closeEvent(self, event) -> None:
        self.save_ui_state()
        super().closeEvent(event)


def ensure_infrastructure() -> tuple[bool, str]:
    script = ROOT / "scripts" / "ensure-infra.ps1"
    if not script.is_file():
        return False, f"Missing infrastructure script: {script}"

    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    try:
        proc = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(script),
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
            creationflags=creationflags,
        )
    except Exception as exc:
        return False, f"Failed to start Qdrant infrastructure: {exc}"

    output = "\n".join(x for x in [proc.stdout.strip(), proc.stderr.strip()] if x)
    if proc.returncode != 0:
        return False, output or f"ensure-infra.ps1 failed with exit code {proc.returncode}"
    return True, output or "Qdrant infrastructure ready."


def launch_mcp_tray() -> tuple[bool, str]:
    """Start the independent tray controller; an existing controller exits its duplicate."""
    if os.name != "nt":
        return False, "The MCP tray controller requires Windows."
    script = ROOT / "scripts" / "rag3-mcp-tray.py"
    if not script.is_file():
        return False, f"Missing MCP tray controller: {script}"
    executable = Path(sys.executable)
    windowless = executable.with_name("pythonw.exe")
    if windowless.is_file():
        executable = windowless
    log_path = ROOT / "data" / "logs" / "mcp-tray.log"
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log:
            flags = (getattr(subprocess, "CREATE_NO_WINDOW", 0)
                     | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
            # A launcher may itself be inside a Windows Job Object. Break away
            # when permitted so closing that launcher cannot stop the tray.
            try:
                subprocess.Popen(
                    [str(executable), str(script)], cwd=str(ROOT),
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    creationflags=flags | getattr(subprocess, "CREATE_BREAKAWAY_FROM_JOB", 0),
                )
            except OSError:
                subprocess.Popen(
                    [str(executable), str(script)], cwd=str(ROOT),
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    creationflags=flags,
                )
    except Exception as exc:
        return False, f"MCP tray controller could not start: {exc}"
    return True, "MCP background controller is starting in the system tray."


def launch_codex_plugin_check() -> tuple[bool, str]:
    """Check/install the local Codex plugin without delaying UI startup."""
    if os.name != "nt":
        return False, "Codex plugin auto-registration requires Windows."
    script = ROOT / "scripts" / "ensure-rag3-codex-plugin.py"
    if not script.is_file():
        return False, f"Missing Codex plugin registration helper: {script}"
    executable = Path(sys.executable)
    windowless = executable.with_name("pythonw.exe")
    if windowless.is_file():
        executable = windowless
    log_path = ROOT / "data" / "logs" / "codex-plugin-install.log"
    status_path = ROOT / "data" / "logs" / "codex-plugin-status.json"
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log:
            flags = (getattr(subprocess, "CREATE_NO_WINDOW", 0)
                     | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
            command = [str(executable), str(script), "--status-file", str(status_path)]
            try:
                subprocess.Popen(
                    command, cwd=str(ROOT), stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT,
                    creationflags=flags | getattr(subprocess, "CREATE_BREAKAWAY_FROM_JOB", 0),
                )
            except OSError:
                subprocess.Popen(
                    command, cwd=str(ROOT), stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, creationflags=flags,
                )
    except Exception as exc:
        return False, f"Codex plugin check could not start: {exc}"
    return True, "Codex plugin registration check started."


def main() -> None:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--smoke-test", action="store_true")
    args, _ = ap.parse_known_args()
    if args.smoke_test:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication(sys.argv[:1])

    infra_ok, infra_message = ensure_infrastructure()
    if not infra_ok:
        QMessageBox.critical(
            None,
            "RAG3 startup failed",
            "Qdrant infrastructure could not be started.\n\n" + infra_message,
        )
        raise SystemExit(1)
    win = IngestionWindow()
    win.statusBar().showMessage("Qdrant ready")
    if args.smoke_test:
        print("PYSIDE6_UI_SMOKE_OK")
        return
    tray_ok, tray_message = launch_mcp_tray()
    if not tray_ok:
        QMessageBox.warning(win, "RAG3 MCP background", tray_message)
    else:
        win.statusBar().showMessage("Qdrant ready; MCP background starting")
    plugin_ok, plugin_message = launch_codex_plugin_check()
    if not plugin_ok and os.name == "nt":
        win.statusBar().showMessage(plugin_message)
    win.show()
    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()
