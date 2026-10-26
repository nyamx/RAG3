from __future__ import annotations

import argparse
import json
import os
import re
import sys
from enum import Enum
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PySide6.QtCore import QEvent, QFileSystemWatcher, QTimer, Qt, QUrl
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

from rag3.source_identity import sha256_file
from rag3.ui_backend_client import UIBackendClient
from rag3.constants import DATA_ROOT

DEFAULT_NATIVE_COLLECTION = "rag3_haystack"
UI_STATE_PATH = DATA_ROOT / "ui-state.json"
CHANGE_SIGNAL_PATH = DATA_ROOT / "ingestion_jobs.signal"


def content_scope(collection: str) -> str:
    return "contents:" + collection

SUPPORTED_EXTENSIONS = {
    ".pdf", ".doc", ".docx", ".pptx", ".xlsx", ".html", ".htm", ".md", ".txt",
    ".json", ".jsonl", ".csv", ".tsv",
    ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp",
}

COL_ID, COL_ORIGIN, COL_COLLECTION, COL_FILE, COL_STAGE, COL_STATUS = range(6)


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

    def showPopup(self):
        callback = getattr(self, 'refresh_popup', None)
        if callback: callback()
        super().showPopup()

    def text(self) -> str:
        return self.currentText()

    def focusOutEvent(self, event):
        super().focusOutEvent(event)
        callback = getattr(self, 'focus_lost', None)
        if callback:
            callback()

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

        self.backend = UIBackendClient(self)
        self._latest_versions: dict[str, int] = {}
        self._jobs_seen_version = -1
        self._collections_seen_version = -1
        self._jobs_loading = False
        self._revisions_loading = False
        self._jobs_again = False
        self._job_write_pending: set[str] = set()
        self.jobs: list[dict] = []
        self._last_complete: set[str] = set()
        self._last_collection_check = 0
        self._job_view: tuple = ()
        self._contents_view: tuple | None = None
        self._displayed_contents_collection: str | None = None
        self._contents_seen: dict[str, int] = {}
        self._contents_loading_version = 0
        self._pending_collection_names: list[str] | None = None
        self._collections_loading_version = 0
        self._collection_background = False

        self._retrieval_pending = False
        self._collection_pending = False
        self.collection_action: str | None = None
        self._collection_background = False
        self.ui_state = UiState.READY

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

        self.create_collection_btn = QPushButton("Create")
        self.create_collection_btn.setToolTip(
            "Create the typed collection with 1024D dense and BM25 indexes used by RAG3."
        )
        self.create_collection_btn.clicked.connect(self.create_collection)
        collection_bar.addWidget(self.create_collection_btn)

        self.manage_collections_btn = QPushButton("Manage...")
        self.manage_collections_btn.setToolTip(
            "Select and delete one or more Qdrant collections."
        )
        self.manage_collections_btn.clicked.connect(self.manage_collections)
        collection_bar.addWidget(self.manage_collections_btn)

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
            ("pause", "Pause", self.pause_current, "Pause at the next safe step; completed chunks remain in Qdrant."),
            ("resume", "Resume", self.resume_selected, "Resume from the Qdrant ingestion progress."),
            ("cancel", "Cancel", self.cancel_current, "Stop the active worker; staged Qdrant chunks remain available for retry."),
            ("retry", "Retry", self.retry_selected, "Retry a failed/cancelled job using staged chunks when available."),
            ("restart", "Restart", self.restart_selected, "Discard staged processing progress and start the source again."),
            ("delete", "Delete", self.remove_selected, "Remove selected rows from this UI queue only; source/Qdrant data are not deleted."),
            ("open_source", "Open Source", self.open_source, "Open the original source file."),
            ("open_artifacts", "Open Source Folder", self.open_artifacts, "Open this source's managed original folder."),
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

        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["Job ID", "Origin", "Collection", "File", "Current stage", "Status"])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(COL_FILE, QHeaderView.ResizeMode.Stretch)
        for c in (COL_ID, COL_ORIGIN, COL_COLLECTION, COL_STAGE, COL_STATUS):
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
        self.retrieval_bm25 = QCheckBox("BM25 (Lexical)")
        self.retrieval_rank_fusion = QCheckBox("Rank Fusion (RRF)")
        for checkbox in (
            self.retrieval_semantic,
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

        self.contents_status = QLabel("Collection contents load automatically when this tab opens.")
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

        # File notifications wake a version read; only the backend reads SQLite.
        self.change_watcher = QFileSystemWatcher(self)
        if CHANGE_SIGNAL_PATH.parent.is_dir():
            self.change_watcher.addPath(str(CHANGE_SIGNAL_PATH.parent))
        self._watch_change_files()
        self.change_watcher.fileChanged.connect(self._change_signal)
        self.change_watcher.directoryChanged.connect(self._change_directory)
        self.change_debounce = QTimer(self)
        self.change_debounce.setSingleShot(True)
        self.change_debounce.setInterval(150)
        self.change_debounce.timeout.connect(self._check_revisions)
        self.fallback_timer = QTimer(self)
        self.fallback_timer.setInterval(15000)
        self.fallback_timer.timeout.connect(self._check_revisions)
        self.fallback_timer.start()
        self.collection.focus_lost = self._collection_focus_lost
        self.load_ui_state()
        self.statusBar().showMessage('Connecting to shared RAG3 backend…')
        self.update_controls()
        self.collection.refresh_popup = self._check_revisions
        self.main_tabs.currentChanged.connect(self.tab_changed)
        QTimer.singleShot(0, self._check_revisions)

    def _change_signal(self, path: str) -> None:
        self._watch_change_files()
        self.change_debounce.start()

    def _change_directory(self, _path: str) -> None:
        if self._watch_change_files():
            self.change_debounce.start()

    def _watch_change_files(self) -> bool:
        watched = set(self.change_watcher.files())
        if CHANGE_SIGNAL_PATH.is_file() and str(CHANGE_SIGNAL_PATH) not in watched:
            self.change_watcher.addPath(str(CHANGE_SIGNAL_PATH))
            return True
        return False

    def _collection_focus_lost(self) -> None:
        if self._pending_collection_names is not None:
            names = self._pending_collection_names
            self._pending_collection_names = None
            self._apply_collection_names(names, background=False)
        self._check_revisions()

    def _collection_is_editing(self) -> bool:
        editor = self.collection.lineEdit()
        return (self.collection.hasFocus() or (editor is not None and editor.hasFocus())
                or self.collection.view().isVisible())

    def _check_revisions(self) -> None:
        if self._revisions_loading:
            return
        self._revisions_loading = True
        requested_collection = self.current_collection()

        def finished(result, error):
            self._revisions_loading = False
            if error:
                self.statusBar().showMessage(f'Backend unavailable: {error}')
                return
            versions = (result or {}).get('versions') or {}
            self._latest_versions.update(versions)
            if versions.get('jobs', 0) != self._jobs_seen_version:
                self._jobs_seen_version = versions.get('jobs', 0)
                self.load_jobs()
            if (versions.get('collections', 0) != self._collections_seen_version
                    and self.ui_state == UiState.READY and not self._collection_is_editing()):
                self.refresh_collections(background=True)
            current = self.current_collection()
            if (current == requested_collection and current
                    and self.collection.findText(current) >= 0
                    and self.ui_state == UiState.READY
                    and self.main_tabs.currentWidget() is self.contents_tab
                    and (self._displayed_contents_collection != current
                         or versions.get(content_scope(current), 0) != self._contents_seen.get(current))):
                self.refresh_contents(background=True)

        self.backend.command('ui.revisions', {'collection': requested_collection}, finished)

    def event(self, event):
        result = super().event(event)
        if event.type() == QEvent.Type.WindowActivate and hasattr(self, 'change_debounce'):
            self.change_debounce.start()
        return result

    def current_collection(self) -> str:
        return self.collection.text().strip()

    def selected_row(self) -> int | None:
        rows = sorted({x.row() for x in self.table.selectedIndexes()})
        return rows[0] if rows else None

    def _manager_action(self, operation: str, job_id: str) -> bool:
        if job_id in self._job_write_pending:
            return False
        self._job_write_pending.add(job_id)

        def finished(result, error):
            self._job_write_pending.discard(job_id)
            if error:
                self.errors.appendPlainText(f'{operation} uncertain/failed: {error}')
                self.statusBar().showMessage(error)
            self.load_jobs()
            self._check_revisions()

        return self.backend.command('jobs.action', {'job_id': job_id, 'operation': operation},
                                    finished, read_only=False)

    def save_jobs(self) -> None:
        """Job storage is exclusively owned by the backend."""
        return

    def load_jobs(self) -> bool:
        if self._jobs_loading:
            self._jobs_again = True
            return False
        self._jobs_loading = True
        data: list[dict] = []

        def load_page(offset: int):
            def finished(result, error):
                if error:
                    self._jobs_loading = False
                    self.statusBar().showMessage(f'Jobs unavailable: {error}')
                    return
                page = (result or {}).get('jobs') or []
                data.extend(page)
                total = (result or {}).get('total', len(data))
                if page and len(data) < total:
                    load_page(len(data))
                    return
                self._jobs_loading = False
                self._apply_jobs(data)
                if self._jobs_again:
                    self._jobs_again = False
                    QTimer.singleShot(0, self.load_jobs)

            self.backend.command('jobs.list', {'limit': 200, 'offset': offset,
                                               'include_hidden': False}, finished)

        load_page(0)
        return True

    def _apply_jobs(self, data: list[dict]) -> None:
        view = tuple((j['job_id'], j['origin'], j['collection'], j['filename'],
                      j['stage'], j['status'], j.get('source_sha256')) for j in data)
        if view == self._job_view:
            return True
        selected = self.selected_row()
        selected_id = self.jobs[selected]['job_id'] if selected is not None and selected < len(self.jobs) else None
        scrollbar = self.table.verticalScrollBar().value()
        self.jobs = [{**j, 'state':j['status'], 'path':j.get('source') or ''} for j in data]
        completed = {j['job_id'] for j in data if j['status']=='COMPLETE'}
        newly_completed = completed - self._last_complete
        current_ids = [self.table.item(i, COL_ID).text() for i in range(self.table.rowCount())]
        target_ids = [j['job_id'] for j in data]
        structural_change = current_ids != target_ids
        # Job list intentionally omits private source paths.
        self.table.blockSignals(True)
        try:
            for i in range(len(current_ids) - 1, -1, -1):
                if current_ids[i] not in target_ids:
                    self.table.removeRow(i)
                    current_ids.pop(i)
            for i, job in enumerate(data):
                if i >= len(current_ids) or current_ids[i] != job['job_id']:
                    self.table.insertRow(i)
                    current_ids.insert(i, job['job_id'])
                values=(job['job_id'],job['origin'],job['collection'],job['filename'],
                        job['stage'],job['status'])
                for c,value in enumerate(values):
                    value = str(value or '')
                    item = self.table.item(i, c)
                    if item is None:
                        self.table.setItem(i,c,QTableWidgetItem(value))
                    elif item.text() != value:
                        item.setText(value)
                file_item = self.table.item(i,COL_FILE)
                tooltip = 'Job: '+job['job_id']
                if file_item.toolTip() != tooltip:
                    file_item.setToolTip(tooltip)
            while len(current_ids) > len(target_ids):
                self.table.removeRow(len(current_ids) - 1)
                current_ids.pop()
            if structural_change and selected_id:
                match=next((i for i,j in enumerate(data) if j['job_id']==selected_id),None)
                if match is not None: self.table.selectRow(match)
            if structural_change:
                self.table.verticalScrollBar().setValue(scrollbar)
        finally:
            self.table.blockSignals(False)
        self._last_complete = completed
        self._job_view = view
        self.update_controls()
        if newly_completed and self.main_tabs.currentWidget() is self.contents_tab:
            if any(j['job_id'] in newly_completed and j['collection'] == self.current_collection() for j in data):
                self.refresh_contents(background=True)


    def set_ui_state(self, state: UiState) -> None:
        """Set the global UI operation state and re-apply the control policy."""
        if self.ui_state == state:
            return
        self.ui_state = state
        self.update_controls()
        if state == UiState.READY and hasattr(self, 'change_debounce'):
            QTimer.singleShot(0, self._check_revisions)

    @staticmethod
    def _set_enabled(widget, enabled: bool) -> None:
        if widget.isEnabled() != enabled:
            widget.setEnabled(enabled)

    def update_controls(self) -> None:
        state=self.ui_state
        row=self.selected_row()
        job=self.jobs[row] if row is not None and row < len(self.jobs) else None
        status=job['status'] if job else ''
        collection_ready=state==UiState.READY
        ingestion_ready=state in {UiState.READY, UiState.RETRIEVAL_RUNNING}
        self._set_enabled(self.collection, collection_ready)
        self._set_enabled(self.create_collection_btn, collection_ready)
        self._set_enabled(self.manage_collections_btn, collection_ready)
        for action in self.document_actions.values():
            self._set_enabled(action, ingestion_ready and bool(self.current_collection()))
        retrieval_ready=state==UiState.READY
        selected_retrievers=sum(cb.isChecked() for cb in
                (self.retrieval_semantic,self.retrieval_bm25))
        for ctrl in (self.query,self.retrieval_top_k,self.retrieval_semantic,
                     self.retrieval_bm25):
            self._set_enabled(ctrl, retrieval_ready)
        self._set_enabled(self.retrieval_rank_fusion, retrieval_ready and selected_retrievers>=2)
        self._set_enabled(self.search_btn, retrieval_ready and selected_retrievers>=1 and bool(self.current_collection()))
        flags = {key: False for key in self.buttons}
        if ingestion_ready:
            flags['run_all'] = any(j['status']=='QUEUED' for j in self.jobs)
            if job:
                flags.update({
                    'run':status=='QUEUED',
                    'pause':status in {'QUEUED','DOWNLOADING','RUNNING','RESUMING'},
                    'resume':status=='PAUSED',
                    'cancel':status in {'QUEUED','DOWNLOADING','RUNNING','RESUMING','PAUSING','PAUSED'},
                    'retry':status in {'FAILED','CANCELLED'},
                    'restart':status in {'FAILED','CANCELLED','PAUSED','COMPLETE'},
                    'delete':status in {'QUEUED','PAUSED','FAILED','CANCELLED','COMPLETE'},
                    'open_source':True,
                    'open_artifacts':bool(job.get('source_sha256')),
                })
        for key, allow in flags.items():
            self._set_enabled(self.buttons[key], allow)

    def add_path(self, path: str) -> None:
        p = Path(path).expanduser().resolve()
        if not p.is_file() or p.suffix.lower() not in SUPPORTED_EXTENSIONS:
            return
        collection = self.current_collection()
        if not collection:
            QMessageBox.warning(self, 'Create collection', 'Create and select a Qdrant collection before ingestion.')
            return
        # No automatic write retry, including on a network timeout.
        key = str(p)
        if key in self._job_write_pending:
            return
        self._job_write_pending.add(key)

        def finished(_result, error):
            self._job_write_pending.discard(key)
            if error:
                self.errors.appendPlainText(f'Queue result unknown/failed for {p.name}: {error}')
                self.statusBar().showMessage(error)
            self.load_jobs()

        self.backend.command('jobs.submit', {'collection': collection, 'path': key},
                             finished, read_only=False)

    def add_files(self) -> None:
        pattern=' '.join(f'*{x}' for x in sorted(SUPPORTED_EXTENSIONS))
        paths,_=QFileDialog.getOpenFileNames(self,'Add RAG3 documents','',
                                            f'Supported documents ({pattern});;All files (*)')
        for path in paths:self.add_path(path)

    def add_folder(self) -> None:
        folder=QFileDialog.getExistingDirectory(self,'Add documents from folder')
        if folder:
            for path in sorted(Path(folder).iterdir()):
                if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                    self.add_path(str(path))

    def selection_changed(self) -> None:
        row=self.selected_row()
        if row is not None and row<len(self.jobs):
            self.inspector.setPlainText(json.dumps(self.jobs[row],ensure_ascii=False,indent=2))
        self.update_controls()

    def run_selected(self) -> None:
        row=self.selected_row()
        if row is not None:self.start_row(row,'run')

    def run_all(self) -> None:
        for job in list(self.jobs):
            if job['status']=='QUEUED':
                self._manager_action('run',job['job_id'])

    def start_row(self, row: int, mode: str) -> None:
        job=self.jobs[row]
        op={'run':'run','resume':'resume','restart':'restart'}.get(mode,mode)
        self._manager_action(op,job['job_id'])

    def pause_current(self) -> None:
        row=self.selected_row()
        if row is not None:self._manager_action('pause',self.jobs[row]['job_id'])

    def resume_selected(self) -> None:
        row=self.selected_row()
        if row is not None:self._manager_action('resume',self.jobs[row]['job_id'])

    def retry_selected(self) -> None:
        row=self.selected_row()
        if row is not None:self._manager_action('retry',self.jobs[row]['job_id'])

    def restart_selected(self) -> None:
        row=self.selected_row()
        if row is not None:self._manager_action('restart',self.jobs[row]['job_id'])

    def cancel_current(self) -> None:
        row=self.selected_row()
        if row is not None:self._manager_action('cancel',self.jobs[row]['job_id'])

    def remove_selected(self) -> None:
        row=self.selected_row()
        if row is not None:self._manager_action('hide',self.jobs[row]['job_id'])

    def open_source(self) -> None:
        row = self.selected_row()
        if row is None:
            return
        job_id = self.jobs[row]['job_id']

        def finished(result, error):
            if error:
                self.statusBar().showMessage(f'Open Source unavailable: {error}')
                return
            path = (result or {}).get('source') or ''
            if path:
                QDesktopServices.openUrl(QUrl.fromLocalFile(path))

        self.backend.command('jobs.local_source', {'job_id': job_id}, finished)

    def open_artifacts(self) -> None:
        row=self.selected_row()
        if row is not None and self.jobs[row].get('source_sha256'):
            folder=DATA_ROOT/'sources'/self.jobs[row]['source_sha256']
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    def periodic_collections(self) -> None:
        self._check_revisions()

    def periodic_contents(self) -> None:
        self._check_revisions()

    def tab_changed(self, _index: int) -> None:
        if self.main_tabs.currentWidget() is self.contents_tab and self.current_collection():
            QTimer.singleShot(0, self._check_revisions)

    def _start_collection_worker(
        self,
        action: str,
        collection: str | None = None,
        collections: list[str] | None = None,
        bm25_settings: tuple[float, float, float] | None = None,
        background: bool = False,
    ) -> bool:
        if self._collection_pending:
            if not background:
                self.statusBar().showMessage('A collection operation is already running.')
            return False
        messages = {'list': 'Loading collections…', 'create': 'Creating collection…',
                    'delete_many': 'Deleting collections…', 'contents': 'Loading contents…'}
        self.collection_action = action
        self._collection_background = background
        if not background:
            self.set_ui_state(UiState.COLLECTION_BUSY)
            self.statusBar().showMessage(messages.get(action, 'Working with collections…'))
        params: dict = {}
        if collection:
            params['collection'] = collection
        if collections:
            params['collections'] = list(collections)
            params['confirm_collections'] = list(collections)
        if bm25_settings is not None:
            params.update(zip(('bm25_k', 'bm25_b', 'bm25_avg_len'), bm25_settings))
        # Stamp response versions and mark in-flight *before* dispatch: the
        # client can fail synchronously if the token/backend is unavailable.
        if action == 'list':
            self._collections_loading_version = self._latest_versions.get('collections', 0)
        elif action == 'contents' and collection:
            self._contents_loading_version = self._latest_versions.get(content_scope(collection), 0)
        self._collection_pending = True
        return self.backend.command('collections.' + action, params,
                                    lambda result, error: self.collection_finished(result, error),
                                    read_only=action in {'list', 'contents'})

    def refresh_collections(self, *, background: bool = False) -> bool:
        return self._start_collection_worker('list', background=background)

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
        self._contents_view = None
        self._displayed_contents_collection = None
        self.contents_status.setText(f"Collection selected: {self.current_collection()}")
        self.update_controls()
        if self.main_tabs.currentWidget() is self.contents_tab and self.current_collection():
            QTimer.singleShot(0, self._check_revisions)

    def refresh_contents(self, *, background: bool = False) -> bool:
        """Request a fresh document-level view from the collection worker."""
        if self.ui_state != UiState.READY:
            return False
        collection = self.current_collection()
        if not collection:
            return False
        return self._start_collection_worker("contents", collection, background=background)

    def _render_collection_contents(self, payload: dict) -> None:
        """Apply only changed cells from the document-level worker output."""
        documents = payload.get("documents") or []
        rows = []
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
            rows.append(tuple(str(value) for value in values))
        status = (
            f"{payload.get('document_count', len(documents))} document(s), "
            f"{payload.get('chunk_count', 0)} chunk(s) | "
            f"collection={payload.get('collection') or self.current_collection()}"
        )
        view = (tuple(rows), status)
        if view == self._contents_view:
            return
        if self.contents_table.rowCount() != len(rows):
            self.contents_table.setRowCount(len(rows))
        for row, values in enumerate(rows):
            for column, value in enumerate(values):
                item = self.contents_table.item(row, column)
                if item is None:
                    self.contents_table.setItem(row, column, QTableWidgetItem(value))
                elif item.text() != value:
                    item.setText(value)
        if self.contents_status.text() != status:
            self.contents_status.setText(status)
        self._contents_view = view

    def _apply_collection_names(self, names: list[str], *, background: bool) -> bool:
        current_names = [self.collection.itemText(i) for i in range(self.collection.count())]
        if names == current_names or (background and self.collection.hasFocus()):
            return False
        current = self.current_collection()
        self.collection.blockSignals(True)
        try:
            for i in range(self.collection.count() - 1, -1, -1):
                if self.collection.itemText(i) not in names:
                    self.collection.removeItem(i)
            for i, name in enumerate(names):
                if i < self.collection.count() and self.collection.itemText(i) == name:
                    continue
                old = self.collection.findText(name)
                if old >= 0:
                    self.collection.removeItem(old)
                self.collection.insertItem(i, name)
            chosen = current if current in names else (names[0] if names else "")
            if self.current_collection() != chosen:
                self.collection.setText(chosen)
        finally:
            self.collection.blockSignals(False)
        if self.current_collection() != current:
            self.collection_changed(self.current_collection())
        return True

    def collection_finished(self, payload: dict | None, error: str | None) -> None:
        action = self.collection_action
        background = self._collection_background
        self._collection_pending = False
        self.collection_action = None
        self._collection_background = False
        if error:
            message = f'Collection {action} failed or unconfirmed: {error}'
            if not background:
                self.set_ui_state(UiState.READY)
            self.statusBar().showMessage(message)
            if action == 'delete_many':
                QMessageBox.critical(self, 'Delete Collections', message)
            elif action == 'contents':
                self.contents_status.setText(message)
            # For timed-out writes, re-query state rather than repeating mutation.
            QTimer.singleShot(0, self._check_revisions)
            return
        payload = payload or {}

        if action == "list":
            names = payload.get("collections") or []
            if background and self._collection_is_editing():
                self._pending_collection_names = names
                changed = False
            else:
                changed = self._apply_collection_names(names, background=background)
            self._collections_seen_version = self._collections_loading_version
            if not background:
                self.set_ui_state(UiState.READY)
                if changed:
                    self.statusBar().showMessage(f"Loaded {len(names)} Qdrant collections.")

        elif action == "create":
            self.set_ui_state(UiState.READY)
            name = str(payload.get("collection") or self.current_collection())
            self.collection.setText(name)
            self.statusBar().showMessage(f"Collection ready: {name}")
            self.refresh_collections()

        elif action == "delete_many":
            self.set_ui_state(UiState.READY)
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
            self.contents_status.setText(f"Deleted {count} collection(s); collection list updating…")
            self.statusBar().showMessage(f"Deleted {count} collection(s); collection list updating…")
            self.refresh_collections()

        elif action == "contents":
            if payload.get('collection') == self.current_collection():
                self._render_collection_contents(payload)
                self._displayed_contents_collection = self.current_collection()
                self._contents_seen[self.current_collection()] = self._contents_loading_version
            if not background:
                self.set_ui_state(UiState.READY)
                self.statusBar().showMessage(
                    f"Loaded contents for {payload.get('collection') or self.current_collection()}."
                )
        else:
            if not background:
                self.set_ui_state(UiState.READY)
        if action in ('list', 'contents'):
            QTimer.singleShot(0, self._check_revisions)

    def run_retrieval(self) -> None:
        if self.ui_state != UiState.READY or self._retrieval_pending:
            return
        query = self.query.text().strip()
        if not query:
            self.retrieval_status.setText('Enter a query.')
            return
        semantic = self.retrieval_semantic.isChecked()
        bm25 = self.retrieval_bm25.isChecked()
        if not (semantic or bm25):
            self.retrieval_status.setText('Select at least one retrieval method.')
            return
        self.retrieval_results.clear()
        self.retrieval_status.setText('Searching…')
        self._retrieval_pending = True
        self.set_ui_state(UiState.RETRIEVAL_RUNNING)
        options = {'semantic': semantic, 'bm25': bm25,
                   'rank_fusion': bool(semantic and bm25 and self.retrieval_rank_fusion.isChecked()),
                   'top_k': self.retrieval_top_k.value()}
        collection = self.current_collection()

        def finished(result, error):
            self._retrieval_pending = False
            self.set_ui_state(UiState.READY)
            if error:
                self.retrieval_status.setText(f'Search failed: {error}')
                return
            self.retrieval_finished(result or {})

        self.backend.command('search', {'query': query, 'collection': collection,
                                        'options': options}, finished)

    def retrieval_finished(self, payload: dict) -> None:
        result = payload.get('ui') or {}
        hits = result.get('hits') or []

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
                    f"sections: {', '.join(hit.get('section_paths') or [])}",
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

        self.collection.setText(str(state.get("collection") or ""))
        self.query.setText(str(state.get("retrieval_query") or ""))

        self.retrieval_semantic.setChecked(
            bool(state.get("retrieval_semantic", True))
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
        self.change_debounce.stop()
        self.fallback_timer.stop()
        # Closing the UI does not stop or restart the shared backend.
        self.save_ui_state()
        super().closeEvent(event)


def main() -> None:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument('--smoke-test', action='store_true')
    args, _ = ap.parse_known_args()
    if args.smoke_test:
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    app = QApplication(sys.argv[:1])
    win = IngestionWindow()
    if args.smoke_test:
        print('PYSIDE6_UI_SMOKE_OK')
        return
    win.show()
    raise SystemExit(app.exec())


if __name__ == '__main__':
    main()
