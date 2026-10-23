"""Offscreen regression check: unchanged data does not mutate widgets or controls."""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PySide6.QtWidgets import QApplication
from rag3 import ingestion_jobs as db

spec = importlib.util.spec_from_file_location('ingest_ui_poll', ROOT/'scripts'/'ingest-ui.py')
ui = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ui)
app = QApplication(sys.argv[:1])
ui.IngestionWindow.refresh_collections = lambda self, **_kwargs: None

with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as temp:
    folder = Path(temp)
    db.DB_PATH = folder/'jobs.sqlite'
    ui.UI_STATE_PATH = folder/'state.json'
    with db.connect() as conn:
        conn.execute(
            """INSERT INTO jobs(job_id,origin,collection,filename,source,status,stage,created_at,updated_at)
               VALUES(?,?,?,?,?,?,?,?,?)""",
            ('1'*32, 'MCP', 'example', 'first.md', str(folder/'first.md'),
             'QUEUED', 'Ready', db.now(), db.now()),
        )
        conn.commit()

    win = ui.IngestionWindow()
    win.collection.addItem('example')
    win.collection.setText('example')
    win.table.selectRow(0)
    item = win.table.item(0, ui.COL_STATUS)
    original_items = [win.table.item(0, col) for col in range(6)]
    controls = []
    original_update = win.update_controls
    win.update_controls = lambda: (controls.append(True), original_update())

    win.load_jobs()
    assert not controls, 'Unchanged poll touched the controls'
    assert win.table.item(0, ui.COL_STATUS) is item

    with db.connect() as conn:
        conn.execute("UPDATE jobs SET status='RUNNING', stage='Embedding' WHERE job_id=?", ('1'*32,))
        conn.commit()
    win.load_jobs()
    assert [win.table.item(0, col) for col in range(6)] == original_items
    assert item.text() == 'RUNNING'
    assert win.table.item(0, ui.COL_STAGE).text() == 'Embedding'
    assert len(controls) == 1

    win.load_jobs()
    assert len(controls) == 1, 'Next unchanged poll mutated controls'

    with db.connect() as conn:
        conn.execute(
            """INSERT INTO jobs(job_id,origin,collection,filename,source,status,stage,created_at,updated_at)
               VALUES(?,?,?,?,?,?,?,?,?)""",
            ('2'*32, 'MCP', 'example', 'second.md', str(folder/'second.md'),
             'QUEUED', 'Ready', '9999-01-01T00:00:00+00:00', db.now()),
        )
        conn.commit()
    win.load_jobs()
    assert win.table.rowCount() == 2
    assert win.table.item(1, ui.COL_STATUS) is item
    assert win.jobs[win.selected_row()]['job_id'] == '1'*32

    payload = {'collection':'example', 'document_count':1, 'chunk_count':1,
               'documents':[{'document_id':'doc-1', 'title':'One', 'chunk_count':1}]}
    win._render_collection_contents(payload)
    content_item = win.contents_table.item(0, 0)
    win._render_collection_contents(payload)
    assert win.contents_table.item(0, 0) is content_item

    win.main_tabs.setCurrentWidget(win.contents_tab)
    requested = []
    win.refresh_contents = lambda **kwargs: requested.append(kwargs)
    win.periodic_contents()
    assert requested[-1] == {'background': True}
    assert win.ui_state == ui.UiState.READY
    assert win.collection.isEnabled() and win.create_collection_btn.isEnabled()

    win.collection_action = 'contents'
    win._collection_background = True
    win.collection_stdout_buffer = json.dumps(payload)
    win.collection_finished(0, None)
    assert win.contents_table.item(0, 0) is content_item
    assert win.ui_state == ui.UiState.READY
    assert win.collection.isEnabled()

    win.collection_action = 'list'
    win._collection_background = True
    win.collection_stdout_buffer = json.dumps({'collections':['example']})
    win.collection_finished(0, None)
    combo_item = win.collection.model().index(0, 0)
    win.collection_action = 'list'
    win._collection_background = True
    win.collection_stdout_buffer = json.dumps({'collections':['example']})
    win.collection_finished(0, None)
    assert win.collection.model().index(0, 0) == combo_item
    assert win.collection.isEnabled()

    class Signal:
        def connect(self, _callback):
            pass

    class FakeProcess:
        readyReadStandardOutput = Signal()
        readyReadStandardError = Signal()
        finished = Signal()

        def __init__(self, _parent):
            self.started = False

        def setProgram(self, _program):
            pass

        def setArguments(self, _arguments):
            pass

        def setWorkingDirectory(self, _directory):
            pass

        def start(self):
            self.started = True

    ui.QProcess = FakeProcess
    assert win._start_collection_worker('list', background=True)
    assert win.collection_process.started
    assert win.ui_state == ui.UiState.READY
    assert win.collection.isEnabled() and win.create_collection_btn.isEnabled()
    win.collection_stdout_buffer = json.dumps({'collections':['example']})
    win.collection_finished(0, None)
    win.close()
    app.processEvents()

print('UI_POLL_DELTA_OK')
