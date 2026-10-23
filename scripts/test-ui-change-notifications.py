"""Offscreen check of cross-process change delivery without idle polling."""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PySide6.QtWidgets import QApplication
from rag3 import ingestion_jobs as db

spec = importlib.util.spec_from_file_location('ingest_ui_events', ROOT/'scripts'/'ingest-ui.py')
ui = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ui)
app = QApplication(sys.argv[:1])
ui.IngestionWindow.refresh_collections = lambda self, **_kwargs: True


def wait_until(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    return False


with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as temp:
    folder = Path(temp)
    db.DB_PATH = folder/'jobs.sqlite'
    ui.UI_STATE_PATH = folder/'state.json'
    win = ui.IngestionWindow()
    assert not any(hasattr(win, name) for name in ('job_timer', 'collection_timer', 'contents_timer'))
    assert win.table.rowCount() == 0

    observed = []
    real_load_jobs = win.load_jobs
    def tracked_load_jobs():
        observed.append(True)
        return real_load_jobs()
    win.load_jobs = tracked_load_jobs

    writer = '''import sys
from pathlib import Path
from rag3 import ingestion_jobs as db
db.DB_PATH = Path(sys.argv[1])
with db.connect() as conn:
    stamp = db.now()
    conn.execute("""INSERT INTO jobs(job_id,origin,collection,filename,status,stage,created_at,updated_at)
                    VALUES(?,?,?,?,?,?,?,?)""",
                 ('1'*32, 'MCP', 'example', 'external.md', 'QUEUED', 'Ready', stamp, stamp))
'''
    subprocess.run([sys.executable, '-c', writer, str(db.DB_PATH)], cwd=ROOT, check=True)
    assert wait_until(lambda: win.table.rowCount() == 1), 'External job change was not delivered'
    assert observed
    observed.clear()
    deadline = time.monotonic() + 0.4
    while time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    assert not observed, 'Idle UI reloaded jobs'

    marker_before = db.change_signal_path().read_text(encoding='ascii')
    old_writer = '''import sqlite3, sys
conn = sqlite3.connect(sys.argv[1])
conn.execute("UPDATE jobs SET status='RUNNING', stage='External update' WHERE job_id=?", ('1'*32,))
conn.commit()
conn.close()
'''
    subprocess.run([sys.executable, '-c', old_writer, str(db.DB_PATH)], cwd=ROOT, check=True)
    assert db.change_signal_path().read_text(encoding='ascii') == marker_before
    assert wait_until(lambda: win.table.item(0, ui.COL_STATUS).text() == 'RUNNING'), (
        'A legacy writer changed SQLite without updating the UI'
    )
    version_reads = []
    original_versions = ui.change_versions
    def tracked_versions(*scopes):
        version_reads.append(scopes)
        return original_versions(*scopes)
    ui.change_versions = tracked_versions
    deadline = time.monotonic() + 0.5
    while time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    version_reads.clear()
    deadline = time.monotonic() + 0.5
    while time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    assert not version_reads, 'Idle filesystem watcher repeatedly queried revisions'
    ui.change_versions = original_versions

    content_requests = []
    def fake_refresh_contents(**_kwargs):
        content_requests.append(True)
        collection = win.current_collection()
        win._displayed_contents_collection = collection
        win._contents_seen[collection] = db.change_versions(db.content_scope(collection))[db.content_scope(collection)]
        return True
    win.refresh_contents = fake_refresh_contents
    win.collection.addItem('example')
    win.collection.setText('example')
    win.main_tabs.setCurrentWidget(win.contents_tab)
    assert wait_until(lambda: bool(content_requests)), 'Opening contents did not request an initial load'
    app.processEvents()
    content_requests.clear()

    db.mark_change(db.content_scope('example'))
    assert wait_until(lambda: bool(content_requests)), 'Collection mutation did not refresh the visible tab'
    content_requests.clear()
    deadline = time.monotonic() + 0.4
    while time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    assert not content_requests, 'Unchanged collection was fetched again'

    win.show()
    win.collection.setFocus()
    app.processEvents()
    assert win._collection_is_editing()
    win._pending_collection_names = ['example', 'new_collection']
    win.query.setFocus()
    assert wait_until(lambda: win.collection.findText('new_collection') >= 0), (
        'Deferred collection names were not applied after editing ended'
    )

    win.close()
    app.processEvents()

print('UI_CHANGE_NOTIFICATIONS_OK')
