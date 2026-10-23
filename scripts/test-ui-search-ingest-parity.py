"""Offscreen regression check for ingestion controls during UI search."""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PySide6.QtWidgets import QApplication

from rag3 import ingestion_jobs as db

spec = importlib.util.spec_from_file_location("ingest_ui_parity", ROOT / "scripts" / "ingest-ui.py")
assert spec and spec.loader
ui = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ui)
ui.IngestionWindow.refresh_collections = lambda self, **_kwargs: None
app = QApplication(sys.argv[:1])

with tempfile.TemporaryDirectory(dir=ROOT / "artifacts") as temp:
    folder = Path(temp)
    db.DB_PATH = folder / "jobs.sqlite"
    ui.UI_STATE_PATH = folder / "ui-state.json"
    with db.connect() as conn:
        for job_id, status in (("1" * 32, "QUEUED"), ("2" * 32, "RUNNING")):
            conn.execute(
                """INSERT INTO jobs(job_id,origin,collection,filename,source,status,stage,created_at,updated_at)
                   VALUES(?,?,?,?,?,?,?,?,?)""",
                (job_id, "UI", "test_collection", f"{status}.md", str(folder / f"{status}.md"),
                 status, status, db.now(), db.now()),
            )
        conn.commit()

    win = ui.IngestionWindow()
    try:
        win.collection.addItem("test_collection")
        win.collection.setText("test_collection")
        win.update_controls()
        assert win.search_btn.isEnabled()

        actions = []
        ui.manager_client = lambda action, **params: actions.append((action, params))
        win.set_ui_state(ui.UiState.RETRIEVAL_RUNNING)
        assert not win.search_btn.isEnabled()
        assert all(action.isEnabled() for action in win.document_actions.values())
        assert win.buttons["run_all"].isEnabled()

        queued_row = next(i for i, job in enumerate(win.jobs) if job["status"] == "QUEUED")
        win.table.selectRow(queued_row)
        assert win.buttons["run"].isEnabled()
        win.buttons["run"].click()
        assert actions[-1] == ("action", {"operation": "run", "job_id": "1" * 32})

        running_row = next(i for i, job in enumerate(win.jobs) if job["status"] == "RUNNING")
        win.table.selectRow(running_row)
        assert win.buttons["pause"].isEnabled() and win.buttons["cancel"].isEnabled()
        win.buttons["pause"].click()
        assert actions[-1] == ("action", {"operation": "pause", "job_id": "2" * 32})

        win.retrieval_stdout_buffer = json.dumps({"type": "error", "message": "sample backend failure"})
        win.retrieval_process = object()
        win.retrieval_finished(1, None)
        assert win.ui_state == ui.UiState.READY
        assert "sample backend failure" in win.retrieval_status.text()

        win.query.setText("test query")
        ui.ROOT = folder / "missing-program-folder"
        win.run_retrieval()
        deadline = time.monotonic() + 3
        while win.ui_state != ui.UiState.READY and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(.01)
        assert win.ui_state == ui.UiState.READY
        assert win.retrieval_process is None
        assert win.retrieval_status.text().startswith("Search could not start:")
    finally:
        ui.ROOT = ROOT
        win.close()
        app.processEvents()

print("UI_SEARCH_INGEST_PARITY_OK")
