from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag3.ui_state import JobState

ui_path = ROOT / "scripts" / "ingest-ui.py"
spec = importlib.util.spec_from_file_location("rag3_ingest_ui_e2e", ui_path)
if spec is None or spec.loader is None:
    raise SystemExit("Could not load ingest-ui.py")
ui = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ui)

fixture_dir = ROOT / "artifacts" / "ui-worker-e2e"
fixture_dir.mkdir(parents=True, exist_ok=True)
source = fixture_dir / "ui-worker-e2e-20261016-0151.txt"
source.write_text(
    "RAG3 UI worker evidence phrase: sapphire orchard omega.\n",
    encoding="utf-8",
)

app = QApplication(sys.argv[:1])
win = ui.IngestionWindow()
win.collection.setText("rag3_haystack_ui_worker_e2e")
win.add_path(str(source))

if len(win.jobs) != 1:
    raise SystemExit(f"Expected one UI job, got {len(win.jobs)}")

win.table.selectRow(0)
win.run_selected()

result = {"timed_out": False}

def watchdog():
    result["timed_out"] = True
    if win.process is not None:
        win.process.kill()
    app.quit()

timer = QTimer()
timer.setSingleShot(True)
timer.timeout.connect(watchdog)
timer.start(120000)

poll = QTimer()

def check_done():
    state = JobState(win.jobs[0]["state"])
    if state in {JobState.COMPLETE, JobState.FAILED, JobState.CANCELLED}:
        timer.stop()
        poll.stop()
        app.quit()

poll.timeout.connect(check_done)
poll.start(200)

app.exec()

state = JobState(win.jobs[0]["state"])
summary = {
    "state": state.value,
    "stage": win.jobs[0]["stage"],
    "timed_out": result["timed_out"],
    "result": win.jobs[0].get("result"),
    "log_tail": win.log.toPlainText().splitlines()[-12:],
    "error_tail": win.errors.toPlainText().splitlines()[-12:],
}
print(json.dumps(summary, ensure_ascii=False, indent=2))

if result["timed_out"]:
    raise SystemExit("UI QProcess worker E2E timed out")
if state != JobState.COMPLETE:
    raise SystemExit(f"UI QProcess worker did not complete: {state.value}")
if not isinstance(win.jobs[0].get("result"), dict):
    raise SystemExit("UI did not capture worker result")
