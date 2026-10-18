from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("rag3_ui_state_test", ROOT / "scripts" / "ingest-ui.py")
if spec is None or spec.loader is None:
    raise SystemExit("Could not load ingest-ui.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

from PySide6.QtWidgets import QApplication

app = QApplication(sys.argv[:1])

def drain_collection_worker(win, timeout_sec=10):
    deadline = time.time() + timeout_sec
    while getattr(win, "collection_process", None) is not None and time.time() < deadline:
        app.processEvents()
        time.sleep(0.05)

w = mod.IngestionWindow()
drain_collection_worker(w)
w.collection.setText("rag3_haystack")
w.query.setText("state persistence probe")
w.retrieval_mode.setCurrentText("sparse")
w.retrieval_top_k.setValue(7)
w.main_tabs.setCurrentIndex(1)
w.save_ui_state()
w.close()
app.processEvents()

w2 = mod.IngestionWindow()
drain_collection_worker(w2)
state = {
    "collection": w2.current_collection(),
    "query": w2.query.text(),
    "mode": w2.retrieval_mode.currentText(),
    "top_k": w2.retrieval_top_k.value(),
    "tab": w2.main_tabs.currentIndex(),
}
print(json.dumps(state, ensure_ascii=False))
assert state["collection"] == "rag3_haystack"
assert state["query"] == "state persistence probe"
assert state["mode"] == "sparse"
assert state["top_k"] == 7
assert state["tab"] == 1
w2.close()
app.processEvents()
print("UI_STATE_PERSISTENCE_OK")
