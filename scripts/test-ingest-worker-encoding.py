"""Worker events must survive the Windows cp949 console encoding."""
from __future__ import annotations

import importlib.util
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("ingest_worker_encoding", ROOT / "scripts" / "ingest-worker.py")
assert spec and spec.loader
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)

output = io.BytesIO()
cp949_stdout = io.TextIOWrapper(output, encoding="cp949", errors="strict", write_through=True)
message = "한글 문서 \ufffd 출력"
with redirect_stdout(cp949_stdout):
    worker.emit("error", message=message)

wire = output.getvalue()
assert wire.isascii()
assert json.loads(wire.decode("utf-8")) == {"type": "error", "message": message}
print("INGEST_WORKER_ENCODING_OK")
