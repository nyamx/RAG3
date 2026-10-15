from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag3.ui_state import JobState, allowed_actions

required = {
    JobState.RUNNING: {"pause", "cancel"},
    JobState.PAUSED: {"resume", "restart"},
    JobState.FAILED: {"retry", "restart"},
    JobState.CANCELLED: {"resume", "restart"},
    JobState.COMPLETE: {"restart"},
}
for state, actions in required.items():
    missing = actions - set(allowed_actions(state))
    if missing:
        raise SystemExit(f"{state}: missing {sorted(missing)}")

env = os.environ.copy()
env["QT_QPA_PLATFORM"] = "offscreen"
cp = subprocess.run(
    [str(ROOT / ".venv" / "Scripts" / "python.exe"), str(ROOT / "scripts" / "ingest-ui.py"), "--smoke-test"],
    cwd=ROOT,
    env=env,
    capture_output=True,
    text=True,
    timeout=30,
)
print(cp.stdout.strip())
if cp.returncode != 0:
    print(cp.stderr)
    raise SystemExit(cp.returncode)
if "PYSIDE6_UI_SMOKE_OK" not in cp.stdout:
    raise SystemExit("UI smoke marker missing")
print("UI_STATE_POLICY_OK")
