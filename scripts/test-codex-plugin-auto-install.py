"""Exercise missing, install, and repeat-launch paths in an isolated Codex home."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "scripts" / "ensure-rag3-codex-plugin.py"


def run_helper(env: dict[str, str], *args: str) -> dict:
    completed = subprocess.run(
        [sys.executable, str(HELPER), *args], cwd=ROOT, env=env,
        capture_output=True, text=True, encoding="utf-8", timeout=240,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if completed.returncode:
        raise AssertionError(f"helper exited {completed.returncode}: {completed.stdout} {completed.stderr}")
    return json.loads(completed.stdout)


def main() -> None:
    artifacts = ROOT / "artifacts"
    artifacts.mkdir(exist_ok=True)
    test_dir = Path(tempfile.mkdtemp(prefix="codex-plugin-install-", dir=artifacts)).resolve()
    try:
        env = os.environ.copy()
        codex_home = test_dir / "codex-home"
        codex_home.mkdir()
        env["CODEX_HOME"] = str(codex_home)
        status_file = test_dir / "codex-plugin-status.json"
        assert run_helper(env, "--check-only")["status"] == "missing"
        installed = run_helper(env, "--status-file", str(status_file))
        assert installed["status"] == "installed", installed
        assert installed["version"] == "0.1.1", installed
        assert json.loads(status_file.read_text(encoding="utf-8"))["status"] == "installed"
        repeated = run_helper(env)
        assert repeated["status"] == "already_installed", repeated
        config_file = codex_home / "config.toml"
        config = config_file.read_text(encoding="utf-8")
        assert 'enabled = true' in config
        config_file.write_text(config.replace('enabled = true', 'enabled = false'), encoding="utf-8")
        disabled = run_helper(env)
        assert disabled["status"] == "disabled", disabled
        assert 'enabled = false' in config_file.read_text(encoding="utf-8")
        print("RAG3_CODEX_PLUGIN_AUTO_INSTALL_OK")
    finally:
        if not test_dir.is_relative_to(ROOT.resolve()):
            raise RuntimeError(f"Refusing to remove an unexpected test path: {test_dir}")
        shutil.rmtree(test_dir)


if __name__ == "__main__":
    main()
