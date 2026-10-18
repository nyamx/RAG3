"""Register the local RAG3 plugin with Codex when it is missing.

This helper is launched independently of the PySide UI so registration cannot
delay the window or be interrupted when the window closes.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MARKETPLACE_NAME = "rag3-local"
PLUGIN_ID = "rag3@rag3-local"


def same_path(left: str | Path, right: str | Path) -> bool:
    """Compare Windows paths, including Codex's extended-length path form."""
    def normalized(value: str | Path) -> str:
        raw = str(value)
        if raw.startswith("\\\\?\\"):
            raw = raw[4:]
        return os.path.normcase(os.path.abspath(raw))

    return normalized(left) == normalized(right)


def find_codex() -> Path | None:
    from_path = shutil.which("codex")
    if from_path:
        return Path(from_path)
    local_app_data = os.environ.get("LOCALAPPDATA")
    if not local_app_data:
        return None
    candidates = list((Path(local_app_data) / "OpenAI" / "Codex" / "bin").glob("*/codex.exe"))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item.stat().st_mtime)


def run_codex(codex: Path, *args: str, json_output: bool = False) -> dict | None:
    command = [str(codex), *args]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=75,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip()[-600:] or completed.stdout.strip()[-600:]
        raise RuntimeError(f"codex {' '.join(args)} exited {completed.returncode}: {detail}")
    if not json_output:
        return None
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"codex {' '.join(args)} did not return valid JSON") from exc


def registration_status(codex: Path) -> tuple[dict | None, dict | None]:
    markets = run_codex(codex, "plugin", "marketplace", "list", "--json", json_output=True)
    plugins = run_codex(codex, "plugin", "list", "--json", json_output=True)
    market = next((item for item in markets.get("marketplaces", [])
                   if item.get("name") == MARKETPLACE_NAME), None)
    plugin = next((item for item in plugins.get("installed", [])
                   if item.get("pluginId") == PLUGIN_ID), None)
    return market, plugin


def ensure_registration(check_only: bool) -> dict:
    manifest = ROOT / ".agents" / "plugins" / "marketplace.json"
    plugin_manifest = ROOT / "plugins" / "rag3" / "plugin.json"
    mcp_manifest = ROOT / "plugins" / "rag3" / "mcp.json"
    if not all(path.is_file() for path in (manifest, plugin_manifest, mcp_manifest)):
        return {"status": "invalid_package", "detail": "RAG3 plugin files are missing"}

    configured_root = (json.loads(mcp_manifest.read_text(encoding="utf-8"))
                       ["mcpServers"]["rag3"]["env"]["RAG3_PROJECT_ROOT"])
    if not same_path(configured_root, ROOT):
        return {"status": "project_path_mismatch", "configured_root": configured_root,
                "actual_root": str(ROOT)}

    codex = find_codex()
    if codex is None:
        return {"status": "codex_missing", "detail": "Codex CLI was not found on this PC"}

    market, plugin = registration_status(codex)
    if market and (not market.get("root") or not same_path(market["root"], ROOT)):
        return {"status": "marketplace_conflict", "existing_root": market.get("root"),
                "expected_root": str(ROOT)}
    if plugin and plugin.get("installed") and not plugin.get("enabled", False):
        return {"status": "disabled", "version": plugin.get("version"),
                "detail": "The user disabled this plugin; its setting was preserved"}
    if check_only:
        if plugin and plugin.get("installed") and market:
            return {"status": "already_installed", "version": plugin.get("version")}
        return {"status": "missing", "marketplace_registered": bool(market),
                "plugin_installed": bool(plugin and plugin.get("installed"))}

    market_was_missing = market is None
    if market_was_missing:
        run_codex(codex, "plugin", "marketplace", "add", str(ROOT))
    if plugin and plugin.get("installed"):
        if market_was_missing:
            market, _ = registration_status(codex)
            if not market or not market.get("root") or not same_path(market["root"], ROOT):
                raise RuntimeError("Codex did not register the RAG3 marketplace")
        return {"status": "marketplace_registered" if market_was_missing else "already_installed",
                "version": plugin.get("version")}
    run_codex(codex, "plugin", "add", PLUGIN_ID)
    market, plugin = registration_status(codex)
    if not market or not market.get("root") or not same_path(market["root"], ROOT):
        raise RuntimeError("Codex did not register the RAG3 marketplace")
    if not plugin or not plugin.get("installed") or not plugin.get("enabled"):
        raise RuntimeError("Codex did not enable the RAG3 plugin")
    return {"status": "installed", "version": plugin.get("version")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--status-file", type=Path)
    args = parser.parse_args()
    try:
        result = ensure_registration(args.check_only)
    except Exception as exc:
        result = {"status": "error", "detail": str(exc)}
    result["checked_at_utc"] = datetime.now(timezone.utc).isoformat()
    output = json.dumps(result, ensure_ascii=True)
    print(output, flush=True)
    if args.status_file:
        status_file = args.status_file.resolve()
        if ROOT not in status_file.parents:
            raise SystemExit("Status file must be inside the RAG3 project")
        status_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = status_file.with_suffix(status_file.suffix + ".tmp")
        temporary.write_text(output + "\n", encoding="utf-8")
        temporary.replace(status_file)
    return 1 if result["status"] in {"error", "invalid_package", "project_path_mismatch",
                                      "marketplace_conflict"} else 0


if __name__ == "__main__":
    sys.exit(main())
