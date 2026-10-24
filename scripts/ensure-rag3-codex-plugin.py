"""Manage the RAG3 local Codex plugin without silently replacing installed transport.

A backend readiness probe and an explicit --activate-http are required before
replacing an installed stdio plugin with the shared-backend HTTP plugin.
An administrator/user-disabled plugin is never enabled automatically.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
MARKETPLACE_NAME = "rag3-local"
PLUGIN_ID = "rag3@rag3-local"


def same_path(left: str | Path, right: str | Path) -> bool:
    def normalized(value: str | Path) -> str:
        raw = str(value)
        if raw.startswith("\\?\\"):
            raw = raw[4:]
        return os.path.normcase(os.path.abspath(raw))
    return normalized(left) == normalized(right)


def find_codex() -> Path | None:
    from_path = shutil.which('codex')
    if from_path:
        return Path(from_path)
    local_app_data = os.environ.get('LOCALAPPDATA')
    if not local_app_data:
        return None
    base = Path(local_app_data) / 'OpenAI' / 'Codex' / 'bin'
    candidates = list(base.glob('*/codex.exe')) if base.is_dir() else []
    return max(candidates, key=lambda item: item.stat().st_mtime) if candidates else None


def run_codex(codex: Path, *args: str, json_output: bool = False) -> dict | None:
    command = [str(codex), *args]
    completed = subprocess.run(
        command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace',
        timeout=75, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
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
    markets = run_codex(codex, 'plugin', 'marketplace', 'list', '--json', json_output=True)
    plugins = run_codex(codex, 'plugin', 'list', '--json', json_output=True)
    market = next((item for item in markets.get('marketplaces', [])
                   if item.get('name') == MARKETPLACE_NAME), None)
    plugin = next((item for item in plugins.get('installed', [])
                   if item.get('pluginId') == PLUGIN_ID), None)
    return market, plugin


def desired_http_manifest() -> tuple[str, str]:
    mcp_path = ROOT / 'plugins' / 'rag3' / 'mcp.json'
    plugin_path = ROOT / 'plugins' / 'rag3' / 'plugin.json'
    data = json.loads(mcp_path.read_text('utf-8-sig'))
    if set(data) != {'$schema', 'mcpServers'} or set(data['mcpServers']) != {'rag3'}:
        raise ValueError('Unexpected MCP manifest structure')
    server = data['mcpServers']['rag3']
    if set(server) != {'type','url'} or server['type'] != 'streamable-http':
        raise ValueError('RAG3 plugin must use only type=streamable-http and url')
    url = server['url']
    parsed = urlsplit(url)
    if (parsed.scheme != 'http' or parsed.hostname != '127.0.0.1' or
            parsed.port is None or not 1 <= parsed.port <= 65535 or
            parsed.path != '/mcp' or parsed.username or parsed.password or
            parsed.fragment or parsed.query):
        raise ValueError('Plugin requires a secret-free http://127.0.0.1:<port>/mcp endpoint')
    return url, str(json.loads(plugin_path.read_text('utf-8-sig'))['version'])


def installed_transport(version: str | None) -> tuple[str | None, str | None]:
    """Inspect the local Codex plugin cache, never edit it here."""
    if not version:
        return None, None
    cache = Path.home() / '.codex' / 'plugins' / 'cache' / MARKETPLACE_NAME / 'rag3' / version / 'mcp.json'
    if not cache.is_file():
        return None, None
    try:
        entry = json.loads(cache.read_text('utf-8-sig'))['mcpServers']['rag3']
        return entry.get('type'), entry.get('url')
    except (OSError, ValueError, KeyError, TypeError):
        return None, None


async def _check_live_mcp(url: str) -> None:
    """MCP initialize + tool listing, not a potentially misleading /health probe."""
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client
    async with asyncio.timeout(20):
        async with streamable_http_client(url) as streams:
            async with ClientSession(streams[0], streams[1]) as session:
                await session.initialize()
                listing = await session.list_tools()
                names = {tool.name for tool in listing.tools}
                required = {'rag3_list_collections', 'rag3_search', 'rag3_ingestion_status'}
                if not required <= names:
                    raise RuntimeError('MCP endpoint did not advertise required RAG3 tools')


def ensure_registration(check_only: bool, activate_http: bool = False) -> dict:
    files = (ROOT/'.agents/plugins/marketplace.json', ROOT/'plugins/rag3/plugin.json',
             ROOT/'plugins/rag3/mcp.json')
    if not all(path.is_file() for path in files):
        return {'status': 'invalid_package', 'detail': 'RAG3 plugin files are missing'}
    try:
        url, desired_version = desired_http_manifest()
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {'status': 'invalid_package', 'detail': str(exc)}

    codex = find_codex()
    if codex is None:
        return {'status': 'codex_missing', 'detail': 'Codex CLI was not found'}
    market, plugin = registration_status(codex)
    if market and (not market.get('root') or not same_path(market['root'], ROOT)):
        return {'status': 'marketplace_conflict', 'existing_root': market.get('root'),
                'expected_root': str(ROOT)}
    if plugin and plugin.get('installed') and not plugin.get('enabled', False):
        return {'status': 'disabled', 'version': plugin.get('version'),
                'detail': 'The user disabled RAG3; no installation state was changed'}

    current_version = str(plugin.get('version', '')) if plugin and plugin.get('installed') else None
    current_type, current_url = installed_transport(current_version)
    # On an uninspectable cache, rely on CLI version but mark the verification incomplete.
    is_current = (current_version == desired_version and current_type == 'streamable-http'
                  and current_url == url)
    if check_only or not activate_http:
        if is_current:
            return {'status': 'already_installed', 'version': desired_version, 'transport': 'streamable-http'}
        return {'status': 'update_pending' if current_version else 'missing',
                'installed_version': current_version, 'desired_version': desired_version,
                'transport': current_type or 'unverified', 'activate_http_required': True}

    # Explicit cutover, only after the HTTP MCP endpoint actually answers.
    try:
        asyncio.run(_check_live_mcp(url))
    except Exception as exc:
        return {'status': 'backend_not_ready', 'detail': type(exc).__name__,
                'installed_version': current_version}
    if is_current:
        return {'status': 'already_installed', 'version': current_version, 'transport': 'streamable-http'}
    if market is None:
        run_codex(codex, 'plugin', 'marketplace', 'add', str(ROOT))
        market, _ = registration_status(codex)
        if not market or not same_path(market.get('root', ''), ROOT):
            raise RuntimeError('Marketplace registration verification failed')
    old_version = current_version
    if current_version:
        run_codex(codex, 'plugin', 'remove', PLUGIN_ID)
    try:
        run_codex(codex, 'plugin', 'add', PLUGIN_ID)
        new_market, new_plugin = registration_status(codex)
        if not new_market or not new_plugin or not new_plugin.get('installed') or not new_plugin.get('enabled'):
            raise RuntimeError('Post-install plugin identity and enabled state not confirmed')
        if str(new_plugin.get('version')) != desired_version:
            raise RuntimeError('Post-install version differs from source manifest')
        installed_type, installed_url = installed_transport(desired_version)
        if installed_type != 'streamable-http' or installed_url != url:
            raise RuntimeError('Installed plugin cache does not match HTTP manifest')
    except Exception as exc:
        # No speculative reinstall of the old stdio server. Surface uncertain state.
        return {'status': 'refresh_failed', 'previous_version': old_version,
                'detail': type(exc).__name__ + ': ' + str(exc)[:300],
                'manual_recovery_required': True}
    return {'status': 'refreshed' if old_version else 'installed',
            'version': desired_version, 'transport': 'streamable-http'}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--activate-http', action='store_true',
                        help='Explicitly replace installed stdio plugin; requires live HTTP MCP probe')
    parser.add_argument('--status-file', type=Path)
    args = parser.parse_args()
    try:
        result = ensure_registration(args.check_only, args.activate_http)
    except Exception as exc:
        result = {'status': 'error', 'detail': str(exc)}
    result['checked_at_utc'] = datetime.now(timezone.utc).isoformat()
    output = json.dumps(result, ensure_ascii=True)
    print(output, flush=True)
    if args.status_file:
        status_file = args.status_file.resolve()
        if ROOT not in status_file.parents:
            raise SystemExit('Status file must be inside the RAG3 project')
        status_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = status_file.with_suffix(status_file.suffix + '.tmp')
        temporary.write_text(output+'\n', encoding='utf-8')
        temporary.replace(status_file)
    return 1 if result['status'] in {'error','invalid_package','marketplace_conflict','refresh_failed'} else 0


if __name__ == '__main__':
    sys.exit(main())
