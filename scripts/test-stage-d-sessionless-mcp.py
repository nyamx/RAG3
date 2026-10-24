"""Sessionless tunnel calls use no shared mutable collection selection.

This test imports the MCP module with an isolated job DB and stubs Qdrant reads.
It exercises the tool handlers with a header-less Context shape.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def body(result):
    return result.structured_content or json.loads(result.content[0].text)


async def exercise(mod):
    from rag3.backend_boundary import SessionCollections

    mod._SESSION_COLLECTIONS = SessionCollections('alpha')
    missing = SimpleNamespace(session_id=None, headers={})
    client_a = SimpleNamespace(session_id='client-a', headers={'mcp-session-id': 'client-a'})
    client_b = SimpleNamespace(session_id='client-b', headers={'mcp-session-id': 'client-b'})
    assert mod._session_key(missing) == mod._SESSIONLESS_HTTP
    assert mod._session_key(client_a) == 'mcp:client-a'

    with patch.object(mod, 'list_collections', return_value=['alpha', 'beta']):
        first = await mod.rag3_list_collections(missing)
        assert not first.is_error
        assert body(first) == {'collections': ['alpha', 'beta'], 'selected': 'alpha',
                               'selection_persistent': False}
        refused = await mod.rag3_select_collection('beta', missing)
        assert refused.is_error
        assert body(refused)['error']['code'] == 'SESSION_REQUIRED'
        second = await mod.rag3_list_collections(missing)
        assert body(second)['selected'] == 'alpha'
        assert mod._SESSIONLESS_HTTP not in mod._SESSION_COLLECTIONS._selection

        selected = await mod.rag3_select_collection('beta', client_a)
        assert not selected.is_error and body(selected)['selected'] == 'beta'
        assert body(await mod.rag3_list_collections(client_a))['selected'] == 'beta'
        assert body(await mod.rag3_list_collections(client_b))['selected'] == 'alpha'
        assert body(await mod.rag3_list_collections(missing))['selected'] == 'alpha'
        print('SESSIONLESS_SELECTION_ISOLATION_PASS', flush=True)

        calls = []
        def fake_search(query, options=None, collection=None, *, include_ui=False):
            calls.append((query, collection, mod._CALL_SESSION.get()))
            return {'results': []}
        with patch.object(mod, '_locked_search', fake_search):
            searched = await mod.rag3_search('sample', missing, collection='beta')
            assert not searched.is_error
            assert calls == [('sample', 'beta', mod._SESSIONLESS_HTTP)]
        print('SESSIONLESS_EXPLICIT_SEARCH_PASS', flush=True)

    mod._SESSION_COLLECTIONS = SessionCollections(None)
    with patch.object(mod, 'list_collections', return_value=['alpha', 'beta']):
        no_default = await mod.rag3_list_collections(missing)
        assert body(no_default)['selected'] is None
        assert body(no_default)['selection_persistent'] is False
    with patch.object(mod, 'list_collections', return_value=['alpha']):
        sole = await mod.rag3_list_collections(missing)
        assert body(sole)['selected'] == 'alpha'
    print('SESSIONLESS_AMBIGUOUS_DEFAULT_PASS', flush=True)


def main():
    with tempfile.TemporaryDirectory(prefix='rag3-sessionless-') as directory:
        os.environ['RAG3_TEST_DATA_ROOT'] = directory
        os.environ['RAG3_BACKEND_TEST_INSTANCE'] = 'sessionless_unit'
        spec = importlib.util.spec_from_file_location(
            'stage_d_sessionless_server', ROOT / 'scripts' / 'rag3-mcp-server.py')
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        asyncio.run(exercise(mod))
    print('STAGE_D_SESSIONLESS_MCP_PASS', flush=True)


if __name__ == '__main__':
    main()
