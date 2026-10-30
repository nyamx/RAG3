"""Sessionless tunnel calls restore the prior process-wide collection selection.

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
    mod._SESSIONLESS_SELECTED = 'alpha'
    missing = SimpleNamespace(session_id=None, headers={})
    another_missing = SimpleNamespace(session_id=None, headers={})
    client_a = SimpleNamespace(session_id='client-a', headers={'mcp-session-id': 'client-a'})
    client_b = SimpleNamespace(session_id='client-b', headers={'mcp-session-id': 'client-b'})
    assert mod._session_key(missing) == mod._SESSIONLESS_HTTP
    assert mod._session_key(client_a) == 'mcp:client-a'

    with patch.object(mod, 'list_collections', return_value=['alpha', 'beta']):
        first = await mod.rag3_list_collections(missing)
        assert not first.is_error
        assert body(first) == {'collections': ['alpha', 'beta'], 'selected': 'alpha',
                               'selection_persistent': True,
                               'selection_scope': 'server_process'}
        changed = await mod.rag3_select_collection('beta', missing)
        assert not changed.is_error
        assert body(changed) == {'selected': 'beta', 'selection_scope': 'server_process'}
        second = await mod.rag3_list_collections(missing)
        assert body(second)['selected'] == 'beta'
        assert body(await mod.rag3_list_collections(another_missing))['selected'] == 'beta'
        assert mod._SESSIONLESS_HTTP not in mod._SESSION_COLLECTIONS._selection

        selected = await mod.rag3_select_collection('beta', client_a)
        assert not selected.is_error and body(selected)['selected'] == 'beta'
        assert body(await mod.rag3_list_collections(client_a))['selected'] == 'beta'
        assert body(await mod.rag3_list_collections(client_b))['selected'] == 'alpha'
        assert body(await mod.rag3_list_collections(missing))['selected'] == 'beta'
        print('SESSIONLESS_SERVER_SELECTION_PASS', flush=True)

        calls = []
        embeddings = []
        def fake_search(query, options=None, *, include_ui=False, dense_encoder=None):
            dense_encoder(query)
            collection = mod._current_collection()
            calls.append((query, collection, mod._CALL_SESSION.get()))
            return {'results': [{'id': collection, 'collection': collection}]}
        with patch.object(mod, '_search_impl', fake_search), patch.object(
                mod, '_encode_query_gpu', lambda query: embeddings.append(query) or [0.1]):
            searched = await mod.rag3_search('sample', ['alpha', 'beta'], missing)
            assert not searched.is_error
            assert [hit['collection'] for hit in body(searched)['results']] == ['alpha', 'beta']
            assert calls == [('sample', 'alpha', mod._SESSIONLESS_HTTP),
                             ('sample', 'beta', mod._SESSIONLESS_HTTP)]
            assert embeddings == ['sample'], embeddings
            assert body(await mod.rag3_list_collections(missing))['selected'] == 'beta'
            missing_names = await mod.rag3_search('sample', [], missing)
            assert missing_names.is_error
            assert 'collections=[' in body(missing_names)['error']['message']
            unknown = await mod.rag3_search('sample', ['unknown'], missing)
            assert unknown.is_error
            assert body(unknown)['error']['code'] == 'COLLECTION_NOT_FOUND'
        print('SESSIONLESS_EXPLICIT_MULTI_SEARCH_PASS', flush=True)

    mod._SESSION_COLLECTIONS = SessionCollections(None)
    mod._SESSIONLESS_SELECTED = None
    with patch.object(mod, 'list_collections', return_value=['alpha', 'beta']):
        no_default = await mod.rag3_list_collections(missing)
        assert body(no_default)['selected'] is None
        assert body(no_default)['selection_persistent'] is True
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
