"""Offline validation of Stage D HTTP plugin transport and safe refresh gates."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
from unittest import mock

ROOT=Path(__file__).resolve().parents[1]
file=ROOT/'scripts/ensure-rag3-codex-plugin.py'
spec=importlib.util.spec_from_file_location('stage_d_plugin_register',file)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
url,version=module.desired_http_manifest()
assert url == 'http://127.0.0.1:8000/mcp',url
assert version=='0.1.2',version
fake_market={'name':'rag3-local','root':str(ROOT)}
old={'pluginId':module.PLUGIN_ID,'installed':True,'enabled':True,'version':'0.1.1'}
with mock.patch.object(module,'find_codex',return_value=Path('codex.exe')) as _, \
     mock.patch.object(module,'registration_status',return_value=(fake_market,old)), \
     mock.patch.object(module,'installed_transport',return_value=('stdio',None)), \
     mock.patch.object(module,'run_codex',side_effect=AssertionError('Unexpected mutating CLI')):
    result=module.ensure_registration(check_only=True)
    assert result['status']=='update_pending',result
    result=module.ensure_registration(check_only=False)
    assert result['status']=='update_pending',result
old_disabled=dict(old, enabled=False)
with mock.patch.object(module,'find_codex',return_value=Path('codex.exe')), \
     mock.patch.object(module,'registration_status',return_value=(fake_market,old_disabled)), \
     mock.patch.object(module,'run_codex',side_effect=AssertionError('Disabled plugin modified')):
    assert module.ensure_registration(check_only=False,activate_http=True)['status']=='disabled'
with mock.patch.object(module,'find_codex',return_value=Path('codex.exe')), \
     mock.patch.object(module,'registration_status',return_value=(fake_market,old)), \
     mock.patch.object(module,'installed_transport',return_value=('stdio',None)), \
     mock.patch.object(module,'_check_live_mcp',side_effect=ConnectionRefusedError()), \
     mock.patch.object(module,'run_codex',side_effect=AssertionError('Offline backend allowed CLI mutation')):
    assert module.ensure_registration(check_only=False,activate_http=True)['status']=='backend_not_ready'
print('STAGE_D_HTTP_PLUGIN_OFFLINE_PASS checks=4')
