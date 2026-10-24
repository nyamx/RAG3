"""Pure validation of the Stage D Task Scheduler backend entry (no service starts)."""
from __future__ import annotations
import importlib.util
import json
import tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('rag3_login_entry',ROOT/'scripts/rag3-backend-login.py')
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
base={'port':8000,'qdrant_url':'http://127.0.0.1:16333','write_enabled':True,
      'delete_enabled':True,'bootstrap_infra':True,'tunnel_autostart':True}
assert mod.validate_config(base)==base
for invalid in [dict(base,port=True),dict(base,port=0),dict(base,port=65536),
                dict(base,qdrant_url='http://user:secret@127.0.0.1:16333'),
                dict(base,qdrant_url='http://127.0.0.1:16333/mcp'),
                dict(base,qdrant_url='https://example.net:6333'),
                dict(base,delete_enabled='1'),dict(base,unrecognized=True)]:
    try:mod.validate_config(invalid)
    except (ValueError,TypeError):pass
    else:raise AssertionError('Invalid config permitted')
nonlocal_qdrant=dict(base,qdrant_url='http://127.0.0.1:16555',bootstrap_infra=False)
assert mod.validate_config(nonlocal_qdrant)==nonlocal_qdrant
with tempfile.TemporaryDirectory(prefix='rag3-login-stage-d-') as d:
    f=Path(d)/'cfg.json';f.write_text(json.dumps(base),encoding='utf-8')
    assert mod.read_config(f)==base
    assert mod.main(['--validate','--config',str(f)])==0
    assert mod.main(['--run','--config',str(Path(d)/'missing.json')])==2
print('STAGE_D_LOGIN_ENTRY_OFFLINE_PASS checks=13 no_server_started=YES')
