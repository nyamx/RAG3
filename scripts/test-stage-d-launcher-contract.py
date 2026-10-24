"""Offline Stage D launcher contract: no UI-owned service startup or shutdown."""
from pathlib import Path
import py_compile
ROOT=Path(__file__).resolve().parents[1]
cs=(ROOT/'launcher'/'Program.cs').read_text(encoding='utf-8-sig')
e2e=(ROOT/'scripts'/'test-launcher-e2e.py').read_text(encoding='utf-8-sig')
assert 'Process.Start(start)' in cs
assert 'scripts", "ingest-ui.py' in cs
assert '공용 RAG3 HTTP 백엔드' in cs
assert 'Qdrant 실행 상태와 Python 가상환경' not in cs
assert 'rag3-mcp-tray.py' not in e2e
assert 'RAG3_TEST_DATA_ROOT' not in e2e
assert 'backend_health' in e2e and 'backend owner' in e2e
assert 'stop_mcp' not in e2e and 'TRAY' not in e2e
assert 'Refusing to close pre-existing user UI' in e2e
py_compile.compile(str(ROOT/'scripts'/'test-launcher-e2e.py'),doraise=True)
print('STAGE_D_LAUNCHER_OFFLINE_CONTRACT=PASS')
