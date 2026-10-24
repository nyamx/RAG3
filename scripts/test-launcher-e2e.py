"""Windows UI launcher lifecycle test against an ALREADY RUNNING shared backend.

Only opens/closes the UI. Never starts/stops the backend, Qdrant or MCP tray.
The caller must choose a time when no user UI is open.
"""
from __future__ import annotations
import ctypes
import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
EXE=ROOT/'RAG3.exe'
TITLE='RAG3 Haystack'
PORT=int(os.environ.get('RAG3_MCP_PORT','8000'))


def window_handle()->int:
    user32=ctypes.WinDLL('user32',use_last_error=True)
    find=user32.FindWindowW
    find.argtypes=(ctypes.c_wchar_p,ctypes.c_wchar_p)
    find.restype=ctypes.c_void_p
    return find(None,TITLE) or 0


def backend_health()->dict:
    req=urllib.request.Request(f'http://127.0.0.1:{PORT}/health')
    with urllib.request.urlopen(req,timeout=6) as resp:
        state=json.load(resp)
    assert state.get('backend')=='ready','shared backend not ready'
    assert isinstance(state.get('pid'),int),'backend PID unavailable'
    return state


def wait_for(predicate,seconds:float,what:str)->None:
    deadline=time.monotonic()+seconds
    while time.monotonic()<deadline:
        if predicate(): return
        time.sleep(0.2)
    raise AssertionError(f'timeout waiting for {what}')


def close_ui()->None:
    hwnd=window_handle()
    if not hwnd: return
    user32=ctypes.WinDLL('user32',use_last_error=True)
    post=user32.PostMessageW
    post.argtypes=(ctypes.c_void_p,ctypes.c_uint,ctypes.c_size_t,ctypes.c_ssize_t)
    post.restype=ctypes.c_int
    if not post(ctypes.c_void_p(hwnd),0x0010,0,0):
        raise OSError(ctypes.get_last_error(),'WM_CLOSE failed')


def main()->None:
    if os.name!='nt': raise SystemExit('Windows only')
    if not EXE.is_file(): raise SystemExit('Build RAG3.exe first')
    if window_handle(): raise SystemExit('Refusing to close pre-existing user UI')
    before=backend_health()
    pid=before['pid']
    launcher=subprocess.Popen([str(EXE)],cwd=ROOT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    created=False
    try:
        wait_for(lambda:bool(window_handle()),90,'RAG3 UI window')
        created=True
        launcher.wait(timeout=20)
        assert launcher.returncode==0,'launcher exit code'
        second=subprocess.run([str(EXE)],cwd=ROOT,timeout=15,
                              creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        assert second.returncode==0,'second launch exit code'
        assert backend_health()['pid']==pid,'UI launch changed backend owner'
    finally:
        if created:
            close_ui()
            wait_for(lambda:not window_handle(),30,'UI shutdown')
        if launcher.poll() is None:
            launcher.terminate()
            launcher.wait(timeout=10)
    after=backend_health()
    assert after['pid']==pid,'closing UI changed backend owner'
    print('STAGE_D_LAUNCHER_LIFECYCLE=PASS backend_pid_unchanged=True')

if __name__=='__main__': main()
