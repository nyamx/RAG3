"""Prepare a one-process Windows Python executable inside the existing RAG3 venv.

Windows venv's stock pythonw.exe can be a redirector that keeps a launcher and
the base interpreter alive. A byte-for-byte copy of the matching base Python
executable and its Python DLLs under venv/Scripts uses the same venv packages
in one process, independent of the login task's PATH.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / '.venv'
DIRECT = VENV / 'Scripts' / 'rag3-pythonw-direct.exe'


def _settings() -> tuple[dict[Path, Path], str]:
    values = {}
    for line in (VENV / 'pyvenv.cfg').read_text('utf-8-sig').splitlines():
        if '=' in line:
            key, value = line.split('=', 1)
            values[key.strip().lower()] = value.strip()
    home = Path(values['home'])
    version = values['version']
    dll = f'python{sys.version_info.major}{sys.version_info.minor}.dll'
    names = {'rag3-pythonw-direct.exe': 'pythonw.exe',
             'python3.dll': 'python3.dll', dll: dll}
    if not home.is_dir() or not all((home / source).is_file() for source in names.values()):
        raise RuntimeError('pyvenv.cfg base Python is unavailable')
    if version != '.'.join(map(str, sys.version_info[:3])):
        raise RuntimeError('pyvenv.cfg version differs from the executing venv Python')
    if Path(sys.prefix).resolve() != VENV.resolve():
        raise RuntimeError('This preparer must run with the RAG3 venv Python')
    if Path(sys.base_prefix).resolve() != home.resolve():
        raise RuntimeError('pyvenv.cfg home differs from the active base Python')
    return {VENV / 'Scripts' / target: home / source
            for target, source in names.items()}, version


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _smoke(version: str) -> None:
    with tempfile.TemporaryDirectory(prefix='rag3-direct-pythonw-') as directory:
        report = Path(directory) / 'report.json'
        code = (
            'import json,os,sys,mcp,pywintypes;from pathlib import Path;'
            'Path(sys.argv[1]).write_text(json.dumps({'
            '"pid":os.getpid(),"prefix":sys.prefix,"base_prefix":sys.base_prefix,'
            '"version":".".join(map(str,sys.version_info[:3])),'
            '"mcp":mcp.__file__,"pywintypes":pywintypes.__file__}),encoding="utf-8")'
        )
        env = os.environ.copy()
        for name in ('PYTHONHOME', 'PYTHONPATH', 'VIRTUAL_ENV'):
            env.pop(name, None)
        windows = Path(env.get('SystemRoot', r'C:\Windows'))
        env['PATH'] = str(windows / 'System32') + os.pathsep + str(windows)
        process = subprocess.Popen(
            [str(DIRECT), '-c', code, str(report)], cwd=ROOT,
            env=env,
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
        )
        try:
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline and not report.exists() and process.poll() is None:
                time.sleep(0.1)
            if not report.exists():
                raise RuntimeError('Direct Pythonw smoke produced no report')
            result = json.loads(report.read_text('utf-8'))
            if result['pid'] != process.pid:
                raise RuntimeError('Direct Pythonw still creates a second process')
            if Path(result['prefix']).resolve() != VENV.resolve():
                raise RuntimeError('Direct Pythonw did not select the RAG3 venv')
            if Path(result['base_prefix']).resolve() != Path(sys.base_prefix).resolve():
                raise RuntimeError('Direct Pythonw selected a different base Python')
            if result['version'] != version:
                raise RuntimeError('Base Pythonw version differs from pyvenv.cfg')
            packages = (VENV / 'Lib' / 'site-packages').resolve()
            if not all(packages in Path(result[name]).resolve().parents
                       for name in ('mcp', 'pywintypes')):
                raise RuntimeError('Direct Pythonw did not import RAG3 venv packages')
            process.wait(timeout=10)
            if process.returncode:
                raise RuntimeError('Direct Pythonw smoke exited with an error')
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if os.name != 'nt':
        raise RuntimeError('Direct Pythonw preparation is Windows-only')
    files, version = _settings()
    for target, source in files.items():
        if target.exists() and _digest(target) != _digest(source):
            raise RuntimeError('Existing direct Python component differs from pyvenv.cfg base: '
                               + target.name)
    if not args.check_only:
        DIRECT.parent.mkdir(parents=True, exist_ok=True)
        for target, source in files.items():
            if target.exists():
                continue
            temporary = target.with_name(target.name + '.stage-d-new')
            if temporary.exists():
                raise RuntimeError('Temporary direct Python component already exists: '
                                   + target.name)
            try:
                shutil.copy2(source, temporary)
                if _digest(temporary) != _digest(source):
                    raise RuntimeError('Direct Python component copy checksum differs: '
                                       + target.name)
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
    present = all(target.is_file() for target in files)
    if present:
        _smoke(version)
    print(json.dumps({'status': 'ready' if present else 'copy_required',
                      'venv_version': version, 'single_process_smoke': present,
                      'venv_packages': present}, ensure_ascii=True))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, KeyError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
        print(json.dumps({'status': 'error', 'reason': type(exc).__name__ + ': ' + str(exc)},
                         ensure_ascii=True), file=sys.stderr)
        raise SystemExit(1)
