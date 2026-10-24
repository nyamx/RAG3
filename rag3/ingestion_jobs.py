"""Shared, lightweight SQLite journal for UI and MCP ingestion jobs."""
from __future__ import annotations

import json
import os
import threading
import re
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from rag3.constants import ROOT, DATA_ROOT

DB_PATH = DATA_ROOT / 'ingestion_jobs.sqlite'
ACTIVE = frozenset({'QUEUED', 'DOWNLOADING', 'RUNNING', 'PAUSING', 'RESUMING', 'CANCELLING'})
TERMINAL = frozenset({'COMPLETE', 'CANCELLED', 'FAILED'})


def change_signal_path() -> Path:
    return DB_PATH.with_suffix('.signal')


def ensure_change_signal() -> Path:
    path = change_signal_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text('0', encoding='ascii')
    return path


def _emit_change_signal() -> None:
    try:
        ensure_change_signal().write_text(str(time.time_ns()), encoding='ascii')
    except OSError:
        # The committed revision is authoritative; startup/activation can recover
        # from a missed filesystem notification.
        pass


def content_scope(collection: str) -> str:
    return 'contents:' + collection


def now():
    return datetime.now(timezone.utc).isoformat()


_SCHEMA_LOCK = threading.RLock()
_SCHEMA_READY = set()  # (resolved path, filesystem identity, process id)


def _file_identity(path):
    stat = path.stat()
    return (str(path.resolve()), stat.st_dev, stat.st_ino, stat.st_size if stat.st_size == 0 else None, os.getpid())


def _initialize_schema(db):
    # WAL is persistent, but PRAGMA journal_mode is not valid inside a tx.
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('BEGIN IMMEDIATE')
    try:
        db.execute('''CREATE TABLE IF NOT EXISTS jobs (
            job_id TEXT PRIMARY KEY, origin TEXT NOT NULL, collection TEXT NOT NULL,
            filename TEXT NOT NULL, source TEXT, source_sha256 TEXT,
            status TEXT NOT NULL, stage TEXT NOT NULL DEFAULT '',
            result TEXT, error TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
            requested INTEGER NOT NULL DEFAULT 0, mode TEXT NOT NULL DEFAULT 'run',
            file_size INTEGER, downloaded_bytes INTEGER NOT NULL DEFAULT 0,
            hidden_ui INTEGER NOT NULL DEFAULT 0
        )''')
        db.execute('CREATE INDEX IF NOT EXISTS job_queue ON jobs (requested,status,created_at)')
        db.execute('CREATE INDEX IF NOT EXISTS job_digest ON jobs (collection,source_sha256)')
        index_row = db.execute("SELECT sql FROM sqlite_master WHERE type='index' AND name='job_one_active_digest'").fetchone()
        if index_row is not None and ('(collection, source_sha256)' not in index_row['sql'] or
                                      "'FAILED'" not in index_row['sql']):
            db.execute('DROP INDEX job_one_active_digest')
            index_row = None
        if index_row is None:
            db.execute("CREATE UNIQUE INDEX job_one_active_digest ON jobs (collection, source_sha256) WHERE source_sha256 IS NOT NULL AND status NOT IN ('COMPLETE','CANCELLED','FAILED')")
        db.execute('CREATE TABLE IF NOT EXISTS ui_revisions (scope TEXT PRIMARY KEY, version INTEGER NOT NULL DEFAULT 0)')
        db.execute('''CREATE TABLE IF NOT EXISTS ingestion_progress (
            collection TEXT NOT NULL, source_sha256 TEXT NOT NULL,
            identity TEXT NOT NULL, phase TEXT NOT NULL,
            total_chunks INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (collection, source_sha256)
        )''')
        for operation in ('INSERT', 'UPDATE', 'DELETE'):
            db.execute(f'''CREATE TRIGGER IF NOT EXISTS jobs_revision_{operation.lower()}
                AFTER {operation} ON jobs BEGIN
                INSERT INTO ui_revisions(scope, version) VALUES('jobs', 1)
                ON CONFLICT(scope) DO UPDATE SET version=version+1;
                END''')
        db.commit()
    except BaseException:
        db.rollback()
        raise


@contextmanager
def connect():
    """Independent journal transactions; schema DDL only for a new DB identity."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _SCHEMA_LOCK:
        db = sqlite3.connect(str(DB_PATH), timeout=20)
        try:
            db.row_factory = sqlite3.Row
            db.execute('PRAGMA busy_timeout=20000')
            key = _file_identity(DB_PATH)
            if key not in _SCHEMA_READY:
                _initialize_schema(db)
                # Commit succeeded; only now cache the observed file identity.
                _SCHEMA_READY.add(_file_identity(DB_PATH))
        except BaseException:
            db.close()
            raise
    try:
        before = db.total_changes
        try:
            with db:
                yield db
        finally:
            if db.total_changes > before:
                _emit_change_signal()
    finally:
        db.close()


def change_versions(*scopes: str) -> dict[str, int]:
    if not scopes:
        return {}
    with connect() as db:
        placeholders = ','.join('?' for _ in scopes)
        rows = db.execute(f'SELECT scope, version FROM ui_revisions WHERE scope IN ({placeholders})', scopes)
        found = {row['scope']: row['version'] for row in rows}
    return {scope: found.get(scope, 0) for scope in scopes}


def mark_change(scope: str) -> None:
    if not scope:
        raise ValueError('change scope is required')
    with connect() as db:
        db.execute('''INSERT INTO ui_revisions(scope, version) VALUES(?, 1)
                      ON CONFLICT(scope) DO UPDATE SET version=version+1''', (scope,))

def check_id(job_id):
    if not isinstance(job_id, str) or not re.fullmatch('[0-9a-f]{32}', job_id):
        raise ValueError('job_id is invalid')
    return job_id


def unpack(row, *, private=False):
    data = dict(row)
    for name in ('result',):
        if isinstance(data.get(name), str):
            data[name] = json.loads(data[name])
    data['requested'] = bool(data['requested'])
    data['checkpoint_available'] = checkpoint_available(data)
    if not private:
        data.pop('source', None)
        data.pop('requested', None)
        data.pop('hidden_ui', None)
        data.pop('mode', None)
    return data


def checkpoint_available(job):
    sha = job.get('source_sha256')
    if not sha or not re.fullmatch('[0-9a-f]{64}', sha):
        return False
    progress = ingestion_progress(job.get('collection', ''), sha)
    return progress is not None and progress['phase'] in {'PARSING', 'CHUNKS_STORED', 'EMBEDDING'}


def ingestion_progress(collection: str, source_sha256: str) -> dict | None:
    with connect() as db:
        row = db.execute('''SELECT identity, phase, total_chunks FROM ingestion_progress
                            WHERE collection=? AND source_sha256=?''',
                         (collection, source_sha256)).fetchone()
    if row is None:
        return None
    return {'identity': json.loads(row['identity']), 'phase': row['phase'],
            'total_chunks': row['total_chunks']}


def set_ingestion_progress(collection: str, source_sha256: str, identity: dict,
                           phase: str, total_chunks: int = 0) -> None:
    if phase not in {'PARSING', 'CHUNKS_STORED', 'EMBEDDING'} or total_chunks < 0:
        raise ValueError('invalid ingestion progress')
    with connect() as db:
        db.execute('''INSERT INTO ingestion_progress
            (collection,source_sha256,identity,phase,total_chunks,updated_at)
            VALUES (?,?,?,?,?,?) ON CONFLICT(collection,source_sha256) DO UPDATE SET
            identity=excluded.identity,phase=excluded.phase,
            total_chunks=excluded.total_chunks,
            updated_at=excluded.updated_at''',
            (collection, source_sha256, json.dumps(identity, sort_keys=True),
             phase, total_chunks, now()))


def clear_ingestion_progress(collection: str, source_sha256: str) -> None:
    with connect() as db:
        db.execute('DELETE FROM ingestion_progress WHERE collection=? AND source_sha256=?',
                   (collection, source_sha256))


def one(job_id, *, private=False):
    with connect() as db:
        row = db.execute('SELECT * FROM jobs WHERE job_id=?', (check_id(job_id),)).fetchone()
    if row is None:
        raise ValueError('job does not exist')
    return unpack(row, private=private)


def list_jobs(*, collection='', status='', limit=50, offset=0, include_hidden=True):
    limit, offset = int(limit), int(offset)
    if limit < 1 or limit > 200 or offset < 0:
        raise ValueError('limit must be 1..200, offset >= 0')
    where, args = [], []
    if collection:
        where.append('collection=?'); args.append(collection)
    if status:
        if status not in ACTIVE | TERMINAL | {'PAUSED'}:
            raise ValueError('unknown status')
        where.append('status=?'); args.append(status)
    if not include_hidden:
        where.append('hidden_ui=0')
    cond = ' WHERE ' + ' AND '.join(where) if where else ''
    with connect() as db:
        total = db.execute('SELECT count(*) FROM jobs' + cond, args).fetchone()[0]
        rows = db.execute('SELECT * FROM jobs' + cond +
                          ' ORDER BY created_at DESC, job_id DESC LIMIT ? OFFSET ?',
                          [*args, limit, offset]).fetchall()
    return {'jobs': [unpack(r) for r in rows], 'total': total, 'limit': limit, 'offset': offset}


def patch(db, job_id, **fields):
    fields['updated_at'] = now()
    allowed = {'status','stage','source','source_sha256','filename','result','error',
               'requested','mode','file_size','downloaded_bytes','hidden_ui','updated_at'}
    if set(fields) - allowed:
        raise ValueError('invalid job update keys')
    if 'result' in fields and fields['result'] is not None:
        fields['result'] = json.dumps(fields['result'], ensure_ascii=False)
    names = list(fields)
    db.execute('UPDATE jobs SET ' + ', '.join(f'{k}=?' for k in names) +
               ' WHERE job_id=?', [*(fields[k] for k in names), check_id(job_id)])
    db.commit()
