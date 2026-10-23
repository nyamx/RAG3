"""One detached stdlib-only local scheduler, worker owner and attachment receiver."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import signal
import secrets
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from rag3.constants import DEFAULT_QDRANT_URL, ROOT
from rag3.ingestion_jobs import (ACTIVE, DB_PATH, TERMINAL, check_id,
                                 checkpoint_available, clear_ingestion_progress,
                                 connect, ingestion_progress, list_jobs, now, one, patch)
from rag3.ingestion_settings import progress_identity

HOST, PORT = '127.0.0.1', int(os.environ.get('RAG3_INGESTION_MANAGER_PORT', '18747'))
TOKEN_FILE = ROOT / 'data' / 'ingestion-manager.token'
PROCESS_PYTHON = ROOT / '.venv' / 'Scripts' / 'pythonw.exe'
JOB_ROOT = ROOT / 'data' / 'mcp_jobs'
SUPPORTED_UI = {'.pdf','.doc','.docx','.md','.pptx','.xlsx','.html','.htm','.txt',
                '.json','.jsonl','.csv','.tsv',
                '.png','.jpg','.jpeg','.tif','.tiff','.bmp'}


def client(action, **params):
    """RPC with startup handshake: no silent, detached fallback worker."""
    payload = json.dumps({'action': action, 'params': params}).encode('utf-8')
    def send():
        token = TOKEN_FILE.read_text(encoding='ascii') if TOKEN_FILE.is_file() else ''
        req = Request(f'http://{HOST}:{PORT}/command', payload,
                      {'Content-Type':'application/json','X-RAG3-Token':token}, method='POST')
        with urlopen(req, timeout=30) as r:
            data = json.loads(r.read())
        if not data.get('ok'):
            raise ValueError(data.get('error') or 'ingestion manager error')
        return data['result']
    try:
        return send()
    except (URLError, ConnectionError, TimeoutError, OSError) as first:
        flags = (getattr(subprocess, 'DETACHED_PROCESS', 0) |
                 getattr(subprocess, 'CREATE_NEW_PROCESS_GROUP', 0)) if os.name == 'nt' else 0
        python = str(PROCESS_PYTHON if PROCESS_PYTHON.exists() else sys.executable)
        log_dir = ROOT / 'data'; log_dir.mkdir(parents=True, exist_ok=True)
        with (log_dir / 'ingestion-manager.log').open('ab') as log:
            subprocess.Popen([python, '-m', 'rag3.ingestion_manager', '--serve'], cwd=ROOT,
                             stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                             creationflags=flags, start_new_session=os.name != 'nt')
        deadline = time.monotonic() + 12
        while time.monotonic() < deadline:
            time.sleep(.2)
            try:
                return send()
            except (URLError, ConnectionError, TimeoutError, OSError):
                pass
        raise RuntimeError(f'Ingestion manager did not start: {first}') from first


class Scheduler:
    def __init__(self):
        self.lock = threading.RLock()
        self.proc = None
        self.active_id = None
        self.downloads = {}  # ephemeral signed URL and control events, never persisted
        self.qdrant_url = os.environ.get('RAG3_QDRANT_URL', DEFAULT_QDRANT_URL)
        self.recover()
        threading.Thread(target=self.pump, daemon=True).start()

    def recover(self):
        with connect() as db:
            rows = db.execute("SELECT * FROM jobs WHERE status IN ('RUNNING','RESUMING','PAUSING','CANCELLING','DOWNLOADING')").fetchall()
            for row in rows:
                job = dict(row)
                if job['status'] == 'DOWNLOADING' or not checkpoint_available(job):
                    patch(db, job['job_id'], status='FAILED', requested=0, error='Manager stopped; no valid checkpoint or incomplete download')
                else:
                    patch(db, job['job_id'], status='PAUSED', requested=0, stage='Interrupted; valid checkpoint preserved')
            db.execute("UPDATE jobs SET status='FAILED', requested=0, error='Interrupted attachment download; submit a fresh reference' WHERE status='PAUSED' AND source_sha256 IS NULL")
            # Old queued requests must not start on a new manager boot.
            db.execute("UPDATE jobs SET requested=0 WHERE status='QUEUED'")
            db.commit()
        self.cleanup_completed()

    def cleanup_completed(self):
        with connect() as db:
            rows = db.execute("SELECT job_id, source_sha256, collection FROM jobs WHERE status='COMPLETE' AND source_sha256 IS NOT NULL").fetchall()
        for row in rows:
            with connect() as db:
                in_flight=db.execute("SELECT 1 FROM jobs WHERE source_sha256=? AND status NOT IN ('COMPLETE','FAILED','CANCELLED') LIMIT 1",(row['source_sha256'],)).fetchone()
            if not in_flight:
                clear_ingestion_progress(row['collection'], row['source_sha256'])
                self.retire_completed_upload(one(row['job_id'], private=True))

    def retire_completed_upload(self, job):
        """Keep the verified managed original and discard a completed MCP job copy."""
        try:
            original = job.get('source')
            result = job.get('result') or {}
            managed_name = result.get('managed_source')
            sha = job.get('source_sha256')
            if not original or not managed_name or not sha:
                return
            job_root = JOB_ROOT.resolve()
            job_folder = job_root / job['job_id']
            source = Path(original).resolve()
            managed = Path(managed_name).resolve()
            expected_managed = (ROOT / 'data' / 'sources' / sha).resolve()
            if (source.parent != job_folder or managed.parent != expected_managed
                    or not managed.is_file() or self.digest(managed) != sha):
                return
            source.unlink(missing_ok=True)
            with connect() as db:
                patch(db, job['job_id'], source=str(managed))
            try: job_folder.rmdir()
            except OSError: pass
        except (OSError, ValueError, sqlite3.Error):
            pass  # Keep the completed job valid; retry cleanup on manager startup.

    def validate_collection(self, collection):
        # A lightweight Qdrant REST check keeps torch/Docling out of this daemon.
        with urlopen(self.qdrant_url.rstrip('/')+'/collections',timeout=10) as response:
            payload=json.loads(response.read())
        names={item['name'] for item in payload['result']['collections']}
        if collection not in names:
            raise ValueError(f'collection does not exist: {collection}; create/select an existing collection first')

    def conflict(self, sha, collection, excluding=''):
        if not sha:
            return
        with connect() as db:
            rows = db.execute("SELECT job_id, collection, status FROM jobs WHERE source_sha256=? AND job_id!=?", (sha, excluding)).fetchall()
        for row in rows:
            if row['collection'] == collection and row['status'] not in {'FAILED','CANCELLED'}:
                raise ValueError('the same SHA is already queued or running in this collection')
            if row['collection'] != collection and row['status'] not in TERMINAL:
                raise ValueError('same SHA is active in another collection; checkpoint path is shared')

    @staticmethod
    def digest(path):
        h = hashlib.sha256()
        with Path(path).open('rb') as f:
            for part in iter(lambda: f.read(1024*1024), b''):
                h.update(part)
        return h.hexdigest()

    def submit(self, *, origin, collection, path='', filename='', content_base64='', attachment=None, requested=True):
        self.validate_collection(collection)
        job_id = uuid.uuid4().hex
        sha, size, downloaded, source = None, None, 0, None
        if attachment is not None:
            from rag3.attachments import validate_url, validate_filename
            if not attachment.get('file_id'):
                raise ValueError('attachment file_id is required')
            validate_url(attachment['download_url'])
            filename = attachment.get('file_name') or filename
            if filename: validate_filename(filename)
            status, stage = 'DOWNLOADING', 'Receiving original attachment'
        elif content_base64:
            from rag3.mcp_jobs import _validate_content, ALLOWED_SUFFIXES
            from rag3.attachments import validate_filename
            validate_filename(filename)
            raw = base64.b64decode(content_base64, validate=True)
            if Path(filename).suffix.lower() not in ALLOWED_SUFFIXES:
                raise ValueError('unsupported document format')
            _validate_content(Path(filename).suffix.lower(), raw)
            sha, size = hashlib.sha256(raw).hexdigest(), len(raw)
            self.conflict(sha, collection)
            folder = JOB_ROOT / job_id; folder.mkdir(parents=True, exist_ok=True)
            source = folder / filename
            source.write_bytes(raw)
            status, stage = 'QUEUED', 'Ready'
        else:
            source = Path(path).expanduser().resolve()
            if not source.is_file() or source.suffix.lower() not in SUPPORTED_UI:
                raise ValueError('unsupported or missing source file')
            filename = source.name
            sha, size = self.digest(source), source.stat().st_size
            self.conflict(sha, collection)
            status, stage = 'QUEUED', 'Ready'
        with connect() as db:
            ts = now()
            db.execute('''INSERT INTO jobs (job_id,origin,collection,filename,source,source_sha256,status,
                          stage,created_at,updated_at,requested,file_size,downloaded_bytes) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                       (job_id,origin,collection,filename,str(source) if source else None,
                        sha,status,stage,ts,ts,int(requested),size,downloaded))
            db.commit()
        if attachment is not None:
            control = {'running':threading.Event(), 'cancelled':threading.Event()}
            control['running'].set()
            with self.lock:
                self.downloads[job_id] = control
            threading.Thread(target=self.receive, args=(job_id,attachment['download_url'],filename,control),daemon=True).start()
        return one(job_id)

    def receive(self, job_id, url, filename, control):
        from rag3.attachments import download_attachment, AttachmentCancelled
        from rag3.mcp_jobs import _validate_source
        def checkpoint():
            while not control['running'].wait(.2):
                if control['cancelled'].is_set():
                    raise AttachmentCancelled
            if control['cancelled'].is_set():
                raise AttachmentCancelled
        def progress(count):
            with connect() as db: patch(db,job_id,downloaded_bytes=count)
        try:
            download_folder = JOB_ROOT / job_id
            download_folder.mkdir(parents=True, exist_ok=True)
            path,name,sha,count = download_attachment(url, download_folder,filename=filename,
                                                      checkpoint=checkpoint, progress=progress)
            _validate_source(path)
            checkpoint()  # Do not wait for download Resume while holding manager lock.
            with self.lock:
                self.conflict(sha, one(job_id,private=True)['collection'], excluding=job_id)
                with connect() as db:
                    patch(db,job_id, source=str(path),filename=name,source_sha256=sha,
                          file_size=count,downloaded_bytes=count,status='QUEUED',stage='Ready')
        except AttachmentCancelled:
            with connect() as db: patch(db,job_id,status='CANCELLED',requested=0,stage='Download cancelled')
        except Exception as exc:
            # Signed URLs and HTTP exception repr must not be placed in journal/log.
            message = str(exc) if isinstance(exc,ValueError) else 'Attachment download failed; submit a new file reference'
            with connect() as db: patch(db,job_id,status='FAILED',requested=0,error=message)
        finally:
            with self.lock: self.downloads.pop(job_id,None)

    def action(self, job_id, operation):
        check_id(job_id)
        with self.lock:
            job = one(job_id,private=True)
            status = job['status']
            dl = self.downloads.get(job_id)
            if operation == 'hide':
                if status not in TERMINAL | {'PAUSED','QUEUED'}:
                    raise ValueError('cannot hide an active job')
                with connect() as db: patch(db,job_id,hidden_ui=1)
            elif operation == 'run':
                if status != 'QUEUED':
                    raise ValueError('job is not queued')
                with connect() as db: patch(db,job_id,requested=1)
            elif operation == 'pause':
                if status == 'DOWNLOADING' and dl:
                    dl['running'].clear()
                    with connect() as db: patch(db,job_id,status='PAUSED',stage='Download paused')
                elif status == 'QUEUED':
                    with connect() as db: patch(db,job_id,status='PAUSED',requested=0,stage='Queued (not yet started)')
                elif status in {'RUNNING','RESUMING'} and job_id == self.active_id:
                    with connect() as db: patch(db,job_id,status='PAUSING',stage='Pause requested; waiting for checkpoint')
                    marker = ROOT/'data'/'checkpoints'/job['source_sha256']/'pause.request'
                    marker.parent.mkdir(parents=True,exist_ok=True)
                    marker.write_text('pause\n', encoding='utf-8')
                else:
                    raise ValueError('job is not running or queued')
            elif operation == 'cancel':
                if status in {'QUEUED','PAUSED'} and not dl:
                    with connect() as db: patch(db,job_id,status='CANCELLED',requested=0)
                elif dl:
                    dl['cancelled'].set();dl['running'].set()
                    with connect() as db: patch(db,job_id,status='CANCELLING',requested=0)
                elif status in ACTIVE and self.active_id == job_id and self.proc:
                    with connect() as db: patch(db,job_id,status='CANCELLING',requested=0)
                    self.kill_worker(self.proc)
                else:
                    raise ValueError('job is not cancellable')
            elif operation in {'resume','retry','restart'}:
                if status not in ({'PAUSED','FAILED','CANCELLED','COMPLETE'} if operation=='restart' else {'PAUSED','FAILED','CANCELLED'}):
                    raise ValueError('job is not paused, failed, or cancelled')
                if dl and status == 'PAUSED':
                    dl['running'].set()
                    with connect() as db: patch(db,job_id,status='DOWNLOADING',requested=1)
                elif status == 'PAUSED' and job['source_sha256'] is None:
                    raise ValueError('download interrupted; submit a fresh attachment')
                else:
                    if operation == 'restart':
                        if job['source_sha256']:
                            clear_ingestion_progress(job['collection'],job['source_sha256'])
                        mode = 'run'
                    elif operation == 'resume' and job['stage'] == 'Queued (not yet started)':
                        mode = 'run'
                    elif operation == 'resume':
                        if not checkpoint_available(job):
                            raise ValueError('Qdrant ingestion progress is required for resume')
                        self.validate_identity(job)
                        mode = 'resume'
                    else:
                        mode = 'resume' if checkpoint_available(job) else 'run'
                        if mode == 'resume': self.validate_identity(job)
                    self.conflict(job['source_sha256'],job['collection'],excluding=job_id)
                    with connect() as db: patch(db,job_id,status='QUEUED',requested=1,
                                                 mode=mode,stage='Queued for '+mode,error=None)
            else:
                raise ValueError('unknown operation')
        return one(job_id)

    @staticmethod
    def validate_identity(job):
        progress = ingestion_progress(job['collection'], job['source_sha256'])
        expected = progress_identity(job['source_sha256'], job['collection'])
        if progress is None or progress['identity'] != expected:
            raise ValueError('No compatible Qdrant ingestion progress for resume')

    @staticmethod
    def kill_worker(proc):
        try:
            if os.name == 'nt':
                subprocess.run(['taskkill','/F','/T','/PID',str(proc.pid)],
                               capture_output=True, timeout=8)
            else:
                try: os.killpg(proc.pid,signal.SIGKILL)
                except ProcessLookupError: pass
                proc.kill()
        except (OSError,subprocess.SubprocessError):
            try: proc.kill()
            except OSError: pass

    def pump(self):
        while True:
            row=None
            try:
                with self.lock:
                    if self.active_id is None:
                        with connect() as db:
                            row = db.execute("SELECT * FROM jobs WHERE status='QUEUED' AND requested=1 ORDER BY created_at,job_id LIMIT 1").fetchone()
                        if row is not None:
                            self.start(dict(row))
            except Exception as exc:
                # Do not leave a faulty task auto-retrying forever.
                with self.lock:
                    if row is not None:
                        with connect() as db: patch(db,row['job_id'],status='FAILED',requested=0,error=str(exc)[:300])
            time.sleep(.25)

    def start(self, job):
        if not job.get('source') or not Path(job['source']).is_file():
            raise FileNotFoundError('source file unavailable')
        self.validate_collection(job['collection'])
        self.conflict(job['source_sha256'],job['collection'],excluding=job['job_id'])
        env = os.environ.copy()
        # Apply before Python imports torch/BLAS; worker imposes its own torch limit too.
        env.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
                   NUMEXPR_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',RAG3_WORKER_CPU_THREADS='2')
        python = str(PROCESS_PYTHON if PROCESS_PYTHON.exists() else sys.executable)
        # Remove stale markers before spawning; later Pause writes a fresh request.
        (ROOT/'data'/'checkpoints'/job['source_sha256']/'pause.request').unlink(missing_ok=True)
        args = [python,str(ROOT/'scripts'/'ingest-worker.py'),job['source'],
                '--collection',job['collection'],'--mode',job['mode'],
                '--source-sha',job['source_sha256']]
        flags = getattr(subprocess,'CREATE_NO_WINDOW',0) if os.name == 'nt' else 0
        proc = subprocess.Popen(args,cwd=ROOT,env=env,stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,text=True,encoding='utf-8',
                                errors='replace',bufsize=1,creationflags=flags,
                                start_new_session=os.name != 'nt')
        self.proc,self.active_id = proc,job['job_id']
        with connect() as db: patch(db,job['job_id'],status='RESUMING' if job['mode']=='resume' else 'RUNNING',
                                     stage='Starting worker')
        threading.Thread(target=self.monitor,args=(job['job_id'],proc),daemon=True).start()

    def monitor(self, job_id, proc):
        result, error = None, None
        qdrant_started = False
        try:
            assert proc.stdout is not None
            for raw in proc.stdout:
                try: event=json.loads(raw)
                except ValueError: continue
                typ = event.get('type')
                with connect() as db:
                    if typ == 'log':
                        message = str(event.get('message',''))[:300]
                        qdrant_started |= message == 'STAGE qdrant RUN'
                        patch(db,job_id,stage=message)
                    elif typ == 'result':
                        result = event.get('result')
                    elif typ == 'error':
                        error = str(event.get('message') or 'worker error')[:1000]
            code = proc.wait()
            with self.lock:
                job = one(job_id,private=True)
                if job['status']=='CANCELLING':
                    status='CANCELLED'
                elif code==75 and isinstance(result,dict) and result.get('status')=='PAUSED':
                    status='PAUSED'
                elif (code==0 and isinstance(result,dict) and result.get('status')=='COMPLETE'
                      and result.get('source_sha256')==job['source_sha256']
                      and result.get('collection')==job['collection']
                      and result.get('source_point_count',0)>0
                      and result.get('source_point_count')==result.get('source_dense_count')
                      and result.get('source_point_count')==result.get('source_bm25_count')):
                    status='COMPLETE'
                else:
                    status='FAILED'
                    error = error or f'Worker exited with code {code}, without verified COMPLETE result'
                with connect() as db: patch(db,job_id,status=status,requested=0,
                                             stage=status,result=result if status=='COMPLETE' else None,error=error)
                if status == 'COMPLETE' or qdrant_started:
                    from rag3.ingestion_jobs import content_scope, mark_change
                    mark_change(content_scope(job['collection']))
                if status=='COMPLETE':
                    clear_ingestion_progress(job['collection'], job['source_sha256'])
                    self.retire_completed_upload(one(job_id,private=True))
                self.proc,self.active_id=None,None
        except Exception as exc:
            with self.lock:
                with connect() as db: patch(db,job_id,status='FAILED',requested=0,error=str(exc)[:400])
                self.proc,self.active_id=None,None


def dispatch(scheduler, request):
    action,p = request['action'],request.get('params',{})
    if action=='health': return {'alive':True,'active_job_id':scheduler.active_id}
    if action=='submit': return scheduler.submit(**p)
    if action=='action': return scheduler.action(p['job_id'],p['operation'])
    if action=='status': return one(p['job_id'])
    if action=='list': return list_jobs(**p)
    if action=='has_active_collection':
        with connect() as db:
            active=db.execute("SELECT 1 FROM jobs WHERE collection=? AND status NOT IN ('COMPLETE','CANCELLED','FAILED') LIMIT 1",(p['collection'],)).fetchone()
        return {'active':bool(active)}
    raise ValueError('invalid manager action')


def serve():
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args): pass
        def do_POST(self):
            if (self.path!='/command' or self.headers.get('Content-Type')!='application/json' or
                self.headers.get('X-RAG3-Token')!=self.server.token):
                self.send_error(403);return
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=48*1024*1024:
                    raise ValueError('request size outside limit')
                req=json.loads(self.rfile.read(length))
                result=dispatch(self.server.scheduler,req)
                payload={'ok':True,'result':result}
            except Exception as exc:
                payload={'ok':False,'error':str(exc)[:500]}
            data=json.dumps(payload,ensure_ascii=False).encode('utf-8')
            self.send_response(200);self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
    # Bind *before* constructing a Scheduler. Binding is the singleton guard.
    http = ThreadingHTTPServer((HOST,PORT),Handler)
    http.token = secrets.token_hex(32)
    TOKEN_FILE.parent.mkdir(parents=True,exist_ok=True)
    TOKEN_FILE.write_text(http.token,encoding='ascii')
    if os.name!='nt': TOKEN_FILE.chmod(0o600)
    http.scheduler = Scheduler()
    http.serve_forever()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--serve',action='store_true')
    args=parser.parse_args()
    if args.serve: serve()
