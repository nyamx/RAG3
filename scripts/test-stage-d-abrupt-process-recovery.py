"""Crash-process regression, isolated DB, fake worker and fake Qdrant.

A synthetic in-flight job is killed through real OS process termination. This
is *not* an end-to-end interrupted Docling/CUDA/Qdrant ingest test.
"""
from __future__ import annotations
import hashlib, json, os, socket, sqlite3, subprocess, sys, tempfile, time, uuid
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
PY=Path(sys.executable)

def assert_isolation(root):
    from rag3 import ingestion_jobs as J, ingestion_backend as C
    target=Path(root).resolve()
    if J.DB_PATH.resolve().is_relative_to((ROOT/'data').resolve()):
        raise AssertionError('would use operational database')
    if not J.DB_PATH.resolve().is_relative_to(target):
        raise AssertionError(f'test DB is not under isolated root: {J.DB_PATH}')
    if not os.environ.get('RAG3_BACKEND_TEST_INSTANCE'):
        raise AssertionError('missing test instance scope')
    assert os.environ.get('RAG3_QDRANT_URL')=='http://127.0.0.1:1'
    return J,C

def seed(job_id, original, sha):
    J,C=assert_isolation(original.parent.parent)
    from rag3.ingestion_jobs import now
    with J.connect() as conn:
        conn.execute("""INSERT INTO jobs (job_id,origin,collection,filename,source,
                 source_sha256,status,stage,created_at,updated_at,requested)
                 VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                 (job_id,'UI','isolated_crash_case',original.name,str(original),
                 sha,'QUEUED','Queued',now(),now(),1))
    from rag3.ingestion_settings import progress_identity
    J.set_ingestion_progress('isolated_crash_case',sha,
        progress_identity(sha,'isolated_crash_case'),'EMBEDDING',8)
    return J,C

def sleep_task(job, marker):
    marker.write_text('worker_started',encoding='ascii')
    while True: time.sleep(.1)

def victim(root, job_id):
    data=Path(root)
    os.environ['RAG3_TEST_DATA_ROOT']=str(data)
    J,C=assert_isolation(data)
    original=data/'original'/'isolated-source.txt'
    sha=hashlib.sha256(original.read_bytes()).hexdigest()
    J,C=seed(job_id,original,sha)
    marker=data/'victim_worker_started'
    def fake_execute(job): sleep_task(job, marker)
    with socket.socket() as s:
        s.bind(('127.0.0.1',0)); legacy_port=s.getsockname()[1]
    scheduler=C.BackendScheduler(object(),legacy_port=legacy_port)
    with patch.object(scheduler,'validate_collection',return_value=None), \
         patch.object(scheduler,'_execute',side_effect=fake_execute):
        scheduler.boot()
        # Startup recovery intentionally clears requested for queued jobs.
        # Request Run only after recovery has completed.
        assert J.one(job_id,private=True)['requested'] is False
        scheduler.action(job_id,'run')
        for _ in range(100):
            if J.one(job_id,private=True)['status']=='RUNNING' and marker.exists():
                (data/'victim_ready').write_text(str(os.getpid()),encoding='ascii')
                break
            time.sleep(.05)
        else:
            state=J.one(job_id,private=True)
            raise AssertionError('fake in-flight worker did not reach RUNNING: '+
                                 repr({k:state.get(k) for k in ('status','stage','error','requested')}))
        while True: time.sleep(.5)

def cold_recovery(data,job_id,sha):
    J,C=assert_isolation(data)
    from rag3.ingestion_jobs import checkpoint_available
    old=J.one(job_id,private=True)
    assert old['status']=='RUNNING',old['status']
    assert old['source_sha256']==sha
    progress_before=J.ingestion_progress('isolated_crash_case',sha)
    assert progress_before and progress_before['phase']=='EMBEDDING'
    scheduler=C.BackendScheduler(object())
    with patch.object(C,'_compatible',side_effect=ConnectionError('injected Qdrant outage')):
        scheduler.recover()
    paused=J.one(job_id,private=True)
    assert paused['status']=='PAUSED',paused
    assert paused['requested']==False
    assert paused['checkpoint_available']
    assert J.ingestion_progress('isolated_crash_case',sha)==progress_before
    assert hashlib.sha256(Path(paused['source']).read_bytes()).hexdigest()==sha
    with patch.object(C,'_compatible',return_value=None):
        result=scheduler.action(job_id,'resume')
    queued=J.one(job_id,private=True)
    assert queued['status']=='QUEUED' and queued['mode']=='resume' and queued['requested']
    assert result['job_id']==job_id
    assert J.ingestion_progress('isolated_crash_case',sha)==progress_before
    with J.connect() as conn:
        assert conn.execute('PRAGMA quick_check').fetchone()[0]=='ok'
        assert conn.execute('SELECT COUNT(*) FROM jobs').fetchone()[0]==1
    return {'real_os_termination':True,'synthetic_running_worker':True,
            'outage_deferred_checkpoint_verification':True,'manual_resume_queued':True,
            'source_sha256_preserved':True,'progress_preserved':True,
            'isolated_db_integrity':'ok'}

def main():
    assert os.name=='nt','Windows-only process tree kill regression'
    with tempfile.TemporaryDirectory(prefix='rag3-stage-d-crash-') as td:
        data=Path(td)/'data'; (data/'original').mkdir(parents=True)
        original=data/'original'/'isolated-source.txt'
        original.write_bytes(b'synthetic original for OS-kill process recovery\n'*4)
        sha=hashlib.sha256(original.read_bytes()).hexdigest()
        job_id=uuid.uuid4().hex
        env=os.environ.copy()
        env.update(RAG3_TEST_DATA_ROOT=str(data), RAG3_BACKEND_TEST_INSTANCE='stage_d_crash_test',
                   RAG3_QDRANT_URL='http://127.0.0.1:1',QDRANT_URL='http://127.0.0.1:1',
                   PYTHONUTF8='1',PYTHONIOENCODING='utf-8',PYTHONPATH=str(ROOT))
        victim_proc=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),
                       '--victim',str(data),job_id],cwd=ROOT,env=env,
                       stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,
                       stderr=subprocess.PIPE,
                       creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        try:
            for _ in range(160):
                if (data/'victim_ready').exists():break
                if victim_proc.poll() is not None:
                    err=victim_proc.stderr.read().decode('utf-8','replace')[-1800:]
                    raise AssertionError(f'victim exit={victim_proc.returncode} before RUNNING: {err}')
                time.sleep(.05)
            else: raise AssertionError('victim startup timeout')
            assert (data/'victim_worker_started').is_file()
            assert victim_proc.poll() is None
            proc_kill=subprocess.run(['taskkill','/F','/T','/PID',str(victim_proc.pid)],
                         capture_output=True,timeout=12)
            if proc_kill.returncode!=0:
                raise AssertionError('failed to terminate test-only process tree')
            victim_proc.wait(timeout=8)
            assert victim_proc.poll() is not None
            os.environ.update(RAG3_TEST_DATA_ROOT=str(data),RAG3_BACKEND_TEST_INSTANCE='stage_d_crash_test',
                              RAG3_QDRANT_URL='http://127.0.0.1:1')
            results=cold_recovery(data,job_id,sha)
            print(json.dumps({'status':'PASS','checks':results},sort_keys=True))
        finally:
            if victim_proc.poll() is None:
                subprocess.run(['taskkill','/F','/T','/PID',str(victim_proc.pid)],
                               capture_output=True,timeout=10)
                victim_proc.wait(timeout=10)

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--victim': victim(sys.argv[2],sys.argv[3])
    else: main()
