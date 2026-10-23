"""Stage C isolated scheduler lifecycle regression. No production DB/Qdrant writes."""
from __future__ import annotations
import hashlib,json,os,socket,sys,tempfile,threading,time,traceback
from pathlib import Path
from unittest.mock import patch
from datetime import datetime,timezone,timedelta
ROOT=Path(r'C:\python_program\RAG3')
sys.path.insert(0,str(ROOT))
from rag3 import ingestion_backend as C, ingestion_jobs as J
from rag3.ingestion_manager import client as jobs_client, install_backend_scheduler

checks=[]
def test(name, fn):
    try:
        info=fn()
        checks.append({'name':name,'ok':True,'details':info})
        print(name,'PASS',json.dumps(info,ensure_ascii=False,default=str))
    except Exception as exc:
        checks.append({'name':name,'ok':False,'error':str(exc),'stack':traceback.format_exc()[-2000:]})
        print(name,'FAIL',repr(exc))

def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1',0));return s.getsockname()[1]

def write_job(db,job_id,status,src,sha,requested=1):
    t=J.now()
    db.execute('''INSERT INTO jobs(job_id,origin,collection,filename,source,source_sha256,status,stage,created_at,updated_at,requested) VALUES(?,?,?,?,?,?,?,?,?,?,?)''',
      (job_id,'UI','test_tmp',src.name,str(src),sha,status,'old',t,t,requested))

def recover_queue_test():
    with tempfile.TemporaryDirectory(prefix='rag3-C-queued-') as td:
      td=Path(td); dbfile=td/'queue.sqlite'
      old=td/'old.txt'; old.write_text('stale queued job',encoding='utf8')
      ui=td/'ui.txt'; ui.write_text('new UI job different',encoding='utf8')
      mcp=td/'mcp.txt'; mcp.write_text('new MCP job different',encoding='utf8')
      sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
      with patch.object(J,'DB_PATH',dbfile), patch.object(C.BackendScheduler,'validate_collection',return_value=None):
        with J.connect() as db:
          write_job(db,'1'*32,'QUEUED',old,sha(old))
          write_job(db,'2'*32,'COMPLETE',old,sha(old),0)
        scheduler=C.BackendScheduler(object(),legacy_port=free_port())
        consumed=[];cond=threading.Event()
        def stub_start(row):
          consumed.append(row['job_id']);
          with J.connect() as db: J.patch(db,row['job_id'],status='COMPLETE',requested=0)
          cond.set()
        with patch.object(scheduler,'start',side_effect=stub_start):
          scheduler.boot()
          stale=J.one('1'*32,private=True)
          assert stale['status']=='QUEUED' and stale['requested']==0,stale
          assert J.one('2'*32,private=True)['status']=='COMPLETE'
          newui=scheduler.submit(origin='UI',collection='test_tmp',path=str(ui),requested=False)
          time.sleep(.12)
          assert newui['status']=='QUEUED' and not J.one(newui['job_id'],private=True)['requested']
          assert consumed==[],consumed
          newmcp=scheduler.submit(origin='MCP',collection='test_tmp',path=str(mcp),requested=True)
          assert cond.wait(3),'MCP event not handled'
          assert consumed==[newmcp['job_id']],consumed
          assert J.one(newui['job_id'],private=True)['status']=='QUEUED'
          cond.clear();scheduler.action(newui['job_id'],'run')
          assert cond.wait(3),'UI Run event not handled'
          assert consumed==[newmcp['job_id'],newui['job_id']],consumed
          assert J.one('1'*32,private=True)['requested']==0
          scheduler.stop(); scheduler._thread.join(2)
          assert not scheduler._thread.is_alive()
          return {'preexisting_queued_reset':True,'mcp_auto':True,'ui_waits_run':True,'run_order':consumed}

def restart_test():
    with tempfile.TemporaryDirectory(prefix='rag3-C-restart-') as td:
      td=Path(td);dbfile=td/'jobs.sqlite';src=td/'doc.txt';src.write_text('original',encoding='utf8')
      sha=hashlib.sha256(src.read_bytes()).hexdigest()
      with patch.object(J,'DB_PATH',dbfile):
        with J.connect() as db:
          write_job(db,'3'*32,'RUNNING',src,sha)
          other=td/'distinct.txt'; other.write_text('another source',encoding='utf8')
          write_job(db,'4'*32,'FAILED',other,hashlib.sha256(other.read_bytes()).hexdigest(),0)
        scheduler=C.BackendScheduler(object(),legacy_port=free_port())
        with patch.object(C,'_client',side_effect=AssertionError('should not access Qdrant')):
          try:scheduler.action('3'*32,'restart')
          except ValueError as e: assert 'not paused' in str(e),str(e)
          else:raise AssertionError('running Restart was allowed')
        fake=object()
        with patch.object(C,'_client',return_value=fake), patch.object(C,'_source_counts',return_value=(1,1,1,1)):
          try:scheduler.action('4'*32,'restart')
          except ValueError as e:assert 'ready' in str(e),str(e)
          else:raise AssertionError('ready-point Restart was allowed')
        assert J.one('4'*32,private=True)['status']=='FAILED'
        return {'running_rejected_before_qdrant':True,'ready_point_preserved':True}

def unavailable_backend_test():
    from rag3 import ingestion_manager as M
    with tempfile.TemporaryDirectory(prefix='rag3-C-backend-') as td:
      with patch.object(M,'_BACKEND_SCHEDULER',None),patch('rag3.backend_boundary._token_file',return_value=Path(td)/'not-present-token'):
        try:M.client('health')
        except RuntimeError as e: assert 'backend is not running' in str(e),str(e)
        else:raise AssertionError('backend unavailable but no error')
    return {'legacy_manager_autostart':False}



def scheduler_start_path_test():
    with tempfile.TemporaryDirectory(prefix='rag3-C-start-') as td:
      td=Path(td); dbfile=td/'start.sqlite'; src=td/'src.txt'
      src.write_text('start should hash Path not string',encoding='utf8')
      sha=hashlib.sha256(src.read_bytes()).hexdigest()
      with patch.object(J,'DB_PATH',dbfile):
        with J.connect() as db:write_job(db,'5'*32,'QUEUED',src,sha)
        scheduler=C.BackendScheduler(object(),legacy_port=free_port())
        row=J.one('5'*32,private=True)
        with patch.object(scheduler,'validate_collection',return_value=None),patch.object(scheduler,'_execute',return_value=None):
          scheduler.start(row)
          assert J.one('5'*32,private=True)['status']=='RUNNING'
          scheduler._task.join(2)
          assert not scheduler._task.is_alive()
        return {'start_validated_real_source_sha':True}

test('scheduler_start_hash_source_path',scheduler_start_path_test)
test('boot_recover_old_queue_then_new_mcp_ui_events',recover_queue_test)
test('restart_ready_points_rejected_without_mutation',restart_test)
test('missing_backend_does_not_autospawn',unavailable_backend_test)
print('RESULT',json.dumps({'checks':checks,'passed':sum(x['ok'] for x in checks),'failed':sum(not x['ok'] for x in checks)},ensure_ascii=False))
if not all(x['ok'] for x in checks):sys.exit(1)
