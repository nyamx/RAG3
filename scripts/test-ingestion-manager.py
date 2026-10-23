"""Isolated shared-DB/queue semantics smoke. No Qdrant, GPU or external file deletion."""
from __future__ import annotations
import tempfile
import threading
from pathlib import Path
from rag3 import ingestion_jobs as db
from rag3.ingestion_manager import Scheduler

with tempfile.TemporaryDirectory() as folder:
    db.DB_PATH = Path(folder) / 'jobs.sqlite'
    m = Scheduler.__new__(Scheduler)
    m.lock=threading.RLock();m.proc=None;m.active_id=None;m.downloads={}
    m.qdrant_url='unused'
    m.validate_collection=lambda collection: None
    source=Path(folder)/'sample.md'; source.write_text('# test\nOne common queue.\n',encoding='utf-8')
    ui=m.submit(origin='UI',collection='test',path=str(source),requested=False)
    assert ui['origin']=='UI' and ui['status']=='QUEUED'
    assert ui['job_id'] in [j['job_id'] for j in db.list_jobs()['jobs']]
    assert not db.one(ui['job_id']).get('source')  # original local path is private
    assert db.one(ui['job_id'],private=True)['source']==str(source)
    try:
        m.submit(origin='MCP',collection='test',path=str(source),requested=True)
    except ValueError as exc:
        assert 'same SHA' in str(exc)
    else: raise AssertionError('duplicate work was accepted')
    paused=m.action(ui['job_id'],'pause')
    assert paused['status']=='PAUSED' and not paused['checkpoint_available']
    m.action(ui['job_id'],'resume')
    assert db.one(ui['job_id'])['status']=='QUEUED'  # queued pause does not pretend to have checkpoint
    m.action(ui['job_id'],'cancel')
    assert db.one(ui['job_id'])['status']=='CANCELLED'
    submitted=m.submit(origin='MCP',collection='test',path=str(source),requested=True)
    assert db.one(submitted['job_id'],private=True)['requested']
    with db.connect() as conn:
        db.patch(conn,submitted['job_id'],status='RUNNING')
    m.recover()
    assert db.one(submitted['job_id'])['status']=='FAILED'
    assert not db.one(submitted['job_id'],private=True)['requested']
    rows=db.list_jobs(collection='test',limit=1,offset=1)
    assert rows['total']==2 and len(rows['jobs'])==1
    assert 'source' not in rows['jobs'][0]
    assert len({j['job_id'] for j in db.list_jobs()['jobs']})==2
    # A second collection can reuse the digest once the previous job is
    # terminal, without violating the per-collection unique constraint.
    another=m.submit(origin='UI',collection='another_collection',path=str(source),requested=False)
    assert another['status']=='QUEUED'
    assert db.one(another['job_id'])['collection']=='another_collection'
    # Distinct sources may coexist as queued jobs; only scheduler.pump runs them.
    second=Path(folder)/'second.md'; second.write_text('Another source\n',encoding='utf-8')
    other=m.submit(origin='MCP',collection='test',path=str(second),requested=True)
    assert db.one(other['job_id'])['status']=='QUEUED'

print('INGESTION_MANAGER_JOURNAL_OK')
