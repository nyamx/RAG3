"""UI and MCP share a single manager job and a brand-new temporary Qdrant collection."""
from __future__ import annotations
import importlib.util
import json
import os
import sys
import time
import uuid
from pathlib import Path
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from PySide6.QtWidgets import QApplication
ROOT=Path(__file__).resolve().parents[1]
from rag3.collections import create_collection,delete_collections
from rag3.ingestion_jobs import one
from rag3.ingestion_manager import client
coll='__rag3_ui_test_'+uuid.uuid4().hex[:12]
create_collection(coll)
spec=importlib.util.spec_from_file_location('ingest_ui_e2e',ROOT/'scripts'/'ingest-ui.py')
ui=importlib.util.module_from_spec(spec);spec.loader.exec_module(ui)
app=QApplication(sys.argv[:1])
win=None
try:
    folder=ROOT/'artifacts'/'ui-worker-e2e';folder.mkdir(parents=True,exist_ok=True)
    source=folder/('ui-'+uuid.uuid4().hex[:12]+'.md')
    source.write_text('# RAG3 shared manager UI evidence\nSapphire orchard omega.\n',encoding='utf-8')
    win=ui.IngestionWindow();win.collection.setText(coll)
    win.add_path(str(source))
    matches=[j for j in win.jobs if j['collection']==coll and j['filename']==source.name]
    assert len(matches)==1 and matches[0]['origin']=='UI'
    job_id=matches[0]['job_id']
    client('action',operation='run',job_id=job_id)
    deadline=time.monotonic()+450
    while time.monotonic()<deadline:
        app.processEvents();win.load_jobs()
        status=one(job_id)
        if status['status'] in {'COMPLETE','FAILED','CANCELLED'}:break
        time.sleep(1)
    assert status['status']=='COMPLETE',status
    assert status['source_sha256'] and not status['checkpoint_available']
    assert any(j['job_id']==job_id and j['status']=='COMPLETE' for j in win.jobs)
    print(json.dumps({'job_id':job_id,'status':status['status'],'result':status['result']},ensure_ascii=False))
finally:
    if win:win.close()
    # Never delete a test collection while a job is active.
    if 'job_id' in locals():
        status=one(job_id)
        if status['status'] not in {'COMPLETE','FAILED','CANCELLED'}:
            client('action',operation='cancel',job_id=job_id)
            for _ in range(30):
                if one(job_id)['status']=='CANCELLED':break
                time.sleep(.5)
    delete_collections([coll])
