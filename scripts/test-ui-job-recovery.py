"""UI displays shared UI/MCP jobs after restart, without touching legacy ui-jobs.json."""
from __future__ import annotations
import importlib.util
import os
import sys
import tempfile
from pathlib import Path
os.environ['QT_QPA_PLATFORM']='offscreen'
ROOT=Path(__file__).resolve().parents[1]
from rag3 import ingestion_jobs as db
from PySide6.QtWidgets import QApplication
spec=importlib.util.spec_from_file_location('ingest_ui_shared',ROOT/'scripts'/'ingest-ui.py')
ui=importlib.util.module_from_spec(spec);spec.loader.exec_module(ui)
app=QApplication(sys.argv[:1])
ui.IngestionWindow.refresh_collections=lambda self: None
ui.IngestionWindow.periodic_collections=lambda self: None
with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as tmp:
    folder=Path(tmp)
    db.DB_PATH=folder/'ingestion_jobs.sqlite'
    ui.UI_STATE_PATH=folder/'ui-state.json'
    now=db.now()
    with db.connect() as c:
        for i,origin in enumerate(('UI','MCP')):
            c.execute('''INSERT INTO jobs(job_id,origin,collection,filename,source,status,stage,created_at,updated_at)
                      VALUES(?,?,?,?,?,?,?,?,?)''', (f'{i+1:032x}',origin,'test_collection',f'{origin}.md',str(folder/f'{origin}.md'),
                      'QUEUED','Ready',now,now))
        c.commit()
    first=ui.IngestionWindow()
    assert {j['origin'] for j in first.jobs}=={'UI','MCP'}
    first.table.selectRow(0)
    first.collection.addItem('test_collection')
    first.collection.setText('test_collection')
    assert len(first.jobs[0]['job_id'])==32
    # The first COMPLETE event must refresh an open contents tab, even if the
    # previous completed-job set was empty. No Qdrant calls are needed here.
    content_refreshes = []
    first.refresh_contents = lambda **_kwargs: content_refreshes.append(True)
    first.main_tabs.setCurrentWidget(first.contents_tab)
    with db.connect() as c:
        c.execute("UPDATE jobs SET status='COMPLETE' WHERE origin='UI'")
        c.commit()
    first.load_jobs()
    assert len(content_refreshes)==1, 'First completion did not refresh contents'
    first.close();app.processEvents()
    second=ui.IngestionWindow()
    assert {j['origin'] for j in second.jobs}=={'UI','MCP'}
    assert not (folder/'ui-jobs.json').exists()
    second.close();app.processEvents()
print('UI_JOB_RECOVERY_OK')
