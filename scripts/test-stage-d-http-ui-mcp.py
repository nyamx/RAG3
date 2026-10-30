"""Stage D offscreen *actual* Qt -> authenticated HTTP + actual MCP over fake isolated Qdrant.
No changes to production DB/collections. Run only within RAG3 venv, offscreen.
"""
from __future__ import annotations
import asyncio, importlib.util, json, os, pathlib, socket, subprocess, sys, tempfile, threading, time, urllib.error, urllib.request, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit
from contextlib import suppress

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
COL = '__rag3_stage_d_wire_' + uuid.uuid4().hex[:10]
POINTID = str(uuid.uuid4())
BODY = 'Stage D isolated live wire context text about pine trees.'

class FakeQdrant(BaseHTTPRequestHandler):
    def respond(self, result):
        raw=json.dumps({'result':result,'status':'ok','time':0.001}).encode()
        self.send_response(200); self.send_header('Content-Type','application/json'); self.send_header('Content-Length',str(len(raw))); self.end_headers(); self.wfile.write(raw)
    def do_GET(self):
        p=urlsplit(self.path).path
        if p == '/collections':
            self.respond({'collections':[{'name':COL}]})
        elif p == '/collections/'+COL+'/exists':
            self.respond({'exists':True})
        elif p == '/collections/'+COL:
            self.respond({'status':'green','optimizer_status':'ok','segments_count':1,
                'config':{'params':{'vectors':{'text-dense':{'size':1024,'distance':'Cosine'}},
                 'sparse_vectors':{'bm25':{'modifier':'idf'}}},
                 'metadata':{'rag3':{'bm25':{'k':1.2,'b':0.75,'avg_len':256}}}} , 'payload_schema':{}})
        else: self.send_error(404)
    def do_POST(self):
        p=urlsplit(self.path).path
        try: payload=json.loads(self.rfile.read(int(self.headers.get('Content-Length',0))))
        except Exception: self.send_error(400);return
        point={'id':POINTID,'version':1,'score':0.85,'payload':{
            'content':BODY,'meta':{'document_id':'stage-d-doc','title':'Stage D fake source',
             'page_start':1,'page_end':1,'chunk_index':0,'section_paths':['1.1'],
             'source_uri':'file:///test_only/stage-d.txt','_rag3_ingest_state':'ready'}}}
        if p == '/collections/'+COL+'/points/query' and payload.get('using')=='bm25':
            self.respond({'points':[point]})
        elif p == '/collections/'+COL+'/points/scroll':
            self.respond({'points':[point],'next_page_offset':None})
        elif p == '/collections/'+COL+'/points':
            self.respond([point] if POINTID in payload.get('ids',[]) else [])
        else: self.send_error(404)
    def log_message(self,*a): pass

def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1',0));return s.getsockname()[1]

def req(base,path,token=None,payload=None):
    headers={'Content-Type':'application/json'}
    if token: headers['X-RAG3-Token']=token
    body=None if payload is None else json.dumps(payload).encode()
    request=urllib.request.Request(base+path,data=body,headers=headers,method='GET' if payload is None else 'POST')
    try:
        with urllib.request.urlopen(request,timeout=15) as response:
            return response.status,json.loads(response.read())
    except urllib.error.HTTPError as e: return e.code,json.loads(e.read())

async def mcp_query(url):
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client
    async with streamable_http_client(url) as streams:
        async with ClientSession(streams[0],streams[1]) as session:
            await session.initialize()
            tools=await session.list_tools()
            assert 'rag3_search_advanced' in {x.name for x in tools.tools}
            response=await session.call_tool('rag3_search_advanced',{'query':'pine',
                'semantic':False,'bm25':True,'rank_fusion':False,'top_k':3,'collections':[COL]})
            assert not response.is_error, str(response)
            return response.structured_content or json.loads(response.content[0].text)

def main():
    q=ThreadingHTTPServer(('127.0.0.1',0),FakeQdrant)
    thread=threading.Thread(target=q.serve_forever,daemon=True);thread.start()
    port=free_port()
    env=os.environ.copy()
    env.update(RAG3_MCP_PORT=str(port),RAG3_BACKEND_TEST_INSTANCE=str(port),RAG3_TEST_DATA_ROOT='',
               RAG3_QDRANT_URL=f'http://127.0.0.1:{q.server_port}', RAG3_MCP_TRANSPORT='streamable-http',
               RAG3_MCP_WRITE_ENABLED='0', RAG3_MCP_DELETE_ENABLED='0', QT_QPA_PLATFORM='offscreen')
    token_file=ROOT/'data'/f'rag3-backend-ui-{port}.token'
    lock_file=ROOT/'data'/f'backend-test-{port}.lock'
    assert not token_file.exists() and not lock_file.exists(), 'Test port already has backend state'
    proc=None
    with tempfile.TemporaryDirectory(prefix='stage-d-wire-') as tmp:
        env['RAG3_TEST_DATA_ROOT']=str(pathlib.Path(tmp)/'isolated-data')
        log=pathlib.Path(tmp)/'server.log'
        try:
            flags=getattr(subprocess,'CREATE_NO_WINDOW',0) if os.name=='nt' else 0
            with log.open('wb') as output:
                proc=subprocess.Popen([sys.executable,str(ROOT/'scripts'/'rag3-mcp-server.py')],
                    cwd=ROOT,env=env,stdout=output,stderr=subprocess.STDOUT,creationflags=flags)
                base=f'http://127.0.0.1:{port}'
                for _ in range(150):
                    assert proc.poll() is None, 'backend failed: '+log.read_text('utf8','replace')[-1600:]
                    try:
                        status,health=req(base,'/health')
                        if status==200: break
                    except (OSError,ValueError):pass
                    time.sleep(.2)
                else: raise AssertionError('Backend health startup timeout')
                assert health['gpu_model']=='not_initialized',health
                assert token_file.exists(),'No port-scoped local token'
                token=token_file.read_text(encoding='ascii').strip()
                status,_=req(base,'/api/v1/command',payload={'action':'collections.list','params':{}})
                assert status==401, status
                status,api=req(base,'/api/v1/command',token,{'action':'search','params':{
                    'query':'pine','collection':COL,'options':{'semantic':False,'bm25':True,'rank_fusion':False,'top_k':3}}})
                assert status==200 and api.get('ok'),(status,api)
                api=api['result']; assert len(api['results'])==1 and len(api['ui']['hits'])==1,api
                assert api['ui']['hits'][0]['content']==BODY,api
                print('WIRE_UI_API_BM25 PASS',flush=True)
                mcp=asyncio.run(mcp_query(base+'/mcp'))
                assert [v['id'] for v in mcp['results']]==[v['id'] for v in api['results']],(mcp,api)
                print('WIRE_MCP_UI_STABLE_ID_PARITY PASS',flush=True)
                os.environ.update({'QT_QPA_PLATFORM':'offscreen','RAG3_MCP_PORT':str(port),'RAG3_BACKEND_TEST_INSTANCE':str(port),'RAG3_TEST_DATA_ROOT':env['RAG3_TEST_DATA_ROOT']})
                from PySide6.QtWidgets import QApplication
                spec=importlib.util.spec_from_file_location('stage_d_live_ui',ROOT/'scripts'/'ingest-ui-stage-d.py')
                ui=importlib.util.module_from_spec(spec);sys.modules[spec.name]=ui;spec.loader.exec_module(ui)
                app=QApplication.instance() or QApplication([])
                window=ui.IngestionWindow();window.collection.setText(COL)
                window.query.setText('pine');window.retrieval_semantic.setChecked(False)
                window.retrieval_bm25.setChecked(True);window.retrieval_rank_fusion.setChecked(False)
                window.retrieval_top_k.setValue(3)
                window.run_retrieval()
                for _ in range(200):
                    app.processEvents()
                    if not window._retrieval_pending:break
                    time.sleep(.05)
                assert not window._retrieval_pending,'Qt GUI HTTP callback timed out'
                text=window.retrieval_results.toPlainText()
                assert BODY in text and '1.1' in text,(window.retrieval_status.text(),text)
                print('WIRE_REAL_QT_UI_RENDER PASS',flush=True)
                window.close()
                status,final_health=req(base,'/health');assert status==200 and final_health['pid']==health['pid'],(health,final_health)
                assert final_health['gpu_model']=='not_initialized',final_health
                print('WIRE_SINGLE_BACKEND_COLD_BM25 PASS',flush=True)
                from rag3 import ingestion_jobs
                db=pathlib.Path(env['RAG3_TEST_DATA_ROOT'])/'ingestion_jobs.sqlite'
                assert db.exists(), 'Isolated sqlite not found'
                assert not (ROOT/'data'/'ingestion_jobs.sqlite').samefile(db)
                print('ISOLATION_DB PASS',flush=True)
        finally:
            if proc and proc.poll() is None:
                proc.terminate()
                try:proc.wait(timeout=8)
                except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=8)
            q.shutdown();q.server_close()
            # Each token is generated for a newly chosen test port only.
            if token_file.exists(): token_file.unlink()
            if lock_file.exists(): lock_file.unlink()
            if log.exists() and proc and proc.returncode not in (0,-15,1,3221225786):
                print('BACKEND_TAIL',log.read_text('utf8','replace')[-700:],flush=True)
    print('STAGE_D_LIVE_WIRE_FAKE_QDRANT PASS; no production collection or job data touched',flush=True)

if __name__=='__main__': main()
