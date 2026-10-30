"""A-stage isolated HTTP/MCP API regression test using a fake Qdrant endpoint.

Does not touch real Qdrant collections or the ingestion DB. The launched HTTP
backend and fake Qdrant are terminated after the test. Run from RAG3 venv.
"""
from __future__ import annotations
import asyncio, http.client, json, os, pathlib, socket, subprocess, sys, tempfile, time, traceback, urllib.request, urllib.error, threading, hashlib
from contextlib import AsyncExitStack
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlsplit
import psutil

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from rag3.backend_boundary import SessionCollections, single_instance

SERVER = ROOT / 'scripts' / 'rag3-mcp-server.py'
VENV = pathlib.Path(sys.executable)
assert SERVER.exists(), 'Expected RAG3 server not found'
print('SOURCE_SHA', hashlib.sha256(SERVER.read_bytes()).hexdigest(), flush=True)

def port_owner(port: int, pid: int) -> bool:
    try:
        return any(c.pid == pid and c.laddr and c.laddr.port == port and c.status == 'LISTEN'
                   for c in psutil.net_connections(kind='tcp'))
    except (psutil.Error, OSError):
        return False

def safe_server_process(pid: int) -> psutil.Process | None:
    try:
        p = psutil.Process(pid)
        args = p.cmdline()
        if len(args) >= 2 and pathlib.Path(args[-1]).resolve() == SERVER.resolve():
            return p
    except (psutil.Error, OSError, ValueError):
        pass
    return None

COLLECTIONS = ['__rag3_A_test_alpha','__rag3_A_test_beta']
FAKE_POINTS = {
    COLLECTIONS[0]: {'id': 101, 'payload': {'content': 'Alpha session source text.',
        'meta': {'document_id': 'alpha-document', 'title': 'Alpha source',
                 'page_start': 1, 'page_end': 1, 'chunk_index': 0,
                 'source_uri': 'file:///fake/alpha.pdf'}}},
    COLLECTIONS[1]: {'id': 202, 'payload': {'content': 'Beta session source text.',
        'meta': {'document_id': 'beta-document', 'title': 'Beta source',
                 'page_start': 2, 'page_end': 2, 'chunk_index': 0,
                 'source_uri': 'file:///fake/beta.pdf'}}},
}
FAKE_COLLECTION_INFO = {
    'status': 'green', 'optimizer_status': 'ok', 'segments_count': 1,
    'config': {'params': {'vectors': {}, 'sparse_vectors': {'bm25': {}}},
               'hnsw_config': {'m': 16, 'ef_construct': 100, 'full_scan_threshold': 10000},
               'optimizer_config': {'default_segment_number': 2, 'flush_interval_sec': 5},
               'metadata': {'rag3': {'bm25': {'k': 1.2, 'b': 0.75, 'avg_len': 100}}}},
    'payload_schema': {},
}


def test_selection_eviction() -> None:
    """An evicted or expired key must not inherit another session's choice."""
    available = COLLECTIONS[:]
    selections = SessionCollections(available[0], limit=2)
    selections.select('session-a', available[1])
    selections.select('session-b', available[1])
    selections.select('session-c', available[1])
    assert selections.get('session-a', available) is None, 'evicted selection was reused'

    expiring = SessionCollections(available[0], ttl=.001)
    expiring.select('expired-session', available[1])
    time.sleep(.02)
    assert expiring.get('expired-session', available) is None, 'expired selection was reused'
    print('SESSION_EVICTION_FAIL_CLOSED_OK', flush=True)


def test_workspace_lock() -> None:
    previous = os.environ.pop('RAG3_BACKEND_TEST_INSTANCE', None)
    try:
        with tempfile.TemporaryDirectory(prefix='rag3-A-lock-') as folder:
            root = pathlib.Path(folder)
            with single_instance(root, 8123):
                try:
                    with single_instance(root, 8124):
                        raise AssertionError('another port acquired the workspace lock')
                except RuntimeError:
                    pass
    finally:
        if previous is not None:
            os.environ['RAG3_BACKEND_TEST_INSTANCE'] = previous
    print('WORKSPACE_GLOBAL_LOCK_OK', flush=True)


class FakeQdrant(BaseHTTPRequestHandler):
    def _reply(self, result, *, status=200):
        body = json.dumps({'result': result, 'status': 'ok', 'time': 0.001}).encode('utf8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == '/collections':
            self._reply({'collections': [{'name': item} for item in COLLECTIONS]})
        elif path.startswith('/collections/'):
            parts = path.split('/')
            collection = parts[2] if len(parts) >= 3 else ''
            if collection not in FAKE_POINTS:
                self.send_error(404)
            elif len(parts) == 4 and parts[3] == 'exists':
                self._reply({'exists': True})
            elif len(parts) == 3:
                self._reply(FAKE_COLLECTION_INFO)
            else:
                self.send_error(404)
        else:
            self.send_error(404)

    def do_POST(self):
        path = urlsplit(self.path).path
        parts = path.split('/')
        if len(parts) < 4 or parts[1] != 'collections' or parts[2] not in FAKE_POINTS or parts[3] != 'points':
            self.send_error(404)
            return
        collection = parts[2]
        try:
            request_body = json.loads(self.rfile.read(int(self.headers.get('Content-Length', '0'))))
        except (ValueError, TypeError):
            self.send_error(400)
            return
        point = FAKE_POINTS[collection]
        if len(parts) == 5 and parts[4] == 'query':
            if request_body.get('using') != 'bm25':
                self.send_error(400, 'fake endpoint accepts BM25 only')
                return
            self._reply({'points': [{'id': point['id'], 'version': 1,
                                     'score': 0.9, 'payload': point['payload']}]})
        elif len(parts) == 5 and parts[4] == 'scroll':
            self._reply({'points': [point], 'next_page_offset': None})
        elif len(parts) == 4:
            ids = request_body.get('ids', [])
            self._reply([point] if point['id'] in ids else [])
        else:
            self.send_error(404)

    def log_message(self, *args):
        pass

test_selection_eviction()
test_workspace_lock()
qserver = ThreadingHTTPServer(('127.0.0.1', 0), FakeQdrant)
qthread = threading.Thread(target=qserver.serve_forever, daemon=True)
qthread.start()
while True:
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]
    if port != 8000:
        break
base = f'http://127.0.0.1:{port}'
env = os.environ.copy()
env.update({'RAG3_MCP_TRANSPORT':'streamable-http','RAG3_MCP_PORT':str(port),
            'RAG3_QDRANT_URL':f'http://127.0.0.1:{qserver.server_port}',
            'RAG3_BACKEND_TEST_INSTANCE':str(port),
            'RAG3_MCP_WRITE_ENABLED':'0', 'RAG3_MCP_DELETE_ENABLED':'0'})
flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0) if os.name == 'nt' else 0
logdir=ROOT/'artifacts'/'analysis';logdir.mkdir(parents=True, exist_ok=True)
logfile=logdir/f'A_runtime_verify_{port}.log'
proc=None
health=None

def request(path, method='GET', payload=None, token=None):
    headers={'Content-Type':'application/json'}
    if token is not None: headers['X-RAG3-Token']=token
    data=json.dumps(payload).encode('utf8') if payload is not None else None
    req=urllib.request.Request(base+path,data=data,headers=headers,method=method)
    try:
        with urllib.request.urlopen(req,timeout=12) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())

def stop_tree(launcher, target_pid):
    ids = set()
    if launcher is not None:
        try:
            p = psutil.Process(launcher.pid)
            ids.add(p.pid)
            for child in p.children(recursive=True): ids.add(child.pid)
        except psutil.Error: pass
    if target_pid and safe_server_process(target_pid) and port_owner(port,target_pid):
        ids.add(target_pid)
    procs=[]
    for pid in ids:
        try:
            p=psutil.Process(pid)
            if pid == launcher.pid or (safe_server_process(pid) is not None and (pid==target_pid or p.ppid()==launcher.pid)):
                procs.append(p)
        except psutil.Error: pass
    # Stop server children first and then launcher; leave unrelated processes alone.
    for p in procs:
        try: p.terminate()
        except psutil.Error: pass
    _, alive = psutil.wait_procs(procs, timeout=8)
    for p in alive:
        try: p.kill()
        except psutil.Error: pass
    psutil.wait_procs(alive, timeout=5)
    if launcher is not None:
        try: launcher.wait(timeout=3)
        except subprocess.TimeoutExpired: launcher.kill(); launcher.wait(timeout=3)

try:
    with logfile.open('wb') as out:
        proc=subprocess.Popen([str(VENV),str(SERVER)],cwd=ROOT,env=env,stdout=out,stderr=subprocess.STDOUT,creationflags=flags)
        for step in range(120):
            if proc.poll() is not None: raise RuntimeError(f'server launcher exited={proc.returncode}; log={logfile.read_text("utf8","replace")[-1500:]}')
            try:
                status,health=request('/health')
                if status==200: break
            except (OSError,ValueError): pass
            time.sleep(.25)
        if not health: raise RuntimeError('health unavailable; log='+logfile.read_text('utf8','replace')[-1500:])
        actual_pid = int(health['pid'])
        assert port_owner(port,actual_pid), f'health PID {actual_pid} does not own listener {port}'
        assert safe_server_process(actual_pid) is not None, 'unexpected listener owner'
        assert health['gpu_model']=='not_initialized',health
        assert health['qdrant_transport']=='reachable',health
        print('HEALTH_OK parent_pid=%s actual_server_pid=%s qdrant=fake_reachable gpu=%s' % (proc.pid,actual_pid,health['gpu_model']),flush=True)

        status, result=request('/api/v1/command','POST',{'action':'collections.list'})
        assert status==401,(status,result)
        status, result=request('/api/v1/command','POST',{'action':'collections.list'},'invalid')
        assert status==401,(status,result)
        token_path=ROOT/'data'/f'rag3-backend-ui-{port}.token'
        token=token_path.read_text(encoding='ascii').strip()
        assert len(token)==64
        status, result=request('/api/v1/command','POST',{'action':23},token)
        assert status==400,(status,result)
        status, result=request('/api/v1/command','POST',{'action':'search','params':{}},token)
        assert status==400 and result.get('error',{}).get('code')=='INVALID_ARGUMENT',(status,result)
        connection = http.client.HTTPConnection('127.0.0.1', port, timeout=12)
        try:
            connection.request('POST', '/api/v1/command',
                               body=iter([b' ' * 131072 for _ in range(3)]),
                               headers={'Content-Type':'application/json', 'X-RAG3-Token':token},
                               encode_chunked=True)
            response = connection.getresponse()
            oversized = json.loads(response.read())
            assert response.status==413 and oversized.get('error',{}).get('code')=='PAYLOAD_TOO_LARGE', (response.status, oversized)
        finally:
            connection.close()
        status, result=request('/api/v1/command','POST',{'action':'collections.list'},token)
        assert status==200 and result.get('ok') and result.get('result',{}).get('collections')==COLLECTIONS,(status,result)
        print('AUTH_AND_API_OK 401/401/400/400/413/200',flush=True)
        # Duplicate server must exit without rotating a valid token.
        with (logdir/f'A_duplicate_{port}.log').open('wb') as duplicate_log:
            duplicate=subprocess.Popen([str(VENV),str(SERVER)],cwd=ROOT,env=env,stdout=duplicate_log,stderr=subprocess.STDOUT,creationflags=flags)
            try:
                dupcode=duplicate.wait(timeout=45)
                assert dupcode!=0, 'duplicate startup succeeded'
            finally:
                if duplicate.poll() is None: stop_tree(duplicate,None)
        assert token_path.read_text('ascii').strip()==token, 'duplicate rotated token'
        assert port_owner(port,actual_pid), 'original listener lost'
        print('DUPLICATE_LOCK_OK no_token_rotation exit_code='+str(dupcode),flush=True)

        with socket.socket() as probe:
            probe.bind(('127.0.0.1', 0))
            other_port = probe.getsockname()[1]
        other_env = env.copy()
        other_env['RAG3_MCP_PORT'] = str(other_port)
        with (logdir/f'A_duplicate_other_port_{other_port}.log').open('wb') as duplicate_log:
            duplicate=subprocess.Popen([str(VENV),str(SERVER)],cwd=ROOT,env=other_env,
                                       stdout=duplicate_log,stderr=subprocess.STDOUT,
                                       creationflags=flags)
            try:
                dupcode=duplicate.wait(timeout=45)
                assert dupcode!=0, 'second backend started on another port'
            finally:
                if duplicate.poll() is None: stop_tree(duplicate,None)
        assert port_owner(port,actual_pid), 'original listener lost after other-port duplicate'
        print('DUPLICATE_OTHER_PORT_LOCK_OK exit_code='+str(dupcode),flush=True)

        async def mcp_probe():
            from mcp import ClientSession
            from mcp.client.streamable_http import streamable_http_client
            async with AsyncExitStack() as stack:
                async def connect():
                    streams=await stack.enter_async_context(streamable_http_client(base+'/mcp'))
                    session=await stack.enter_async_context(ClientSession(streams[0],streams[1]))
                    await session.initialize()
                    return session
                a=await connect(); b=await connect()
                tools=await a.list_tools()
                names={tool.name:tool for tool in tools.tools}
                for name in ['rag3_search','rag3_fetch','rag3_list_collections','rag3_select_collection','rag3_get_source']:
                    assert name in names, ('missing tool',name)
                for name in ['rag3_search','rag3_fetch','rag3_search_advanced','rag3_search_flexible','rag3_select_collection','rag3_list_collections']:
                    assert name in names,('missing tool',name)
                    assert 'ctx' not in names[name].input_schema.get('properties',{}),(name,names[name].input_schema)
                for name in ['rag3_search','rag3_search_advanced','rag3_search_flexible']:
                    assert 'collections' in names[name].input_schema['required'], names[name].input_schema
                assert 'collection' in names['rag3_fetch'].input_schema['required']
                print('MCP_SCHEMA_OK tool_count='+str(len(names)),flush=True)
                async def call(session,name,args=None):
                    reply=await session.call_tool(name,args or {})
                    if reply.is_error: raise RuntimeError(f'{name} failed {reply.content!s}'[:600])
                    body=reply.structured_content
                    if body is None: body=json.loads(reply.content[0].text)
                    return body
                first=await call(a,'rag3_list_collections')
                assert first['collections']==COLLECTIONS,first
                await call(a,'rag3_select_collection',{'collection':COLLECTIONS[0]})
                await call(b,'rag3_select_collection',{'collection':COLLECTIONS[1]})
                for j in range(5):
                    aa,bb=await asyncio.gather(call(a,'rag3_list_collections'),call(b,'rag3_list_collections'))
                    assert aa['selected']==COLLECTIONS[0] and bb['selected']==COLLECTIONS[1],(aa,bb)
                print('MCP_TWO_SESSIONS_ISOLATED_OK calls=10',flush=True)

                async def search(session, collection):
                    args={'query':'source', 'semantic':False, 'bm25':True,
                          'rank_fusion':False, 'top_k':1,
                          'collections':[collection]}
                    result=await call(session,'rag3_search_advanced',args)
                    assert len(result['results'])==1,result
                    return result['results'][0]

                alpha,beta=await asyncio.gather(search(a,COLLECTIONS[0]),search(b,COLLECTIONS[1]))
                assert alpha['title']=='Alpha source' and beta['title']=='Beta source',(alpha,beta)
                explicit_beta=await search(a,COLLECTIONS[1])
                assert explicit_beta['title']=='Beta source',explicit_beta
                assert (await search(a,COLLECTIONS[0]))['id']==alpha['id']
                both=await call(a,'rag3_search_advanced',{'query':'source','semantic':False,
                    'bm25':True,'rank_fusion':False,'top_k':1,'collections':COLLECTIONS})
                assert [hit['collection'] for hit in both['results']]==COLLECTIONS,both
                invalid=await a.call_tool('rag3_search',{'query':'source','collections':['missing']})
                assert invalid.is_error and 'rag3_list_collections' in invalid.content[0].text,invalid

                fetched_a,fetched_b=await asyncio.gather(
                    call(a,'rag3_fetch',{'id':alpha['id'],'collection':COLLECTIONS[0]}),
                    call(b,'rag3_fetch',{'id':beta['id'],'collection':COLLECTIONS[1]}))
                assert fetched_a['text']=='Alpha session source text.',fetched_a
                assert fetched_b['text']=='Beta session source text.',fetched_b
                explicit_fetch=await call(a,'rag3_fetch',{'id':beta['id'],'collection':COLLECTIONS[1]})
                assert explicit_fetch['text']==fetched_b['text'],explicit_fetch
                assert (await call(a,'rag3_fetch',{'id':alpha['id'],'collection':COLLECTIONS[0]}))['text']==fetched_a['text']
                wrong_collection=await a.call_tool('rag3_fetch',{'id':beta['id'],'collection':COLLECTIONS[0]})
                assert wrong_collection.is_error,wrong_collection
                error=json.loads(wrong_collection.content[0].text)['error']
                assert error['code']=='STALE_ID',error
                print('MCP_SEARCH_FETCH_ISOLATED_OK explicit_collection_precedence=verified',flush=True)
        asyncio.run(mcp_probe())
        print('STAGE_A_ISOLATED_RUNTIME_PASS',flush=True)
finally:
    if proc is not None:
        stop_tree(proc, int(health['pid']) if health else None)
    qserver.shutdown();qserver.server_close();qthread.join(timeout=3)
    listener_stopped = False
    try:
        with socket.create_connection(('127.0.0.1',port),timeout=.5):
            print('CLEANUP_FAILED active_port='+str(port),flush=True)
    except OSError:
        listener_stopped = True
        print('TEST_LISTENER_STOPPED',flush=True)
    if (listener_stopped and 'token' in locals() and
            (not health or safe_server_process(int(health['pid'])) is None)):
        test_token = ROOT/'data'/f'rag3-backend-ui-{port}.token'
        if test_token.is_file() and test_token.read_text('ascii').strip() == token:
            test_token.unlink()
    if proc is not None and health:
        pid=health['pid']
        if safe_server_process(pid):print('WARNING_SERVER_STILL_ALIVE pid='+str(pid),flush=True)
