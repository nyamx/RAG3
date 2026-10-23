"""A-stage isolated HTTP/MCP API regression test using a fake Qdrant endpoint.

Does not touch real Qdrant collections or the ingestion DB. The launched HTTP
backend and fake Qdrant are terminated after the test. Run from RAG3 venv.
"""
from __future__ import annotations
import asyncio, json, os, pathlib, socket, subprocess, sys, time, traceback, urllib.request, urllib.error, threading, hashlib
from contextlib import AsyncExitStack
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import psutil

ROOT = pathlib.Path(__file__).resolve().parents[1]
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
class FakeQdrant(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith('/collections'):
            payload = {'result': {'collections': [{'name': item} for item in COLLECTIONS]},
                       'status': 'ok', 'time': 0.001}
            body = json.dumps(payload).encode('utf8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)
    def log_message(self, *args):
        pass

qserver = ThreadingHTTPServer(('127.0.0.1', 0), FakeQdrant)
qthread = threading.Thread(target=qserver.serve_forever, daemon=True)
qthread.start()
with socket.socket() as probe:
    probe.bind(('127.0.0.1', 0))
    port = probe.getsockname()[1]
base = f'http://127.0.0.1:{port}'
env = os.environ.copy()
env.update({'RAG3_MCP_TRANSPORT':'streamable-http','RAG3_MCP_PORT':str(port),
            'RAG3_QDRANT_URL':f'http://127.0.0.1:{qserver.server_port}',
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
        status, result=request('/api/v1/command','POST',{'action':'collections.list'},token)
        assert status==200 and result.get('ok') and result.get('result',{}).get('collections')==COLLECTIONS,(status,result)
        print('AUTH_AND_API_OK 401/401/400/200',flush=True)
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
        asyncio.run(mcp_probe())
        print('STAGE_A_ISOLATED_RUNTIME_PASS',flush=True)
finally:
    if proc is not None:
        stop_tree(proc, int(health['pid']) if health else None)
    qserver.shutdown();qserver.server_close();qthread.join(timeout=3)
    try:
        with socket.create_connection(('127.0.0.1',port),timeout=.5):
            print('CLEANUP_FAILED active_port='+str(port),flush=True)
    except OSError:
        print('TEST_LISTENER_STOPPED',flush=True)
    if proc is not None and health:
        pid=health['pid']
        if safe_server_process(pid):print('WARNING_SERVER_STILL_ALIVE pid='+str(pid),flush=True)
