"""Backend-owned, single-generation Docling parser service.

Only stdlib is imported in the backend. Docling stays inside ingest-worker.py.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from rag3.constants import ROOT


class ParserUnavailable(RuntimeError):
    pass


class PersistentCPUParser:
    def __init__(self, stage_callback=None, *, ready_timeout=240, parse_timeout=3600,
                 device='cpu'):
        if device not in ('cpu', 'cuda'):
            raise ValueError('parser device must be cpu or cuda')
        self._device = device
        self._lock = threading.RLock()
        self._ready = threading.Event()
        self._done = threading.Event()
        self._restart = threading.Event()
        self._stopped = threading.Event()
        self._manager = None
        self._proc = None
        self._job = None
        self._reader = None
        self._logger = None
        self._protocol = None
        self._state = 'stopped'
        self._error = None
        self._generation = 0
        self._restart_count = 0
        self._automatic_restarts = 0
        self._fatal_initialization = False
        self._request_id = None
        self._job_id = None
        self._response = None
        self._stage = stage_callback
        self._ready_timeout = ready_timeout
        self._parse_timeout = parse_timeout
        self._diagnostics = []
        self._prepared = {}

    def snapshot(self):
        with self._lock:
            proc = self._proc
            return {'state': self._state, 'pid': proc.pid if proc is not None and proc.poll() is None else None,
                    'ready': self._state in ('ready','busy'),
                    'device': self._device,
                    'restart_count': self._restart_count, 'error_code': self._error,
                    'generation': self._generation,
                    'prepared_pipeline_count': len(set(self._prepared.values()))}

    def start(self):
        with self._lock:
            if self._manager is not None:
                raise RuntimeError('CPU parser already started')
            self._stopped.clear()
            self._restart.set()
            self._manager = threading.Thread(target=self._supervise, name='rag3-parser-supervisor', daemon=True)
            self._manager.start()

    def _supervise(self):
        while not self._stopped.is_set():
            if not self._restart.wait(timeout=.5):
                continue
            self._restart.clear()
            if self._stopped.is_set():
                break
            try:
                self._retire()
                if self._stopped.is_set():
                    break
                self._spawn()
            except Exception as exc:
                with self._lock:
                    self._state = 'failed'
                    self._error = 'PARSER_START_FAILED'
                    self._ready.set()
                    self._done.set()
                    self._diagnostics.append(type(exc).__name__)
                # No uncontrolled retry loop. Subsequent cancellation/manual restart
                # may request exactly one more generation.

    def _spawn(self):
        with self._lock:
            if self._proc is not None:
                raise RuntimeError('previous parser not retired')
            self._generation += 1
            gen = self._generation
            self._state = 'starting' if gen == 1 else 'restarting'
            self._error = None
            self._ready.clear()
            self._done.clear()
            self._prepared = {}
            self._fatal_initialization = False
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES='0' if self._device == 'cuda' else '-1',
                   RAG3_PARSE_DEVICE=self._device,
                   RAG3_PARSE_CPU_ONLY='1' if self._device == 'cpu' else '0',
                   OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                   HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        rfd, wfd = os.pipe()
        proc = None
        job = None
        try:
            startup = {}
            if os.name == 'nt':
                import msvcrt
                from rag3.windows_job import WindowsParserJob
                job = WindowsParserJob()
                handle = msvcrt.get_osfhandle(wfd)
                os.set_handle_inheritable(handle, True)
                info = subprocess.STARTUPINFO()
                info.lpAttributeList = {'handle_list': [handle]}
                startup = {'startupinfo': info, 'close_fds': True,
                           'creationflags': getattr(subprocess, 'CREATE_NO_WINDOW', 0)}
                env['RAG3_PARSER_RESPONSE_HANDLE'] = str(handle)
                env['RAG3_PARSE_PARENT_GATE'] = '1'
            else:
                os.set_inheritable(wfd, True)
                startup = {'pass_fds': (wfd,), 'close_fds': True, 'start_new_session': True}
                env['RAG3_PARSER_RESPONSE_FD'] = str(wfd)
            proc = subprocess.Popen(
                [sys.executable, str(ROOT/'scripts'/'ingest-worker.py'), '--parser-service'],
                cwd=ROOT, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace',
                bufsize=1, **startup)
            os.close(wfd)
            wfd = -1
            if job is not None:
                job.assign(proc.pid)  # Never open the gate before Job Object assignment.
            with self._lock:
                if self._stopped.is_set():
                    raise RuntimeError('service stopped while launching')
                self._proc = proc
                self._job = job
                self._protocol = os.fdopen(rfd, 'r', encoding='utf-8', errors='replace')
                rfd = -1
                self._reader = threading.Thread(target=self._read_protocol, args=(gen,),
                                                name='rag3-parser-protocol', daemon=True)
                self._logger = threading.Thread(target=self._drain_logs, args=(proc,),
                                                name='rag3-parser-logs', daemon=True)
                self._reader.start()
                self._logger.start()
            if job is not None:
                self._write('G')  # Same stdin: 1-byte parent gate followed by JSON lines.
        except BaseException:
            if proc is not None:
                self._kill(proc, job)
                try: proc.wait(timeout=8)
                except subprocess.TimeoutExpired: proc.kill()
            if job is not None: job.close()
            if rfd >= 0: os.close(rfd)
            if wfd >= 0: os.close(wfd)
            raise

    @staticmethod
    def _kill(proc, job):
        if job is not None:
            try:
                job.terminate()
                return
            except OSError:
                pass
        if proc.poll() is None:
            proc.kill()

    def _write(self, content):
        proc = self._proc
        if proc is None or proc.stdin is None:
            raise ParserUnavailable('CPU parser stdin unavailable')
        proc.stdin.write(content)
        proc.stdin.flush()

    def _drain_logs(self, proc):
        # Native stdout/stderr never share the framing channel.
        for stream in (proc.stdout, proc.stderr):
            if stream is None: continue
            # Drain both streams independently so one filled pipe cannot deadlock.
            threading.Thread(target=self._drain_one, args=(stream,), daemon=True).start()

    def _drain_one(self, stream):
        try:
            for line in stream:
                with self._lock:
                    if len(self._diagnostics) >= 30: self._diagnostics.pop(0)
                    self._diagnostics.append(line[:240])
        except OSError:
            pass

    def _read_protocol(self, gen):
        stream = self._protocol
        try:
            for line in stream:
                if len(line) > 65536:
                    raise RuntimeError('oversized parser protocol event')
                evt = json.loads(line)
                if not isinstance(evt, dict): raise RuntimeError('invalid parser event')
                cb = None
                with self._lock:
                    if self._generation != gen:
                        return
                    kind = evt.get('type')
                    if kind == 'ready' and self._state in ('starting','restarting'):
                        self._prepared = evt.get('pipelines', {}) if isinstance(evt.get('pipelines'), dict) else {}
                        self._state = 'ready'
                        self._ready.set()
                    elif kind == 'fatal':
                        self._fatal_initialization = True
                        self._state = 'failed'
                        self._error = 'PARSER_INITIALIZATION_FAILED'
                        self._ready.set()
                        self._done.set()
                    elif kind not in ('ready','result','error','log','fatal','paused'):
                        raise RuntimeError('unexpected parser protocol event')
                    elif kind in ('result','error','log') and (self._state == 'busy' and
                            evt.get('request_id') == self._request_id):
                        if kind == 'log' and self._stage is not None:
                            cb = (self._job_id, str(evt.get('message', ''))[:200])
                        if kind in ('result','error'):
                            self._response = evt
                            self._done.set()
                if cb:
                    try: self._stage(*cb)
                    except Exception: pass
        except (OSError, ValueError, RuntimeError):
            pass
        finally:
            with self._lock:
                if self._generation == gen and not self._stopped.is_set():
                    # Invalidate this reader before signalling the supervisor.
                    # A late EOF from the retired child cannot request a second
                    # restart while the replacement is being launched.
                    self._generation += 1
                    self._error = 'PARSER_PROTOCOL_CLOSED'
                    self._state = 'failed'
                    self._ready.set()
                    self._done.set()
                    if not self._fatal_initialization and self._automatic_restarts < 1:
                        self._automatic_restarts += 1
                        self._restart_count += 1
                        self._restart.set()

    def parse(self, job, *, cancelled=lambda: False):
        if not self._ready.wait(self._ready_timeout):
            raise ParserUnavailable('CPU parser preparation timed out')
        with self._lock:
            if self._state != 'ready' or self._proc is None or self._proc.poll() is not None:
                raise ParserUnavailable(self._error or 'CPU parser is not ready')
            rid = uuid.uuid4().hex
            gen = self._generation
            self._request_id = rid
            self._job_id = job['job_id']
            self._response = None
            self._done.clear()
            self._state = 'busy'
            command = {'command': 'parse', 'request_id': rid,
                       'job_id': job['job_id'], 'source': str(job['source']),
                       'collection': job['collection'],
                       'source_sha256': job['source_sha256']}
            try:
                self._write(json.dumps(command, ensure_ascii=True, separators=(',', ':')) + '\n')
            except Exception:
                # The failed write owns this restart; invalidate its reader.
                self._generation += 1
                self._state = 'failed'
                self._error = 'PARSER_WRITE_FAILED'
                self._restart.set()
                raise ParserUnavailable('CPU parser command write failed')
        deadline = time.monotonic() + self._parse_timeout
        while not self._done.wait(.2):
            if cancelled():
                self.cancel(job['job_id'])
                return {'status': 'PAUSED'}
            if time.monotonic() >= deadline:
                self.cancel(job['job_id'])
                raise ParserUnavailable('CPU parser request deadline exceeded')
        with self._lock:
            if self._generation != gen or self._request_id != rid:
                return {'status': 'PAUSED'}
            evt = self._response
            if evt is None and not cancelled():
                raise ParserUnavailable(self._error or 'CPU parser stopped without result')
            if evt is not None:
                self._request_id = self._job_id = None
                if self._state == 'busy': self._state = 'ready'
                if cancelled():
                    return {'status': 'PAUSED'}
        if evt is None:
            self.cancel(job['job_id'])
            return {'status': 'PAUSED'}
        if evt.get('type') == 'error':
            raise RuntimeError('CPU parser document error: ' + str(evt.get('code', 'PARSE_FAILED')))
        result = evt.get('result')
        if not isinstance(result, dict) or result.get('status') not in ('PARSED','PAUSED'):
            raise ParserUnavailable('CPU parser invalid result')
        return result

    def cancel(self, job_id):
        with self._lock:
            if self._state != 'busy' or self._job_id != job_id:
                return False
            proc, job = self._proc, self._job
            self._generation += 1  # Invalidate stale responses before killing.
            self._state = 'restarting'
            self._request_id = self._job_id = None
            self._done.set()
            self._ready.clear()
        if proc is not None:
            self._kill(proc, job)
        self._restart_count += 1
        self._restart.set()
        return True

    def _retire(self):
        with self._lock:
            proc, job, reader, logger, protocol = self._proc, self._job, self._reader, self._logger, self._protocol
            self._proc = self._job = self._reader = self._logger = self._protocol = None
        if proc is not None:
            if proc.poll() is None:
                try:
                    if proc.stdin is not None:
                        proc.stdin.write(json.dumps({'command':'shutdown'})+'\n')
                        proc.stdin.flush()
                except Exception:
                    pass
                try: proc.wait(timeout=2)
                except subprocess.TimeoutExpired: self._kill(proc,job)
            try: proc.wait(timeout=8)
            except subprocess.TimeoutExpired: proc.kill(); proc.wait(timeout=8)
            for stream in (proc.stdin, proc.stdout, proc.stderr):
                if stream is not None:
                    try: stream.close()
                    except OSError: pass
        if protocol is not None:
            try: protocol.close()
            except OSError: pass
        if job is not None: job.close()
        for thread in (reader, logger):
            if thread is not None and thread is not threading.current_thread():
                thread.join(timeout=3)

    def stop(self):
        self._stopped.set()
        self._ready.set()
        self._done.set()
        self._restart.set()
        with self._lock:
            proc, job = self._proc, self._job
        if proc is not None: self._kill(proc, job)
        if self._manager is not None:
            self._manager.join(timeout=15)
        self._retire()
        with self._lock: self._state = 'stopped'
