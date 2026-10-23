"""Backend-owned ingestion: event scheduler, CPU parser and shared BGE embedding.

One instance is installed only after the HTTP backend instance lock is held.
No DB writes or model loading happen when this module is imported.
"""
from __future__ import annotations
import copy
import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

from rag3.constants import ROOT
from rag3.ingestion_manager import Scheduler, dispatch, JOB_ROOT, PORT as OLD_PORT
from rag3.ingestion_jobs import (connect, one, patch, ingestion_progress,
                                 set_ingestion_progress, clear_ingestion_progress,
                                 content_scope, mark_change, checkpoint_available)
from rag3.ingestion_settings import progress_identity
from rag3.source_identity import sha256_file
PLAIN_TEXT_SUFFIXES = frozenset({".txt", ".json", ".jsonl", ".tsv"})


def _source_filter(sha: str, state: str | None = None):
    from qdrant_client import models
    must=[models.FieldCondition(key='meta.source_sha256', match=models.MatchValue(value=sha))]
    if state:
        must.append(models.FieldCondition(key='meta._rag3_ingest_state', match=models.MatchValue(value=state)))
    return models.Filter(must=must)


def _source_counts(client, collection, sha):
    from qdrant_client import models
    def count(extra=None):
        f=_source_filter(sha)
        if extra is not None:
            f.must.append(extra)
        return client.count(collection_name=collection, count_filter=f, exact=True).count
    return (count(),
            count(models.HasVectorCondition(has_vector='text-dense')),
            count(models.HasVectorCondition(has_vector='bm25')),
            count(models.FieldCondition(key='meta._rag3_ingest_state',match=models.MatchValue(value='ready'))))


def _client(url):
    from qdrant_client import QdrantClient
    return QdrantClient(url=url)


def _compatible(job, *, url, check_qdrant=True):
    """Fail closed while preserving checkpoint and all Qdrant points."""
    path=Path(job['source']) if job.get('source') else None
    if path is None or not path.is_file():
        raise ValueError('Original file missing')
    if sha256_file(path) != job.get('source_sha256'):
        raise ValueError('Original SHA-256 mismatch')
    progress=ingestion_progress(job['collection'], job['source_sha256'])
    if not progress or progress['identity'] != progress_identity(job['source_sha256'],job['collection']):
        raise ValueError('Checkpoint identity missing or incompatible')
    if progress['phase'] not in {'PARSING','CHUNKS_STORED','EMBEDDING'}:
        raise ValueError('Checkpoint phase invalid')
    if check_qdrant:
        c=_client(url)
        if not c.collection_exists(job['collection']):
            raise ValueError('Checkpoint Qdrant collection missing')
        if progress['phase'] != 'PARSING':
            n=_source_counts(c,job['collection'],job['source_sha256'])[0]
            if n < 1 or n > progress['total_chunks']:
                raise ValueError('Checkpoint Qdrant points/total mismatch')
    return progress


def _named_point(point, vector, bm25_config):
    """Full replacement must not discard unknown named vectors/payload fields."""
    from qdrant_client import models
    existing=point.vector
    if existing is None:
        existing={}
    if not isinstance(existing,dict):
        raise ValueError('Cannot preserve non-named Qdrant vectors')
    extra={}
    for name,value in existing.items():
        if name in {'text-dense','bm25'}:
            continue
        if not isinstance(name,str) or not isinstance(value,(list,models.SparseVector,models.Document)):
            raise ValueError(f'Cannot preserve extra vector {name!r}')
        extra[name]=value
    payload=copy.deepcopy(point.payload or {})
    if not isinstance(payload.get('meta'),dict):
        raise ValueError('Parsed point missing nested meta')
    content=payload.get('content')
    if not isinstance(content,str):
        raise ValueError('Parsed point missing content')
    payload['meta']['_rag3_ingest_state']='ready'
    return models.PointStruct(
        id=point.id, payload=payload,
        vector={**extra, 'text-dense':vector,
                'bm25':models.Document(text=content,model='qdrant/bm25',options=bm25_config)})


def _bm25_config(client, collection):
    """Same metadata and Qdrant settings as collection_bm25_config, no Docling import."""
    from qdrant_client import models
    info=client.get_collection(collection_name=collection).model_dump()
    sparse=info.get('config',{}).get('params',{}).get('sparse_vectors') or {}
    if 'bm25' not in sparse:
        raise ValueError('Qdrant BM25 vector absent')
    settings=(info.get('config',{}).get('metadata') or {}).get('rag3',{}).get('bm25')
    if not settings:
        raise ValueError('Qdrant RAG3 BM25 metadata absent')
    return models.Bm25Config(k=float(settings.get('k',1.2)),b=float(settings.get('b',.75)),
        avg_len=float(settings.get('avg_len',256)),tokenizer=models.TokenizerType.MULTILINGUAL,
        lowercase=True,stopwords=None,
        stemmer=models.DisabledStemmerParams(type=models.NoStemmer.NONE))


def _stage_plain(source,collection,sha,progress,should_stop,emit,url):
    from haystack_integrations.document_stores.qdrant.converters import convert_haystack_documents_to_qdrant_points
    from rag3.ingestion_light import preserve_source,split_plain_text_source,normalize_chunk
    from rag3.ingestion_settings import CHUNK_GROUP_SIZE
    if progress['phase'] in {'CHUNKS_STORED','EMBEDDING'}:
        return
    preserved=preserve_source(source,expected_sha256=sha)
    base_meta=dict(document_id=sha,title=preserved['title'],source_uri=preserved['source_uri'],
        source_sha256=sha,page_start=None,page_end=None,content_type='prose',document_version='1')
    chunks=split_plain_text_source(preserved,base_meta)
    if not chunks:
        raise ValueError('Document produced no searchable chunks')
    c=_client(url)
    for offset in range(0,len(chunks),CHUNK_GROUP_SIZE):
        if should_stop():
            return
        staged=[]
        for i in range(offset,min(len(chunks),offset+CHUNK_GROUP_SIZE)):
            chunk=normalize_chunk(chunks[i],i)
            chunk.meta['_rag3_ingest_state']='parsed'
            staged.append(chunk)
        c.upsert(collection_name=collection,points=convert_haystack_documents_to_qdrant_points(
            staged,use_sparse_embeddings=True),wait=True)
        emit(f'STAGE qdrant STORED {offset+len(staged)}')
    if not should_stop():
        set_ingestion_progress(collection,sha,progress_identity(sha,collection),'CHUNKS_STORED',len(chunks))
        emit('STAGE parse PASS')


def _embed_source(job, gpu, should_stop, emit, url):
    source,collection,sha=Path(job['source']),job['collection'],job['source_sha256']
    c=_client(url)
    cfg=_bm25_config(c,collection)
    while True:
        if should_stop():
            return {'status':'PAUSED'}
        progress=_compatible(job,url=url,check_qdrant=False)
        if progress['phase']=='PARSING':
            raise ValueError('Embedding requires completed parsing')
        points,_=c.scroll(collection_name=collection,scroll_filter=_source_filter(sha,'parsed'),
            limit=8,with_payload=True,with_vectors=True)
        if not points:
            break
        # Check that the source state is unchanged; do not resurrect a ready point.
        for p in points:
            if (p.payload or {}).get('meta',{}).get('_rag3_ingest_state')!='parsed':
                raise RuntimeError('Parsed point state changed during scroll')
        texts=[p.payload['content'] for p in points]
        request=gpu.submit_documents(texts)
        try:
            dense=request.result()
        finally:
            # Expired/cancelled Future may finish before the actual CUDA call.
            # Wait for CUDA owner to move off this request before proceeding.
            request.wait_execution_end()
        if should_stop():
            return {'status':'PAUSED'}
        if len(dense)!=len(points):
            raise RuntimeError('GPU embedding count mismatch')
        ready=[_named_point(p,v,cfg) for p,v in zip(points,dense,strict=True)]
        if should_stop():
            return {'status':'PAUSED'}
        c.upsert(collection_name=collection,points=ready,wait=True)
        emit(f'STAGE embedding {len(ready)} written')
    progress=_compatible(job,url=url,check_qdrant=False)
    total=progress['total_chunks']
    counts=_source_counts(c,collection,sha)
    if any(n!=total for n in counts) or total<=0:
        raise RuntimeError(f'Qdrant source verification failed: expected {total}, got {counts}')
    return {'status':'COMPLETE','source_sha256':sha,'collection':collection,
            'managed_source':str(ROOT/'data'/'sources'/sha/source.name),
            'documents_written':total,'bm25_vectors_written':counts[2],
            'source_point_count':counts[0],'source_dense_count':counts[1],
            'source_bm25_count':counts[2],
            'count_after':c.count(collection_name=collection,exact=True).count}


class BackendScheduler(Scheduler):
    def __init__(self, gpu, *, legacy_port=OLD_PORT):
        # Do not invoke Scheduler.__init__ (legacy recovery, polling, subprocess).
        self.gpu=gpu
        self.lock=threading.RLock()
        self.active_id=None
        self.proc=None   # CPU parser only, never a whole-document worker
        self.downloads={}
        self.qdrant_url=os.environ.get('RAG3_QDRANT_URL','http://127.0.0.1:16333')
        self._wake=threading.Event()
        self._stop=threading.Event()
        self._started=False
        self._legacy_port=legacy_port
        self._task=None
        self._thread=None

    def boot(self):
        if self._started:
            raise RuntimeError('Backend scheduler already started')
        # This is a *distinct* guard from the single-instance HTTP lock.
        with socket.socket() as sock:
            sock.settimeout(.2)
            if sock.connect_ex(('127.0.0.1',self._legacy_port))==0:
                raise RuntimeError('Legacy ingestion manager port is occupied; refuse dual scheduler')
        with self.lock:
            self.recover()
            self._started=True
            self._thread=threading.Thread(target=self.pump,name='rag3-ingestion-scheduler',daemon=True)
            self._thread.start()
            self._wake.set()

    def recover(self):
        with connect() as db:
            rows=db.execute("SELECT * FROM jobs WHERE status IN ('RUNNING','RESUMING','PAUSING','CANCELLING','DOWNLOADING')").fetchall()
            # Runs before a new submission can be accepted, under the backend lock.
            for row in rows:
                job=dict(row)
                if job['status']=='DOWNLOADING':
                    patch(db,job['job_id'],status='FAILED',requested=0,
                          error='Interrupted download; resubmit attachment reference')
                    continue
                try:
                    _compatible(job,url=self.qdrant_url)
                except Exception as exc:
                    patch(db,job['job_id'],status='FAILED',requested=0,
                          error=f'Interrupted; incompatible checkpoint: {exc}'[:400])
                else:
                    patch(db,job['job_id'],status='PAUSED',requested=0,
                          stage='Interrupted; verified checkpoint preserved')
            db.execute("UPDATE jobs SET status='FAILED',requested=0,error='Incomplete attachment' WHERE status='PAUSED' AND source_sha256 IS NULL")
            db.execute("UPDATE jobs SET requested=0 WHERE status='QUEUED'")
            db.commit()
        # No cleanup of any COMPLETE job during recovery.

    def validate_identity(self,job):
        return _compatible(job,url=self.qdrant_url)

    def submit(self, **params):
        if not self._started:
            raise RuntimeError('Backend scheduler is not started')
        with self.lock:
            value=super().submit(**params)
            if value['status']=='QUEUED' and params.get('requested',True):
                self._wake.set()
            return value

    def receive(self,job_id,url,filename,control):
        super().receive(job_id,url,filename,control)
        self._wake.set()

    def action(self,job_id,operation):
        with self.lock:
            job=one(job_id,private=True)
            if operation=='restart':
                # The base action rejects invalid states. Do not destroy even
                # *parsed* points before that eligibility check can run.
                if job['status'] not in {'PAUSED','FAILED','CANCELLED','COMPLETE'}:
                    raise ValueError('job is not paused, failed, or cancelled')
                # Never remove a verified ready chunk even after an explicit restart.
                sha=job.get('source_sha256')
                if sha:
                    if not job.get('source') or sha256_file(Path(job['source'])) != sha:
                        raise ValueError('Restart requires unchanged original SHA-256')
                    self.conflict(sha,job['collection'],excluding=job_id)
                    c=_client(self.qdrant_url)
                    before=_source_counts(c,job['collection'],sha)
                    if before[3]:
                        raise ValueError('Restart would delete existing ready points; use an isolated collection')
                    if before[0]:
                        from qdrant_client import models
                        # The caller explicitly requested Restart. The subset
                        # is restricted to this SHA, collection, and parsed state.
                        c.delete(collection_name=job['collection'],wait=True,
                            points_selector=models.FilterSelector(filter=_source_filter(sha,'parsed')))
                        after=_source_counts(c,job['collection'],sha)
                        if after[0]:
                            raise RuntimeError('Restart blocked: non-parsed source points remain')
            if operation=='cancel' and self.active_id==job_id and job['status'] in {'RUNNING','RESUMING','PAUSING'}:
                with connect() as db: patch(db,job_id,status='CANCELLING',requested=0,
                                             stage='Cancel accepted; finish current GPU batch')
                if self.proc is not None:
                    self.kill_worker(self.proc)
                self._wake.set()
                return one(job_id)
            value=super().action(job_id,operation)
            if operation in {'run','resume','retry','restart','cancel'}:
                self._wake.set()
            return value

    def pump(self):
        while not self._stop.is_set():
            self._wake.wait()
            self._wake.clear()
            if self._stop.is_set():
                break
            with self.lock:
                if self.active_id is not None:
                    continue
                row=None
                try:
                    with connect() as db:
                        row=db.execute("SELECT * FROM jobs WHERE status='QUEUED' AND requested=1 ORDER BY created_at,job_id LIMIT 1").fetchone()
                    if row is not None:
                        self.start(dict(row))
                except Exception as exc:
                    if row is not None:
                        # Keep processing other queued work only after the bad
                        # job was marked FAILED. A DB-read failure with no row
                        # must not become an unbounded event self-wake loop.
                        with connect() as db: patch(db,row['job_id'],status='FAILED',requested=0,error=str(exc)[:400])
                        self._wake.set()

    def start(self,job):
        if not job.get('source') or not Path(job['source']).is_file():
            raise FileNotFoundError('Original file unavailable')
        if sha256_file(Path(job['source']))!=job['source_sha256']:
            raise ValueError('Original SHA-256 changed')
        self.validate_collection(job['collection'])
        self.conflict(job['source_sha256'],job['collection'],excluding=job['job_id'])
        if job['mode']=='resume':
            self.validate_identity(job)
        self.active_id=job['job_id']
        # A previous pause marker is not a new request.
        marker=ROOT/'data'/'checkpoints'/job['source_sha256']/'pause.request'
        marker.unlink(missing_ok=True)
        with connect() as db:
            patch(db,job['job_id'],status='RESUMING' if job['mode']=='resume' else 'RUNNING',stage='Starting backend ingestion')
        self._task=threading.Thread(target=self._execute,args=(job,),name='rag3-ingest-active',daemon=True)
        self._task.start()

    def _stopping(self,job):
        status=one(job['job_id'],private=True)['status']
        marker=ROOT/'data'/'checkpoints'/job['source_sha256']/'pause.request'
        return status in {'CANCELLING','PAUSING'} or marker.exists()

    def _parse_cpu(self,job):
        if self._stopping(job):
            return {'status':'PAUSED'}
        env=os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
                   OPENBLAS_NUM_THREADS='2',RAG3_PARSE_CPU_ONLY='1')
        args=[sys.executable,str(ROOT/'scripts'/'ingest-worker.py'),job['source'],
              '--collection',job['collection'],'--phase','parse','--source-sha',job['source_sha256']]
        creation=getattr(subprocess,'CREATE_NO_WINDOW',0) if os.name=='nt' else 0
        proc=subprocess.Popen(args,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                              stdin=subprocess.DEVNULL,text=True,encoding='utf-8',errors='replace',
                              creationflags=creation,start_new_session=os.name!='nt')
        with self.lock:
            self.proc=proc
            if self._stopping(job):
                self.kill_worker(proc)
        result=None; error=None
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                try: evt=json.loads(line)
                except ValueError: continue
                if evt.get('type')=='result': result=evt.get('result')
                elif evt.get('type')=='error': error=str(evt.get('message'))[:400]
                elif evt.get('type')=='log': self._stage(job, str(evt.get('message'))[:240])
            code=proc.wait()
        finally:
            with self.lock:
                if self.proc is proc: self.proc=None
        if self._stopping(job):
            return {'status':'PAUSED'}
        if code==75 and isinstance(result,dict) and result.get('status')=='PAUSED':
            return result
        if code!=0 or not isinstance(result,dict) or result.get('status')!='PARSED':
            raise RuntimeError(error or f'CPU parser exited {code}')
        return result

    def _stage(self,job,message):
        with connect() as db: patch(db,job['job_id'],stage=message)

    def _execute(self,job):
        result=None; error=None; completed=False
        try:
            sha,collection=job['source_sha256'],job['collection']
            progress=ingestion_progress(collection,sha)
            if job['mode']=='resume':
                progress=_compatible(job,url=self.qdrant_url)
            elif progress is not None:
                raise ValueError('Existing checkpoint: explicit validated Resume/Restart required')
            else:
                c=_client(self.qdrant_url)
                if _source_counts(c,collection,sha)[0]:
                    raise ValueError('Existing points for SHA; refuse implicit delete')
                set_ingestion_progress(collection,sha,progress_identity(sha,collection),'PARSING')
                progress=ingestion_progress(collection,sha)
            if progress['phase']=='PARSING' and not self._stopping(job):
                self._stage(job,'STAGE parse RUN')
                if Path(job['source']).suffix.lower() in PLAIN_TEXT_SUFFIXES:
                    _stage_plain(Path(job['source']),collection,sha,progress,
                                 lambda:self._stopping(job),lambda msg:self._stage(job,msg),self.qdrant_url)
                else:
                    self._parse_cpu(job)
            if not self._stopping(job):
                self._stage(job,'STAGE embedding RUN')
                result=_embed_source(job,self.gpu,lambda:self._stopping(job),
                                     lambda msg:self._stage(job,msg),self.qdrant_url)
                completed=result.get('status')=='COMPLETE'
        except Exception as exc:
            error=f'{type(exc).__name__}: {exc}'[:400]
        finally:
            with self.lock:
                current=one(job['job_id'],private=True)
                if completed:
                    status='COMPLETE'
                elif current['status']=='CANCELLING':
                    status='CANCELLED'
                elif current['status']=='PAUSING' or result and result.get('status')=='PAUSED':
                    status='PAUSED'
                else:
                    status='FAILED'
                    error=error or 'Backend ingestion ended without verified completion'
                with connect() as db:
                    patch(db,job['job_id'],status=status,requested=0,stage=status,
                          result=result if completed else None,error=error)
                if completed:
                    clear_ingestion_progress(job['collection'],job['source_sha256'])
                    self.retire_completed_upload(one(job['job_id'],private=True))
                if completed or ingestion_progress(job['collection'],job['source_sha256']):
                    mark_change(content_scope(job['collection']))
                self.active_id=None
                self.proc=None
                self._task=None
                self._wake.set()

    def stop(self):
        self._stop.set()
        self._wake.set()
