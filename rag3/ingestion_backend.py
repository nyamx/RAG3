"""Backend-owned ingestion: CPU parser, exclusive CUDA PDF parser, and BGE embedding.

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

from rag3.constants import ROOT, DATA_ROOT
from rag3.qdrant_clients import get_client
from rag3.ingestion_manager import (Scheduler, dispatch, JOB_ROOT, PORT as OLD_PORT,
                                    CHATGPT_SHARD_NAME)
from rag3.ingestion_jobs import (connect, one, patch, ingestion_progress,
                                 set_ingestion_progress, clear_ingestion_progress,
                                 content_scope, mark_change, checkpoint_available)
from rag3.ingestion_settings import progress_identity
from rag3.source_identity import sha256_file
PLAIN_TEXT_SUFFIXES = frozenset({".txt", ".json", ".jsonl", ".tsv"})


def _use_cuda_parser(source: str | Path) -> bool:
    """Amortize CUDA parser startup on multi-page PDFs; retain fast CPU small jobs."""
    source = Path(source)
    if source.suffix.lower() != '.pdf':
        return False
    min_pages = int(os.environ.get('RAG3_CUDA_PARSE_MIN_PAGES', '8'))
    if min_pages < 1:
        raise ValueError('RAG3_CUDA_PARSE_MIN_PAGES must be positive')
    if min_pages == 1:
        return True
    from pypdfium2 import PdfDocument
    pdf = PdfDocument(str(source))
    try:
        return len(pdf) >= min_pages
    finally:
        pdf.close()


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


def _completed_chatgpt_sources(collection, current_sha):
    """Only verified jobs may establish that a conversation was already ingested."""
    with connect() as db:
        complete=db.execute('''SELECT source_sha256, filename FROM jobs
            WHERE collection=? AND status='COMPLETE' AND source_sha256 IS NOT NULL''',
            (collection,)).fetchall()
        unfinished=db.execute('''SELECT DISTINCT p.source_sha256, j.filename
            FROM ingestion_progress p JOIN jobs j
              ON j.collection=p.collection AND j.source_sha256=p.source_sha256
            WHERE p.collection=? AND p.source_sha256<>? AND j.status<>'COMPLETE' ''',
            (collection,current_sha)).fetchall()
    if any(CHATGPT_SHARD_NAME.fullmatch(row['filename']) for row in unfinished):
        raise ValueError('Another ChatGPT export has an unfinished checkpoint in this collection; resume or resolve it first')
    return sorted({row['source_sha256'] for row in complete
                   if CHATGPT_SHARD_NAME.fullmatch(row['filename'])})


def _known_chatgpt_conversations(client, collection, current_sha, requested_ids):
    """Return existing conversation IDs without reading stored text or vectors."""
    if not requested_ids:
        return set()
    complete_sources=_completed_chatgpt_sources(collection,current_sha)
    if not complete_sources:
        return set()
    from qdrant_client import models
    fields=('meta.chatgpt_conversation_id','meta.source_sha256','meta._rag3_ingest_state')
    indexed=client.get_collection(collection).payload_schema or {}
    for field in fields:
        if field not in indexed:
            client.create_payload_index(collection_name=collection,field_name=field,
                                        field_schema=models.PayloadSchemaType.KEYWORD,wait=True)
    found=set()
    candidates=sorted(requested_ids)
    for offset in range(0,len(candidates),100):
        batch=candidates[offset:offset+100]
        filters=models.Filter(must=[
            models.FieldCondition(key='meta.chatgpt_conversation_id',
                                  match=models.MatchAny(any=batch)),
            models.FieldCondition(key='meta.source_sha256',
                                  match=models.MatchAny(any=complete_sources)),
            models.FieldCondition(key='meta._rag3_ingest_state',
                                  match=models.MatchValue(value='ready'))])
        result=client.facet(collection_name=collection,key='meta.chatgpt_conversation_id',
                            facet_filter=filters,exact=True,limit=len(batch))
        values={hit.value for hit in result.hits if hit.count > 0}
        if not values.issubset(batch):
            raise RuntimeError('Qdrant returned an unexpected conversation ID')
        found.update(values)
    return found


def _verified_completed_chatgpt_source(client, collection, sha):
    """Recognize an unchanged, already completed shard before staging points."""
    with connect() as db:
        rows=db.execute('''SELECT filename,result FROM jobs WHERE collection=?
            AND source_sha256=? AND status='COMPLETE' ORDER BY created_at DESC''',
            (collection,sha)).fetchall()
    expected=[]
    for row in rows:
        if not CHATGPT_SHARD_NAME.fullmatch(row['filename']) or not row['result']:
            continue
        result=json.loads(row['result'])
        count=result.get('documents_written')
        if isinstance(count,int) and not isinstance(count,bool) and count>=0:
            expected.append(count)
    if not expected:
        return False
    counts=_source_counts(client,collection,sha)
    if counts[0] == 0 and 0 in expected:
        return True
    if counts[0] > 0 and counts[0] in expected and all(n==counts[0] for n in counts):
        return True
    raise ValueError('Previously completed ChatGPT source has missing or incomplete Qdrant points')


def _empty_chatgpt_result(source, collection, sha, client):
    from rag3.ingestion_light import preserve_source
    preserved=preserve_source(source,expected_sha256=sha)
    return {'status':'COMPLETE','source_sha256':sha,'collection':collection,
            'managed_source':str(preserved['managed_source']),
            'documents_written':0,'bm25_vectors_written':0,
            'count_after':client.count(collection_name=collection,exact=True).count}


def _client(url):
    from qdrant_client import QdrantClient
    return get_client(url=url)


def _compatible(job, *, url, check_qdrant=True, source_guard=None):
    """Fail closed while preserving checkpoint and all Qdrant points."""
    path=Path(job['source']) if job.get('source') else None
    if path is None or not path.is_file():
        raise ValueError('Original file missing')
    if source_guard is None or not source_guard.strong:
        if sha256_file(path) != job.get('source_sha256'):
            raise ValueError('Original SHA-256 mismatch')
    elif source_guard.source != path.resolve() or source_guard.expected_sha != job.get('source_sha256'):
        raise ValueError('Original source handle identity mismatch')
    progress=ingestion_progress(job['collection'], job['source_sha256'])
    if not progress or progress['identity'] != progress_identity(job['source_sha256'],job['collection']):
        raise ValueError('Checkpoint identity missing or incompatible')
    if progress['phase'] not in {'PARSING','CHUNKS_STORED','EMBEDDING'}:
        raise ValueError('Checkpoint phase invalid')
    if check_qdrant:
        c=_client(url)
        if not c.collection_exists(job['collection']):
            raise ValueError('Checkpoint Qdrant collection missing')
        if progress['phase'] == 'PARSING':
            if _source_counts(c,job['collection'],job['source_sha256'])[3]:
                raise ValueError('PARSING checkpoint contains ready points')
        else:
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


def _stage_plain(source,collection,sha,progress,should_stop,emit,url,*,commit_batch=None):
    from haystack_integrations.document_stores.qdrant.converters import convert_haystack_documents_to_qdrant_points
    from rag3.ingestion_light import preserve_source,split_plain_text_source,normalize_chunk
    from rag3.ingestion_settings import CHUNK_GROUP_SIZE
    if progress['phase'] in {'CHUNKS_STORED','EMBEDDING'}:
        return
    preserved=preserve_source(source,expected_sha256=sha)
    base_meta=dict(document_id=sha,title=preserved['title'],source_uri=preserved['source_uri'],
        source_sha256=sha,page_start=None,page_end=None,content_type='prose',document_version='1')
    from rag3.chatgpt_export import SHARD, split_shard_source
    c=_client(url)
    if SHARD.fullmatch(Path(source).name):
        chunks=split_shard_source(preserved,base_meta,
            lookup_existing=lambda ids:_known_chatgpt_conversations(c,collection,sha,ids))
    else:
        chunks=split_plain_text_source(preserved,base_meta)
    if not chunks:
        if SHARD.fullmatch(Path(source).name):
            emit('STAGE parse SKIPPED (no new searchable conversations)')
            return 0
        raise ValueError('Document produced no searchable chunks')
    for offset in range(0,len(chunks),CHUNK_GROUP_SIZE):
        if should_stop():
            return
        staged=[]
        for i in range(offset,min(len(chunks),offset+CHUNK_GROUP_SIZE)):
            chunk=normalize_chunk(chunks[i],i)
            chunk.meta['_rag3_ingest_state']='parsed'
            staged.append(chunk)
        points=convert_haystack_documents_to_qdrant_points(staged,use_sparse_embeddings=True)
        if commit_batch is None:
            if should_stop():
                return
            c.upsert(collection_name=collection,points=points,wait=True)
        elif not commit_batch(c,collection,points):
            return
        emit(f'STAGE qdrant STORED {offset+len(staged)}')
    if not should_stop():
        set_ingestion_progress(collection,sha,progress_identity(sha,collection),'CHUNKS_STORED',len(chunks))
        emit('STAGE parse PASS')
    return len(chunks)


def _embed_source(job, gpu, should_stop, emit, url, *, register_request=None,
                  clear_request=None, commit_batch=None, source_guard=None):
    source,collection,sha=Path(job['source']),job['collection'],job['source_sha256']
    c=_client(url)
    cfg=_bm25_config(c,collection)
    completed=c.count(collection_name=collection,
                      count_filter=_source_filter(sha,'ready'),exact=True).count
    while True:
        if should_stop():
            return {'status':'PAUSED'}
        progress=_compatible(job,url=url,check_qdrant=False,source_guard=source_guard)
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
            if register_request is not None:
                register_request(request)
            dense=request.result()
        except BaseException:
            request.cancel()
            raise
        finally:
            # Expired/cancelled Future may finish before the actual CUDA call.
            # Wait for CUDA owner to move off this request before proceeding.
            try:
                request.wait_execution_end()
            finally:
                if clear_request is not None:
                    clear_request(request)
        if should_stop():
            return {'status':'PAUSED'}
        if len(dense)!=len(points):
            raise RuntimeError('GPU embedding count mismatch')
        ready=[_named_point(p,v,cfg) for p,v in zip(points,dense,strict=True)]
        if commit_batch is None:
            if should_stop():
                return {'status':'PAUSED'}
            c.upsert(collection_name=collection,points=ready,wait=True)
        elif not commit_batch(c, collection, ready):
            return {'status':'PAUSED'}
        completed += len(ready)
        emit(f'STAGE embedding {completed}/{progress["total_chunks"]} written')
    progress=_compatible(job,url=url,check_qdrant=False,source_guard=source_guard)
    total=progress['total_chunks']
    counts=_source_counts(c,collection,sha)
    if any(n!=total for n in counts) or total<=0:
        raise RuntimeError(f'Qdrant source verification failed: expected {total}, got {counts}')
    return {'status':'COMPLETE','source_sha256':sha,'collection':collection,
            'managed_source':str(DATA_ROOT/'sources'/sha/source.name),
            'documents_written':total,'bm25_vectors_written':counts[2],
            'source_point_count':counts[0],'source_dense_count':counts[1],
            'source_bm25_count':counts[2],
            'count_after':c.count(collection_name=collection,exact=True).count}


def _verify_reparsed_points(client, collection, sha, expected):
    """Fail closed before embedding if old/new parsed ID sets do not coincide."""
    import hashlib
    import uuid
    if not isinstance(expected, dict) or expected.get('status') != 'PARSED':
        raise ValueError('CPU parser did not return verified chunk count')
    count = expected.get('total_chunks')
    digest = expected.get('chunk_ids_sha256')
    if not isinstance(count,int) or count < 1 or not isinstance(digest,str) or len(digest)!=64:
        raise ValueError('CPU parser omitted parsed-ID verification')
    points=[]
    cursor=None
    while True:
        batch,cursor=client.scroll(collection_name=collection,
            scroll_filter=_source_filter(sha), limit=256, offset=cursor,
            with_payload=True, with_vectors=False)
        for point in batch:
            if (point.payload or {}).get('meta',{}).get('_rag3_ingest_state')!='parsed':
                raise ValueError('Cannot resume: ready or foreign source points present')
            points.append(str(uuid.UUID(str(point.id))))
        if cursor is None: break
    computed=hashlib.sha256(('\n'.join(sorted(points))).encode('utf-8')).hexdigest()
    if len(points)!=count or computed!=digest:
        raise ValueError('Reparsed point IDs differ; explicit Restart required')


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
        self._gpu_request=None
        self._parser_job=None
        from rag3.persistent_parser import PersistentCPUParser
        self._persistent_parser = PersistentCPUParser(self._parser_stage)
        self._gpu_parser = None  # CUDA child exists only while a PDF is parsed.

    def boot(self):
        if self._started:
            raise RuntimeError('Backend scheduler already started')
        # This is a *distinct* guard from the single-instance HTTP lock.
        with socket.socket() as sock:
            sock.settimeout(.2)
            if sock.connect_ex(('127.0.0.1',self._legacy_port))==0:
                raise RuntimeError('Legacy ingestion manager port is occupied; refuse dual scheduler')
        try:
            with self.lock:
                self._persistent_parser.start()  # Qdrant outage cannot delay parser startup.
                self.recover()
                self._started=True
                self._thread=threading.Thread(target=self.pump,name='rag3-ingestion-scheduler',daemon=True)
                self._thread.start()
                self._wake.set()
        except BaseException:
            self._stop.set()
            self._wake.set()
            self._started=False
            self._persistent_parser.stop()  # Boot has no caller-owned finally yet.
            raise

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
                except ValueError as exc:
                    patch(db,job['job_id'],status='FAILED',requested=0,
                          error=f'Interrupted; incompatible checkpoint: {exc}'[:400])
                except Exception:
                    # A temporary Qdrant/model/IO outage does not prove that
                    # the saved checkpoint is incompatible. Resume validates it.
                    patch(db,job['job_id'],status='PAUSED',requested=0,
                          stage='Interrupted; checkpoint verification deferred',
                          error='Checkpoint could not be verified during startup')
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
                if self._gpu_request is not None:
                    self._gpu_request.cancel()
                self._persistent_parser.cancel(job_id)
                if self._gpu_parser is not None:
                    self._gpu_parser.cancel(job_id)
                self._wake.set()
                return one(job_id)
            value=super().action(job_id,operation)
            if operation=='pause' and self.active_id==job_id and self._gpu_request is not None:
                self._gpu_request.cancel()
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
                        try:
                            with connect() as db:
                                patch(db,row['job_id'],status='FAILED',requested=0,error=str(exc)[:400])
                        except Exception:
                            pass  # The journal is unavailable; do not spin.
                        else:
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
        try:
            # A previous pause marker is not a new request.
            marker=DATA_ROOT/'checkpoints'/job['source_sha256']/'pause.request'
            marker.unlink(missing_ok=True)
            with connect() as db:
                patch(db,job['job_id'],status='RESUMING' if job['mode']=='resume' else 'RUNNING',stage='Starting backend ingestion')
            self._task=threading.Thread(target=self._execute,args=(job,),name='rag3-ingest-active',daemon=True)
            self._task.start()
        except Exception as exc:
            self._task=None
            self.active_id=None
            try:
                with connect() as db:
                    patch(db,job['job_id'],status='FAILED',requested=0,
                          error=f'Could not start ingestion: {exc}'[:400])
            except Exception:
                pass  # Preserve the original startup error if SQLite is down.
            raise

    def _stopping(self,job):
        if self._stop.is_set():
            return True
        status=one(job['job_id'],private=True)['status']
        marker=DATA_ROOT/'checkpoints'/job['source_sha256']/'pause.request'
        return status in {'CANCELLING','PAUSING'} or marker.exists()

    def _terminate_parser(self, proc, parser_job):
        if parser_job is not None:
            try:
                parser_job.terminate()
                return
            except OSError:
                pass
        self.kill_worker(proc)

    def parser_snapshot(self):
        snapshot = self._persistent_parser.snapshot()
        with self.lock:
            gpu_parser = self._gpu_parser
        snapshot['cuda_parser'] = gpu_parser.snapshot() if gpu_parser is not None else None
        return snapshot

    def _parser_stage(self,job_id,message):
        # The callback runs on the protocol reader. Do not write a cancelled
        # job's progress into a later or unrelated ingestion.
        if self.active_id == job_id:
            with connect() as db:
                patch(db,job_id,stage=message)

    def _parse_cpu(self,job):
        if self._stopping(job):
            return {'status':'PAUSED'}
        return self._persistent_parser.parse(job, cancelled=lambda:self._stopping(job))

    def _parse_cuda(self, job):
        """Give CUDA to one short-lived PDF parser, then return it to BGE."""
        from rag3.persistent_parser import PersistentCPUParser

        parser = None
        self._stage(job, 'STAGE GPU release BGE')
        try:
            self.gpu.reserve_for_parsing()
            if self._stopping(job):
                return {'status': 'PAUSED'}
            parser = PersistentCPUParser(self._parser_stage, device='cuda')
            with self.lock:
                self._gpu_parser = parser
            self._stage(job, 'STAGE CUDA parse RUN')
            parser.start()
            return parser.parse(job, cancelled=lambda:self._stopping(job))
        finally:
            try:
                if parser is not None:
                    parser.stop()  # The CUDA context must exit before BGE reloads.
            finally:
                with self.lock:
                    self._gpu_parser = None
                self.gpu.release_after_parsing()

    def _stage(self,job,message):
        with connect() as db: patch(db,job['job_id'],stage=message)

    def _register_gpu_request(self, job, request):
        with self.lock:
            self._gpu_request=request
            try:
                stopping=self._stopping(job)
            except Exception:
                request.cancel()
                raise
            if stopping:
                request.cancel()

    def _clear_gpu_request(self, request):
        with self.lock:
            if self._gpu_request is request:
                self._gpu_request=None

    def _commit_batch(self, job, client, collection, points):
        # Cancel cannot be acknowledged between the final state check and
        # starting a parsed or ready Qdrant write. An active write cannot be interrupted.
        with self.lock:
            if self._stopping(job):
                return False
            client.upsert(collection_name=collection,points=points,wait=True)
            return True

    def _execute(self,job):
        from rag3.source_guard import OriginalSourceGuard
        result=None; error=None; completed=False; source_guard=None
        job_started=time.perf_counter()
        parse_elapsed=None; parser_detail={}
        no_new_chatgpt=False
        try:
            source_guard=OriginalSourceGuard(job['source'], job['source_sha256'])
            sha,collection=job['source_sha256'],job['collection']
            is_chatgpt=bool(CHATGPT_SHARD_NAME.fullmatch(Path(job['source']).name))
            progress=ingestion_progress(collection,sha)
            if job['mode']=='resume':
                progress=_compatible(job,url=self.qdrant_url,source_guard=source_guard)
            elif progress is not None:
                raise ValueError('Existing checkpoint: explicit validated Resume/Restart required')
            else:
                c=_client(self.qdrant_url)
                if is_chatgpt and _verified_completed_chatgpt_source(c,collection,sha):
                    result=_empty_chatgpt_result(Path(job['source']),collection,sha,c)
                    completed=True
                    return
                if _source_counts(c,collection,sha)[0]:
                    raise ValueError('Existing points for SHA; refuse implicit delete')
                set_ingestion_progress(collection,sha,progress_identity(sha,collection),'PARSING')
                progress=ingestion_progress(collection,sha)
            if progress['phase']=='PARSING' and not self._stopping(job):
                parse_started=time.perf_counter()
                self._stage(job,'STAGE parse RUN')
                if Path(job['source']).suffix.lower() in PLAIN_TEXT_SUFFIXES:
                    staged_count=_stage_plain(Path(job['source']),collection,sha,progress,
                                 lambda:self._stopping(job),lambda msg:self._stage(job,msg),self.qdrant_url,
                                 commit_batch=lambda client,collection,points:
                                     self._commit_batch(job,client,collection,points))
                    no_new_chatgpt=is_chatgpt and staged_count==0
                else:
                    parsed=(self._parse_cuda(job) if _use_cuda_parser(job['source'])
                            else self._parse_cpu(job))
                    parser_detail={key:parsed[key] for key in ('conversion_seconds','chunk_stage_seconds')
                                   if key in parsed}
                    if not self._stopping(job):
                        _verify_reparsed_points(_client(self.qdrant_url),collection,sha,parsed)
                parse_elapsed=round(time.perf_counter()-parse_started,3)
            if not self._stopping(job):
                if no_new_chatgpt:
                    result=_empty_chatgpt_result(Path(job['source']),collection,sha,
                                                 _client(self.qdrant_url))
                    completed=True
                    embed_elapsed=0.0
                else:
                    embed_started=time.perf_counter()
                    self._stage(job,'STAGE embedding RUN')
                    result=_embed_source(job,self.gpu,lambda:self._stopping(job),
                                         lambda msg:self._stage(job,msg),self.qdrant_url,
                                         register_request=lambda request:self._register_gpu_request(job,request),
                                         clear_request=self._clear_gpu_request, source_guard=source_guard,
                                         commit_batch=lambda client,collection,points:
                                             self._commit_batch(job,client,collection,points))
                    completed=result.get('status')=='COMPLETE'
                    embed_elapsed=round(time.perf_counter()-embed_started,3)
                if completed:
                    result['timings']={
                        'parse_phase_seconds':parse_elapsed,
                        'embedding_phase_seconds':embed_elapsed,
                        'total_seconds':round(time.perf_counter()-job_started,3),
                        **parser_detail,
                    }
        except Exception as exc:
            error=f'{type(exc).__name__}: {exc}'[:400]
        finally:
            if source_guard is not None:
                source_guard.close()
            with self.lock:
                try:
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
                finally:
                    self.active_id=None
                    self._task=None
                    self._gpu_request=None
                    self._wake.set()

    def stop(self):
        with self.lock:
            self._stop.set()
            self._wake.set()
            if self.active_id is not None:
                try:
                    current=one(self.active_id,private=True)
                    if current['status'] in {'RUNNING','RESUMING'}:
                        with connect() as db:
                            patch(db,self.active_id,status='PAUSING',requested=0,
                                  stage='Backend stopping; preserve checkpoint')
                except Exception:
                    pass  # Startup recovery will reconcile the unchanged journal.
            if self._gpu_request is not None:
                self._gpu_request.cancel()
            gpu_parser = self._gpu_parser
        self._persistent_parser.stop()
        if gpu_parser is not None:
            gpu_parser.stop()
