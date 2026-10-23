"""Stage C: deterministic safety checks; never touch production DB/Qdrant."""
from __future__ import annotations
import copy
import os
import tempfile
import threading
import time
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from qdrant_client import models
from rag3.gpu_model_service import GPUModelService, GPUServiceConfig
from rag3 import ingestion_backend as C
from rag3 import ingestion_jobs as J


class FakePoint:
    def __init__(self, index, sha):
        self.id=str(uuid.uuid5(uuid.NAMESPACE_URL,f'{sha}:{index}'))
        self.payload={'content':f'chunk {index}', 'meta':{
            'source_sha256':sha,'_rag3_ingest_state':'parsed',
            'section_paths':['chapter',str(index)],'chunk_index':index,
            'other':{'retain':1}},'other_payload':{'x':[1,2]}}
        self.vector={'text-sparse':models.SparseVector(indices=[1,5],values=[1.0,2.0])}


class FakeQdrant:
    def __init__(self, size,sha):
        self.sha=sha
        self.points=[FakePoint(i,sha) for i in range(size)]
        self.before={p.id:(copy.deepcopy(p.payload),copy.deepcopy(p.vector)) for p in self.points}
        self.upserts=[]
    def get_collection(self,**_):
        return SimpleNamespace(model_dump=lambda:{'config':{'params':{'sparse_vectors':{'bm25':{},'text-sparse':{}}},
            'metadata':{'rag3':{'bm25':{'k':1.2,'b':0.75,'avg_len':256}}}}})
    def scroll(self, **kw):
        assert kw['limit']==8 and kw['with_vectors'] and kw['with_payload']
        return [p for p in self.points if p.payload['meta']['_rag3_ingest_state']=='parsed'][:8],None
    def upsert(self, **kw):
        assert kw['wait'] and len(kw['points'])<=8
        self.upserts.append(kw['points'])
        for new in kw['points']:
            old=next(p for p in self.points if p.id==new.id)
            original, vector=self.before[new.id]
            assert new.payload['content']==original['content']
            assert new.payload['other_payload']==original['other_payload']
            assert new.payload['meta']['section_paths']==original['meta']['section_paths']
            assert new.payload['meta']['other']==original['meta']['other']
            assert new.payload['meta']['_rag3_ingest_state']=='ready'
            assert new.vector['text-sparse']==vector['text-sparse']
            assert len(new.vector['text-dense'])==2
            assert isinstance(new.vector['bm25'],models.Document)
            old.payload=copy.deepcopy(new.payload)
            old.vector=copy.deepcopy(new.vector)
    def count(self, **kw):
        filt=kw.get('count_filter')
        points=self.points
        if filt:
            for clause in filt.must:
                if isinstance(clause,models.HasVectorCondition):
                    points=[p for p in points if clause.has_vector in p.vector]
                elif isinstance(clause,models.FieldCondition):
                    key=clause.key.split('.')[-1]
                    if key=='_rag3_ingest_state':
                        points=[p for p in points if p.payload['meta'][key]==clause.match.value]
                    elif key=='source_sha256':
                        points=[p for p in points if p.payload['meta'][key]==clause.match.value]
        return SimpleNamespace(count=len(points))


class Tests(unittest.TestCase):
    def test_01_point_replacement_preserves_extra_vectors_payload(self):
        sha='a'*64
        p=FakePoint(0,sha)
        cfg=models.Bm25Config(k=1.2,b=0.75,avg_len=256)
        target=C._named_point(p,[1.,2.],cfg)
        self.assertEqual(target.payload['meta']['_rag3_ingest_state'],'ready')
        self.assertEqual(target.payload['other_payload'],p.payload['other_payload'])
        self.assertEqual(target.vector['text-sparse'],p.vector['text-sparse'])
        self.assertEqual(p.payload['meta']['_rag3_ingest_state'],'parsed')

    def test_02_batch_17_full_upsert(self):
        sha='b'*64
        fake=FakeQdrant(17,sha)
        jobs={'source':'fake.txt','collection':'test','source_sha256':sha}
        class GPU:
            def __init__(self): self.batch=[]
            def submit_documents(self,texts):
                self.batch.append(len(texts))
                class Req:
                    def result(self): return [[.5,.5] for _ in texts]
                    def wait_execution_end(self): pass
                return Req()
        gpu=GPU()
        with patch.object(C,'_client',return_value=fake), patch.object(C,'_compatible',return_value={'phase':'CHUNKS_STORED','total_chunks':17}):
            result=C._embed_source(jobs,gpu,lambda:False,lambda msg:None,'fake')
        self.assertEqual(result['status'],'COMPLETE')
        self.assertEqual(gpu.batch,[8,8,1])
        self.assertEqual([len(v) for v in fake.upserts],[8,8,1])
        self.assertEqual(result['source_point_count'],17)
        self.assertEqual(result['source_dense_count'],17)
        self.assertEqual(result['source_bm25_count'],17)

    def test_03_cancel_during_nonpreemptible_cuda_no_write(self):
        sha='c'*64
        fake=FakeQdrant(17,sha)
        started,release,cancel=threading.Event(),threading.Event(),threading.Event()
        def encode(_model,batch,**_):
            started.set()
            if not release.wait(timeout=8):
                raise RuntimeError('test timed out')
            return [[1.,0.] for _ in batch]
        gpu=GPUModelService(GPUServiceConfig(),model_factory=lambda:object(),encode_fn=encode)
        job={'source':'fake.txt','collection':'tmp','source_sha256':sha}
        output={}
        def run():
            try:
                output['result']=C._embed_source(job,gpu,cancel.is_set,lambda msg:None,'fake')
            except Exception as e:
                output['error']=e
        with patch.object(C,'_client',return_value=fake), patch.object(C,'_compatible',return_value={'phase':'CHUNKS_STORED','total_chunks':17}):
            thread=threading.Thread(target=run,daemon=True)
            thread.start()
            self.assertTrue(started.wait(7),'GPU call not started')
            cancel.set()
            release.set()
            thread.join(7)
        self.assertFalse(thread.is_alive())
        self.assertNotIn('error',output)
        self.assertEqual(output['result']['status'],'PAUSED')
        self.assertEqual(fake.upserts,[])
        self.assertEqual(gpu.metrics()['document_batches'],[8])
        gpu.close()

    def test_04_early_future_completion_waits_for_cuda(self):
        busy,release,ended=threading.Event(),threading.Event(),threading.Event()
        def encode(_model,batch,**_):
            busy.set()
            if not release.wait(7): raise RuntimeError('test blocked')
            return [[1.,0.] for _ in batch]
        gpu=GPUModelService(GPUServiceConfig(),model_factory=lambda:object(),encode_fn=encode)
        request=gpu.submit_documents(['x'])
        self.assertTrue(busy.wait(7))
        request.cancel()
        waiting=threading.Thread(target=lambda:(request.wait_execution_end(),ended.set()))
        waiting.start()
        self.assertFalse(ended.wait(.06),'Future done but CUDA still busy')
        release.set()
        self.assertTrue(ended.wait(7))
        waiting.join(3)
        gpu.close()

    def test_05_recovery_is_scoped_and_preserves_completed(self):
        with tempfile.TemporaryDirectory() as td,patch.object(J,'DB_PATH',Path(td)/'jobs.sqlite'):
            with J.connect() as db:
                for status,ident in [('RUNNING','1'),('DOWNLOADING','2'),('QUEUED','3'),('COMPLETE','4')]:
                    db.execute('''INSERT INTO jobs(job_id,origin,collection,filename,source,source_sha256,
                        status,stage,created_at,updated_at,requested) VALUES (?,?,?,?,?,?,?,?,?,?,?)''',
                        (ident*32,'UI','tmp','fake.txt','fake.txt',ident*64,status,'old',J.now(),J.now(),1))
            scheduler=C.BackendScheduler(object())
            def check(job,**_):
                if job['job_id']=='1'*32: return {'phase':'CHUNKS_STORED','total_chunks':17}
                raise ValueError('missing')
            with patch.object(C,'_compatible',side_effect=check):
                scheduler.recover()
            statuses={s:J.one(ident*32,private=True) for s,ident in [('RUNNING','1'),('DOWNLOADING','2'),('QUEUED','3'),('COMPLETE','4')]}
            self.assertEqual(statuses['RUNNING']['status'],'PAUSED')
            self.assertEqual(statuses['DOWNLOADING']['status'],'FAILED')
            self.assertEqual(statuses['QUEUED']['status'],'QUEUED')
            self.assertFalse(statuses['QUEUED']['requested'])
            self.assertEqual(statuses['COMPLETE']['status'],'COMPLETE')

    def test_06_invalid_restart_does_not_delete_parsed_points(self):
        # A running job must be rejected *before* Qdrant deletion.
        fake=SimpleNamespace()
        sha='d'*64
        job={'job_id':'d'*32,'status':'RUNNING','source_sha256':sha,
             'collection':'safe_col','source':'missing_source.txt'}
        scheduler=C.BackendScheduler(object())
        with patch.object(C,'one',return_value=job), \
             patch.object(C,'_client',return_value=fake), \
             patch.object(C,'sha256_file',side_effect=AssertionError('must not hash')):
            with self.assertRaisesRegex(ValueError,'job is not paused'):
                scheduler.action(job['job_id'],'restart')

    def test_07_db_read_failure_does_not_spin(self):
        scheduler=C.BackendScheduler(object())
        reads=[0]
        def bad_connect():
            reads[0]+=1
            raise RuntimeError('test DB unavailable')
        with patch.object(C,'connect',side_effect=bad_connect):
            thread=threading.Thread(target=scheduler.pump,daemon=True)
            thread.start()
            scheduler._wake.set()
            time.sleep(.15)
            self.assertLessEqual(reads[0],2,'unbounded self-wake on DB failure')
            scheduler._stop.set(); scheduler._wake.set()
            thread.join(2)
            self.assertFalse(thread.is_alive())

if __name__=='__main__':
    unittest.main(verbosity=2)
