"""Stage C hardening regressions using only temporary SQLite and fake GPU/Qdrant.

The tests deliberately exercise the backend scheduler's public action path.
They never contact the configured Qdrant server or load BGE/Docling.
"""
from __future__ import annotations

import hashlib
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from qdrant_client import models

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag3 import ingestion_backend as C
from rag3 import ingestion_jobs as J
from rag3.gpu_model_service import GPUModelService, GPUServiceConfig


def _job(db, source: Path, *, job_id: str = 'a' * 32,
         status: str = 'RUNNING', mode: str = 'resume') -> dict:
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    with db.connect() as conn:
        conn.execute('''INSERT INTO jobs
            (job_id,origin,collection,filename,source,source_sha256,status,stage,
             created_at,updated_at,requested,mode)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)''',
            (job_id, 'MCP', 'test_hardening', source.name, str(source), sha,
             status, 'test', db.now(), db.now(), 1, mode))
    return db.one(job_id, private=True)


def _wait(predicate, timeout=4):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(.01)
    return predicate()


class FakeQdrant:
    def __init__(self, sha):
        self.point = SimpleNamespace(
            id='11111111-1111-4111-8111-111111111111',
            payload={'content': 'one chunk', 'meta': {
                'source_sha256': sha, '_rag3_ingest_state': 'parsed'}},
            vector={})
        self.upserts = 0
        self.write_entered = None
        self.write_release = None
        self.accepted_cancel = None
        self.committed_after_cancel = False

    def collection_exists(self, _collection):
        return True

    def get_collection(self, **_):
        return SimpleNamespace(model_dump=lambda: {'config': {
            'params': {'sparse_vectors': {'bm25': {}}},
            'metadata': {'rag3': {'bm25': {'k': 1.2, 'b': .75, 'avg_len': 256}}}}})

    def scroll(self, **_):
        if self.point.payload['meta']['_rag3_ingest_state'] == 'parsed':
            return [self.point], None
        return [], None

    def upsert(self, *, points, **_):
        if self.write_entered is not None:
            self.write_entered.set()
            if not self.write_release.wait(4):
                raise TimeoutError('test did not release fake Qdrant write')
        self.committed_after_cancel = bool(
            self.accepted_cancel and self.accepted_cancel.is_set())
        self.upserts += 1
        self.point.payload = points[0].payload
        self.point.vector = points[0].vector

    def count(self, *, count_filter=None, **_):
        point = self.point
        keep = True
        if count_filter is not None:
            for clause in count_filter.must:
                if isinstance(clause, models.HasVectorCondition):
                    keep &= clause.has_vector in point.vector
                elif isinstance(clause, models.FieldCondition):
                    key = clause.key.split('.')[-1]
                    keep &= point.payload['meta'].get(key) == clause.match.value
        return SimpleNamespace(count=int(bool(keep)))


class HardeningTests(unittest.TestCase):
    def test_thread_start_failure_releases_scheduler(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / 'source.txt'
            source.write_text('thread startup failure', encoding='utf-8')
            with patch.object(J, 'DB_PATH', root / 'jobs.sqlite'), \
                 patch.object(C, 'ROOT', root):
                job = _job(J, source, status='QUEUED', mode='run')
                scheduler = C.BackendScheduler(object())

                class FailingThread:
                    def __init__(self, **_):
                        pass
                    def start(self):
                        raise RuntimeError('forced Thread.start failure')

                with patch.object(scheduler, 'validate_collection'), \
                     patch.object(scheduler, 'conflict'), \
                     patch.object(C.threading, 'Thread', FailingThread):
                    with self.assertRaisesRegex(RuntimeError, 'forced Thread.start failure'):
                        scheduler.start(job)
                self.assertIsNone(scheduler.active_id)
                self.assertIsNone(scheduler._task)
                self.assertNotIn(J.one(job['job_id'])['status'], {'RUNNING', 'RESUMING'})

    def test_finalize_failure_releases_scheduler(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / 'source.txt'
            source.write_text('finalize failure', encoding='utf-8')
            with patch.object(J, 'DB_PATH', root / 'jobs.sqlite'), \
                 patch.object(C, 'ROOT', root):
                job = _job(J, source)
                J.set_ingestion_progress(job['collection'], job['source_sha256'],
                                         {'test': True}, 'CHUNKS_STORED', 1)
                scheduler = C.BackendScheduler(object())
                scheduler.active_id = job['job_id']
                result = {'status': 'COMPLETE'}
                with patch.object(C, '_compatible', return_value={
                         'phase': 'CHUNKS_STORED', 'total_chunks': 1}), \
                     patch.object(C, '_embed_source', return_value=result), \
                     patch.object(C, 'clear_ingestion_progress',
                                  side_effect=RuntimeError('forced cleanup failure')):
                    try:
                        scheduler._execute(job)
                    except RuntimeError as exc:
                        self.assertIn('forced cleanup failure', str(exc))
                self.assertIsNone(scheduler.active_id)
                self.assertIsNone(scheduler._task)
                self.assertTrue(scheduler._wake.is_set())
                self.assertEqual(J.one(job['job_id'])['status'], 'COMPLETE')

    def test_cancel_removes_waiting_document_gpu_request(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / 'source.txt'
            source.write_text('queued GPU cancellation', encoding='utf-8')
            with patch.object(J, 'DB_PATH', root / 'jobs.sqlite'), \
                 patch.object(C, 'ROOT', root):
                job = _job(J, source)
                J.set_ingestion_progress(job['collection'], job['source_sha256'],
                                         {'test': True}, 'CHUNKS_STORED', 1)
                qdrant = FakeQdrant(job['source_sha256'])
                query_running, release_query = threading.Event(), threading.Event()
                encoded_documents = []

                def encode(_model, batch, *, query, **_):
                    if query:
                        query_running.set()
                        if not release_query.wait(5):
                            raise TimeoutError('query blocker was not released')
                    else:
                        encoded_documents.extend(batch)
                    return [[.1, .2] for _ in batch]

                gpu = GPUModelService(GPUServiceConfig(), model_factory=object,
                                      encode_fn=encode)
                scheduler = C.BackendScheduler(gpu)
                scheduler.active_id = job['job_id']
                errors = []
                with patch.object(C, '_client', return_value=qdrant), \
                     patch.object(C, '_compatible', return_value={
                         'phase': 'CHUNKS_STORED', 'total_chunks': 1}):
                    query = gpu.submit_query('occupy owner')
                    self.assertTrue(query_running.wait(4))
                    worker = threading.Thread(target=lambda: _run(scheduler, job, errors))
                    worker.start()
                    try:
                        self.assertTrue(_wait(lambda: gpu.snapshot()['document_waiting'] == 1),
                                        'document request never entered GPU queue')
                        self.assertEqual(scheduler.action(job['job_id'], 'cancel')['status'],
                                         'CANCELLING')
                    finally:
                        release_query.set()
                    query.result(timeout=4)
                    worker.join(5)
                gpu.close()
                self.assertFalse(worker.is_alive())
                self.assertEqual(errors, [])
                self.assertEqual(encoded_documents, [])
                self.assertEqual(qdrant.upserts, 0)
                self.assertEqual(J.one(job['job_id'])['status'], 'CANCELLED')

    def test_cancel_cannot_commit_after_acceptance(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / 'source.txt'
            source.write_text('cancel while upsert is about to commit', encoding='utf-8')
            with patch.object(J, 'DB_PATH', root / 'jobs.sqlite'), \
                 patch.object(C, 'ROOT', root):
                job = _job(J, source)
                J.set_ingestion_progress(job['collection'], job['source_sha256'],
                                         {'test': True}, 'CHUNKS_STORED', 1)
                qdrant = FakeQdrant(job['source_sha256'])
                qdrant.write_entered = threading.Event()
                qdrant.write_release = threading.Event()
                qdrant.accepted_cancel = threading.Event()

                class GPU:
                    def submit_documents(self, _texts):
                        return SimpleNamespace(result=lambda: [[.1, .2]],
                                               wait_execution_end=lambda: None,
                                               cancel=lambda: True)

                scheduler = C.BackendScheduler(GPU())
                scheduler.active_id = job['job_id']
                errors, cancel_errors = [], []
                with patch.object(C, '_client', return_value=qdrant), \
                     patch.object(C, '_compatible', return_value={
                         'phase': 'CHUNKS_STORED', 'total_chunks': 1}):
                    worker = threading.Thread(target=lambda: _run(scheduler, job, errors))
                    worker.start()
                    try:
                        self.assertTrue(qdrant.write_entered.wait(4),
                                        'ready upsert was never reached')
                        cancel_entered = threading.Event()

                        def cancel():
                            cancel_entered.set()
                            try:
                                scheduler.action(job['job_id'], 'cancel')
                            except ValueError as exc:
                                # A completed write may make the later Cancel ineligible.
                                cancel_errors.append(exc)
                            else:
                                qdrant.accepted_cancel.set()

                        canceller = threading.Thread(target=cancel)
                        canceller.start()
                        self.assertTrue(cancel_entered.wait(2))
                        # Give an unlocked Cancel a chance to return before
                        # releasing the fake write; a locked one must wait.
                        qdrant.accepted_cancel.wait(.2)
                    finally:
                        qdrant.write_release.set()
                    worker.join(5)
                    canceller.join(5)
                self.assertFalse(worker.is_alive())
                self.assertFalse(canceller.is_alive())
                self.assertEqual(errors, [])
                self.assertFalse(qdrant.committed_after_cancel,
                                 'ready point was written after Cancel returned successfully')

    def test_qdrant_outage_defers_checkpoint_verification(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / 'source.txt'
            source.write_text('Qdrant outage', encoding='utf-8')
            with patch.object(J, 'DB_PATH', root / 'jobs.sqlite'), \
                 patch.object(C, 'ROOT', root):
                job = _job(J, source)
                identity = {'must': 'remain'}
                J.set_ingestion_progress(job['collection'], job['source_sha256'],
                                         identity, 'CHUNKS_STORED', 1)
                scheduler = C.BackendScheduler(object())
                with patch.object(C, '_compatible',
                                  side_effect=ConnectionError('Qdrant offline')):
                    scheduler.recover()
                recovered = J.one(job['job_id'], private=True)
                self.assertEqual(recovered['status'], 'PAUSED')
                self.assertFalse(recovered['requested'])
                self.assertIn('defer', recovered['stage'].lower())
                self.assertEqual(J.ingestion_progress(job['collection'],
                    job['source_sha256'])['identity'], identity)

    def test_plain_parsed_upsert_finishes_before_cancel_ack(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / 'source.txt'
            source.write_text('plain parsed write', encoding='utf-8')
            with patch.object(J, 'DB_PATH', root / 'jobs.sqlite'), \
                 patch.object(C, 'ROOT', root):
                job = _job(J, source)
                J.set_ingestion_progress(job['collection'], job['source_sha256'],
                                         {'test': True}, 'PARSING')
                qdrant = FakeQdrant(job['source_sha256'])
                qdrant.write_entered = threading.Event()
                qdrant.write_release = threading.Event()
                qdrant.accepted_cancel = threading.Event()
                scheduler = C.BackendScheduler(object())
                scheduler.active_id = job['job_id']
                errors = []
                fake_chunk = SimpleNamespace(meta={})
                fake_point = SimpleNamespace(payload={
                    'content': 'plain parsed write', 'meta': {
                        'source_sha256': job['source_sha256'],
                        '_rag3_ingest_state': 'parsed'}}, vector={})

                def cancel():
                    try:
                        scheduler.action(job['job_id'], 'cancel')
                    except ValueError:
                        # The already completed parse may make Cancel ineligible.
                        pass
                    else:
                        qdrant.accepted_cancel.set()

                with patch.object(C, '_client', return_value=qdrant), \
                     patch.object(C, '_compatible', return_value={
                         'phase': 'PARSING', 'total_chunks': 0}), \
                     patch('rag3.ingestion_light.preserve_source', return_value={
                         'title': source.stem, 'source_uri': str(source)}), \
                     patch('rag3.ingestion_light.split_plain_text_source',
                           return_value=[fake_chunk]), \
                     patch('rag3.ingestion_light.normalize_chunk',
                           return_value=fake_chunk), \
                     patch('haystack_integrations.document_stores.qdrant.converters'
                           '.convert_haystack_documents_to_qdrant_points',
                           return_value=[fake_point]):
                    worker = threading.Thread(target=lambda: _run(scheduler, job, errors))
                    worker.start()
                    try:
                        self.assertTrue(qdrant.write_entered.wait(4),
                                        'parsed upsert was never reached')
                        canceller = threading.Thread(target=cancel)
                        canceller.start()
                        qdrant.accepted_cancel.wait(.2)
                    finally:
                        qdrant.write_release.set()
                    worker.join(5)
                    canceller.join(5)
                self.assertFalse(worker.is_alive())
                self.assertFalse(canceller.is_alive())
                self.assertEqual(errors, [])
                self.assertEqual(qdrant.upserts, 1)
                self.assertFalse(qdrant.committed_after_cancel,
                                 'parsed point was written after Cancel returned')

    def test_parsing_checkpoint_with_ready_point_rejects_resume(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / 'source.txt'
            source.write_text('inconsistent parsing checkpoint', encoding='utf-8')
            with patch.object(J, 'DB_PATH', root / 'jobs.sqlite'), \
                 patch.object(C, 'ROOT', root):
                job = _job(J, source, status='PAUSED')
                identity = C.progress_identity(job['source_sha256'], job['collection'])
                J.set_ingestion_progress(job['collection'], job['source_sha256'],
                                         identity, 'PARSING')
                with J.connect() as db:
                    J.patch(db, job['job_id'], requested=0)
                qdrant = FakeQdrant(job['source_sha256'])
                qdrant.point.payload['meta']['_rag3_ingest_state'] = 'ready'
                scheduler = C.BackendScheduler(object())
                with patch.object(C, '_client', return_value=qdrant):
                    with self.assertRaisesRegex(ValueError,
                                                'PARSING checkpoint contains ready points'):
                        scheduler.action(job['job_id'], 'resume')
                saved = J.one(job['job_id'], private=True)
                self.assertEqual(saved['status'], 'PAUSED')
                self.assertFalse(saved['requested'])
                self.assertEqual(qdrant.upserts, 0)
                self.assertEqual(qdrant.point.payload['meta']['_rag3_ingest_state'],
                                 'ready')
                self.assertEqual(J.ingestion_progress(job['collection'],
                    job['source_sha256'])['identity'], identity)

    def test_graceful_stop_marks_active_job_pausing_and_cancels_gpu(self):
        for status in ('RUNNING', 'RESUMING'):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                source = root / 'source.txt'
                source.write_text(f'backend stop {status}', encoding='utf-8')
                with patch.object(J, 'DB_PATH', root / 'jobs.sqlite'), \
                     patch.object(C, 'ROOT', root):
                    job = _job(J, source, status=status)
                    identity = {'checkpoint': status}
                    J.set_ingestion_progress(job['collection'], job['source_sha256'],
                                             identity, 'CHUNKS_STORED', 1)

                    class Request:
                        cancelled = 0
                        def cancel(self):
                            self.cancelled += 1
                            return True

                    request = Request()
                    scheduler = C.BackendScheduler(object())
                    scheduler.active_id = job['job_id']
                    scheduler._gpu_request = request
                    scheduler.stop()

                    saved = J.one(job['job_id'], private=True)
                    self.assertEqual(saved['status'], 'PAUSING')
                    self.assertFalse(saved['requested'])
                    self.assertIn('preserve checkpoint', saved['stage'])
                    self.assertEqual(request.cancelled, 1)
                    self.assertTrue(scheduler._stop.is_set())
                    self.assertEqual(J.ingestion_progress(job['collection'],
                        job['source_sha256'])['identity'], identity)


def _run(scheduler, job, errors):
    try:
        scheduler._execute(job)
    except Exception as exc:
        errors.append(exc)


if __name__ == '__main__':
    unittest.main(verbosity=2)
