"""Deterministic Stage B contract tests; isolated fakes, no live collections."""
import asyncio
import importlib
import math
import sys
import threading
import unittest
from collections import deque
from concurrent.futures import TimeoutError
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rag3.bge_dense import encode_dense
from rag3.gpu_model_service import (GPUModelService, GPUServiceConfig,
                                    GPURequestExpired, GPUServiceError)


def service(encode, *, factory=None, **opts):
    config = GPUServiceConfig(**opts)
    return GPUModelService(config, model_factory=factory or (lambda: object()), encode_fn=encode)


class Blocker:
    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.calls = []
        self.lock = threading.Lock()

    def __call__(self, _model, texts, *, query, **kwargs):
        with self.lock:
            self.calls.append(('q' if query else 'd', len(texts), list(texts)))
            first = len(self.calls) == 1
        if first:
            self.entered.set()
            if not self.release.wait(5):
                raise AssertionError('test blocker timeout')
        return [[float(len(t))] for t in texts]


class ServiceTest(unittest.TestCase):
    def test_cold_creation_and_same_model_owner(self):
        modules_before = {x for x in sys.modules if x.split('.')[0] in
                          ('torch', 'FlagEmbedding', 'docling')}
        factory_models = []
        owner_ids = []

        def factory():
            model = object()
            factory_models.append(model)
            return model

        def encode(model, texts, *, query, **_):
            self.assertIs(model, factory_models[0])
            owner_ids.append(threading.get_ident())
            return [[1.0, 0.0] for _ in texts]

        s = service(encode, factory=factory)
        self.assertEqual(s.snapshot()['state'], 'not_initialized')
        self.assertEqual(modules_before, {x for x in sys.modules if x.split('.')[0]
                         in ('torch', 'FlagEmbedding', 'docling')})
        try:
            results = []
            threads = [threading.Thread(target=lambda: results.append(s.submit_query('test').result()))
                       for _ in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(4)
            self.assertEqual(len(results), 8)
            self.assertEqual(s.submit_documents(['a', 'b']).result(), [[1.0, 0.0]] * 2)
            self.assertEqual(len(factory_models), 1)
            self.assertEqual(len(set(owner_ids)), 1)
            self.assertEqual(s.snapshot()['model_load_count'], 1)
        finally:
            s.close()

    def test_17_batches_with_interleaved_query(self):
        b = Blocker()
        s = service(b)
        try:
            doc = s.submit_documents(['d'] * 17)
            self.assertTrue(b.entered.wait(2))
            query = s.submit_query('q')
            b.release.set()
            self.assertEqual(len(doc.result(timeout=4)), 17)
            self.assertEqual(query.result(timeout=4), [1.0])
            self.assertEqual([(kind, size) for kind, size, _ in b.calls],
                             [('d', 8), ('q', 1), ('d', 8), ('d', 1)])
        finally:
            b.release.set(); s.close()

    def test_query_burst_bound_and_fifo(self):
        b = Blocker()
        s = service(b, max_query_burst=3)
        try:
            active = s.submit_query('hold')
            self.assertTrue(b.entered.wait(2))
            queries = [s.submit_query(str(i)) for i in range(6)]
            doc = s.submit_documents(['doc'])
            b.release.set()
            for handle in queries + [doc, active]:
                handle.result(timeout=4)
            order = [a[0] for a in b.calls]
            # The active "hold" query began while no document waited; only
            # subsequent queries consume the fairness burst.
            self.assertEqual(order[:5], ['q', 'q', 'q', 'q', 'd'])
            self.assertEqual([a[2][0] for a in b.calls if a[0] == 'q'],
                             ['hold'] + [str(i) for i in range(6)])
        finally:
            b.release.set(); s.close()

    def test_uncontended_query_history_does_not_consume_priority(self):
        hold_entered, release_hold = threading.Event(), threading.Event()
        calls = []
        lock = threading.Lock()

        def encode(_model, texts, *, query, **_):
            with lock:
                calls.append(('q' if query else 'd', texts[0]))
            if query and texts == ['hold']:
                hold_entered.set()
                if not release_hold.wait(5):
                    raise AssertionError('held query was not released')
            return [[1.0] for _ in texts]

        s = service(encode, max_query_burst=2)
        try:
            s.submit_query('historical').result(timeout=4)
            hold = s.submit_query('hold')
            self.assertTrue(hold_entered.wait(2))
            document = s.submit_documents(['doc'])
            fresh_query = s.submit_query('fresh')
            release_hold.set()
            for request in (hold, fresh_query, document):
                request.result(timeout=4)
            self.assertEqual(calls,
                             [('q', 'historical'), ('q', 'hold'),
                              ('q', 'fresh'), ('d', 'doc')])
        finally:
            release_hold.set(); s.close()

    def test_document_round_robin(self):
        b = Blocker()
        s = service(b)
        try:
            a = s.submit_documents(['a'] * 17)
            self.assertTrue(b.entered.wait(2))
            z = s.submit_documents(['z'] * 9)
            b.release.set()
            a.result(timeout=4); z.result(timeout=4)
            self.assertEqual([v[2][0] for v in b.calls], ['a', 'z', 'a', 'z', 'a'])
            self.assertEqual([v[1] for v in b.calls], [8, 8, 8, 1, 1])
        finally:
            b.release.set(); s.close()

    def test_two_documents_rotate_with_queue_limit_one(self):
        first_entered, release_first = threading.Event(), threading.Event()
        calls = []
        queue_sizes = []

        def encode(_model, texts, *, query, **_):
            self.assertFalse(query)
            calls.append((texts[0], len(texts)))
            if len(calls) == 1:
                first_entered.set()
                if not release_first.wait(5):
                    raise AssertionError('first document batch was not released')
            return [[1.0 if text == 'a' else 2.0] for text in texts]

        s = service(encode, queue_limit=1)

        class ObservedDocuments(deque):
            def append(self, request):
                super().append(request)
                queue_sizes.append(len(s._queries) + len(s._documents))

        s._documents = ObservedDocuments()
        try:
            a = s.submit_documents(['a'] * 17)
            self.assertTrue(first_entered.wait(2))
            b = s.submit_documents(['b'] * 9)
            release_first.set()
            self.assertEqual(a.result(timeout=4), [[1.0]] * 17)
            self.assertEqual(b.result(timeout=4), [[2.0]] * 9)
            self.assertEqual(calls, [('a', 8), ('b', 8), ('a', 8),
                                     ('b', 1), ('a', 1)])
            self.assertLessEqual(max(queue_sizes), 1)
            self.assertFalse(s.snapshot()['document_deferred'])
        finally:
            release_first.set(); s.close()

    def test_deferred_document_precedes_later_document(self):
        first_entered, release_first = threading.Event(), threading.Event()
        query_entered, release_query = threading.Event(), threading.Event()
        calls = []

        def encode(_model, texts, *, query, **_):
            calls.append(('q' if query else texts[0], len(texts)))
            if len(calls) == 1:
                first_entered.set()
                if not release_first.wait(5):
                    raise AssertionError('first document batch was not released')
            elif query:
                query_entered.set()
                if not release_query.wait(5):
                    raise AssertionError('query was not released')
            value = 0.0 if query else (1.0 if texts[0] == 'a' else 2.0)
            return [[value] for _ in texts]

        s = service(encode, queue_limit=1)
        try:
            a = s.submit_documents(['a'] * 17)
            self.assertTrue(first_entered.wait(2))
            query = s.submit_query('query')
            release_first.set()
            self.assertTrue(query_entered.wait(2))
            self.assertTrue(s.snapshot()['document_deferred'])
            b = s.submit_documents(['b'] * 9)
            release_query.set()
            self.assertEqual(query.result(timeout=4), [0.0])
            self.assertEqual(a.result(timeout=4), [[1.0]] * 17)
            self.assertEqual(b.result(timeout=4), [[2.0]] * 9)
            self.assertEqual(calls, [('a', 8), ('q', 1), ('a', 8),
                                     ('b', 8), ('a', 1), ('b', 1)])
        finally:
            release_first.set(); release_query.set(); s.close()

    def test_saturation_and_cancel_queued(self):
        b = Blocker()
        s = service(b, queue_limit=2)
        try:
            first = s.submit_documents(['a'])
            self.assertTrue(b.entered.wait(2))
            one = s.submit_query('1'); two = s.submit_query('2')
            with self.assertRaisesRegex(GPUServiceError, 'QUEUE_FULL'):
                s.submit_query('3')
            self.assertTrue(one.cancel())
            with self.assertRaisesRegex(GPUServiceError, 'CANCELLED'):
                one.future.result(timeout=.1)
            third = s.submit_query('3')
            b.release.set()
            first.result(timeout=4); two.result(timeout=4); third.result(timeout=4)
            self.assertNotIn('1', [v[2][0] for v in b.calls])
        finally:
            b.release.set(); s.close()

    def test_expired_pending_recovers_slot(self):
        b = Blocker()
        s = service(b, queue_limit=1)
        try:
            first = s.submit_documents(['a'])
            self.assertTrue(b.entered.wait(2))
            expired = s.submit_query('old', deadline_seconds=.015)
            threading.Event().wait(.03)
            third = s.submit_query('new')
            b.release.set(); first.result(timeout=4); third.result(timeout=4)
            with self.assertRaisesRegex(GPUServiceError, 'DEADLINE_EXCEEDED'):
                expired.future.result(timeout=2)
            self.assertNotIn('old', [x[2][0] for x in b.calls])
        finally:
            b.release.set(); s.close()

    def test_queued_deadline_completes_future_while_factory_is_blocked(self):
        loading, release_factory = threading.Event(), threading.Event()
        encoded = []

        def factory():
            loading.set()
            if not release_factory.wait(5):
                raise AssertionError('model factory was not released')
            return object()

        def encode(_model, texts, *, query, **_):
            encoded.extend(texts)
            return [[1.0] for _ in texts]

        s = service(encode, factory=factory)
        try:
            active = s.submit_query('active', deadline_seconds=.09)
            self.assertTrue(loading.wait(2))
            queued = s.submit_query('expired', deadline_seconds=.03)
            # Deliberately use Future.result, without GPURequest.result or another
            # submission. The service itself must retire both the queued request
            # and the active request, even while the model factory is blocked.
            with self.assertRaises(GPURequestExpired):
                queued.future.result(timeout=2)
            with self.assertRaises(GPURequestExpired):
                active.future.result(timeout=2)
            self.assertEqual(s.snapshot()['query_waiting'], 0)
            self.assertNotIn('expired', encoded)
            release_factory.set()
        finally:
            release_factory.set(); s.close()
        self.assertEqual(encoded, [])

    def test_document_continuation_never_exceeds_waiting_limit(self):
        first_entered = threading.Event()
        release_first = threading.Event()
        query_entered = threading.Event()
        release_query = threading.Event()
        calls = []
        queue_sizes = []

        def encode(_model, texts, *, query, **_):
            calls.append(('q' if query else 'd', len(texts)))
            if len(calls) == 1:
                first_entered.set()
                if not release_first.wait(5):
                    raise AssertionError('first document batch was not released')
            elif query:
                query_entered.set()
                if not release_query.wait(5):
                    raise AssertionError('query was not released')
            return [[1.0] for _ in texts]

        s = service(encode, queue_limit=1)

        class ObservedDocuments(deque):
            def append(self, request):
                super().append(request)
                # append executes under the service Condition. This records the
                # otherwise invisible instant before the owner pops the next job.
                queue_sizes.append(len(s._queries) + len(s._documents))

        s._documents = ObservedDocuments()
        try:
            document = s.submit_documents(['doc'] * 17)
            self.assertTrue(first_entered.wait(2))
            query = s.submit_query('query')
            release_first.set()
            self.assertTrue(query_entered.wait(2))
            self.assertTrue(s.snapshot()['document_deferred'])
            self.assertLessEqual(max(queue_sizes), 1)
            release_query.set()
            self.assertEqual(query.result(timeout=4), [1.0])
            self.assertEqual(len(document.result(timeout=4)), 17)
            self.assertEqual(calls, [('d', 8), ('q', 1), ('d', 8), ('d', 1)])
        finally:
            release_first.set(); release_query.set(); s.close()

    def test_running_cancel_stops_following_batch(self):
        b = Blocker()
        s = service(b)
        try:
            doc = s.submit_documents(['a'] * 17)
            self.assertTrue(b.entered.wait(2))
            self.assertTrue(doc.cancel())
            b.release.set()
            with self.assertRaisesRegex(GPUServiceError, 'CANCELLED'):
                doc.future.result(timeout=4)
            self.assertEqual([v[1] for v in b.calls], [8])
        finally:
            b.release.set(); s.close()

    def test_load_failure_fails_all_pending(self):
        entered, release = threading.Event(), threading.Event()
        def broken():
            entered.set()
            if not release.wait(5):
                raise AssertionError('timeout')
            raise RuntimeError('broken load')
        s = service(lambda *_args, **_kwargs: [], factory=broken)
        try:
            first = s.submit_query('a')
            self.assertTrue(entered.wait(2))
            second = s.submit_documents(['b'])
            release.set()
            for req in (first, second):
                with self.assertRaisesRegex(GPUServiceError, 'GPU_INIT_FAILED'):
                    req.future.result(timeout=4)
            self.assertEqual(s.snapshot()['state'], 'failed')
            self.assertEqual(s.snapshot()['query_waiting'], 0)
        finally:
            release.set(); s.close()

    def test_cuda_oom_fails_all(self):
        b = Blocker()
        def oom(*a, **kw):
            b(*a, **kw)
            raise RuntimeError('CUDA out of memory')
        s = service(oom)
        try:
            first = s.submit_query('a')
            self.assertTrue(b.entered.wait(2))
            second = s.submit_documents(['b'])
            b.release.set()
            for req in (first, second):
                with self.assertRaisesRegex(GPUServiceError, 'CUDA_OOM'):
                    req.future.result(timeout=4)
            self.assertEqual(s.snapshot()['state'], 'failed')
        finally:
            b.release.set(); s.close()

    def test_close_cancels_queued_and_current(self):
        b = Blocker()
        s = service(b)
        active = s.submit_query('a')
        self.assertTrue(b.entered.wait(2))
        queued = s.submit_query('b')
        s.close(wait=False)
        b.release.set()
        for req in (active, queued):
            with self.assertRaises(GPUServiceError):
                req.future.result(timeout=4)
        s.close()

    def test_close_without_wait_releases_active_future_before_gpu_returns(self):
        blocker = Blocker()
        s = service(blocker)
        try:
            active = s.submit_query('active')
            self.assertTrue(blocker.entered.wait(2))
            s.close(wait=False)
            self.assertFalse(blocker.release.is_set())
            with self.assertRaisesRegex(GPUServiceError, 'SERVICE_STOPPED'):
                active.future.result(timeout=.5)
        finally:
            blocker.release.set(); s.close()

    def test_input_and_deadline_bounds(self):
        s = service(lambda *_args, **_kwargs: [], max_document_texts=17)
        try:
            for operation in (lambda: s.submit_query(''),
                              lambda: s.submit_documents(['x'] * 18),
                              lambda: s.submit_query('hi', deadline_seconds=-1),
                              lambda: s.submit_documents([])):
                with self.assertRaises(GPUServiceError):
                    operation()
            self.assertEqual(s.snapshot()['state'], 'not_initialized')
        finally:
            s.close()

    def test_input_memory_and_finite_deadlines(self):
        s = service(lambda *_a, **_k: [], max_text_chars=8)
        try:
            invalid = [lambda: s.submit_query('x' * 9),
                       lambda: s.submit_documents(['x' * 9]),
                       lambda: s.submit_query('ok', deadline_seconds=float('nan')),
                       lambda: s.submit_query('ok', deadline_seconds=float('inf')),
                       lambda: s.submit_query('ok', deadline_monotonic=float('nan')),
                       lambda: s.submit_query('ok', deadline_monotonic=float('inf')),
                       lambda: s.submit_query('ok', deadline_monotonic='bad')]
            for action in invalid:
                with self.assertRaisesRegex(GPUServiceError, 'INVALID_ARGUMENT'):
                    action()
            self.assertEqual(s.snapshot()['state'], 'not_initialized')
            with self.assertRaises(ValueError):
                GPUServiceConfig(default_deadline_seconds=float('inf'))
        finally:
            s.close()

    def test_close_during_loading_does_not_mark_ready(self):
        entered, release = threading.Event(), threading.Event()
        call_count = []
        def slow_factory():
            entered.set()
            if not release.wait(5):
                raise AssertionError('timeout')
            call_count.append(1)
            return object()
        s = service(lambda *_a, **_k: [[1.0]], factory=slow_factory)
        req = s.submit_query('x')
        self.assertTrue(entered.wait(2))
        s.close(wait=False)
        release.set()
        with self.assertRaises(GPUServiceError):
            req.future.result(timeout=4)
        s.close()
        self.assertEqual(s.snapshot()['state'], 'stopped')
        self.assertEqual(s.snapshot()['model_load_count'], 1)
        self.assertEqual(call_count, [1])

    def test_deadline_while_loading_skips_encoding(self):
        entered, release = threading.Event(), threading.Event()
        encoded = []
        def factory():
            entered.set()
            if not release.wait(5):
                raise AssertionError('timeout')
            return object()
        def encode(*args, **kwargs):
            encoded.append(1)
            return [[1.0]]
        s = service(encode, factory=factory)
        req = s.submit_query('test', deadline_seconds=.02)
        self.assertTrue(entered.wait(2))
        self.assertTrue(threading.Event().wait(.035) is False)
        release.set()
        with self.assertRaisesRegex(GPUServiceError, 'DEADLINE_EXCEEDED'):
            req.future.result(timeout=4)
        self.assertEqual(encoded, [])
        s.close()

    def test_query_corpus_normalization_contract(self):
        import numpy as np
        class Fake:
            def __init__(self): self.calls = []
            def encode_queries(self, t, **k):
                self.calls.append(('query', k))
                return {'dense_vecs': [[3.0, 4.0]]}
            def encode_corpus(self, t, **k):
                self.calls.append(('corpus', k))
                return {'dense_vecs': [[0.0, 2.0], [0.0, 0.0]]}
        model = Fake()
        self.assertEqual(encode_dense(model, ['x'], query=True), [[.6000000238418579, .800000011920929]])
        self.assertEqual(encode_dense(model, ['x', 'y'], query=False), [[0.0, 1.0], [0.0, 0.0]])
        self.assertEqual([c[0] for c in model.calls], ['query', 'corpus'])
        self.assertTrue(all(c[1]['return_sparse'] is False and c[1]['return_colbert_vecs'] is False
                            for c in model.calls))


if __name__ == '__main__':
    unittest.main(verbosity=2)
