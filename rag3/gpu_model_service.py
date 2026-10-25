"""Stage B: one lazy GPU BGE-M3 owner and bounded, fair batch scheduler.

No torch, FlagEmbedding, numpy, Docling, OCR import occurs on module import,
construction, or status inspection. Legacy retrieval/ingestion paths are NOT
connected until stages C/D.
"""
from __future__ import annotations

import os
import math
import gc
import threading
import time
from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Callable


class GPUServiceError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(f'{code}: {message}')
        self.code = code


class GPURequestCancelled(GPUServiceError):
    def __init__(self):
        super().__init__('CANCELLED', 'GPU encoding request was cancelled')


class GPURequestExpired(GPUServiceError):
    def __init__(self):
        super().__init__('DEADLINE_EXCEEDED', 'GPU encoding request deadline exceeded')


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


@dataclass(frozen=True)
class GPUServiceConfig:
    # 8 matches ingestion BGE_BATCH_SIZE; limits are per in-flight *request*,
    # with only waiting requests counted in queue_limit.
    batch_size: int = field(default_factory=lambda: _env_int('RAG3_GPU_BATCH_SIZE', 8))
    queue_limit: int = field(default_factory=lambda: _env_int('RAG3_GPU_QUEUE_LIMIT', 32))
    max_document_texts: int = field(default_factory=lambda: _env_int('RAG3_GPU_MAX_DOC_TEXTS', 256))
    max_text_chars: int = field(default_factory=lambda: _env_int('RAG3_GPU_MAX_TEXT_CHARS', 16384))
    max_query_burst: int = field(default_factory=lambda: _env_int('RAG3_GPU_QUERY_BURST', 3))
    default_deadline_seconds: float = field(default_factory=lambda: _env_float('RAG3_GPU_DEADLINE_SECONDS', 180.0))
    max_length: int = 8192

    def __post_init__(self):
        if not (1 <= self.batch_size <= 8 and self.queue_limit >= 1
                and self.max_document_texts >= 1 and self.max_text_chars >= 1
                and self.max_query_burst >= 1 and math.isfinite(self.default_deadline_seconds)
                and self.default_deadline_seconds > 0 and self.max_length >= 1):
            raise ValueError('Invalid GPUServiceConfig bounds')


@dataclass(eq=False)
class _Request:
    kind: str
    texts: list[str]
    future: Future
    submitted: float
    deadline: float
    cancelled: bool = False
    offset: int = 0
    vectors: list[list[float]] = field(default_factory=list)
    last_queued: float = 0.0
    document_sequence: int = 0


class GPURequest:
    """The Future, cooperative cancellation and deadline-aware wait handle.

    cancel() works even after Future.set_running_or_notify_cancel(): an active
    <=8-text GPU call is not interrupted; its result is discarded at boundary.
    Future callbacks must not synchronously wait for another request on this
    service, because completion may run on the sole GPU owner thread.
    """
    def __init__(self, service: 'GPUModelService', request: _Request):
        self.future = request.future
        self._service = service
        self._request = request

    def cancel(self) -> bool:
        return self._service.cancel(self._request)

    def wait_execution_end(self) -> None:
        """Wait until a non-preemptible CUDA call has actually finished.

        The Future may be marked done earlier by timeout or cancellation.
        This must only be called from outside the GPU owner thread.
        """
        if threading.get_ident() == self._service._owner_thread_id:
            raise RuntimeError('GPU owner cannot wait on itself')
        with self._service._condition:
            while self._service._current is self._request:
                self._service._condition.wait()

    def result(self, timeout: float | None = None):
        # An external deadline applies even while model loading runs.
        remaining = max(0.0, self._request.deadline - self._service._clock())
        if timeout is not None:
            remaining = min(remaining, max(0.0, timeout))
        try:
            return self.future.result(timeout=remaining)
        except TimeoutError:
            if self._service._clock() >= self._request.deadline:
                self._service.expire(self._request)
                raise GPURequestExpired() from None
            raise


class GPUModelService:
    def __init__(self, config: GPUServiceConfig | None = None, *,
                 model_factory: Callable[[], object] | None = None,
                 encode_fn: Callable | None = None,
                 cuda_cache_clear: Callable[[], None] | None = None,
                 monotonic: Callable[[], float] = time.monotonic):
        self.config = config or GPUServiceConfig()
        self._factory = model_factory or self._load_cuda_model
        self._encode_fn = encode_fn  # deferred import by owner thread by default
        self._cuda_cache_clear = cuda_cache_clear or self._clear_cuda_cache
        self._clock = monotonic
        self._condition = threading.Condition()
        self._queries: deque[_Request] = deque()
        self._documents: deque[_Request] = deque()
        # A partial document request remains in flight between batches when
        # the waiting queue is full. This one explicit continuation slot lets
        # a query enter even with queue_limit=1, without overflowing the queue.
        self._deferred_document: _Request | None = None
        self._next_document_sequence = 0
        self._current: _Request | None = None
        self._worker: threading.Thread | None = None
        self._deadline_worker: threading.Thread | None = None
        self._reset_query_burst = False
        self._stop = False
        self._state = 'not_initialized'
        self._error_code: str | None = None
        self._error_summary: str | None = None
        self._model = None
        self._parse_reservation = False
        self._parse_parked = False
        self._model_load_count = 0
        self._owner_thread_id: int | None = None
        self._load_seconds: float | None = None
        self._queue_wait_seconds = 0.0
        self._query_encode_seconds = 0.0
        self._doc_encode_seconds = 0.0
        self._query_batches = 0
        self._document_batches: list[int] = []

    def snapshot(self) -> dict:
        # A health request never expires work, spins up a worker or imports ML.
        with self._condition:
            return dict(state=self._state, scope='shared_backend',
                        query_waiting=len(self._queries), document_waiting=len(self._documents),
                        document_deferred=self._deferred_document is not None,
                        active_kind=self._current.kind if self._current else None,
                        error_code=self._error_code, error_summary=self._error_summary,
                        model_load_count=self._model_load_count,
                        parser_has_gpu=self._parse_parked)

    def metrics(self) -> dict:
        with self._condition:
            return dict(model_load_count=self._model_load_count,
                        owner_thread_id=self._owner_thread_id,
                        load_seconds=self._load_seconds,
                        queue_wait_seconds=self._queue_wait_seconds,
                        query_encode_seconds=self._query_encode_seconds,
                        document_encode_seconds=self._doc_encode_seconds,
                        query_batches=self._query_batches,
                        document_batches=self._document_batches[:])

    def reserve_for_parsing(self, *, timeout: float = 180.0) -> None:
        """Drain the GPU owner and release BGE before a CUDA parser starts."""
        if timeout <= 0 or not math.isfinite(timeout):
            raise ValueError('timeout must be finite and positive')
        deadline = time.monotonic() + timeout
        with self._condition:
            if self._stop or self._state == 'failed':
                raise GPUServiceError('GPU_FAILED', 'GPU service cannot be reserved')
            if self._parse_reservation:
                raise GPUServiceError('GPU_RESERVED_FOR_PARSING', 'GPU is already reserved')
            self._parse_reservation = True
            error = GPUServiceError('GPU_RESERVED_FOR_PARSING', 'GPU is reserved for PDF parsing')
            for req in tuple(self._queries) + tuple(self._documents):
                if not req.future.done():
                    req.future.set_exception(error)
            if self._deferred_document and not self._deferred_document.future.done():
                self._deferred_document.future.set_exception(error)
            self._queries.clear()
            self._documents.clear()
            self._deferred_document = None
            # Let one already-running CUDA batch finish before releasing BGE.
            # No new request is accepted after the reservation starts.
            if self._worker is None:
                self._parse_parked = True
                self._state = 'reserved_for_parsing'
            self._condition.notify_all()
            while not self._parse_parked and self._state != 'failed' and not self._stop:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    # The caller must still invoke release_after_parsing in its
                    # finally block; a late CUDA call can finish after timeout.
                    raise GPUServiceError('GPU_RESERVATION_TIMEOUT', 'BGE did not release GPU in time')
                self._condition.wait(remaining)
            if self._state == 'failed' or self._stop:
                raise GPUServiceError('GPU_FAILED', 'GPU service failed while reserving')

    def release_after_parsing(self, *, timeout: float = 180.0) -> None:
        """Permit the next semantic query or embedding to load BGE again."""
        deadline = time.monotonic() + timeout
        with self._condition:
            if not self._parse_reservation:
                return
            while not self._parse_parked and not self._stop and self._state != 'failed':
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise GPUServiceError('GPU_RESERVATION_TIMEOUT', 'BGE release is still in progress')
                self._condition.wait(remaining)
            self._parse_reservation = False
            self._parse_parked = False
            if not self._stop and self._state != 'failed':
                self._state = 'not_initialized'
            self._condition.notify_all()

    def submit_query(self, text: str, *, deadline_seconds: float | None = None,
                     deadline_monotonic: float | None = None) -> GPURequest:
        if (not isinstance(text, str) or not text.strip()
                or len(text) > self.config.max_text_chars):
            raise GPUServiceError('INVALID_ARGUMENT', 'query must be nonempty bounded text')
        return self._submit('query', [text], deadline_seconds, deadline_monotonic)

    def submit_documents(self, texts: list[str], *, deadline_seconds: float | None = None,
                         deadline_monotonic: float | None = None) -> GPURequest:
        if (not isinstance(texts, list) or not texts
                or len(texts) > self.config.max_document_texts
                or any(not isinstance(t, str) or len(t) > self.config.max_text_chars
                       for t in texts)):
            raise GPUServiceError('INVALID_ARGUMENT', 'document input exceeds bounds or is invalid')
        return self._submit('document', list(texts), deadline_seconds, deadline_monotonic)

    def _submit(self, kind, texts, seconds, absolute):
        if absolute is not None and seconds is not None:
            raise GPUServiceError('INVALID_ARGUMENT', 'specify only one deadline')
        if seconds is not None and (not isinstance(seconds, (int, float))
                                    or not math.isfinite(seconds) or seconds <= 0):
            raise GPUServiceError('INVALID_ARGUMENT', 'deadline_seconds must be finite and positive')
        if absolute is not None and (not isinstance(absolute, (int, float))
                                     or not math.isfinite(absolute)):
            raise GPUServiceError('INVALID_ARGUMENT', 'deadline_monotonic must be finite')
        now = self._clock()
        deadline = float(absolute) if absolute is not None else now + (
            seconds if seconds is not None else self.config.default_deadline_seconds)
        if not math.isfinite(deadline):
            raise GPUServiceError('INVALID_ARGUMENT', 'deadline must be finite')
        if deadline <= now:
            raise GPURequestExpired()
        req = _Request(kind, texts, Future(), now, deadline, last_queued=now)
        with self._condition:
            self._prune_locked(now)
            if self._stop:
                raise GPUServiceError('SERVICE_STOPPED', 'GPU service is stopped')
            if self._state == 'failed':
                raise GPUServiceError(self._error_code or 'GPU_FAILED',
                                      'GPU service is in failed state')
            if self._parse_reservation:
                raise GPUServiceError('GPU_RESERVED_FOR_PARSING',
                                      'GPU is reserved for PDF parsing')
            if len(self._queries) + len(self._documents) >= self.config.queue_limit:
                raise GPUServiceError('QUEUE_FULL', 'GPU request queue at capacity')
            if kind == 'query':
                self._queries.append(req)
            else:
                self._stamp_document_locked(req)
                self._documents.append(req)
            if self._worker is None:
                self._worker = threading.Thread(target=self._run, name='rag3-bge-gpu-owner', daemon=True)
                self._worker.start()
            if self._deadline_worker is None:
                # The GPU owner cannot expire Futures while blocked in model
                # loading or CUDA. One shared timer serves all requests.
                self._deadline_worker = threading.Thread(
                    target=self._watch_deadlines, name='rag3-bge-deadlines', daemon=True)
                self._deadline_worker.start()
            self._condition.notify_all()
        return GPURequest(self, req)

    def _stamp_document_locked(self, req: _Request):
        self._next_document_sequence += 1
        req.document_sequence = self._next_document_sequence

    def _insert_document_locked(self, req: _Request):
        # The deferred continuation can be older than documents submitted
        # while a query runs. Restore it at its original FIFO position.
        for index, queued in enumerate(self._documents):
            if req.document_sequence < queued.document_sequence:
                self._documents.insert(index, req)
                return
        self._documents.append(req)

    def _prune_locked(self, now):
        for queue in (self._queries, self._documents):
            for req in tuple(queue):
                if req.cancelled or now >= req.deadline or req.future.done():
                    queue.remove(req)
                    if not req.future.done():
                        req.future.set_exception(GPURequestCancelled() if req.cancelled else GPURequestExpired())
        req = self._deferred_document
        if req is not None and (req.cancelled or now >= req.deadline or req.future.done()):
            self._deferred_document = None
            if not req.future.done():
                req.future.set_exception(GPURequestCancelled() if req.cancelled else GPURequestExpired())
        req = self._current
        if req is not None and (req.cancelled or now >= req.deadline) and not req.future.done():
            # The in-flight CUDA call remains non-preemptible. Its caller is
            # released at the deadline and the owner discards the late result.
            req.future.set_exception(GPURequestCancelled() if req.cancelled else GPURequestExpired())
        if not self._documents and self._deferred_document is None:
            # A later document starts a fresh contention period. Old queries
            # cannot consume its priority budget.
            self._reset_query_burst = True

    def _watch_deadlines(self):
        with self._condition:
            while not self._stop and self._state != 'failed':
                now = self._clock()
                self._prune_locked(now)
                pending = (tuple(self._queries) + tuple(self._documents)
                           + ((self._deferred_document,) if self._deferred_document else ())
                           + ((self._current,) if self._current else ()))
                deadlines = [req.deadline for req in pending if not req.future.done()]
                self._condition.wait(timeout=max(0.0, min(deadlines) - now) if deadlines else None)

    def cancel(self, req: _Request) -> bool:
        with self._condition:
            if req.future.done():
                return False
            req.cancelled = True
            self._prune_locked(self._clock())
            self._condition.notify_all()
            return True

    def expire(self, req: _Request) -> None:
        with self._condition:
            self._prune_locked(self._clock())
            self._condition.notify_all()

    def close(self, *, wait: bool = True, timeout: float | None = None):
        with self._condition:
            if not self._stop:
                self._stop = True
                for req in tuple(self._queries) + tuple(self._documents):
                    if not req.future.done():
                        req.future.set_exception(GPUServiceError('SERVICE_STOPPED', 'GPU service stopped'))
                if self._deferred_document and not self._deferred_document.future.done():
                    self._deferred_document.future.set_exception(
                        GPUServiceError('SERVICE_STOPPED', 'GPU service stopped'))
                self._queries.clear()
                self._documents.clear()
                self._deferred_document = None
                if self._current:
                    self._current.cancelled = True
                    if not self._current.future.done():
                        # GPU work cannot be interrupted, but callers must not
                        # wait for a stuck call after service shutdown.
                        self._current.future.set_exception(
                            GPUServiceError('SERVICE_STOPPED', 'GPU service stopped'))
                self._condition.notify_all()
            worker = self._worker
            deadline_worker = self._deadline_worker
            if self._state != 'failed':
                self._state = 'stopped'
        if wait and worker and threading.current_thread() is not worker:
            worker.join(timeout)
        if wait and deadline_worker and threading.current_thread() is not deadline_worker:
            deadline_worker.join(timeout)

    def _load_cuda_model(self):
        from rag3.ingestion_settings import find_bge_m3
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
            raise GPUServiceError('CUDA_UNAVAILABLE', 'cuda:0 is unavailable')
        from FlagEmbedding import BGEM3FlagModel
        path = str(find_bge_m3())
        model = BGEM3FlagModel(path, use_fp16=True, pooling_method='cls', devices='cuda:0')
        # Verify the real parameter device rather than trusting constructor flags.
        parameters = getattr(model, 'model', None)
        if parameters is None or not hasattr(parameters, 'parameters'):
            raise GPUServiceError('GPU_DEVICE_UNVERIFIED', 'BGE device could not be verified')
        # FlagEmbedding initializes the weights on CPU even with devices='cuda:0'.
        # Move the same instance to CUDA once; its encode_* methods reuse this
        # instance. Never create a CPU inference fallback or an extra model.
        parameters.to(torch.device('cuda:0'))
        inspected = 0
        for param in parameters.parameters():
            inspected += 1
            if param.device.type != 'cuda' or param.device.index != 0:
                raise GPUServiceError('GPU_DEVICE_INVALID', 'BGE model is not on cuda:0')
        if inspected == 0:
            raise GPUServiceError('GPU_DEVICE_UNVERIFIED', 'BGE parameters are unavailable')
        return model

    @staticmethod
    def _clear_cuda_cache() -> None:
        import torch
        torch.cuda.empty_cache()

    def _load_if_needed(self):
        if self._model is not None:
            return self._model
        with self._condition:
            self._state = 'loading'
        started = self._clock()
        model = self._factory()
        if model is None:
            raise GPUServiceError('GPU_INIT_FAILED', 'GPU factory returned no model')
        took = self._clock() - started
        with self._condition:
            self._model = model
            self._model_load_count += 1
            self._load_seconds = took
            if not self._stop:
                self._state = 'ready'
        return model

    def _failure(self, code: str, summary: str, current: _Request | None):
        err = GPUServiceError(code, summary)
        with self._condition:
            self._state = 'failed'
            self._error_code = code
            self._error_summary = summary  # fixed safe strings, no model paths or traceback
            requests = list(self._queries) + list(self._documents)
            if self._deferred_document:
                requests.append(self._deferred_document)
            if current:
                requests.append(current)
            self._queries.clear()
            self._documents.clear()
            self._deferred_document = None
            self._current = None
            for req in requests:
                if not req.future.done():
                    req.future.set_exception(err)
            self._condition.notify_all()

    def _choose_locked(self, burst: int):
        documents_waiting = bool(self._documents or self._deferred_document)
        if self._queries and (burst < self.config.max_query_burst or not documents_waiting):
            # Queries served while no document waits do not consume fairness.
            return self._queries.popleft(), burst + 1 if documents_waiting else 0
        if self._documents:
            if (self._deferred_document is not None
                    and self._deferred_document.document_sequence
                    < self._documents[0].document_sequence):
                req = self._deferred_document
                self._deferred_document = None
                return req, 0
            req = self._documents.popleft()
            if self._deferred_document:
                # The freed queue slot restores the older continuation in
                # sequence order, even if later documents were submitted.
                self._insert_document_locked(self._deferred_document)
                self._deferred_document = None
            return req, 0
        if self._deferred_document:
            req = self._deferred_document
            self._deferred_document = None
            return req, 0
        return None, 0

    def _run(self):
        self._owner_thread_id = threading.get_ident()
        burst = 0
        while True:
            # Drop the previous request's local model reference before release.
            model = None
            with self._condition:
                self._prune_locked(self._clock())
                while (not self._stop and not self._queries and not self._documents
                       and not self._deferred_document and not self._parse_reservation
                       and self._state != 'failed'):
                    self._condition.wait()
                    self._prune_locked(self._clock())
                if self._stop or self._state == 'failed':
                    return
                if self._parse_reservation:
                    if not self._parse_parked:
                        old_model = self._model
                        self._model = None
                        self._state = 'releasing_for_parsing'
                    else:
                        self._condition.wait()
                        continue
                else:
                    old_model = None
                if self._parse_reservation:
                    req = None
                else:
                    if self._reset_query_burst:
                        burst = 0
                        self._reset_query_burst = False
                    req, burst = self._choose_locked(burst)
                    self._current = req
                    # set_running occurs only on the first batch. Cooperative cancel
                    # works independently of Future.cancel once running.
                    if req.offset == 0 and not req.future.set_running_or_notify_cancel():
                        self._current = None
                        continue
                    now = self._clock()
                    self._queue_wait_seconds += now - req.last_queued
            if req is None:
                try:
                    if old_model is not None:
                        del old_model
                        gc.collect()
                        self._cuda_cache_clear()
                except BaseException:
                    self._failure('GPU_RELEASE_FAILED', 'BGE release failed', None)
                    return
                with self._condition:
                    if not self._stop:
                        self._parse_parked = True
                        self._state = 'reserved_for_parsing'
                    self._condition.notify_all()
                continue
            if req.cancelled or self._clock() >= req.deadline:
                self._finish_error(req, GPURequestCancelled() if req.cancelled else GPURequestExpired())
                continue
            try:
                model = self._load_if_needed()
            except BaseException as exc:
                code = exc.code if isinstance(exc, GPUServiceError) else (
                    'CUDA_OOM' if 'out of memory' in str(exc).lower() else 'GPU_INIT_FAILED')
                self._failure(code, 'GPU model initialization failed', req)
                return
            if req.cancelled or self._clock() >= req.deadline:
                self._finish_error(req, GPURequestCancelled() if req.cancelled else GPURequestExpired())
                continue
            batch = req.texts[req.offset:req.offset + self.config.batch_size]
            try:
                if self._encode_fn is None:
                    from rag3.bge_dense import encode_dense
                    self._encode_fn = encode_dense
                started = self._clock()
                result = self._encode_fn(model, batch, query=req.kind == 'query',
                                         batch_size=self.config.batch_size,
                                         max_length=self.config.max_length)
                duration = self._clock() - started
                if len(result) != len(batch):
                    raise ValueError('BGE returned incorrect vector count')
            except BaseException as exc:
                code = 'CUDA_OOM' if 'out of memory' in str(exc).lower() else 'GPU_EXECUTION_FAILED'
                self._failure(code, 'GPU encoding failed', req)
                return
            with self._condition:
                if req.kind == 'query':
                    self._query_batches += 1
                    self._query_encode_seconds += duration
                else:
                    self._document_batches.append(len(batch))
                    self._doc_encode_seconds += duration
            if req.future.done() or req.cancelled or self._clock() >= req.deadline or self._stop:
                self._finish_error(req, GPURequestCancelled() if req.cancelled else GPURequestExpired())
                continue
            req.vectors.extend(result)
            req.offset += len(batch)
            with self._condition:
                parsing = self._parse_reservation
            if parsing and req.kind == 'document' and req.offset < len(req.texts):
                self._finish_error(req, GPUServiceError(
                    'GPU_RESERVED_FOR_PARSING', 'GPU is reserved for PDF parsing'))
                continue
            if req.offset < len(req.texts):
                with self._condition:
                    if self._stop:
                        req.cancelled = True
                    elif self._parse_reservation:
                        req.cancelled = True
                    elif req.future.done() or req.cancelled or self._clock() >= req.deadline:
                        pass
                    else:
                        req.last_queued = self._clock()
                        self._stamp_document_locked(req)
                        if len(self._queries) + len(self._documents) < self.config.queue_limit:
                            self._documents.append(req)  # rotation across document requests
                        else:
                            # Never append beyond queue_limit. A partial active
                            # document retains one visible continuation slot.
                            assert self._deferred_document is None
                            self._deferred_document = req
                    self._current = None
                    self._condition.notify_all()
                if req.cancelled or req.future.done() or self._clock() >= req.deadline:
                    with self._condition:
                        self._prune_locked(self._clock())
                    self._finish_error(req, GPURequestCancelled() if req.cancelled else GPURequestExpired())
            else:
                with self._condition:
                    self._current = None
                    if not req.future.done():
                        if req.cancelled or self._stop:
                            req.future.set_exception(GPURequestCancelled())
                        elif self._clock() >= req.deadline:
                            req.future.set_exception(GPURequestExpired())
                        else:
                            req.future.set_result(req.vectors[0] if req.kind == 'query' else req.vectors)
                    self._condition.notify_all()

    def _finish_error(self, req: _Request, exc: GPUServiceError):
        with self._condition:
            if self._current is req:
                self._current = None
            if not req.future.done():
                req.future.set_exception(exc)
            self._condition.notify_all()
