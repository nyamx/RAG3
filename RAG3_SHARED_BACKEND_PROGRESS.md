# RAG3 Shared Backend — implementation progress

## Stage A — isolated runtime acceptance PASS (2026-10-23)

Baseline: `codex/shared-backend`, HEAD before work `e886e04`. Existing untracked material was preserved. Bridge logs: `run_output_20261023_161959_779.zip` (patch), `run_output_20261023_162427_255.zip` (safety change), `run_output_20261023_164557_063.zip` (session fix and successful verification).

### Implemented
- `scripts/rag3-mcp-server.py`: shared MCP HTTP backend with `/health`, authenticated `/api/v1/command`, and HTTP MCP session-aware collection selection for default-selection tools; existing MCP required tool arguments preserved.
- `rag3/backend_boundary.py`: session selection store, stable `ctx.session_id` key, token-file UI API authentication, workspace-wide single-instance lock, health route. UI API token is created **after** acquiring the lock so failed duplicate starts cannot rotate the active token.
- Legacy ingestion-manager RPC remains until C. UI's direct workers remain until D. No real collection or existing ingestion record was intentionally mutated.
- A reproducible mock-Qdrant runtime regression test is included in `scripts/test-shared-backend-A-http-e2e.py`.

### Verified on Windows through Bridge
- Python syntax checks and `git diff --check`: PASS.
- HTTP `/health` on an actual server process: HTTP 200, `gpu_model=not_initialized`; backend start did not load BGE.
- UI API authentication: missing/incorrect token -> 401, malformed request -> 400, authorized collection-list -> 200 with fake Qdrant.
- Duplicate startup: rejected; first server PID remained and its token did not rotate.
- MCP tools: 22 discovered; `Context` injection is not exposed as a user parameter.
- Two simultaneous independent HTTP MCP sessions: different selected collections persisted correctly across 10 interleaved calls with fake Qdrant.
- Own test listener cleaned up; `git diff --check` returned zero.

### Not yet independently verified
- Real Qdrant was unavailable during verification. Live existing-collection search/fetch and reconnection are not validated in Stage A; do not claim real data-path success.
- Existing installed plugin, Secure MCP Tunnel and Windows-login integration are D-stage tasks.
- Actual GPU single instance / 8-chunk queue is B-stage; ingestion consolidation C-stage; UI migration D-stage.
- Mock test does not measure production latency/VRAM or prove that all error paths are correct under concurrent load.

### A review corrections (2026-10-23)
- Windows UI API token files are now created with a private ACL for the current user, SYSTEM and Administrators; the generated file's ACL was inspected after the HTTP test. This closes the inherited `data/` permission gap found in review.
- An expired or evicted MCP session selection now requires explicit reselection instead of silently reverting to the default collection. The instance lock is workspace-wide for normal operation, including when the listen port changes; isolated fake-Qdrant tests use an explicit test scope.
- UI API missing/invalid required fields return `400 INVALID_ARGUMENT`. The collection-create API preserves the existing UI worker's `reuse_existing=True` behavior.
- The revised fake-Qdrant HTTP test (`.venv\Scripts\python.exe scripts\test-shared-backend-A-http-e2e.py`) passed outside the socket-restricted inspection sandbox: authentication, missing-argument `400`, chunked oversized-body `413`, cold health without BGE, production-scope workspace lock, same-port and different-port duplicate rejection, 22 MCP tool schemas, two session selections, BM25 searches, fetches, explicit collection precedence, stale IDs, and listener cleanup. `py_compile` and `git diff --check` passed. No real collection or ingestion job was changed.
- `/health.gpu_model` still reports only whether `torch` has been imported after startup; Stage B must replace this with the actual single-model service state. Real Qdrant and existing-collection search/fetch remain to be checked when the full path is integrated.
- The local UI API still uses a 90-second `wait_for(to_thread(...))` timeout: a timed-out write can continue in the thread. Before D exposes this path through the UI, make timeout responses explicitly indeterminate and reconcile the operation's final state; do not report a timeout as proof that a mutation was cancelled. The installation directory also inherits broad local-user write permissions, so the protected token file alone is not a complete multi-user installation boundary.

### Next execution unit — Stage B
1. Re-read this log and check HEAD/worktree; confirm no active ingestion must be disturbed.
2. Inspect exact `_encode`, `warm_up`, document/query paths in `rag3/haystack_native.py` and `rag3/retrieval.py` before changing them.
3. Implement single lazy `cuda:0` BGE owner thread with bounded fair query/document queues, deadlines and cancellation; batch documents into <=8 actual `_encode` calls.
4. Unit-test the BGE service and run its GPU-level isolated acceptance tests; **do not** convert ingestion/UI or modify live Qdrant in B.
5. Update this progress file with accurate tests, changed paths, model load counts, metrics and the exact next task for C.

## Data and rollback

- Backups were created under `artifacts/backups/shared-backend-A-*`.
- Existing `data/`, collections, source files and previously untracked artifacts must not be deleted.
- The A changes are additive and can be reverted by restoring the above backup and the original MCP server entry point; do not roll back C/D data with old code once their schema changes land.

## Stage B — isolated GPU model service acceptance PASS (2026-10-23 21:30 KST)

### Baseline and scope
- Branch: `codex/shared-backend`; prior HEAD: `f08e67c`. B-only commit containing this record: resolve with `git log -1 --format=%H` on this branch (the commit hash cannot be embedded in its own content without changing it).
- Changed: `rag3/gpu_model_service.py` (new), `rag3/bge_dense.py` (new), `rag3/haystack_native.py`, `rag3/backend_boundary.py`, `scripts/rag3-mcp-server.py`, `scripts/test-shared-backend-B-gpu-service.py` (new), `scripts/test-shared-backend-B-real-gpu.py` (new), this progress file.
- Existing untracked materials, Qdrant collections, source files and ingestion SQLite DB were not edited, deleted or cleaned. Only Stage B code + documentation committed.

### Implementation and API
- One `GPUModelService` is constructed by the backend; construction/import/repeated `/health` uses no torch, FlagEmbedding, Docling or OCR and loads no model. `/health.gpu_model` and `/health.gpu_model_service` report the **Stage B service only**, not all legacy GPU usage, without absolute model paths or tracebacks. Status includes `not_initialized`, `loading`, `ready`, `failed`, pending query/doc counts, safe error code and summary.
- `submit_query(text, deadline_seconds=None, deadline_monotonic=None)` returns `GPURequest`, whose `result(timeout=None)` yields a dense vector; `submit_documents(texts, ...)` yields ordered vectors. Request has `.future` and cooperative `.cancel()`. Service exposes `.snapshot()`, `.metrics()`, `.close(wait=True, timeout=None)`. `GPUServiceError.code` is stable (`INVALID_ARGUMENT`, `QUEUE_FULL`, `DEADLINE_EXCEEDED`, `CANCELLED`, `GPU_INIT_FAILED`, `GPU_EXECUTION_FAILED`, `CUDA_OOM`, etc.).
- The owner thread is lazy and unique. First work loads the fixed local BGE-M3 into `cuda:0` exactly once. FlagEmbedding initially creates CPU-located parameters even when `devices='cuda:0'`; same model is explicitly moved to GPU once, then verified on device. No secondary CPU model, CPU fallback, device shuttling or cache-emptying.
- Shared `rag3.bge_dense.encode_dense` preserves query `encode_queries` vs document `encode_corpus`, float32 dense L2 normalization, vector order and disabled sparse/ColBERT. Legacy `BGEM3OnePassEmbedder` uses the same helper but its model ownership remains unchanged until C/D.
- Two `deque` queues under `threading.Condition`, Future-based responses. Actual model calls are <= 8 documents at a time; documents rotate to the end of the document queue per batch. Waiting queries have batch-boundary priority limited by a three-query burst before a waiting doc batch. Query/doc GPU calls never hold the Condition lock. No Qdrant I/O or CPU parsing in this owner.
- Submitted requests carry monotonic absolute deadlines including load/queue/inference. Pending cancellations/expirations release queue capacity; a running <=8 batch is non-preemptible but its result is discarded on cancellation/deadline and remaining batches are skipped. Fatal GPU load/encode errors fail all outstanding Futures and latch `failed` until restart; no retry storm or silent CPU downgrade.

### Config defaults and their evidence/limits
- `RAG3_GPU_BATCH_SIZE=8` (hard maximum 8): existing ingestion BGE internal batch size and deterministic 17-doc `8,8,1` test.
- `RAG3_GPU_QUEUE_LIMIT=32` waiting requests; `RAG3_GPU_MAX_DOC_TEXTS=256` per document request; `RAG3_GPU_MAX_TEXT_CHARS=16384` per text; these are explicit defensive memory bounds, **not** tuned from production concurrency/load benchmarks. Active request is accounted separately.
- `RAG3_GPU_QUERY_BURST=3` maximum consecutive query batches while documents wait: deterministic starvation/FIFO verification, not latency-optimized.
- `RAG3_GPU_DEADLINE_SECONDS=180` including lazy model load; service-level default, not yet validated against production p95; real GPU acceptance used an explicit 300s safety deadline.
- `GPUServiceConfig.max_length=8192` retains previous BGE max token length. Model path comes only from fixed local `find_bge_m3()`; arbitrary request-side paths are not accepted. Invalid/nonfinite deadlines and oversized inputs are rejected.

### Acceptance on target Windows venv
- `.venv\Scripts\python.exe scripts\test-shared-backend-B-gpu-service.py`: **PASS 15/15**; injected fake model/synchronization/clock covers one owner, 17-doc interleaving with query, 8-call boundaries, FIFO/rotation, query burst bound, input/queue limits, deadline while loading/in queued work, cancellation while running, stop behavior, load failure and CUDA OOM.
- `.venv\Scripts\python.exe -c <cold health smoke embedded in Bridge finalization>`: **PASS**; 3 health calls, heavy imports=0, model load=0; no A-stage integration regression suite rerun per instruction.
- `.venv\Scripts\python.exe scripts\test-shared-backend-B-real-gpu.py`: **PASS**, isolated B-only process, device `Quadro T1000`, actual parameters `cuda:0`, `BGEM3FlagModel` instance/model load count **1**, owner thread ID `74392`, query/17-doc/second query; dense 1024, norms ~1, document actual call sizes `[8, 8, 1]`. No CPU fallback.
- Legacy `rag3/retrieval.py`: unchanged. Static branch guard confirms BM25-only `semantic=False` does not call `_embed_query`; previous Stage A mock BM25 verification exists, but live BM25 search **not rerun** in B. Live Qdrant mutation: **not performed**.
- `git diff --check` and `git diff --cached --check`: PASS (checked before commit).

### Measurements (single isolated run, not load benchmarks)
- GPU model first load `20.563 s`; aggregate queue wait `0.000 s` (sequential requests, **not** a contended-queue latency benchmark).
- Aggregate query encoding (2 calls) `0.750 s`; aggregate document encoding (3 batches / 17 docs) `1.484 s`. CPU parsing: **미측정**; Qdrant I/O: **미측정**.
- Process RSS before `577.1 MiB`, after `2690.1 MiB`. Device VRAM total `4095.7 MiB`, available before `3298.5 MiB`, available after `2166.5 MiB`, observed decline `1132.0 MiB`. These are process/container snapshots, not per-component allocation accounting.
- Separate Bridge evidence: implementation `run_output_20261023_210522_692.zip`, initial GPU fault `run_output_20261023_210900_461.zip`, CUDA ownership fix / successful GPU `run_output_20261023_211504_764.zip`, hardened 15 tests and cold health `run_output_20261023_211911_463.zip`, audit `run_output_20261023_212144_156.zip`; final run output provides new measured numbers.

### Remaining paths and next Stage C action
- `rag3/retrieval.py` still has `_QUERY_EMBEDDERS` and `build_retrieval_pipeline` direct BGE initialization; `scripts/ingest-worker.py` continues to create its own embedder and submit 32 texts to its own `_encode`, and UI search worker/MCP semantic search have not been migrated. Hence real operation **can still instantiate multiple BGE models**. Stage B completion proves only isolated B service semantics and GPU residency, not shared end-to-end model, concurrent ingestion/search or finalized process topology.
- Start C by inspecting live ingestion activity, SQLite job state/checkpoints and consistency of an intact backup **before** any scheduler/worker transition. Then connect ingestion embedding to the B service via the backend's own process, validate temporary collections only, and reconcile cancellation/deadline semantics. Do not alter existing collections or live jobs during these preparations. UI/MCP service integration remains D.

### Stage B review corrections (2026-10-23)

- A post-completion code review reproduced three scheduler edge cases in `daa689c`: queries processed before any document was waiting consumed the later query-priority budget; expired queued/active Futures did not finish autonomously while the GPU owner was blocked; a partial document requeue could make the waiting deque exceed `queue_limit` by one. The initial 15 tests missed these cases.
- The GPU service now counts the query burst only during document contention. One lightweight deadline watcher thread starts lazily on the first request and completes expired Futures while the single GPU owner remains busy; it does not run the model or interrupt an active CUDA call. A partial document can occupy one explicit deferred continuation slot, reported in health as `document_deferred`, while the waiting deques remain within `queue_limit`. This permits a query to enter when `queue_limit=1` and a document batch is active.
- The new tests failed against the old scheduler and now pass: `.venv\Scripts\python.exe scripts\test-shared-backend-B-gpu-service.py` **21/21 PASS**. They cover priority after uncontended queries, autonomous expiry of queued and active Futures during blocked model loading, and one/two document continuations at `queue_limit=1`, including interleaved queries, FIFO rotation and the queue bound. A further review caught active Future completion on `close(wait=False)` and an older deferred document being overtaken by a later submission; both were reproduced before correction and covered by the 21 tests. Python syntax and `git diff --check` passed.
- The Bridge ZIP's three cold-health calls used `DummyMCP`, so they did not prove actual HTTP startup. The focused `scripts/test-shared-backend-B-http-health.py` test then started the real server with a fake Qdrant transport and test instance scope: HTTP `/health` 200, backend PID owned the listener, BGE service `not_initialized`, model load count 0. Its final isolated loopback run used port 37564, started in 3.78 s with 114.9 MiB server RSS, and cleaned up the server, listener, token and lock. The initial restricted-sandbox HTTP attempt was inconclusive; the passing run used normal local process/socket access.
- `GPURequest.future` is exposed for result delivery. As with a single-worker executor, a done callback must not synchronously wait for another request from this same GPU service; such callbacks would block its only GPU owner. Callers should use `GPURequest.result()` from another thread or nonblocking callbacks. This API limit is documented in the request handle.
- Model loader and dense encoding code were not changed in this correction. The isolated real GPU result from `daa689c` (one CUDA model, `8/8/1`, 20.563 s load, ~1132 MiB observed VRAM decline) remains the available GPU evidence; a second model load was not performed solely for these scheduler fixes. Legacy ingestion, UI and MCP semantic paths remain unconverted until C/D. No live Qdrant collection or SQLite job was changed.
