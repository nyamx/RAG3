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

## Stage C — shared backend isolated acceptance (2026-10-24 KST)

### Scope and transition
- Baseline `codex/shared-backend` HEAD `63cc00f` (Stage C handoff). Stage C commit hash: see the next C-scoped commit in `git log`; a self-referential hash is deliberately not embedded.
- Changed paths: `rag3/gpu_model_service.py`, `rag3/ingestion_manager.py`, `rag3/retrieval.py`, `scripts/ingest-worker.py`, `scripts/rag3-mcp-server.py`; new `rag3/ingestion_backend.py`, `rag3/ingestion_light.py`, `scripts/test-shared-backend-C-isolated.py`, `scripts/test-shared-backend-C-lifecycle.py`; this progress file.
- The HTTP backend owns a single event-woken ingestion scheduler, SQLite job state, document embedding through the same backend `GPU_BGE_SERVICE` as MCP/API semantic search, and Qdrant upserts. One *temporary, CPU-only* Docling parsing child is allowed. No legacy manager server, persistent full ingestion worker or separate embedding subprocess is started on the C backend path; Qdrant remains a separate service. Backend health/cold import do not load torch/Docling/FlagEmbedding.
- The `ingestion_manager.client()` compatibility path must not auto-start a second manager when the backend is unavailable. Existing UI search/collection worker and plugin/tunnel/login wiring remain **Stage D**, and are not evidence of a single GPU model across all UI processes yet.
- Existing SQLite job identifiers/status semantics and `parsed` to incremental `ready` Qdrant state are preserved. Staged 32-chunk `parsed` records remain; each backend embedding request uses at most 8 texts. Point updates preserve entire payload and extra named vector(s), and commit dense/BM25/ready in a single upsert; unsupported vector schemas fail rather than silently erase fields.

### Data protection and recovery
- Before implementation, an SQLite online backup was saved at `artifacts/backups/shared-backend-C-20261024_015200/ingestion_jobs.sqlite.snapshot.sqlite`, tested with `PRAGMA integrity_check=ok` and a successful read of 82 jobs. The same count was observed in the final read-only audit (`run_output_20261024_030522_825.zip`): `COMPLETE=72`, `CANCELLED=5`, `FAILED=5`; active jobs=0 and progress rows=0. Legacy port 18747 closed. Existing Qdrant collections and user sources were not deliberately changed; isolated Qdrant tests created and removed distinct temporary collections only. The existing untracked materials remain untouched.
- Cold recovery in temporary DB: stale `RUNNING` without valid checkpoint becomes `FAILED`; pre-boot `QUEUED` requires explicit Run; jobs retained. Newly submitted MCP work is eligible immediately; UI submission waits for Run; missing backend never auto-starts the legacy manager. Invalid Restart is rejected **before** deleting existing ready/parsed points. Both start and Restart validate SHA against `Path(source)` rather than a raw string. DB read failure does not busy-loop.
- With an inflight non-preemptible fake GPU call, Pause preserves checkpoint and prevents the blocked batch from writing. Resume completes 17 chunks without duplicates; Cancel executes no later GPU batch and does not upsert the cancelled result. The process cannot preempt a CUDA kernel that is already executing.

### Evidence and bounded acceptance tests
- Bridge outputs: read-only probe `run_output_20261024_014722_280.zip`, online backup `run_output_20261024_015157_891.zip`, baseline compile/regression `run_output_20261024_021149_969.zip`, actual Qdrant fake-vector checks `run_output_20261024_022401_504.zip`, actual GPU/Qdrant/backend-handler `run_output_20261024_022849_562.zip`, lifecycle and regression `run_output_20261024_024350_720.zip`, HTTP backend `run_output_20261024_024629_878.zip`, actual MCP and recovery `run_output_20261024_025328_432.zip`, CPU parser `run_output_20261024_025914_171.zip`, Pause/Resume/Cancel `run_output_20261024_030257_352.zip`, final read-only audit `run_output_20261024_030522_825.zip`.
- `.venv\Scripts\python.exe scripts\test-shared-backend-C-isolated.py`: **PASS 7/7** (extra vectors/payload, 17/8 batching, cancellation and early-Future completion, recovery, protected Restart, no DB-error busy loop). `scripts\test-shared-backend-C-lifecycle.py`: **PASS 4/4** (SHA handling, old/new queue distinction, Restart protection, no manager auto-start). Stage B GPU scheduler regression: **PASS 21/21**. Compile + `git diff --check`: **PASS**.
- Real Qdrant temporary collection: `parsed` points and a complete ready upsert preserved point IDs, nested metadata, `text-sparse` and BM25; semantic, BM25 and hybrid searches each returned 2 hits. Test collection removed.
- Actual BGE-M3 on NVIDIA Quadro T1000 `cuda:0`: **one GPU model load**, 1024-dimensional dense vectors, same backend object used for simulated ingestion + MCP/API search **handler** calls. BM25-only submitted 0 model requests. Concurrent search returned results while embedding work was active; 5 chunks progressively reached `ready`; MCP/API handler results matched (3 hits). Test collection removed. This is **not** a full live MCP+UI HTTP semantic search while ingestion is in progress.
- Actual HTTP process `/health` 200, unauthenticated `/api/v1/command` 401, authenticated `jobs.list`, isolated SQLite and scheduler PASS; cold health remained `not_initialized`. Actual Streamable HTTP MCP connection: 22 tools, `rag3_list_ingestion_jobs` returned two temporary jobs; cold recovery PASS; server/token/lock cleaned.
- Actual Docling CPU parser child: `CUDA_VISIBLE_DEVICES=-1`, 1 chunk, SQLite `CHUNKS_STORED`, Qdrant `parsed`, parse elapsed **29.48 seconds**, temporary collection and input cleaned. For 17 chunks, fake GPU Pause/Resume verified 8/8/8/1 attempts across pause/resume with 0 paused-result writes; for 9 chunks, inflight Cancel verified only one 8-document request and 0 cancelled-result writes (**12/12 checks PASS**).

### Measurements and limits
- Actual GPU/Qdrant-handler run: model loading in script 32.73 s, service-recorded load 24.328 s; total query encode 0.952 s; total document encode 38.469 s; aggregate queue wait 29.999 s (one isolated run, **not** latency p95). PyTorch GPU allocated 1094.2 MiB, reserved 1328.0 MiB. Full backend RSS, model-wide VRAM peak, p50/p95, process count for arbitrary production workloads: **미측정**. CPU parsing one document 29.48 s. Do not treat these as production capacity measurements.
- The actual CUDA/Qdrant run invoked MCP/API semantic search **handlers in process**, not simultaneous semantic requests over two real wire protocols; a separate real MCP wire test exercised tool listing and job queries. Multi-format production ingestion, Windows cancellation subtree stress, real long-running production recovery, timeout/late writes under all races, actual UI window equality, existing installed plugin, Secure MCP tunnel, login startup are **not certified here**. Re-test these before claiming deployed end-to-end parity. The Stage C code integration is accepted in *isolated scope*; the full UI/plugin rollout remains D.

#### Actual MCP + HTTP live-wire acceptance (2026-10-24 KST)
- Bridge run `run_output_20261024_031649_333.zip` (`bridge_stage_c_wire_acceptance_20261024_26.py_`): **PASS**, exit 0, stderr empty. Separate test-owned SQLite DB/input root and temporary Qdrant collection `__rag3_stage_c_wire_81891131887c`; no user job or production collection was deliberately modified. One live backend process with the standard backend lock, then clean termination.
- Real Streamable HTTP MCP `rag3_ingest_file` submitted job `f307888f50bf46048093a84fb1847948`; in state `RUNNING`, Qdrant already contained **8 `ready` points**. While ingestion was active, semantic+BM25 hybrid search over **both actual MCP and authenticated `/api/v1/command`** returned the same ordered 3 IDs. BM25-only MCP/API searches likewise returned the same ordered 3 IDs. The job ended `COMPLETE`.
- `/health` reported exactly **1** GPU BGE model load and state `ready` after the live run (cold start had zero loads). The run did not separately count BM25-only GPU submissions; BM25-only non-use of BGE was established in prior isolated tests. Search p50/p95, RAM peak, exact per-request queue wait and full multi-process production capacity remain **미측정**.
- Cleanup verified: backend stopped, token removed, backend lock removed, isolated temp DB/source removed, temporary Qdrant collection removed. These are observed final states for this specific test, not a claim of tested UI desktop/plug-in parity.
- This new test **closes the earlier handler-only wire coverage gap for Stage C**. Stage D still owns desktop UI search worker, installed plugin/tunnel/login integration and final UI-to-MCP parity. Other previously enumerated edge cases remain outside tested scope.

### Stage D handoff
- Start from the Stage C commit, recheck live jobs/owners before UI cutover. Complete UI search and collection-worker routing to authenticated backend APIs; verify real UI-to-MCP results against the same temporary collection, 90s timeout reconciliation for writes, installed plugin + tunnel + login orchestration, and ensure all UI processes stop instantiating duplicate BGE models.
- Do not alter existing jobs, Qdrant collections or backups to reproduce the isolated acceptance tests; use temporary sources/DB/collections. Maintain the single-backend lock and no legacy manager auto-spawn contract. Keep uncovered edge cases clearly labeled rather than inferring complete production acceptance.

### Stage C post-review hardening (2026-10-24 KST)

- Independent review reproduced cancellation and startup/recovery races after the original C commit. The backend now keeps the active GPU request handle and cancels queued work on Cancel/Pause/stop; it waits for the non-preemptible GPU call to finish before leaving the ingestion thread. Parsed plain-text and ready-vector Qdrant upserts share the scheduler lock with Cancel acceptance, so a new backend-owned upsert cannot start after Cancel returns. An upsert already in progress completes before Cancel can return.
- CPU Docling parsing remains a temporary child. On Windows it is placed in a kill-on-close Job Object before the parent opens a one-byte start gate. The child and its ordinary descendants are terminated if the backend dies, even without a Python cleanup callback. A failed assignment leaves the gate closed and the child is reaped. The actual parser's EOF gate was checked with a missing temporary input path and exited before parsing.
- A failure during scheduler thread startup or job finalization now releases the active slot; a transient Qdrant verification failure during cold recovery pauses the job for explicit validation on Resume instead of declaring its checkpoint incompatible. A `PARSING` checkpoint with already-ready points is rejected before a resume could overwrite them. A graceful backend stop marks active work `PAUSING` before cancelling it, so restart can recover the checkpoint. No existing job or collection was modified for these checks.
- MCP and authenticated HTTP searches now pass the outer 90-second request deadline into each GPU query (with a 0.25-second margin), cancel waiting GPU requests when their client times out or disconnects, and return bounded GPU failure codes without internal error text. A clean HTTP backend exit calls `scheduler.stop()`.
- Isolated regression results: Stage C hardening 8/8; Windows parser Job Object 3/3, including a child-created grandchild; search budget 5/5; original Stage C isolated 7/7 and lifecycle 4/4; Stage B deterministic GPU 21/21; Stage A fake-Qdrant HTTP E2E PASS. The actual GPU model, live Qdrant and desktop UI were not rerun for this hardening pass; Stage D still owns UI/plugin parity.
- A Qdrant request already sent by the CPU parser when termination begins may finish at the server. Such an in-flight write is treated as non-preemptible; cancellation prevents later backend batches and keeps checkpoint/point state available for an explicit Resume or Restart decision.

<!-- STAGE_D_RUNTIME_EVIDENCE_20261024 -->
### Stage D deployment/acceptance checkpoint — 2026-10-24 13:25:28 KST

**PARTIAL, NOT FINAL PASS.** Operational SQLite: `quick_check=ok`, jobs **82**, active **0** at record time. HTTP backend and Qt UI were exercised; installed Codex HTTP MCP plugin `0.1.2` was exercised in a new Codex task. Actual semantic/RRF/nonfused UI↔MCP search parity **7/7**, single GPU BGE load **1**, RRF stable tie-break and 5/5 offline acceptance tests PASS. Live backend was restarted safely (PID `18140` to `22688`) with token rotation; process/DB preservation checked. The exact evidence and unresolved conditions are documented in `artifacts/stage-d-verification-20261024/STAGE_D_ACCEPTANCE_STATUS.md`.

**Do not declare Stage D complete:** remote tunnel external MCP E2E and post-restart reconnect, genuine Windows relogin, full live failure-recovery matrix, precutover performance/resource baseline and deployment cleanup remain unverified. A ChatGPT RAG3 tool response does not alone establish that the response traversed the saved Secure MCP Tunnel.
