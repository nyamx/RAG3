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
