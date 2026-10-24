# RAG3 shared HTTP/MCP backend

**Stage D operational status (2026-10-24): functional rollout verified; formal performance comparison incomplete.** Local backend, Qt UI, installed Codex HTTP plugin, the existing remote tunnel's read calls, isolated fault recovery, and actual Windows sign-out/sign-in have passed. A comparable precutover performance baseline was not recorded; see [RAG3_SHARED_BACKEND_PROGRESS.md](RAG3_SHARED_BACKEND_PROGRESS.md) and `STAGE_D_ACCEPTANCE_STATUS.md`.

`scripts/rag3-mcp-server.py` hosts one persistent HTTP/MCP backend and the authenticated local UI command API. The backend, rather than Qt UI or stdio clients, owns job state, Qdrant access, retrieval, and the single CUDA BGE-M3 service. A temporary CPU parser child can appear during ingestion. Qdrant and the Secure MCP Tunnel runtime are separate services and are not included in the resident RAG3 Python backend count.

See [RAG3_PLUGIN_SETUP.md](RAG3_PLUGIN_SETUP.md) for Windows login registration, local Codex installation, and the distinct remote tunnel procedure.

## Starting and stopping on Windows

On this PC, the user-scope Scheduled Task **`RAG3 Shared Backend`** launches `scripts/rag3-backend-login.py` through `.venv\Scripts\rag3-pythonw-direct.exe`. The direct executable uses the existing virtual environment without the stock Windows venv `pythonw.exe` redirector process. After an actual Windows sign-out/sign-in on 2026-10-24, the task started at 17:07:40 KST with one backend Python PID, reachable Qdrant, and zero cold BGE loads; the saved tunnel reconnected at 17:09. The login entry checks or starts local Qdrant when configured, then runs the HTTP/MCP backend in that same PID. Opening a UI window does not start the persistent backend. The saved, non-secret configuration is `data/backend-login-config.json`. `scripts/manage-rag3-login.ps1` supports `-Operation Inspect`, `Plan`, `Prepare`, `Upgrade`, and `Install`; `Install` registers a missing task, while `Upgrade` changes only a recognized legacy action. Neither is an arbitrary force-restart command.

```powershell
cd C:\python_program\RAG3
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\manage-rag3-login.ps1 -Operation Inspect
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\manage-rag3-login.ps1 -Operation Plan
# Only if Inspect shows the recognized legacy pythonw.exe action:
# powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\manage-rag3-login.ps1 -Operation Upgrade
# Only if no task is registered:
# powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\manage-rag3-login.ps1 -Operation Install
Get-ScheduledTask -TaskName 'RAG3 Shared Backend'
```

`RAG3.exe` launches the PySide6 UI only; the launcher requires .NET 8 and the project virtual environment. A second launch focuses the current UI, and closing the UI does not terminate the backend or remote tunnel. Build with `scripts/build-launcher.ps1` when changing `launcher/Program.cs`. To verify the service, query `http://127.0.0.1:8000/health` and check the returned backend PID/ownership; a listening port alone is not proof that the correct backend owns it.

The old `scripts/rag3-mcp-tray.py`, stdio server startup, and independent search/collection workers are **not part of the normal Stage D startup path**. Do not use the old tray menu to stop or restart the current service. Before any restart, inspect active ingestion/download jobs and the queue. To apply backend code/config changes, use the Scheduled Task owner and verify the same SQLite data, a rotated local API token, and the backend PID afterward.

## Local HTTP and authentication boundaries

The MCP endpoint is Streamable HTTP at `http://127.0.0.1:8000/mcp` by default; the UI calls `http://127.0.0.1:8000/api/v1/command`. MCP is not authenticated using the UI token: the loopback-only UI API sends `X-RAG3-Token` loaded from the backend's port-specific local token file (`rag3/backend_boundary.py`). Never put that token in URLs, logs, the plugin manifest, or tunnel configuration. Backend restarts rotate it. The UI client reloads it for safe read retries; a timed-out write is treated as indeterminate and is reconciled without automatic duplicate submission.

Installed local Codex plugin `rag3@rag3-local` (verified version `0.1.2` on 2026-10-24) uses an HTTP manifest:

```json
{
  "mcpServers": {
    "rag3": {
      "type": "streamable-http",
      "url": "http://127.0.0.1:8000/mcp"
    }
  }
}
```

The local plugin does **not** spawn a separate Python stdio server. Start a **new** Codex task after changing the installed plugin; the live `rag3_list_collections` tool call was verified in a new task. The Python installation helper `scripts/ensure-rag3-codex-plugin.py --check-only` can inspect the installation; `--activate-http` explicitly updates it only after a live HTTP MCP readiness probe and must not reactivate a user-disabled plugin. The UI does not perform plugin registration or update on startup.

Windows login configuration controls `RAG3_MCP_PORT`, `RAG3_QDRANT_URL`, `RAG3_MCP_WRITE_ENABLED`, `RAG3_MCP_DELETE_ENABLED`, and optional `default_collection` for the **whole shared backend**, not per plugin or per remote client. The current registered configuration was verified with port 8000, Qdrant `http://127.0.0.1:16333`, write and delete enabled, and default collection `rag3_main_20261020`. `RAG3_MCP_WRITE_ENABLED=0` denies mutation, while `RAG3_MCP_DELETE_ENABLED=0` denies deletion tools; deletion also requires exact confirmation values. Never treat a tunnel connection as an independent per-client policy boundary. Restarting the UI alone does not change backend settings.

For ChatGPT web/cloud, use the existing authenticated Secure MCP Tunnel pointing to the same local `/mcp`. `scripts/rag3-tunnel-autostart.py` is a one-shot reconnector triggered after backend health is ready, reusing existing saved DPAPI credentials and profile. The tunnel runtime is separate and may remain resident. On 2026-10-24, this chat's connected RAG3 app returned the live collection list, shared ingestion jobs, and search results after backend restarts; the tunnel dispatch log increased for the connected calls. The app's MCP initialization exchange and external URL were not separately inspected. Preserve existing profiles, aliases, and credentials; do not publish them.

The login configuration's `default_collection` sets `RAG3_COLLECTION`, the preferred initial collection for a new MCP session when it exists. `rag3_select_collection` changes **that session's** selection; one client's choice does not change another client's choice or the UI collection. The current remote connector sends no persistent MCP session ID. Its `rag3_list_collections` reports `selection_persistent=false`; it can search the configured default, and `rag3_search_flexible` accepts an explicit collection for another collection. `rag3_select_collection` returns `SESSION_REQUIRED` on that connector so a selection is never falsely shared between users. Search/fetch IDs remain collection-scoped.

Qdrant new collections store BGE-M3 `text-dense` and Qdrant-native sparse `bm25` vectors. The old `text-sparse` vectors may remain on historical collections but are not queried. Pure BM25 does not require a CUDA BGE load. Semantic and hybrid searches reuse the shared CUDA service; CUDA failures are surfaced rather than silently falling back to CPU. Retrieval deduplicates results across methods; RRF score ties use a stable `point_id` order.

RAG3 does not impose an application file-size cap for original attachments; original files are fetched as verified disk-backed data rather than included in an MCP request body. Inline legacy Base64 transport remains subject to tunnel/message limits.

## Tools

| Area | Tools |
| --- | --- |
| Collections | `rag3_list_collections`, `rag3_select_collection`, `rag3_create_collection`, `rag3_delete_collection` |
| Ingestion | `rag3_ingest_file`, `rag3_ingest_text`, `rag3_ingestion_status`, `rag3_list_ingestion_jobs`, `rag3_pause_ingestion`, `rag3_resume_ingestion`, `rag3_cancel_ingestion` |
| Documents | `rag3_list_documents`, `rag3_get_document`, `rag3_update_chunk_metadata`, `rag3_delete_document` |
| Retrieval | `rag3_search`, `rag3_search_advanced`, `rag3_search_flexible`, `rag3_fetch` |

`rag3_ingest_file` accepts the original PDF, DOC, DOCX, MD, TXT, JSON, JSONL, CSV, or TSV attachment (text formats require UTF-8) in `file`, declared by `_meta["openai/fileParams"]`. The file object requires `download_url` and `file_id`, with optional `file_name` and `mime_type`. ChatGPT supplies this reference; users do not encode files or replace PDFs with extracted text. RAG3 returns a `job_id` immediately, downloads the original bytes to disk, validates the file, and queues the work in the shared backend job scheduler. Temporary signed URLs stay in memory and are not saved in job records or errors. Downloads allow only public HTTPS destinations, revalidate redirects, and pin connections to checked addresses.

`filename` and `content_base64` remain optional for legacy local clients; they are not the ChatGPT attachment path. Neither input has a RAG3 file-size cap. The Secure MCP Tunnel's own message limit still applies to legacy inline Base64 payloads, while attachment references remain small regardless of the original file size. Refresh the ChatGPT plugin's tool definitions after updating the server, then start a new conversation.

TXT, JSON, JSONL, and TSV retain their original bytes and go directly to the CPU token splitter without a parser copy; CSV uses Docling's CSV reader. JSON and JSONL are indexed as text without structural field extraction. Legacy DOC ingestion requires Microsoft Word on Windows: the shared ingestion path retains the original and creates a DOCX parser copy before Docling. `rag3_ingest_text` stores generated text as a managed Markdown source. Poll `rag3_ingestion_status` through `DOWNLOADING` and worker stages until `COMPLETE`, `FAILED`, `PAUSED`, or `CANCELLED`; `downloaded_bytes` reports download progress. Downloads can pause, resume, or cancel before ingestion. A download interrupted by a server restart needs a fresh attachment reference. For document processing, pause takes effect after conversion or a staged chunk group, or before the next embedding group. Resume uses Qdrant points and a small SQLite progress record. Cancel preserves staged Qdrant points for retry.

UI and MCP ingestion share the backend scheduler. Parsing can use one temporary CPU parser child; the CUDA BGE service remains resident in the backend process. Docling converts complete supported documents on CPU, then the CPU token splitter creates 1024-token chunks with 128-token overlap. The parser writes chunk text and metadata to Qdrant without vectors, and the backend then uses its existing GPU model service to embed staged chunks. The embedding process reads at most 32 chunks at a time (four BGE batches of eight), updates dense and BM25 vectors without resending the text payload, and marks those points ready. Ingestion does not create `snapshot.json`. PDF OCR and table structure remain enabled; table analysis uses Docling FAST mode, while picture image generation is disabled and image scaling is 1.0. A pause during full-document conversion takes effect after conversion or at the next staged chunk group; restarting an interrupted conversion reprocesses the source. Completed MCP uploads retain the verified managed original and discard the temporary upload copy.

`rag3_get_document` returns document metadata and a page of full chunk text. Use `chunk_offset` and `chunk_limit` (1–10) to continue. `rag3_search` combines BGE-M3 semantic and BM25 results with RRF and returns up to eight results. `rag3_search_advanced` exposes the UI's semantic, BM25, rank fusion, and final `top_k` controls. `rag3_fetch` reads a chunk by stable ID and offset. Physical page numbers are null for formats without pages; PDF chunks require valid page provenance.

`rag3_search_flexible` adds a per-request `collection`, `candidate_k`, and a Qdrant `metadata_filter` to the same search path. Filter keys refer to stored chunk metadata under `meta.`, for example `{"must":[{"key":"meta.document_id","match":{"value":"DOCUMENT_ID"}}]}`. Qdrant `must`, `should`, and `must_not` clauses allow AND, OR, and NOT combinations; range and match conditions can target any stored metadata field. The filter runs in each selected Dense and BM25 candidate search before RRF fusion. An omitted collection uses the server selection. A field with no stored value cannot match; `section_paths` is populated only after a user-requested annotation.

`rag3_update_chunk_metadata` lets an AI annotate existing chunks after the user requests it, without reingestion or reembedding. Read chunks with `rag3_get_document`, then supply their `point_id` values and numeric dotted `section_paths`, for example `[{"point_id":"...","section_paths":["2.3","2.3.1"]}]`. A chunk crossing a section boundary can list multiple paths. `mode=add` merges new paths; `mode=replace` corrects or clears them. The tool derives `section_scopes` with all ancestor numbers, so a filter matching `meta.section_scopes` to `2.3` includes chunks labeled `2.3.1`. A filter on `meta.section_paths` matches the recorded section number exactly. The tool previews changes by default (`dry_run=true`); pass `dry_run=false` to apply. It verifies that all point IDs belong to the requested document before writing only these two nested metadata fields, and rejects writes while the collection has an active ingestion job. Batch writes are not atomic; retrying the same request is safe.

UI and MCP job history is saved in `data/ingestion_jobs.sqlite`. Downloaded MCP attachments remain under `data/mcp_jobs`. The shared backend owns processing, so closing an MCP connection or UI window does not stop a running job. After backend interruption, resumable staged work is shown as paused and does not run again until requested.

The UI submits a local file as `QUEUED` and starts it only when Run is selected; MCP submissions request execution immediately. UI Run, Pause, Resume, Retry, Restart, Cancel, and Hide use the same backend scheduler and job IDs. Hide removes a row from the UI list without deleting its source or Qdrant points; MCP history still includes it. The UI reads revisions, job status, collection contents, and the local-only original path through the authenticated command API rather than opening the job SQLite database. A request that times out after a write is **completion unknown**: refresh the job/collection list before retrying, since the backend may still commit it.

## Verification

`scripts/test-mcp-attachments.py` checks streamed original bytes above the previous 25 MiB cap, all supported file formats, download controls, interrupted downloads, and the ChatGPT file-input schema. `scripts/test-mcp-attachment-http-e2e.py` downloads a public W3C PDF through the actual HTTP tool, runs ingestion, verifies its original hash and page metadata, reads its text, and removes only its disposable collection. It also verifies that existing collections and the selected collection are unchanged.

Other live client smoke scripts are `scripts/test-mcp-expanded-e2e.py`, `scripts/test-mcp-files-lifecycle-e2e.py`, `scripts/test-mcp-permissions.py`, and `scripts/test-mcp-http-client.py`. **Do not run live server tests on the production job database.** Even when the Qdrant collection is disposable, a child server can trigger recovery in the production SQLite DB. Inject and verify a separate `RAG3_TEST_DATA_ROOT` and `RAG3_BACKEND_TEST_INSTANCE` for all subprocesses before starting a test server; isolate source paths and collection names too.

`rag3_list_ingestion_jobs` reads shared UI/MCP SQLite job history (collection/status filters, limit/offset), without source paths or signed URLs. Ingestion execution concurrency defaults to one; backend restarts do not auto-resume interrupted jobs.
