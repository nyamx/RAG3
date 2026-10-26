# RAG3 project handoff

Basis: repository code and public documentation as of October 26, 2026. Writing this handoff did not restart the application or rerun full ingestion tests. Interpret measurements and historical experiments within the scope stated in the [verification notes](verification.md) and [file-handoff record](chatgpt-file-handoff.md). A Korean version is available in [프로젝트 인계서](project-handoff.ko.md).

## 1. What the project delivers

RAG3 ingests and indexes documents on a Windows PC, then makes the same collections searchable from a desktop UI, a local Codex MCP plugin, and a ChatGPT web MCP plugin. Where available, it retains PDF page ranges and the managed original so a retrieved passage can be checked against its source. ChatGPT web reaches the local MCP server through a Secure MCP Tunnel and a custom plugin configured for that user.

There is an important boundary between **giving the AI a whole original document as a conversation attachment** and **retrieving relevant indexed content for the AI**. The latter is the working RAG3 path. The former reached ChatGPT host file storage in experiments, but did not establish conversation attachment registration or direct model reading of the original PDF.

## 2. Runtime architecture

~~~text
Windows user
  ├─ PySide6 UI ── authenticated /api/v1/command ┐
  ├─ local Codex MCP ── /mcp ──────────────────────┤
  └─ ChatGPT web ── Secure MCP Tunnel ─ /mcp ─────┤
                                                  ▼
                                  shared HTTP/MCP backend (Python)
                                  ├─ SQLite job journal and scheduler
                                  ├─ resident CPU Docling parser child
                                  ├─ temporary CUDA parser child for long PDFs
                                  ├─ BGE-M3 GPU model service
                                  ├─ ingestion, retrieval, source serving
                                  └─ Qdrant (Docker in WSL2)
                                         ├─ dense: text-dense
                                         └─ sparse: bm25
~~~

The normal HTTP entry point is one backend process at [scripts/rag3-mcp-server.py](../scripts/rag3-mcp-server.py). The UI is a separate process but does not own job or retrieval state. [RAG3.exe](../launcher/Program.cs) is an optional .NET UI launcher, not a backend replacement. Qdrant, the tunnel runtime, and parser children are separate processes. The former stdio server, tray, and standalone search workers are not part of the current normal startup path.

## 3. Technology choices and reasons

| Technology | Used for | Reason and implementation consequence |
| --- | --- | --- |
| Python 3.11, PySide6 | Backend and desktop UI | Connect the document and ML libraries in one language while keeping the Windows UI a separate client. |
| Docling, pypdfium2 | Structured-document parsing and page rendering | Enable PDF OCR and table structure in Docling; use PDFium to count pages and render a physical original page. Some plain-text formats bypass Docling. |
| Haystack, tiktoken | Conversion, token splitting, Qdrant document-store integration | Reuse ingestion components and split chunks with o200k_base. RAG3's retrieval code controls multi-method search and fusion. |
| BGE-M3, FlagEmbedding, PyTorch CUDA | 1,024-dimensional dense embeddings | Perform local semantic retrieval. A shared GPU service loads the model on demand and reuses it. |
| Qdrant 1.19.2 | Collections, dense and native BM25 sparse vectors, filters, RRF | Apply semantic and lexical search to the same chunks with server-side filtering and rank fusion. |
| SQLite WAL | Jobs, stage/checkpoint records, UI change versions | Persist ingestion without another database service. Recovery also checks Qdrant chunk state and the original hash. |
| MCP Streamable HTTP | Codex and ChatGPT clients | Share one backend and tool boundary across the UI and AI clients. ChatGPT web needs a separate tunnel and plugin setup. |
| WSL2 and Docker Compose | Qdrant on Windows | Keep the vector service and persistent volume separate from application code. PostgreSQL is defined but not started by the default ingestion/search path. |
| SHA-256 managed originals | Source preservation, reading, recovery | Verify that indexed chunks still correspond to the original file and reject implicit resume after a source change. |

For pinned versions and prerequisites, use [requirements.txt](../requirements.txt), [requirements-ui.txt](../requirements-ui.txt), [compose.yaml](../compose.yaml), and the [Windows installation guide](install-windows.md). Model weights, personal documents, data volumes, and credentials are absent from Git.

RapidOCR is pinned as a dependency, but the current pipeline enables Docling OCR without explicitly selecting RapidOCR as its engine. Verify the effective OCR backend on a fresh installation.

## 4. Ingestion implementation

1. The UI queues a local file and waits for the user to click **Run**. MCP file or text submission requests execution immediately. Both routes use the shared scheduler and job IDs.
2. The backend preserves the original under data/sources and records its SHA-256. For MCP file references, [attachments.py](../rag3/attachments.py) streams only from checked public HTTPS addresses. It validates URLs, redirects, filenames, types, and incomplete downloads. Signed URLs are not persisted in job records.
3. TXT/JSON/JSONL/TSV take a lightweight plain-text splitting path. Docling converts PDFs and other supported structured formats. Legacy DOC needs Microsoft Word to make a DOCX parser copy.
4. The current defaults in [ingestion_settings.py](../rag3/ingestion_settings.py) are 1,024-token chunks, 128-token overlap, BGE batches of eight, and chunk groups of 32. Parsed text, document ID, source hash, and available page metadata are first staged in Qdrant.
5. The shared GPU service computes dense chunk vectors. Qdrant produces native BM25 sparse vectors from the same content. Completion requires verification of the per-source dense, BM25, and ready counts after chunks become ready.
6. SQLite jobs/checkpoints and Qdrant staged/ready points form the recovery record. After interruption, the server does not automatically submit the work again; compatible work remains PAUSED. A mismatch in source hash, pipeline identity, chunk counts, or point state blocks implicit recovery.

The 4 GiB GPU design avoids keeping the parser and embedding model loaded together. By default, a PDF of at least eight physical pages makes RAG3 drain BGE work, release the model, run a temporary CUDA parser child, and terminate that child. The next embedding or semantic query reloads BGE. Shorter PDFs and other structured formats use a resident CPU parser. RAG3_CUDA_PARSE_MIN_PAGES changes the threshold. Semantic search may wait or return an error while the parser owns the GPU; BM25-only search can run without BGE.

## 5. Retrieval and reading

- [retrieval.py](../rag3/retrieval.py) optionally runs BGE-M3 dense and Qdrant native BM25 queries. Qdrant RRF fuses both methods; equal scores use point ID for stable ordering. Metadata filtering is applied during candidate retrieval.
- The UI and MCP call the same retrieval path. MCP rag3_search_flexible accepts a collection, candidate count, and metadata filter per call, which matters for remote clients without a persistent session ID.
- [rag3-mcp-server.py](../scripts/rag3-mcp-server.py) formats hits with source information and chunk IDs. rag3_fetch reads one chunk by character offset. A search excerpt is not evidence that the full source was read.
- [collections.py](../rag3/collections.py) lists collections/documents, reads and deletes documents, and corrects section metadata. Section updates default to a dry run and verify that the points belong to the requested document before writing.
- [sources.py](../rag3/sources.py) resolves a collection/document ID to a managed original and verifies its path and hash. rag3_get_source provides user-facing original-download information; rag3_get_page renders a physical PDF page.

## 6. Module map for maintainers

| Area | Main files | Responsibility and change impact |
| --- | --- | --- |
| Entry point and MCP | [rag3-mcp-server.py](../scripts/rag3-mcp-server.py) | Streamable HTTP startup, tool schemas, search/source responses, UI command routing. Check local and remote clients after tool changes. |
| Backend boundary | [backend_boundary.py](../rag3/backend_boundary.py) | Single-instance lock, per-session collection selection, request budget, /health, token-protected local UI API. |
| Ingestion scheduler | [ingestion_backend.py](../rag3/ingestion_backend.py), [ingestion_manager.py](../rag3/ingestion_manager.py) | One execution slot, stage transitions, pause/resume/cancel/recovery, CPU/CUDA parser and GPU coordination. The current backend reuses the earlier manager interface. |
| Job journal | [ingestion_jobs.py](../rag3/ingestion_jobs.py), [mcp_jobs.py](../rag3/mcp_jobs.py) | SQLite WAL, progress identities, status reads, conversion of MCP inputs into shared jobs. Check recovery when changing the schema. |
| Parsing | [persistent_parser.py](../rag3/persistent_parser.py), [ingest-worker.py](../scripts/ingest-worker.py), [windows_job.py](../rag3/windows_job.py) | Resident CPU/temporary CUDA parser child lifecycle, Docling execution, Windows child-process cleanup. |
| Ingestion pipeline | [haystack_native.py](../rag3/haystack_native.py), [ingestion_light.py](../rag3/ingestion_light.py), [ingestion_settings.py](../rag3/ingestion_settings.py) | Conversion, chunk normalization, source preservation, plain-text path, pipeline identity. Review checkpoint compatibility after changing chunk settings. |
| Embedding and GPU | [gpu_model_service.py](../rag3/gpu_model_service.py), [bge_dense.py](../rag3/bge_dense.py), [retrieval_embedding.py](../rag3/retrieval_embedding.py) | BGE ownership, query/document queues, deadlines/cancellation, GPU handoff to the CUDA parser. |
| Vector storage and search | [qdrant_store.py](../rag3/qdrant_store.py), [qdrant_bm25.py](../rag3/qdrant_bm25.py), [qdrant_clients.py](../rag3/qdrant_clients.py), [retrieval.py](../rag3/retrieval.py) | Collection schema, BM25 configuration, client reuse, dense/BM25/RRF retrieval. |
| Source and attachment security | [attachments.py](../rag3/attachments.py), [sources.py](../rag3/sources.py), [source_identity.py](../rag3/source_identity.py), [source_guard.py](../rag3/source_guard.py), [source-viewer.html](../rag3/source-viewer.html) | Attachment-download validation, SHA-256, managed-path validation, original parts, PDF page rendering, and the MCP UI download card. The card is not a model conversation attachment. |
| Desktop | [ingest-ui.py](../scripts/ingest-ui.py), [ui_backend_client.py](../rag3/ui_backend_client.py), [ui_state.py](../rag3/ui_state.py), [Program.cs](../launcher/Program.cs) | PySide6 view/state, asynchronous authenticated HTTP client, optional single-window launcher. A timed-out write is not automatically replayed. |
| Startup and connections | [rag3-backend-login.py](../scripts/rag3-backend-login.py), [manage-rag3-login.ps1](../scripts/manage-rag3-login.ps1), [ensure-infra.ps1](../scripts/ensure-infra.ps1), [rag3-tunnel-autostart.py](../scripts/rag3-tunnel-autostart.py), [ensure-rag3-codex-plugin.py](../scripts/ensure-rag3-codex-plugin.py), [mcp.json](../plugins/rag3/mcp.json) | Windows login task, WSL Docker Qdrant startup, saved-tunnel reconnection, and local Codex plugin setup/address. |

## 7. Data, configuration, and security boundaries

| Item | Location and meaning |
| --- | --- |
| SQLite job journal | data/ingestion_jobs.sqlite. Because WAL is used, take an SQLite online backup rather than copying only the main DB file. |
| Managed originals | Under data/sources/hash/. A Qdrant snapshot alone cannot restore original downloads. |
| MCP download jobs | Temporary and progress files under data/mcp_jobs/. An interrupted signed-URL download needs a new attachment reference. |
| Qdrant data | The rag3_qdrant_data Docker volume; back up or migrate it separately from the code checkout. |
| Models | Local BGE-M3 and Docling files under data/models/. They are not committed. |
| Login settings | data/backend-login-config.json contains non-secret port, Qdrant address, and backend-wide write/delete policy. |
| Tunnel credentials | User-specific local saved profile; never put them in the repository, documentation, or logs. |

The HTTP server binds to 127.0.0.1 by default. The UI command API uses a backend-issued X-RAG3-Token, rotated on backend restart. MCP does not use that UI token. RAG3_MCP_WRITE_ENABLED and RAG3_MCP_DELETE_ENABLED control write/delete permission for the **whole backend**. In a manual server launch, the code defaults both to allowed, so explicitly review the policy before connecting a remote client. Sessionless MCP clients should pass the collection on each call; do not assume one user's selection carries to another. Compose currently publishes Qdrant ports without a host-IP restriction, so inspect network exposure separately.

## 8. Startup, verification, and transfer sequence

1. Follow the [Windows installation guide](install-windows.md) for dependencies, models, WSL2 Docker, and Qdrant. The current infrastructure script expects C:\python_program\RAG3.
2. Follow [usage](usage.md) to start the backend and UI; check http://127.0.0.1:8000/health for the correct backend PID and status. Closing the UI leaves the backend running.
3. Connect the local Codex plugin and the ChatGPT web tunnel separately. ChatGPT web requires each user's own tunnel and workspace access.
4. Use [worked examples](examples.md) with a disposable collection. For full ingestion regression, follow the [verification script guide](verification.md).
5. Check active ingestion jobs before changing production data or running live-server tests. Give test servers a separate RAG3_TEST_DATA_ROOT and RAG3_BACKEND_TEST_INSTANCE in every child process, plus a disposable Qdrant collection. A test server that shares production SQLite can alter production jobs during recovery.
6. Plan backup/migration of the SQLite online backup, Qdrant snapshot or volume, managed originals, and model files together. Transfer credentials separately through a secure channel.

## 9. Change-impact guide

- **Adding an input format:** Review MCP suffixes in [constants.py](../rag3/constants.py), attachment validation, the UI picker, the Docling/plain-text branch, and source MIME serving together. Do not assume that the UI and MCP expose identical formats.
- **Changing chunks, embeddings, or collection schema:** Treat the pipeline identity, Qdrant vector names/dimensions, existing collection compatibility, and resume checkpoints as one design change. Do not silently read old ingested data under a new schema.
- **Changing an MCP tool:** Check its server schema, read/write/delete policy, and fresh local Codex and ChatGPT web plugin connections separately. Updated server code does not imply that an already-open AI chat has refreshed tool definitions.
- **Changing recovery or backup:** Verify consistency across SQLite, Qdrant points, managed originals, and manual resume after interruption. A timed-out UI write has unknown completion, so refresh state before submitting it again.

## 10. Verified results and remaining work

- One recorded local run on October 25, 2026 indexed a 535-page PDF into 252 chunks in 30.7 minutes; dense, BM25, and hybrid search all found the source. Maximum device-wide use on the 4 GiB Quadro T1000 was 3,873 MiB. This is one machine/document measurement, not a general performance guarantee.
- The shared backend, UI, local Codex MCP, and ChatGPT web read path through the existing tunnel were exercised on the development PC. **A fresh end-to-end installation on another PC has not been verified.**
- Outbound original-PDF experiments confirmed ChatGPT host storage, re-download, and a matching hash only. The file was not found in the Library UI; current-conversation attachment registration and direct model reading failed. Internal Library registration remains unknown. See [the detailed record](chatgpt-file-handoff.md).
- Maintenance priorities are reproducing installation on a second PC, removing the hard-coded infrastructure path, reviewing Qdrant port exposure, verifying backup/restore operations, and measuring a comparable before/after performance baseline. Before adding functionality, check ingestion-checkpoint compatibility and preservation of source hashes and provenance.

Source code is available under the [MIT License](../LICENSE).
