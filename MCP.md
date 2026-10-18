# RAG3 MCP server

`scripts/rag3-mcp-server.py` exposes the existing RAG3 ingestion worker, Haystack/Qdrant collection view, and core retrieval through MCP. It does not run a separate ingestion or retrieval pipeline.

Codex installation, ChatGPT desktop/web connection, and UI startup auto-registration are documented in [RAG3_PLUGIN_SETUP.md](RAG3_PLUGIN_SETUP.md).

## Use RAG3 by name in local AI chats

This repository includes a local plugin at `plugins/rag3` and a marketplace at `.agents/plugins/marketplace.json`. The normal UI startup checks and installs this local Codex plugin when missing. To install it manually on this Windows PC:

```powershell
codex plugin marketplace add C:\python_program\RAG3
codex plugin add rag3@rag3-local
```

Start a **new local** Codex or ChatGPT desktop conversation and select `@RAG3`, or ask, for example, “RAG3의 컬렉션을 보여줘” or “RAG3의 my_rag1에서 이 주제를 찾아 출처와 함께 요약해 줘.” The plugin starts the existing MCP server over stdio when needed. The HTTP tray server may be stopped; Qdrant and the project's `.venv` must be available. The plugin currently points to `C:\python_program\RAG3`; update `plugins/rag3/mcp.json` and reinstall if the project moves.

The local plugin enables ingestion and job control. It also enables deletion tools, which still require exact confirmation fields and advertise destructive actions to the client. Installing a plugin during an existing conversation does not add its tools to that conversation; start a new one.

ChatGPT web and cloud conversations cannot reach this PC's stdio process. To use RAG3 there, connect the local HTTP endpoint through an authenticated Secure MCP Tunnel and register it as a custom MCP plugin in ChatGPT. The local plugin installation does not configure that tunnel.

## Start

For a normal Windows launch, double-click `RAG3.exe` in the project root. It opens the existing PySide6 UI without a console window; the UI then starts the MCP tray controller. A second double-click focuses the existing UI. This is a small launcher, so keep it beside the project's `.venv` and `scripts` directories. It requires the installed .NET 8 runtime. Rebuild it with `powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/build-launcher.ps1`.

On Windows, starting `scripts/ingest-ui.py` also starts a separate RAG3 MCP controller in the system tray. The UI window can close while the controller and its Streamable HTTP server continue running. Starting the UI again reuses the existing controller. Double-click the tray icon to open or focus the UI; its menu shows server status and offers restart, stop, and exit. **Exit RAG3 background** stops the MCP server and removes the tray icon. Server output is written to `data/logs/mcp-server.log`; controller errors go to `data/logs/mcp-tray.log`. The controller inherits `RAG3_MCP_PORT`, `RAG3_MCP_WRITE_ENABLED`, and `RAG3_MCP_DELETE_ENABLED` from the UI launch environment. The UI smoke-test mode does not start it.

Use the project virtual environment. For a local stdio client, configure:

```json
{
  "mcpServers": {
    "rag3": {
      "command": "C:\\python_program\\RAG3\\.venv\\Scripts\\python.exe",
      "args": ["C:\\python_program\\RAG3\\scripts\\rag3-mcp-server.py"],
      "env": {
        "RAG3_MCP_TRANSPORT": "stdio",
        "RAG3_MCP_WRITE_ENABLED": "1"
      }
    }
  }
}
```

Writes and deletion are enabled by default for both stdio and the HTTP tray server. Set `RAG3_MCP_WRITE_ENABLED=0` to reject all eight mutation tools with `ACCESS_DENIED`, or set `RAG3_MCP_DELETE_ENABLED=0` to reject only the two deletion tools. Deletion still requires an exact confirmation value and retains the selected-collection and active-job checks. The server does not delete or rebuild existing collections on startup. Restart the MCP tray server after changing these settings or updating the server code.

The default transport is Streamable HTTP at `http://127.0.0.1:8000/mcp`. Set `RAG3_MCP_PORT` to choose a different local port. The server binds to loopback; exposing it outside the machine requires an authenticated gateway. RAG3 imposes no file-size cap. Original attachments are downloaded directly to disk, outside the MCP request body; the old 25 MiB file cap and 40 MiB application request cap have been removed.

`RAG3_QDRANT_URL` sets the Qdrant endpoint. `RAG3_COLLECTION` sets the initial selected collection. `rag3_select_collection` changes the selection for the server process, so clients sharing one process should coordinate selection. Search and fetch IDs are bound to the selected collection; select the same collection before fetching a prior search result.

## Tools

| Area | Tools |
| --- | --- |
| Collections | `rag3_list_collections`, `rag3_select_collection`, `rag3_create_collection`, `rag3_delete_collection` |
| Ingestion | `rag3_ingest_file`, `rag3_ingest_text`, `rag3_ingestion_status`, `rag3_pause_ingestion`, `rag3_resume_ingestion`, `rag3_cancel_ingestion` |
| Documents | `rag3_list_documents`, `rag3_get_document`, `rag3_delete_document` |
| Retrieval | `rag3_search`, `rag3_search_advanced`, `rag3_fetch` |

`rag3_ingest_file` accepts the original PDF, DOC, DOCX, or UTF-8 MD attachment in `file`, declared by `_meta["openai/fileParams"]`. The file object requires `download_url` and `file_id`, with optional `file_name` and `mime_type`. ChatGPT supplies this reference; users do not encode files or replace PDFs with extracted text. RAG3 returns a `job_id` immediately, downloads the original bytes to disk, validates the file, and starts the existing ingestion worker. Temporary signed URLs stay in memory and are not saved in job records or errors. Downloads allow only public HTTPS destinations, revalidate redirects, and pin connections to checked addresses.

`filename` and `content_base64` remain optional for legacy local clients; they are not the ChatGPT attachment path. Neither input has a RAG3 file-size cap. The Secure MCP Tunnel's own message limit still applies to legacy inline Base64 payloads, while attachment references remain small regardless of the original file size. Refresh the ChatGPT plugin's tool definitions after updating the server, then start a new conversation.

Legacy DOC ingestion requires Microsoft Word on Windows: the shared ingestion path retains the original and creates a DOCX parser copy before Docling. `rag3_ingest_text` stores generated text as a managed Markdown source. Poll `rag3_ingestion_status` through `DOWNLOADING` and worker stages until `COMPLETE`, `FAILED`, `PAUSED`, or `CANCELLED`; `downloaded_bytes` reports download progress. Downloads can pause, resume, or cancel before ingestion. A download interrupted by a server restart needs a fresh attachment reference. Worker pause takes effect at an existing checkpoint, and worker resume requires a saved checkpoint. Cancel preserves any checkpoint already written.

`rag3_get_document` returns document metadata and a page of full chunk text. Use `chunk_offset` and `chunk_limit` (1–10) to continue. `rag3_search` keeps the v1.2 default hybrid search and returns up to eight results. `rag3_search_advanced` exposes the UI's semantic, BGE lexical, BM25, rank fusion, and final `top_k` controls. `rag3_fetch` reads a chunk by stable ID and offset. Physical page numbers are null for formats without pages; PDF chunks require valid page provenance.

MCP jobs are saved under `data/mcp_jobs`. If the MCP server exits during a job, a later status check reports failure unless the worker recorded completion. Resume can use a checkpoint when one exists. The existing UI worker and its checkpoint format remain in use.

## Verification

`scripts/test-mcp-attachments.py` checks streamed original bytes above the previous 25 MiB cap, all supported file formats, download controls, interrupted downloads, and the ChatGPT file-input schema. `scripts/test-mcp-attachment-http-e2e.py` downloads a public W3C PDF through the actual HTTP tool, runs ingestion, verifies its original hash and page metadata, reads its text, and removes only its disposable collection. It also verifies that existing collections and the selected collection are unchanged.

Other live client smoke scripts are `scripts/test-mcp-expanded-e2e.py`, `scripts/test-mcp-files-lifecycle-e2e.py`, `scripts/test-mcp-permissions.py`, and `scripts/test-mcp-http-client.py`. The E2E scripts create uniquely named temporary collections and delete only those test collections.
