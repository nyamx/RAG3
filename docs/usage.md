# Using RAG3

Start Qdrant and the shared backend as described in the [Windows installation guide](install-windows.md). The backend must be running before the UI or an MCP client can use RAG3.

## Desktop UI

1. Open `scripts/ingest-ui.py` or the optional `RAG3.exe` launcher.
2. Create or choose a Qdrant collection.
3. Add a local file to the job list and choose **Run**. The UI initially queues local files; adding one does not start ingestion by itself.
4. Watch the job status until it is **COMPLETE**. Pause, resume, retry, and cancel controls operate on the same backend job journal.
5. Search the collection using semantic, BM25, or hybrid retrieval. Results show stored text and source metadata. Select a job and use **Open Source** to open its original file.

The UI accepts PDF, DOC, DOCX, PPTX, XLSX, HTML, Markdown, TXT, JSON, JSONL, CSV, TSV, and common image formats. Output quality depends on the source and parser. `.doc` requires Microsoft Word on Windows; plain UTF-8 text does not need Docling conversion.

## MCP clients

The local server uses Streamable HTTP at `http://127.0.0.1:8000/mcp`. The included [plugin manifest](../plugins/rag3/mcp.json) points to that address. It connects to the running shared backend; it does not start another Python server.

### Codex on this PC

With the backend running, install or refresh the bundled local plugin from the repository root:

```powershell
.\.venv\Scripts\python.exe scripts\ensure-rag3-codex-plugin.py --check-only
.\.venv\Scripts\python.exe scripts\ensure-rag3-codex-plugin.py --activate-http
codex plugin list --json
```

The helper checks the live MCP endpoint before changing the plugin and leaves a user-disabled installation disabled. The Codex CLI must be installed and available. Open a new Codex task after installation, then ask it to list RAG3 collections. The local plugin does not connect ChatGPT on the web.

### ChatGPT on the web

RAG3 was exercised from ChatGPT web through a custom MCP plugin and OpenAI Secure MCP Tunnel. The tunnel lets ChatGPT reach the private `http://127.0.0.1:8000/mcp` endpoint while RAG3 keeps running on your PC. ChatGPT web does not read the local Codex plugin manifest. This is a per-user setup, subject to ChatGPT workspace and Platform tunnel permissions.

1. Complete the [Windows installation](install-windows.md), start the backend, and verify `http://127.0.0.1:8000/health` locally.
2. Follow the [Secure MCP Tunnel guide](https://developers.openai.com/api/docs/guides/secure-mcp-tunnels) to create a tunnel associated with the ChatGPT workspace, install `tunnel-client` on the RAG3 PC, point its HTTP MCP profile to `http://127.0.0.1:8000/mcp`, run `tunnel-client doctor`, and keep the client running. Keep tunnel IDs and runtime keys out of this repository.
3. In [ChatGPT Plugins](https://chatgpt.com/plugins), choose **Add custom MCP server** and **Tunnel** as the connection, select your tunnel, configure authentication, create and install the plugin. See the [official ChatGPT custom MCP guide](https://developers.openai.com/api/docs/guides/custom-mcp-server).
4. Start a new ChatGPT web conversation, select the installed plugin with `@`, and first ask it to list your RAG3 collections. Then search a named collection. For a remote connection without a session ID, use `rag3_search_flexible(collection=...)` or configure a backend default collection; `rag3_select_collection` needs a session.

The repository's tunnel helper restores an existing local tunnel profile. It does not create another user's tunnel, key, or ChatGPT plugin. A local Codex plugin installation alone does not enable ChatGPT web access. Secure MCP Tunnel is for a private connection; publishing this GitHub source repository does not publish an MCP plugin to everyone.

**Original PDF delivery to the AI:** `rag3_get_source` can provide a download card or source URI, but this does not make the PDF a native attachment in the current ChatGPT conversation. In the [October 2026 file-handoff experiments](chatgpt-file-handoff.md), a PDF reached ChatGPT host storage and was downloaded back with a matching SHA-256, yet the AI could not read it by the returned `fileId`. Use `rag3_search`, `rag3_get_document`, and `rag3_fetch` for text, or `rag3_get_page` for a page image. This limitation concerns RAG3-to-ChatGPT delivery; the separate `rag3_ingest_file` tool ingests an input file into RAG3 when an MCP client supplies a valid file reference.

The main MCP tool groups are:

| Task | Tools |
| --- | --- |
| Collections | `rag3_list_collections`, `rag3_select_collection`, `rag3_create_collection` |
| Ingestion | `rag3_ingest_file`, `rag3_ingest_text`, `rag3_ingestion_status`, `rag3_list_ingestion_jobs` |
| Reading | `rag3_list_documents`, `rag3_get_document`, `rag3_fetch` |
| Search | `rag3_search`, `rag3_search_advanced`, `rag3_search_flexible` |

The UI and MCP clients share Qdrant collections and ingestion jobs. An MCP file submission starts work immediately; a UI file submission waits for **Run**. The MCP file attachment path accepts PDF, DOC, DOCX, MD, TXT, JSON, JSONL, CSV, and TSV. Other formats available in the UI are not all exposed as MCP attachments.

For any other remote MCP client, the loopback address needs a separately configured connection. Tunnel profiles and credentials are not included in this repository.
