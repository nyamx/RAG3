# RAG3 — search your documents from ChatGPT on the web

**Index a PDF on your Windows PC. Ask about it from ChatGPT, Codex, or the desktop UI. Trace an answer back to the source.**

RAG3 turns local documents into a searchable collection. Docling parses PDFs and other formats; BGE-M3 and Qdrant index both meaning and exact terms. A shared backend serves the desktop UI and MCP clients, so a document added in one place is searchable from the others after ingestion completes.

## Picture the workflow

1. **Add a document.** Create a `demo` collection in the desktop UI, add a PDF, click **Run**, and wait for **COMPLETE**. RAG3 keeps a verified source copy and indexes its text with page metadata when available.
2. **Ask a real question.** Search in the UI, or connect the MCP plugin and ask ChatGPT on the web:

   > Search my `demo` collection for the main findings. Read the matching chunks and show the document title and available source pages. If the evidence is thin, tell me what is missing.

3. **Inspect the evidence.** Read the returned chunks, compare semantic, BM25, and fused results, or request a PDF page image for a figure or table.

This ChatGPT web path was exercised on the development PC. It uses an [OpenAI Secure MCP Tunnel](https://developers.openai.com/api/docs/guides/secure-mcp-tunnels) and a custom MCP plugin; each user needs their own tunnel and ChatGPT workspace access. RAG3 supplies retrieved content to the model on request. The original PDF does not automatically become a ChatGPT conversation attachment.

## Try it

| Step | Guide |
| --- | --- |
| 1. Install Python, WSL2, Docker, Qdrant, and local models | [Windows installation](docs/install-windows.md) |
| 2. Start RAG3 and connect your preferred client | [Desktop, Codex, and ChatGPT web setup](docs/usage.md) |
| 3. Follow a PDF-to-answer walkthrough | [Worked examples and copyable prompts](docs/examples.md) |

Want the internals? Read the [architecture](docs/architecture.md), [verification results](docs/verification.md), and [file-handoff experiments](docs/chatgpt-file-handoff.md).

## Engineering highlights

One measured local run indexed a **535-page PDF into 252 chunks in 30.7 minutes** on a 4 GiB Quadro T1000. Dense, BM25, and hybrid searches all found the source in that run. The [verification notes](docs/verification.md) show what was measured and what was not.

- **One backend for three clients.** The PySide6 UI, local Codex plugin, and ChatGPT web plugin use the same collections, ingestion jobs, and retrieval path. Closing the UI leaves the backend running.
- **GPU ownership on a 4 GiB card.** Long PDFs use a short-lived CUDA parser. RAG3 releases BGE-M3 before parsing and reloads it for embedding or semantic search, while BM25-only search stays available without the dense model.
- **Recoverable ingestion.** A SQLite job journal records progress and state; the UI can pause, resume, retry, and cancel jobs. Managed source copies carry SHA-256 checks for source retrieval.
- **Inspectable retrieval.** BGE-M3 dense vectors and Qdrant's native BM25 sparse vectors support semantic, lexical, and reciprocal-rank-fusion search. Results retain source metadata and PDF page information when available.
- **Format-aware parsing.** Docling handles PDF OCR and tables plus structured formats; plain text has a direct path. Legacy `.doc` conversion uses Microsoft Word when installed.

```text
PySide6 desktop UI -------------------+
Codex local MCP plugin ---------------+--> Shared RAG3 backend
ChatGPT web --> Secure MCP Tunnel ----+      |-- SQLite job journal
                                            |-- Docling parser and BGE-M3
                                            |-- Qdrant dense and BM25 index
                                            +-- managed source files
```

The [architecture notes](docs/architecture.md) explain the data flow, GPU handoff, and retrieval behavior.

## Requirements at a glance

| For | Required components |
| --- | --- |
| Desktop UI and backend | Windows x64, Python 3.11, NVIDIA GPU and compatible driver, WSL2 Ubuntu, Docker Engine and Compose in Ubuntu, Qdrant, local BGE-M3 and Docling models |
| Optional launcher | .NET 8 runtime; .NET 8 SDK only to build `RAG3.exe` |
| Local Codex MCP plugin | Running backend and Codex with local plugin support |
| ChatGPT web MCP plugin | Running backend, Secure MCP Tunnel client and permissions, ChatGPT custom MCP plugin access |

Microsoft Word is needed only for legacy `.doc` conversion. The [Windows installation guide](docs/install-windows.md) lists the setup and checks for every component.

## Quick start

Follow the [Windows installation guide](docs/install-windows.md) to set up WSL2, Docker, Qdrant, Python packages, and local model files. The launcher is optional.

Once the environment and Qdrant are ready, start the shared backend in one PowerShell window:

```powershell
.\.venv\Scripts\python.exe scripts\rag3-mcp-server.py
```

In another window, open the UI:

```powershell
.\.venv\Scripts\python.exe scripts\ingest-ui.py
```

Create a collection, add a PDF, run its job, then search the collection. The backend also serves `http://127.0.0.1:8000/mcp` for local MCP clients. See [Using RAG3](docs/usage.md) for the UI and MCP paths.

## Verification and limits

The [verification notes](docs/verification.md) include a repeatable disposable PDF-to-search check and the measurements behind the run above. Device-wide GPU usage reached 3,873 MiB. These numbers describe one document and PC, not a general performance guarantee.

**ChatGPT original-file boundary:** Experiments uploaded a PDF to ChatGPT host file storage and downloaded matching bytes, but did not register it as an attachment in the current conversation or make it directly readable by the AI. The uploaded file was not found in the Library UI; internal Library registration remains unknown. Search, chunk reads, and page images are the demonstrated ways to provide evidence to ChatGPT. See the [experiment record](docs/chatgpt-file-handoff.md).

The code and workflow have been exercised on the development PC. A clean installation on a second PC has not yet been completed. The bundled infrastructure script currently expects the checkout at `C:\python_program\RAG3`; other locations require editing that script.

## License and published content

The source code and documentation are available under the [MIT License](LICENSE). The repository does not include your indexed documents, model weights, local `.env`, Qdrant data, or tunnel credentials. ChatGPT and other MCP clients receive document content only when you invoke the corresponding tools; review access and write permissions before connecting a client.
