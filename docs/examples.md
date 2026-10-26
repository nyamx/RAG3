# Worked examples

These examples start after the [Windows installation](install-windows.md). Keep Qdrant and the RAG3 backend running. Replace `demo`, the sample question, and any returned IDs with values from your own documents. The repository does not include a sample PDF, and the prompts below are examples of how to use the tools, not recorded output from a particular document.

## 1. Index a PDF and answer from it in the desktop UI

1. Start the backend and UI as described in the [quick start](../README.md#quick-start).
2. In the UI, create a collection named `demo` or choose one you already created.
3. Add a local PDF that you have permission to process. Select the queued job and click **Run**; adding a file in the UI alone does not start ingestion.
4. Wait for **COMPLETE**. Search for a phrase or question related to that PDF. Select **BGE-M3 (Semantic)** and **BM25 (Lexical)** with **Rank Fusion (RRF)** to combine their results, then compare either method separately if needed.
5. Open a result to inspect its stored text and source metadata. Use **Open Source** on the job to open the managed original. For figures or tables in an MCP client, request the relevant PDF page with `rag3_get_page`.

This workflow keeps the documents and Qdrant index on the RAG3 PC. A search result is a retrieved chunk; check the source before treating it as a complete answer.

## 2. Ask ChatGPT on the web about an indexed document

First complete the [ChatGPT web MCP setup](usage.md#chatgpt-on-the-web). In a new chat, select your installed RAG3 plugin with `@`. A useful first request is:

> Use my RAG3 plugin to list collections. Then search the `demo` collection for "What are the main findings?" Read the relevant result chunks before answering. Include document titles and PDF page ranges when the results provide them. If the retrieved text is insufficient, say what is missing.

For a sessionless web connection, the client should pass the collection on each call. The following illustrates the core read flow; `<id from search>` is the unchanged ID returned by the search tool:

```text
rag3_list_collections({})
rag3_search_flexible({"query":"What are the main findings?","collection":"demo","top_k":5})
rag3_fetch({"id":"<id from search>","offset":0,"collection":"demo"})
```

`rag3_fetch` reads up to 12,000 characters from one chunk. If it returns `next_offset`, fetch the same ID again at that offset to continue. To inspect a figure, use the `document_id` from a result and request the physical PDF page number (starting at 1):

```text
rag3_get_page({"document_id":"<document_id from result>","page":3,"collection":"demo"})
```

The page call only works for an indexed PDF with that page. `rag3_get_source` may provide a user-facing download of the original; it does **not** make the whole PDF an attachment readable by ChatGPT's AI. See the [file-handoff experiment record](chatgpt-file-handoff.md).

## 3. Use the local Codex MCP plugin

After [installing the local plugin](usage.md#codex-on-this-pc), open a new Codex task and try:

> List my RAG3 collections. Search the `demo` collection for the document's recommendations, then read the best matching chunks. Give a short summary with the document title and page ranges available in the results. Distinguish quoted source content from your own summary.

The same `rag3_search_flexible` and `rag3_fetch` calls above apply. Codex and the desktop UI use the same backend and collection data. Creating the collection and indexing a PDF in the UI is enough to make it searchable from Codex once the job finishes.

## 4. Save text produced in a chat

With a collection already created and MCP write access enabled, an MCP client can submit generated text as a managed Markdown source:

```text
rag3_ingest_text({"collection":"demo","title":"Meeting summary","text":"# Meeting summary\nDecisions and follow-up actions..."})
rag3_ingestion_status({"job_id":"<job_id from ingest_text>"})
```

Wait for the ingestion job to finish before searching for that text. This is an inbound write **to RAG3**. It does not transfer an original RAG3 file into the current ChatGPT conversation.
