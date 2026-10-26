# RAG3-to-ChatGPT file handoff experiments

This record describes **outbound** transfer of an original RAG3 file through MCP to ChatGPT. It is separate from **inbound** ingestion, where `rag3_ingest_file` receives a file reference from an MCP client and adds that file to RAG3.

## Evidence and conclusion

The October 18–19 results below were reconstructed from RAG3 development conversations. Some experiment logs survive only as summaries; the historical probe code was not rerun for this document. The later Library UI search is a direct observation reported by the user on October 26, 2026. These results apply to the tested ChatGPT and RAG3 setup, not to every possible MCP client or future ChatGPT release.

**Observed result:** RAG3 uploaded a PDF to ChatGPT host file storage, received a `fileId`, then downloaded the file back with a matching SHA-256. `uploadFile(file, {library: true})` returned success, but a direct search in the ChatGPT Library UI did not find the file. That file was **not registered as an attachment in the current conversation**. An attempted AI read using the returned `fileId` failed with a file-reference visibility error. Native whole-PDF reading by the AI was not demonstrated. The observations do not distinguish failed internal Library registration from successful registration that was not exposed in the UI, and they do not establish intentional blocking by OpenAI.

| Date | Experiment | Observation |
| --- | --- | --- |
| Oct 18 | Return the original PDF URI with `rag3_get_source` | URI returned; model access to the original was not established. |
| Oct 18 | Show a PDF viewer and file in MCP UI | Server and UI response succeeded; model visibility was not established. |
| Oct 18 | Call `uploadFile(File)` with a PDF | Upload succeeded and a `fileId` was issued. |
| Oct 18 | Call `getFileDownloadUrl()` and download the upload | Download succeeded; SHA-256 matched the original. |
| Oct 18 | Have the AI read using the returned `fileId` | Failed with a file-reference visibility error. The file had not become a current-conversation attachment. |
| Oct 18 | Upload a ZIP | Rejected as `Unsupported file type`. |
| Oct 19 | Call `uploadFile(file, {library: true})` | Success response; this did not register a current-conversation attachment. |
| Oct 19 | Probe model context and file delivery | Host upload succeeded; direct AI reading still failed. |
| Oct 26 | Search for the uploaded file in the ChatGPT Library UI | File not found. Internal registration failure versus UI non-exposure remains unresolved. |

Two further probes explored alternatives:

- **One-page `EmbeddedResource`: partial success.** The historical `rag3_direct_pdf_resource_probe` returned a real 24,615-byte PDF page from a 40-page source as `application/pdf`. A ChatGPT MCP call reportedly extracted a sentence from it. The record does not establish that ChatGPT's native PDF-processing path performed the extraction. This was an experiment, not a shipped RAG3 tool.
- **Large PDF as MCP content: unsuccessful.** The historical `rag3_direct_pdf_max_probe` encoded a 358-page, roughly 7.32 MB PDF into roughly 9.76 MB of Base64. Although it fit the tested MCP transfer-size condition, the ChatGPT conversation-length limit was reached. Model receipt and processing were not confirmed. This was also an experimental probe, not a shipped RAG3 tool.

The following paths were considered but **not verified end to end** in these experiments: `ui/update-model-context`, `ui/message`, `setWidgetState(modelContent)`, `openai/files/open`, and `_meta["openai/fileParams"]`. The last item describes a file **input** parameter and does not by itself prove an outbound file becomes an AI-readable conversation attachment.

## Practical boundary

| Stage | Result in the tested setup |
| --- | --- |
| RAG3 to ChatGPT host file storage | Confirmed by `fileId` and matching downloaded hash. |
| ChatGPT Library | `library: true` returned success, but the file was not found in the Library UI. Internal registration status remains unknown. |
| Current-conversation attachment registration | Did not occur. |
| Native AI reading of the original PDF | Not demonstrated; the direct `fileId` attempt failed. |

For questions about indexed documents, use RAG3's `rag3_search`, `rag3_fetch`, and `rag3_get_document` text results. Use `rag3_get_page` for a specific PDF page image and `rag3_get_source` for a user-facing original download. Do not present the download card, returned URI, or host `fileId` as proof that the whole original PDF entered the model's context.

OpenAI's [plugin file API reference](https://developers.openai.com/plugins/reference#file-apis) documents file upload and download helpers. Their existence explains how the host-storage steps could succeed; the conversation-attachment and AI-visibility result above comes from the RAG3 experiment record.
