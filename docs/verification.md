# Verification

This check parses a PDF, embeds its chunks, stores both vector types, and retrieves the source with dense, BM25, and hybrid search. The verification script uses a unique Qdrant collection and an isolated job directory, then removes that collection when it finishes.

Run it on a machine with the [Windows setup](install-windows.md) complete. Avoid running another GPU-heavy ingestion job at the same time.

```powershell
.\.venv\Scripts\python.exe scripts\verify-rag3-pdf-gpu.py
```

With no `--pdf`, the script generates a nine-page PDF. It forces CUDA parsing, checks that BGE-M3 is released and reloaded, and tests all three search modes. It writes a JSON report under `artifacts/verify-pdf-gpu-*/report.json`.

To test a document you are allowed to use locally:

```powershell
.\.venv\Scripts\python.exe scripts\verify-rag3-pdf-gpu.py --pdf C:\path\to\document.pdf --pages 0 --timeout-minutes 60
```

`--pages 0` processes the complete PDF. The default is the first 24 pages. The script retains a managed source copy and report under `artifacts/`; inspect those files before sharing them. Do not commit private or copyrighted input PDFs, paths, or unedited reports.

## Measured local run

One private 535-page PDF was tested on 2026-10-25. The source file and raw report are intentionally omitted from the public documentation.

| Measurement | Result |
| --- | ---: |
| Input | 535 pages, 33,473,585 bytes |
| Indexed chunks | 252 |
| Parse phase | 303.0 s |
| Embedding phase | 1,539.8 s |
| Reported ingestion total | 1,843.0 s (30.7 min) |
| Dense / BM25 / hybrid search | Source found in all three |
| GPU | NVIDIA Quadro T1000, 4,096 MiB |
| Highest observed device-wide GPU memory use | 3,873 MiB |
| Lowest observed free GPU memory | 65 MiB |
| Highest observed test-process-tree RSS | 7,631.5 MiB |

The sampler read `nvidia-smi` and process memory every 0.5 seconds. GPU memory numbers cover the **whole device**, including unrelated processes; they are not a per-model allocation. The run verifies this document and configuration, not a minimum hardware specification or a worst-case limit. Different page layouts, OCR load, images, and other GPU users can change memory use and runtime. A comparable before/after performance baseline was not recorded.
