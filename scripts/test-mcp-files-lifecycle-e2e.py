"""Live MCP file formats and pause/resume/cancel smoke in a temp collection."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import uuid
from pathlib import Path

from docx import Document
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logging.getLogger("httpx").setLevel(logging.WARNING)
ROOT = Path(__file__).resolve().parents[1]


def make_pdf(line: str) -> bytes:
    parts = [b"<< /Type /Catalog /Pages 2 0 R >>",
             b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
             b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
             b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>"]
    stream = f"BT /F1 18 Tf 72 720 Td ({line}) Tj ET".encode("ascii")
    parts.append(b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n"
                 + stream + b"\nendstream")
    parts.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, part in enumerate(parts, 1):
        offsets.append(len(out))
        out.extend(f"{number} 0 obj\n".encode() + part + b"\nendobj\n")
    xref = len(out)
    out.extend(f"xref\n0 {len(parts)+1}\n".encode() + b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        out.extend(f"{offset:010d} 00000 n \n".encode())
    out.extend(f"trailer\n<< /Size {len(parts)+1} /Root 1 0 R >>\n"
               f"startxref\n{xref}\n%%EOF\n".encode())
    return bytes(out)


async def main() -> None:
    suffix = uuid.uuid4().hex[:12]
    collection = "__rag3_mcp_files_" + suffix
    env = os.environ.copy()
    env.update(RAG3_MCP_TRANSPORT="stdio", RAG3_MCP_WRITE_ENABLED="1",
               RAG3_MCP_DELETE_ENABLED="1", RAG3_COLLECTION="")
    params = StdioServerParameters(command=os.sys.executable,
        args=[str(ROOT / "scripts" / "rag3-mcp-server.py")], env=env, cwd=str(ROOT))
    async with stdio_client(params) as streams:
        async with ClientSession(*streams) as session:
            await session.initialize()

            async def call(tool: str, args: dict, *, expect_error: bool = False) -> dict:
                response = await session.call_tool(tool, args)
                body = response.structured_content or json.loads(response.content[0].text)
                if bool(response.is_error) != expect_error:
                    raise AssertionError(f"{tool}: {body}")
                return body

            async def wait_job(job_id: str, terminal: set[str], timeout: int = 300) -> dict:
                for _ in range(timeout // 2):
                    body = await call("rag3_ingestion_status", {"job_id": job_id})
                    if body["status"] in terminal:
                        return body
                    await asyncio.sleep(2)
                raise AssertionError(f"job {job_id} timed out")

            created = False
            jobs: list[str] = []
            try:
                existing = await call("rag3_list_collections", {})
                assert existing.get("collections"), "Create one default RAG3 collection before tests"
                await call("rag3_create_collection", {"collection": collection})
                created = True
                await call("rag3_select_collection", {"collection": collection})
                bad = await call("rag3_ingest_file", {"collection": collection,
                    "filename": "bad.pdf", "content_base64": "AAAA"}, expect_error=True)
                assert bad["error"]["code"] == "INVALID_ARGUMENT"

                phrase = "Violet ocean " + suffix
                pdf = make_pdf(phrase)
                submitted = await call("rag3_ingest_file", {"collection": collection,
                    "filename": "mcp-pause.pdf",
                    "content_base64": base64.b64encode(pdf).decode()})
                jobs.append(submitted["job_id"])
                index = await call("rag3_list_ingestion_jobs", {"collection": collection, "limit": 20})
                assert submitted["job_id"] in {j["job_id"] for j in index["jobs"]}
                paused_request = await call("rag3_pause_ingestion", {"job_id": submitted["job_id"]})
                assert paused_request["status"] in {"PAUSED", "PAUSING"}
                paused = await wait_job(submitted["job_id"], {"PAUSED", "FAILED", "COMPLETE"})
                assert paused["status"] == "PAUSED", paused
                await call("rag3_resume_ingestion", {"job_id": submitted["job_id"]})
                complete = await wait_job(submitted["job_id"], {"COMPLETE", "FAILED"})
                assert complete["status"] == "COMPLETE", complete
                assert complete["origin"] == "MCP" and not complete["checkpoint_available"]

                docx = Document()
                docx.add_paragraph("Amber forest " + suffix)
                stream = io.BytesIO()
                docx.save(stream)
                submitted_docx = await call("rag3_ingest_file", {"collection": collection,
                    "filename": "mcp-file.docx",
                    "content_base64": base64.b64encode(stream.getvalue()).decode()})
                jobs.append(submitted_docx["job_id"])
                docx_done = await wait_job(submitted_docx["job_id"], {"COMPLETE", "FAILED"})
                assert docx_done["status"] == "COMPLETE", docx_done

                doc_bytes = (ROOT / "artifacts" / "e2e-formats" / "e2e.doc").read_bytes()
                submitted_doc = await call("rag3_ingest_file", {"collection": collection,
                    "filename": "mcp-file.doc",
                    "content_base64": base64.b64encode(doc_bytes).decode()})
                jobs.append(submitted_doc["job_id"])
                doc_done = await wait_job(submitted_doc["job_id"], {"COMPLETE", "FAILED"})
                assert doc_done["status"] == "COMPLETE", doc_done

                md = ("# Markdown fixture\n\nSilver mountain " + suffix + "\n").encode()
                submitted_md = await call("rag3_ingest_file", {"collection": collection,
                    "filename": "mcp-file.md",
                    "content_base64": base64.b64encode(md).decode()})
                jobs.append(submitted_md["job_id"])
                md_done = await wait_job(submitted_md["job_id"], {"COMPLETE", "FAILED"})
                assert md_done["status"] == "COMPLETE", md_done

                docs = await call("rag3_list_documents", {"collection": collection})
                assert docs["document_count"] == 4, docs
                pages = {doc["title"]: (doc["page_start"], doc["page_end"])
                         for doc in docs["documents"]}
                assert pages["mcp-pause"] == (1, 1), pages
                hits = await call("rag3_search_advanced", {"query": phrase,
                    "semantic": False, "bm25": True,
                    "rank_fusion": False, "top_k": 3})
                assert hits["results"] and hits["results"][0]["page_start"] == 1, hits
                fetched = await call("rag3_fetch", {"id": hits["results"][0]["id"]})
                assert phrase in fetched["text"] and fetched["page_start"] == 1, fetched

                cancel_job = await call("rag3_ingest_text", {"collection": collection,
                    "title": "Cancel " + suffix, "text": "Cancel test " + suffix})
                jobs.append(cancel_job["job_id"])
                await call("rag3_cancel_ingestion", {"job_id": cancel_job["job_id"]})
                cancelled = await wait_job(cancel_job["job_id"], {"CANCELLED", "COMPLETE", "FAILED"})
                assert cancelled["status"] == "CANCELLED", cancelled
                print(json.dumps({"pdf": complete["status"], "doc": doc_done["status"],
                    "docx": docx_done["status"],
                    "md": md_done["status"], "cancel": cancelled["status"],
                    "pdf_pages": pages["mcp-pause"]}, ensure_ascii=False), flush=True)
            finally:
                for job_id in jobs:
                    status = await call("rag3_ingestion_status", {"job_id": job_id})
                    if status["status"] in {"RUNNING", "RESUMING", "PAUSING"}:
                        await call("rag3_cancel_ingestion", {"job_id": job_id})
                        await wait_job(job_id, {"CANCELLED", "COMPLETE", "FAILED"}, 30)
                others = await call("rag3_list_collections", {})
                alternate = next((name for name in others.get("collections", []) if name != collection), None)
                if alternate is None:
                    raise AssertionError("A non-test collection is needed for safe deletion")
                await call("rag3_select_collection", {"collection": alternate})
                if created:
                    await call("rag3_delete_collection", {"collection": collection,
                        "confirm_collection": collection})


if __name__ == "__main__":
    asyncio.run(main())
