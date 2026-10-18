"""Receive a real HTTPS PDF reference and ingest it through Streamable HTTP.

Uses the public W3C PDF fixture and a disposable collection; existing user
collections and the selected collection must remain unchanged.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

ROOT = Path(__file__).resolve().parents[1]
PDF_URL = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"


async def main() -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    env = os.environ.copy()
    env.update(RAG3_MCP_TRANSPORT="streamable-http", RAG3_MCP_PORT=str(port),
               RAG3_MCP_WRITE_ENABLED="1", RAG3_MCP_DELETE_ENABLED="1")
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    process = subprocess.Popen([sys.executable, str(ROOT / "scripts/rag3-mcp-server.py")],
        cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        creationflags=flags)
    collection = "__rag3_attachment_" + uuid.uuid4().hex[:12]
    job_id = None
    created = False
    try:
        async with httpx.AsyncClient(timeout=2) as client:
            for _ in range(90):
                if process.poll() is not None:
                    raise AssertionError("MCP server exited before startup")
                try:
                    response = await client.get(f"http://127.0.0.1:{port}/")
                    if response.status_code in {200, 404}:
                        break
                except httpx.RequestError:
                    pass
                await asyncio.sleep(1)
            else:
                raise AssertionError("MCP server did not start")
        async with streamable_http_client(f"http://127.0.0.1:{port}/mcp") as streams:
            async with ClientSession(streams[0], streams[1]) as session:
                await session.initialize()

                async def call(name: str, args: dict) -> dict:
                    result = await session.call_tool(name, args)
                    body = result.structured_content or json.loads(result.content[0].text)
                    assert not result.is_error, (name, body)
                    return body

                tools = (await session.list_tools()).tools
                ingest = next(tool for tool in tools if tool.name == "rag3_ingest_file")
                assert ingest.meta["openai/fileParams"] == ["file"], ingest.meta
                before = await call("rag3_list_collections", {})
                try:
                    await call("rag3_create_collection", {"collection": collection})
                    created = True
                    started = time.monotonic()
                    job = await call("rag3_ingest_file", {"collection": collection, "file": {
                        "download_url": PDF_URL, "file_id": "public-w3c-pdf-fixture",
                        "file_name": "original-attachment.pdf", "mime_type": "application/pdf",
                    }})
                    job_id = job["job_id"]
                    assert job["status"] == "DOWNLOADING", job
                    assert time.monotonic() - started < 10, "download must run in the background"
                    print("ATTACHMENT_ACCEPTED original_file=true inline_bytes=false", flush=True)
                    previous = None
                    deadline = time.monotonic() + 360
                    while time.monotonic() < deadline:
                        job = await call("rag3_ingestion_status", {"job_id": job_id})
                        stage = (job["status"], job["stage"])
                        if stage != previous:
                            print(json.dumps({"status": stage[0], "stage": stage[1]},
                                             ensure_ascii=False), flush=True)
                            previous = stage
                        if job["status"] in {"COMPLETE", "FAILED", "CANCELLED"}:
                            break
                        await asyncio.sleep(2)
                    assert job["status"] == "COMPLETE", job
                    path = ROOT / "data/mcp_jobs" / job_id / "original-attachment.pdf"
                    with path.open("rb") as source:
                        assert source.read(5) == b"%PDF-"
                        source.seek(0)
                        assert hashlib.file_digest(source, "sha256").hexdigest() == job["source_sha256"]
                    assert path.stat().st_size == job["file_size"] == job["downloaded_bytes"]
                    docs = await call("rag3_list_documents", {"collection": collection})
                    assert docs["document_count"] == 1, docs
                    document = docs["documents"][0]
                    assert document["source_sha256"] == job["source_sha256"], document
                    assert (document["page_start"], document["page_end"]) == (1, 1), document
                    fetched = await call("rag3_get_document", {"collection": collection,
                        "document_id": document["document_id"]})
                    assert "Dummy PDF file" in " ".join(chunk["content"] for chunk in fetched["chunks"]), fetched
                    print(json.dumps({"HTTP_ATTACHMENT_E2E": "OK", "file_bytes": job["file_size"],
                        "original_sha256": job["source_sha256"], "pdf_pages": [1, 1],
                        "document_count": 1}, ensure_ascii=False), flush=True)
                finally:
                    if job_id:
                        status = await call("rag3_ingestion_status", {"job_id": job_id})
                        if status["status"] in {"QUEUED", "DOWNLOADING", "RUNNING", "RESUMING", "PAUSING"}:
                            await call("rag3_cancel_ingestion", {"job_id": job_id})
                            for _ in range(40):
                                status = await call("rag3_ingestion_status", {"job_id": job_id})
                                if status["status"] not in {"CANCELLING", "RUNNING", "DOWNLOADING"}:
                                    break
                                await asyncio.sleep(1)
                    if created:
                        await call("rag3_delete_collection", {"collection": collection,
                            "confirm_collection": collection})
                    after = await call("rag3_list_collections", {})
                    assert set(after["collections"]) == set(before["collections"]), after
                    assert after["selected"] == before["selected"], after
                    print("USER_COLLECTIONS_UNCHANGED", flush=True)
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


if __name__ == "__main__":
    asyncio.run(main())
