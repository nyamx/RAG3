"""Read an existing managed PDF through real MCP HTTP without modifying it."""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import os
import socket
import subprocess
import sys
import uuid
from pathlib import Path

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from qdrant_client import QdrantClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rag3.constants import DEFAULT_QDRANT_URL


async def main(collection: str, live: bool) -> None:
    client = QdrantClient(url=DEFAULT_QDRANT_URL)
    cursor = None
    metadata = None
    try:
        while True:
            points, cursor = client.scroll(collection_name=collection, limit=256,
                offset=cursor, with_payload=True, with_vectors=False)
            for point in points:
                meta = (point.payload or {}).get("meta") or {}
                if str(meta.get("source_uri", "")).lower().endswith(".pdf"):
                    metadata = meta
                    break
            if metadata or cursor is None:
                break
    finally:
        client.close()
    assert metadata, "Collection needs an existing managed PDF for this read-only check"
    document_id = metadata["document_id"]
    process = None
    port = 8000
    if not live:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        env = os.environ.copy()
        env.update(RAG3_MCP_TRANSPORT="streamable-http", RAG3_MCP_PORT=str(port),
                   RAG3_COLLECTION=collection, RAG3_MCP_WRITE_ENABLED="0", RAG3_MCP_DELETE_ENABLED="0")
        process = subprocess.Popen([sys.executable, str(ROOT / "scripts/rag3-mcp-server.py")],
            cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    try:
        async with httpx.AsyncClient(timeout=2) as client:
            for _ in range(90):
                if process and process.poll() is not None:
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

                async def call(name, args):
                    result = await session.call_tool(name, args)
                    assert not result.is_error, result
                    return result

                tools = {tool.name: tool for tool in (await session.list_tools()).tools}
                assert len(tools) == 19
                assert tools["rag3_get_source"].annotations.read_only_hint
                # Explicit collection works even when a different collection is selected.
                result = await call("rag3_get_source", {"document_id": document_id, "collection": collection})
                source = result.structured_content
                assert source["source_sha256"] == metadata["source_sha256"]
                assert "data_base64" not in source
                assert str(ROOT) not in json.dumps(source)
                ui = await session.read_resource(tools["rag3_get_source"].meta["ui"]["resourceUri"])
                assert "ui/download-file" in ui.contents[0].text
                assert "openai/widgetCSP" not in ui.contents[0].meta
                assert "openExternal" not in ui.contents[0].text
                assert "<button" not in ui.contents[0].text
                output = ROOT / "data/logs/source-readback" / uuid.uuid4().hex[:12]
                output.mkdir(parents=True)
                target = output / source["filename"]
                digest = hashlib.sha256()
                offset = 0
                blocks = 0
                with target.open("wb") as destination:
                    while True:
                        uri = source["first_part_uri"].rsplit("/", 1)[0] + "/" + str(offset)
                        resource = await session.read_resource(uri)
                        wire = resource.model_dump_json(by_alias=True).encode()
                        assert len(wire) < 1024 * 1024
                        part = json.loads(resource.contents[0].text)
                        assert part["offset"] == offset
                        assert part["source_sha256"] == source["source_sha256"]
                        content = base64.b64decode(part["data_base64"], validate=True)
                        assert hashlib.sha256(content).hexdigest() == part["part_sha256"]
                        if offset == 0:
                            via_tool = await call("rag3_read_source_part", {"collection": collection,
                                "document_id": document_id, "offset": offset})
                            assert "data_base64" not in json.dumps(via_tool.structured_content)
                            assert "data_base64" not in json.dumps([item.model_dump() for item in via_tool.content])
                            assert via_tool.meta["rag3/sourcePart"] == part
                            assert len(via_tool.model_dump_json(by_alias=True).encode()) < 1024 * 1024
                        destination.write(content)
                        digest.update(content)
                        blocks += 1
                        offset = part["end_offset"]
                        if part["next_offset"] is None:
                            break
                assert target.stat().st_size == source["file_size"] == offset
                assert digest.hexdigest() == source["source_sha256"]
                page = await call("rag3_get_page", {"collection": collection,
                    "document_id": document_id, "page": 1, "max_dimension": 1600})
                image = next(item for item in page.content if item.type == "image")
                assert len(page.model_dump_json(by_alias=True).encode()) < 1024 * 1024
                (output / "page-001.jpg").write_bytes(base64.b64decode(image.data))
                if page.structured_content["page_count"] >= 15:
                    detail = await call("rag3_get_page", {"collection": collection,
                        "document_id": document_id, "page": 15, "max_dimension": 2000})
                    image = next(item for item in detail.content if item.type == "image")
                    (output / "page-015.jpg").write_bytes(base64.b64decode(image.data))
                report = {"HTTP_SOURCE_E2E": "OK", "live_tray": live,
                    "file_bytes": source["file_size"], "source_sha256": digest.hexdigest(),
                    "blocks": blocks, "page_count": page.structured_content["page_count"],
                    "read_only": True, "output": str(output)}
                (ROOT / "data/logs/mcp-source-verification.json").write_text(
                    json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
                print(json.dumps(report, ensure_ascii=False), flush=True)
    finally:
        if process:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", required=True)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    asyncio.run(main(args.collection, args.live))
