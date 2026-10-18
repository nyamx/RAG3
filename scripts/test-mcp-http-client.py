"""Verify default MCP writes and deletion confirmation over Streamable HTTP."""

from __future__ import annotations

import asyncio
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

ROOT = Path(__file__).resolve().parents[1]


async def main() -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    env = os.environ.copy()
    env.update(RAG3_MCP_TRANSPORT="streamable-http", RAG3_MCP_PORT=str(port),
               RAG3_COLLECTION="my_rag1")
    env.pop("RAG3_MCP_WRITE_ENABLED", None)
    env.pop("RAG3_MCP_DELETE_ENABLED", None)
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    process = subprocess.Popen([sys.executable, str(ROOT / "scripts" / "rag3-mcp-server.py")],
        cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        creationflags=flags)
    url = f"http://127.0.0.1:{port}/mcp"
    try:
        async with httpx.AsyncClient(timeout=2) as client:
            for _ in range(90):
                if process.poll() is not None:
                    raise AssertionError(f"server exited early: {process.returncode}")
                try:
                    response = await client.get(f"http://127.0.0.1:{port}/")
                    if response.status_code in {200, 404}:
                        break
                except httpx.RequestError:
                    pass
                await asyncio.sleep(1)
            else:
                raise AssertionError("HTTP server did not start")
        async with streamable_http_client(url) as streams:
            async with ClientSession(streams[0], streams[1]) as session:
                init = await session.initialize()
                tools = await session.list_tools()
                names = {tool.name for tool in tools.tools}
                assert "rag3_search" in names and "rag3_ingest_file" in names

                async def call(tool: str, args: dict, *, expect_error: bool = False) -> dict:
                    result = await session.call_tool(tool, args)
                    body = result.structured_content or json.loads(result.content[0].text)
                    assert bool(result.is_error) == expect_error, (tool, body)
                    return body

                collection = "__rag3_http_write_" + uuid.uuid4().hex[:12]
                before = await call("rag3_list_collections", {})
                assert collection not in before["collections"], before
                creation_attempted = False
                deleted = False
                try:
                    creation_attempted = True
                    created = await call("rag3_create_collection", {"collection": collection})
                    assert created["collection"] == collection and created["count"] == 0, created
                    listed = await call("rag3_list_collections", {})
                    assert collection in listed["collections"], listed
                    assert listed["selected"] == before["selected"], listed
                    rejected = await call("rag3_delete_collection", {
                        "collection": collection, "confirm_collection": collection + "_wrong",
                    }, expect_error=True)
                    assert rejected["error"]["code"] == "INVALID_ARGUMENT", rejected
                    preserved = await call("rag3_list_collections", {})
                    assert collection in preserved["collections"], preserved
                    removed = await call("rag3_delete_collection", {
                        "collection": collection, "confirm_collection": collection,
                    })
                    deleted = True
                    assert removed["deleted"] == [collection], removed
                    after = await call("rag3_list_collections", {})
                    assert set(after["collections"]) == set(before["collections"]), after
                    assert after["selected"] == before["selected"], after
                    print(f"HTTP_MCP_OK server={init.server_info.name} tools={len(names)} "
                          f"port={port} default_writes=allowed delete_confirmation=checked")
                finally:
                    if creation_attempted and not deleted:
                        remaining = await call("rag3_list_collections", {})
                        if collection in remaining["collections"]:
                            await call("rag3_delete_collection", {
                                "collection": collection, "confirm_collection": collection,
                            })
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


if __name__ == "__main__":
    asyncio.run(main())
