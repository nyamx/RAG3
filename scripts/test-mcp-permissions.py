"""Verify explicit MCP write and delete restrictions remain effective."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parents[1]


async def check_denied(*, write_enabled: str, delete_enabled: str,
                       checks: dict[str, dict]) -> None:
    env = os.environ.copy()
    env.update(RAG3_MCP_TRANSPORT="stdio", RAG3_MCP_WRITE_ENABLED=write_enabled,
               RAG3_MCP_DELETE_ENABLED=delete_enabled)
    params = StdioServerParameters(command=os.sys.executable,
        args=[str(ROOT / "scripts" / "rag3-mcp-server.py")], env=env, cwd=str(ROOT))
    async with stdio_client(params) as streams:
        async with ClientSession(*streams) as session:
            await session.initialize()
            for tool, args in checks.items():
                result = await session.call_tool(tool, args)
                body = json.loads(result.content[0].text)
                assert result.is_error and body["error"]["code"] == "ACCESS_DENIED", (tool, body)
            print(json.dumps({"write_enabled": write_enabled,
                              "delete_enabled": delete_enabled,
                              "denied_tools": list(checks), "code": "ACCESS_DENIED"}))


async def main() -> None:
    # Invalid targets keep a permission regression from modifying existing data.
    collection = "__permission/probe__"
    checks = {
        "rag3_create_collection": {"collection": collection},
        "rag3_ingest_file": {"collection": collection, "filename": "probe.md",
                             "content_base64": "cHJvYmU="},
        "rag3_ingest_text": {"collection": collection, "title": "probe", "text": "probe"},
        "rag3_pause_ingestion": {"job_id": "not-a-job"},
        "rag3_resume_ingestion": {"job_id": "not-a-job"},
        "rag3_cancel_ingestion": {"job_id": "not-a-job"},
        "rag3_delete_collection": {"collection": collection, "confirm_collection": collection},
        "rag3_delete_document": {"collection": collection, "document_id": "probe",
                                 "confirm_document_id": "probe"},
    }
    await check_denied(write_enabled="0", delete_enabled="1", checks=checks)
    deletes = {name: args for name, args in checks.items() if name.startswith("rag3_delete_")}
    await check_denied(write_enabled="1", delete_enabled="0", checks=deletes)


if __name__ == "__main__":
    asyncio.run(main())
