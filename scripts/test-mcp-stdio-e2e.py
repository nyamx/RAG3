from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parents[1]
PYTHON = str(ROOT / ".venv" / "Scripts" / "python.exe")
SERVER = str(ROOT / "scripts" / "rag3-mcp-server.py")
COLLECTION = "rag3_haystack_e2e_formats"


async def main() -> None:
    env = os.environ.copy()
    env["RAG3_COLLECTION"] = COLLECTION

    params = StdioServerParameters(
        command=PYTHON,
        args=[SERVER],
        env=env,
        cwd=str(ROOT),
    )

    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            init = await session.initialize()
            tools = await session.list_tools()
            names = [t.name for t in tools.tools]
            if "search_rag3" not in names:
                raise SystemExit(f"search_rag3 tool missing: {names}")

            result = await session.call_tool(
                "search_rag3",
                {
                    "query": "Which fixture mentions amber forest beta?",
                    "mode": "hybrid",
                    "top_k": 3,
                },
            )

            payload = None
            for item in getattr(result, "content", []) or []:
                text = getattr(item, "text", None)
                if text:
                    try:
                        payload = json.loads(text)
                    except Exception:
                        continue
                    if isinstance(payload, dict):
                        break

            if not isinstance(payload, dict):
                structured = getattr(result, "structuredContent", None)
                if isinstance(structured, dict):
                    payload = structured

            if not isinstance(payload, dict):
                raise SystemExit(f"Could not decode MCP tool result: {result!r}")

            hits = payload.get("hits") or []
            if not hits:
                raise SystemExit("MCP stdio search returned no hits")

            out = {
                "server_name": getattr(init, "serverInfo", None).name if getattr(init, "serverInfo", None) else None,
                "tools": names,
                "hit_count": len(hits),
                "top_title": hits[0].get("title"),
                "top_content": hits[0].get("content"),
            }
            print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
