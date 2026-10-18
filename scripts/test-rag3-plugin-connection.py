"""Verify that the local RAG3 plugin launches and advertises its MCP tools."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parents[1]


async def main() -> None:
    plugin_root = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "plugins" / "rag3"
    config = json.loads((plugin_root / "mcp.json").read_text(encoding="utf-8"))
    server = config["mcpServers"]["rag3"]
    args = [value.replace("${PLUGIN_ROOT}", str(plugin_root)) for value in server["args"]]
    env = os.environ.copy()
    env.update(server["env"])
    print("Starting plugin MCP process", flush=True)
    params = StdioServerParameters(command=server["command"], args=args,
                                   env=env, cwd=str(plugin_root))
    async with stdio_client(params) as streams:
        print("MCP process started", flush=True)
        async with ClientSession(*streams) as session:
            info = await asyncio.wait_for(session.initialize(), timeout=90)
            tools = await asyncio.wait_for(session.list_tools(), timeout=30)
            names = {tool.name for tool in tools.tools}
            assert len(names) == 16, sorted(names)
            assert {"rag3_search", "rag3_ingest_text", "rag3_delete_document"} <= names
            print(f"PLUGIN_MCP_OK server={info.server_info.name} tools={len(names)}")


if __name__ == "__main__":
    asyncio.run(main())
