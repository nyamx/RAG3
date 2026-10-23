"""Live MCP stdio smoke against an isolated temporary Qdrant collection."""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "scripts" / "rag3-mcp-server.py"


async def main() -> None:
    name = "__rag3_mcp_expanded_" + uuid.uuid4().hex[:12]
    env = os.environ.copy()
    env.update(RAG3_MCP_TRANSPORT="stdio", RAG3_MCP_WRITE_ENABLED="1",
               RAG3_MCP_DELETE_ENABLED="1", RAG3_COLLECTION="my_rag1")
    params = StdioServerParameters(command=os.sys.executable, args=[str(SERVER)],
                                   env=env, cwd=str(ROOT))
    async with stdio_client(params) as streams:
        async with ClientSession(*streams) as session:
            await session.initialize()
            tools = await session.list_tools()
            names = {tool.name for tool in tools.tools}
            required = {"rag3_search", "rag3_fetch", "rag3_search_advanced",
                        "rag3_list_collections", "rag3_select_collection",
                        "rag3_create_collection", "rag3_delete_collection",
                        "rag3_ingest_file", "rag3_ingest_text", "rag3_ingestion_status",
                        "rag3_pause_ingestion", "rag3_resume_ingestion",
                        "rag3_cancel_ingestion", "rag3_list_documents",
                        "rag3_get_document", "rag3_delete_document"}
            assert required <= names, sorted(required - names)

            async def call(tool: str, args: dict) -> dict:
                result = await session.call_tool(tool, args)
                payload = result.structured_content
                if payload is None:
                    payload = json.loads(result.content[0].text)
                if result.is_error:
                    raise AssertionError(f"{tool}: {payload}")
                return payload

            created = False
            selected = None
            try:
                before = await call("rag3_list_collections", {})
                selected = before["selected"]
                await call("rag3_create_collection", {"collection": name})
                created = True
                await call("rag3_select_collection", {"collection": name})
                listing = await call("rag3_list_collections", {})
                assert name in listing["collections"] and listing["selected"] == name
                job = await call("rag3_ingest_text", {
                    "collection": name, "title": "MCP integration note",
                    "text": "# MCP integration note\n\nAmber forest beta is a unique test phrase.\n",
                })
                job_id = job["job_id"]
                for _ in range(180):
                    status = await call("rag3_ingestion_status", {"job_id": job_id})
                    if status["status"] in {"COMPLETE", "FAILED", "CANCELLED", "PAUSED"}:
                        break
                    await asyncio.sleep(2)
                else:
                    raise AssertionError("ingestion job did not finish within six minutes")
                assert status["status"] == "COMPLETE", status
                docs = await call("rag3_list_documents", {"collection": name})
                assert docs["document_count"] == 1, docs
                document_id = docs["documents"][0]["document_id"]
                doc = await call("rag3_get_document", {"collection": name,
                                                        "document_id": document_id})
                assert "Amber forest beta" in doc["chunks"][0]["content"], doc
                hits = await call("rag3_search_advanced", {
                    "query": "Amber forest beta", "semantic": False,
                    "bm25": True,
                    "rank_fusion": False, "top_k": 3,
                })
                assert hits["results"], hits
                fetched = await call("rag3_fetch", {"id": hits["results"][0]["id"]})
                assert "Amber forest beta" in fetched["text"], fetched
                default_hits = await call("rag3_search", {"query": "Amber forest beta"})
                assert default_hits["results"], default_hits
                deleted = await call("rag3_delete_document", {
                    "collection": name, "document_id": document_id,
                    "confirm_document_id": document_id})
                assert deleted["chunks_deleted"] >= 1, deleted
                after = await call("rag3_list_documents", {"collection": name})
                assert after["document_count"] == 0, after
                print(json.dumps({"tools": len(names), "collection": name,
                                  "job_status": status["status"],
                                  "search_results": len(hits["results"]),
                                  "default_search_results": len(default_hits["results"]),
                                  "deleted_chunks": deleted["chunks_deleted"]},
                                 ensure_ascii=False), flush=True)
            finally:
                if selected and selected != name:
                    await call("rag3_select_collection", {"collection": selected})
                if created:
                    await call("rag3_delete_collection", {
                        "collection": name, "confirm_collection": name})


if __name__ == "__main__":
    asyncio.run(main())
