from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcp.server import MCPServer
from rag3.haystack_native import DEFAULT_NATIVE_COLLECTION
from rag3.retrieval import search

COLLECTION = os.environ.get("RAG3_COLLECTION", DEFAULT_NATIVE_COLLECTION)

mcp = MCPServer(
    "RAG3 Retrieval",
    instructions=(
        "Search the local RAG3 corpus through the Haystack-native Qdrant retrieval pipeline. "
        "Use hybrid unless a caller explicitly needs dense-only or sparse-only retrieval."
    ),
)


@mcp.tool()
def search_rag3(
    query: str,
    mode: Literal["dense", "sparse", "hybrid"] = "hybrid",
    top_k: int = 5,
) -> dict:
    """Retrieve evidence chunks from the Haystack-native RAG3 corpus."""
    return search(query, mode=mode, top_k=top_k, collection=COLLECTION)


if __name__ == "__main__":
    mcp.run()
