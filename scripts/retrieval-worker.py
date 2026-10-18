from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag3.retrieval import search


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True)
    ap.add_argument("--collection", required=True)
    ap.add_argument("--semantic", action="store_true")
    ap.add_argument("--bge-lexical", action="store_true")
    ap.add_argument("--bm25", action="store_true")
    ap.add_argument("--rank-fusion", action="store_true")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--mode", choices=["hybrid", "dense", "sparse"])
    args = ap.parse_args()

    try:
        semantic = args.semantic
        bge_lexical = args.bge_lexical
        bm25 = args.bm25
        rank_fusion = args.rank_fusion

        if not any((semantic, bge_lexical, bm25)) and args.mode is None:
            semantic = bge_lexical = bm25 = True
            rank_fusion = True

        result = search(
            args.query,
            semantic=semantic,
            bge_lexical=bge_lexical,
            bm25=bm25,
            rank_fusion=rank_fusion,
            top_k=args.top_k,
            collection=args.collection,
            mode=args.mode,
        )
        print(json.dumps({"type": "result", "result": result}, ensure_ascii=False), flush=True)
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {"type": "error", "message": str(exc), "traceback": traceback.format_exc()},
                ensure_ascii=False,
            ),
            flush=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
