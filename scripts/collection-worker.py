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

DEFAULT_QDRANT_URL = "http://127.0.0.1:16333"


from rag3.collections import collection_contents, create_collection, delete_collections, list_collections


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--action", choices=["list", "create", "delete_many", "contents"], required=True)
    ap.add_argument("--collection")
    ap.add_argument("--collections-json")
    ap.add_argument("--bm25-k", type=float, default=1.2)
    ap.add_argument("--bm25-b", type=float, default=0.75)
    ap.add_argument("--bm25-avg-len", type=float, default=256.0)
    args = ap.parse_args()

    try:
        if args.action == "list":
            names = list_collections()
            print(json.dumps({"collections": names}, ensure_ascii=False), flush=True)
            return 0

        if args.action == "delete_many":
            # Validate the full request before deleting anything.
            names = json.loads(args.collections_json or "[]")
            if not isinstance(names, list) or not names:
                raise ValueError("at least one collection is required")
            names = list(dict.fromkeys(str(item).strip() for item in names if str(item).strip()))
            if not names:
                raise ValueError("at least one collection is required")
            print(json.dumps(delete_collections(names), ensure_ascii=False), flush=True)
            return 0

        name = (args.collection or "").strip()
        if not name:
            raise ValueError("collection name is required")

        if args.action == "contents":
            print(json.dumps(collection_contents(name), ensure_ascii=False), flush=True)
            return 0

        result = create_collection(
            name,
            bm25_k=args.bm25_k,
            bm25_b=args.bm25_b,
            bm25_avg_len=args.bm25_avg_len,
            reuse_existing=True,
        )
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {"error": str(exc), "traceback": traceback.format_exc()},
                ensure_ascii=False,
            ),
            flush=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
