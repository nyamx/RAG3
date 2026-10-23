"""Remove the retired meta.section_path key from one Qdrant collection."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from qdrant_client import QdrantClient, models

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag3.constants import DEFAULT_QDRANT_URL


def scan(client: QdrantClient, collection: str) -> tuple[int, list[int | str], int]:
    scanned = 0
    matching: list[int | str] = []
    nonempty = 0
    cursor = None
    while True:
        points, cursor = client.scroll(
            collection_name=collection, limit=256, offset=cursor,
            with_payload=models.PayloadSelectorInclude(include=["meta"]),
            with_vectors=False,
        )
        scanned += len(points)
        for point in points:
            meta = (point.payload or {}).get("meta")
            if isinstance(meta, dict) and "section_path" in meta:
                matching.append(point.id)
                if meta["section_path"] not in (None, [], ""):
                    nonempty += 1
        if cursor is None:
            return scanned, matching, nonempty


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--apply", action="store_true", help="Delete the key after previewing")
    args = parser.parse_args()

    client = QdrantClient(url=DEFAULT_QDRANT_URL)
    if not client.collection_exists(args.collection):
        raise ValueError(f"collection does not exist: {args.collection}")
    scanned, matching, nonempty = scan(client, args.collection)
    print(f"collection={args.collection} scanned={scanned} section_path={len(matching)} nonempty={nonempty}")
    if not args.apply:
        return 0
    if nonempty:
        raise ValueError("nonempty section_path values need review before deletion")
    for offset in range(0, len(matching), 256):
        client.delete_payload(collection_name=args.collection,
                              keys=["meta.section_path"],
                              points=matching[offset:offset + 256], wait=True)
    rescanned, remaining, _ = scan(client, args.collection)
    print(f"verified_scanned={rescanned} remaining_section_path={len(remaining)}")
    if remaining:
        raise RuntimeError("section_path still exists on some points")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
