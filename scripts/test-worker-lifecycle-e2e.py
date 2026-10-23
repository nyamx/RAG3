"""Compatibility entry point for the current staged worker lifecycle test."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def main() -> None:
    test_path = Path(__file__).with_name("test-staged-ingestion-e2e.py")
    spec = importlib.util.spec_from_file_location("rag3_staged_ingestion_test", test_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {test_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()


if __name__ == "__main__":
    main()
