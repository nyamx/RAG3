from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from haystack.dataclasses import PipelineSnapshot

ROOT = Path(__file__).resolve().parents[1]
PIPELINE_SCHEMA_VERSION = "haystack-native-v1-bm25"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class CheckpointIdentity:
    source_sha256: str
    pipeline_schema_version: str
    split_length: int
    split_overlap: int
    embedding_model: str
    target_collection: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class IncompatibleCheckpointError(RuntimeError):
    pass


class CheckpointStore:
    def __init__(self, source_sha256: str) -> None:
        self.source_sha256 = source_sha256
        self.root = ROOT / "data" / "checkpoints" / source_sha256
        self.snapshot_path = self.root / "snapshot.json"
        self.manifest_path = self.root / "manifest.json"

    def _atomic_json_write(self, path: Path, payload: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8", newline="\n") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(path)

    def save_snapshot(
        self,
        snapshot: PipelineSnapshot,
        *,
        identity: CheckpointIdentity,
        state: str,
    ) -> str:
        snapshot_payload = snapshot.to_dict()
        self._atomic_json_write(self.snapshot_path, snapshot_payload)
        self._atomic_json_write(
            self.manifest_path,
            {
                "state": state,
                "identity": identity.to_dict(),
                "snapshot_file": self.snapshot_path.name,
                "updated_at": _utc_now(),
            },
        )
        return str(self.snapshot_path)

    def callback(
        self,
        *,
        identity: CheckpointIdentity,
        state: str = "PAUSED",
    ) -> Callable[[PipelineSnapshot], str]:
        def _save(snapshot: PipelineSnapshot) -> str:
            return self.save_snapshot(snapshot, identity=identity, state=state)

        return _save

    def load_manifest(self) -> dict[str, Any] | None:
        if not self.manifest_path.is_file():
            return None
        with self.manifest_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def has_checkpoint(self) -> bool:
        return self.snapshot_path.is_file() and self.manifest_path.is_file()

    def load_snapshot(self, expected_identity: CheckpointIdentity) -> PipelineSnapshot:
        manifest = self.load_manifest()
        if manifest is None or not self.snapshot_path.is_file():
            raise FileNotFoundError(f"No checkpoint for {self.source_sha256}")

        actual = manifest.get("identity") or {}
        expected = expected_identity.to_dict()
        if actual != expected:
            mismatches = {
                key: {"expected": expected.get(key), "actual": actual.get(key)}
                for key in expected
                if actual.get(key) != expected.get(key)
            }
            raise IncompatibleCheckpointError(
                "Checkpoint is incompatible with the current pipeline: "
                + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
            )

        with self.snapshot_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return PipelineSnapshot.from_dict(payload)

    def set_state(self, state: str, *, identity: CheckpointIdentity) -> None:
        manifest = self.load_manifest() or {}
        manifest.update(
            {
                "state": state,
                "identity": identity.to_dict(),
                "snapshot_file": self.snapshot_path.name if self.snapshot_path.exists() else None,
                "updated_at": _utc_now(),
            }
        )
        self._atomic_json_write(self.manifest_path, manifest)

    def discard_processing_state(self) -> None:
        self.snapshot_path.unlink(missing_ok=True)
        self.manifest_path.unlink(missing_ok=True)
        try:
            self.root.rmdir()
        except OSError:
            pass
