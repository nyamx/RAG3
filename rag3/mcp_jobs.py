"""MCP job orchestration around the existing checkpoint-aware ingestion worker."""

from __future__ import annotations

import hashlib
import codecs
import json
import os
import re
import subprocess
import sys
import threading
import uuid
import zipfile
import time
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

from rag3.collections import list_collections
from rag3.constants import DEFAULT_QDRANT_URL, ROOT
from rag3.attachments import (AttachmentCancelled, download_attachment,
                              validate_filename, validate_url)

JOB_ROOT = ROOT / "data" / "mcp_jobs"
MAX_TEXT_CHARS = 1_000_000
ALLOWED_SUFFIXES = {".pdf", ".doc", ".docx", ".md"}
ACTIVE = {"QUEUED", "DOWNLOADING", "RUNNING", "RESUMING", "PAUSING", "CANCELLING"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_content(suffix: str, content: bytes) -> None:
    if not content:
        raise ValueError("file must not be empty")
    if suffix == ".pdf" and not content.startswith(b"%PDF-"):
        raise ValueError("PDF header is invalid")
    if suffix == ".doc" and not content.startswith(bytes.fromhex("d0cf11e0a1b11ae1")):
        raise ValueError("DOC header is invalid")
    if suffix == ".docx":
        try:
            with zipfile.ZipFile(BytesIO(content)) as archive:
                if "word/document.xml" not in archive.namelist():
                    raise ValueError("DOCX document part is missing")
        except zipfile.BadZipFile as exc:
            raise ValueError("DOCX archive is invalid") from exc
    if suffix == ".md":
        content.decode("utf-8")


def _validate_source(source: Path) -> None:
    with source.open("rb") as stream:
        header = stream.read(8)
        if not header:
            raise ValueError("file must not be empty")
        suffix = source.suffix.lower()
        if suffix == ".pdf" and not header.startswith(b"%PDF-"):
            raise ValueError("PDF header is invalid")
        if suffix == ".doc" and header != bytes.fromhex("d0cf11e0a1b11ae1"):
            raise ValueError("DOC header is invalid")
        if suffix == ".md":
            stream.seek(0)
            decoder = codecs.getincrementaldecoder("utf-8")()
            while chunk := stream.read(1024 * 1024):
                decoder.decode(chunk)
            decoder.decode(b"", final=True)
    if source.suffix.lower() == ".docx":
        try:
            with zipfile.ZipFile(source) as archive:
                if "word/document.xml" not in archive.namelist():
                    raise ValueError("DOCX document part is missing")
        except zipfile.BadZipFile as exc:
            raise ValueError("DOCX archive is invalid") from exc


class _DownloadControl:
    def __init__(self) -> None:
        self.running = threading.Event()
        self.running.set()
        self.cancelled = threading.Event()

    def checkpoint(self) -> None:
        while not self.running.wait(0.2):
            if self.cancelled.is_set():
                raise AttachmentCancelled
        if self.cancelled.is_set():
            raise AttachmentCancelled


class JobManager:
    def __init__(self, *, qdrant_url: str = DEFAULT_QDRANT_URL) -> None:
        self.qdrant_url = qdrant_url
        self._lock = threading.RLock()
        self._processes: dict[str, subprocess.Popen] = {}
        self._downloads: dict[str, _DownloadControl] = {}

    @staticmethod
    def _path(job_id: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{32}", job_id):
            raise ValueError("job_id is invalid")
        return JOB_ROOT / job_id / "job.json"

    def _read(self, job_id: str) -> dict:
        path = self._path(job_id)
        if not path.is_file():
            raise ValueError("job does not exist")
        return json.loads(path.read_text(encoding="utf-8"))

    def _save(self, job: dict) -> None:
        path = self._path(job["job_id"])
        path.parent.mkdir(parents=True, exist_ok=True)
        job["updated_at"] = _now()
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)

    @staticmethod
    def _public(job: dict) -> dict:
        return {key: job.get(key) for key in (
            "job_id", "collection", "filename", "source_sha256", "status",
            "stage", "error", "result", "created_at", "updated_at",
            "downloaded_bytes", "file_size",
        )}

    def submit_file(self, collection: str, filename: str, content: bytes) -> dict:
        if collection not in list_collections(url=self.qdrant_url):
            raise ValueError(f"collection does not exist: {collection}")
        validate_filename(filename)
        suffix = Path(filename).suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            raise ValueError("supported formats: PDF, DOC, DOCX, MD")
        _validate_content(suffix, content)
        source_sha = hashlib.sha256(content).hexdigest()
        self._check_checkpoint(source_sha, collection)
        with self._lock:
            self._check_duplicate(source_sha)
            job_id = uuid.uuid4().hex
            folder = JOB_ROOT / job_id
            folder.mkdir(parents=True)
            source = folder / filename
            source.write_bytes(content)
            job = {"job_id": job_id, "collection": collection, "filename": filename,
                   "source": str(source), "source_sha256": source_sha,
                   "status": "QUEUED", "stage": "Starting", "error": None,
                   "result": None, "created_at": _now(), "file_size": len(content)}
            self._save(job)
            self._launch(job, "run")
            return self._public(job)

    def _check_checkpoint(self, source_sha: str, collection: str) -> None:
        old_manifest = ROOT / "data" / "checkpoints" / source_sha / "manifest.json"
        if old_manifest.is_file():
            previous = json.loads(old_manifest.read_text(encoding="utf-8"))
            previous_collection = (previous.get("identity") or {}).get("target_collection")
            if (previous.get("state") != "COMPLETE" and previous_collection != collection
                    and previous_collection in list_collections(url=self.qdrant_url)):
                raise ValueError("this source has an unfinished checkpoint for another collection")

    def _check_duplicate(self, source_sha: str, *, excluding: str = "") -> None:
        for job_id in self._processes.keys() | self._downloads.keys():
            if job_id == excluding:
                continue
            other = self._read(job_id)
            if other["source_sha256"] == source_sha and (other["status"] in ACTIVE or job_id in self._downloads):
                raise ValueError("the same source is already being ingested")

    def submit_attachment(self, collection: str, attachment: dict, filename: str = "") -> dict:
        if collection not in list_collections(url=self.qdrant_url):
            raise ValueError(f"collection does not exist: {collection}")
        url = attachment["download_url"]
        validate_url(url)
        if not attachment.get("file_id"):
            raise ValueError("attachment file_id is required")
        name = attachment.get("file_name") or filename
        if name:
            validate_filename(name)
        with self._lock:
            job_id = uuid.uuid4().hex
            folder = JOB_ROOT / job_id
            folder.mkdir(parents=True)
            job = {"job_id": job_id, "collection": collection, "filename": name,
                   "source": None, "source_sha256": None, "status": "DOWNLOADING",
                   "stage": "Receiving original attachment", "error": None,
                   "result": None, "created_at": _now(), "downloaded_bytes": 0}
            self._save(job)
            control = _DownloadControl()
            self._downloads[job_id] = control
            # Keep the temporary signed download URL in memory only.
            threading.Thread(target=self._receive_attachment,
                             args=(job_id, url, name, control), daemon=True).start()
            return self._public(job)

    def _receive_attachment(self, job_id: str, url: str, name: str, control: _DownloadControl) -> None:
        last_update = 0.0

        def progress(count: int) -> None:
            nonlocal last_update
            if time.monotonic() - last_update < 1:
                return
            last_update = time.monotonic()
            with self._lock:
                job = self._read(job_id)
                job["downloaded_bytes"] = count
                self._save(job)

        try:
            source, name, source_sha, count = download_attachment(
                url, JOB_ROOT / job_id, filename=name,
                checkpoint=control.checkpoint, progress=progress)
            _validate_source(source)
            while True:
                # Do not wait for resume while holding the lock needed by resume().
                control.checkpoint()
                with self._lock:
                    if not control.running.is_set():
                        continue
                    if control.cancelled.is_set():
                        raise AttachmentCancelled
                    self._check_checkpoint(source_sha, self._read(job_id)["collection"])
                    self._check_duplicate(source_sha, excluding=job_id)
                    job = self._read(job_id)
                    job.update(source=str(source), filename=name, source_sha256=source_sha,
                               downloaded_bytes=count, file_size=count)
                    # Switch ownership atomically so pause/cancel targets the worker,
                    # rather than a download that has already finished.
                    self._downloads.pop(job_id, None)
                    self._launch(job, "run")
                    break
        except AttachmentCancelled:
            with self._lock:
                job = self._read(job_id)
                job.update(status="CANCELLED", stage="Attachment download cancelled")
                self._save(job)
        except Exception as exc:
            with self._lock:
                job = self._read(job_id)
                # Never persist exception strings from HTTP clients or signed URLs.
                message = (str(exc) if isinstance(exc, ValueError) else
                           "Could not receive the original attachment; submit a fresh file reference.")
                job.update(status="FAILED", stage="Attachment download failed", error=message)
                self._save(job)
        finally:
            with self._lock:
                self._downloads.pop(job_id, None)

    def submit_text(self, collection: str, title: str, content: str) -> dict:
        if not isinstance(content, str) or not content.strip() or len(content) > MAX_TEXT_CHARS:
            raise ValueError(f"text must be nonempty and at most {MAX_TEXT_CHARS} characters")
        if not isinstance(title, str) or not title.strip() or len(title) > 200:
            raise ValueError("title must be 1..200 characters")
        safe_title = re.sub(r"[^\w. -]", "_", title.strip()).strip(" .") or "generated"
        return self.submit_file(collection, safe_title + ".md", content.encode("utf-8"))

    def _launch(self, job: dict, mode: str) -> None:
        args = [sys.executable, str(ROOT / "scripts" / "ingest-worker.py"),
                job["source"], "--collection", job["collection"], "--mode", mode]
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        process = subprocess.Popen(args, cwd=ROOT, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True,
                                   encoding="utf-8", errors="replace", bufsize=1,
                                   creationflags=flags)
        self._processes[job["job_id"]] = process
        job["status"] = "RESUMING" if mode == "resume" else "RUNNING"
        job["stage"] = "Starting"
        job["error"] = None
        self._save(job)
        threading.Thread(target=self._monitor, args=(job["job_id"], process), daemon=True).start()

    def _monitor(self, job_id: str, process: subprocess.Popen) -> None:
        assert process.stdout is not None
        for line in process.stdout:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            with self._lock:
                job = self._read(job_id)
                if event.get("type") == "log":
                    job["stage"] = str(event.get("message", ""))[:300]
                    if job.pop("_pause_pending", False):
                        self._pause_marker(job).write_text("pause\n", encoding="utf-8")
                elif event.get("type") == "result":
                    job["result"] = event.get("result")
                elif event.get("type") == "error":
                    job["error"] = str(event.get("message") or "Worker error")[:1000]
                self._save(job)
        exit_code = process.wait()
        with self._lock:
            job = self._read(job_id)
            if job["status"] == "CANCELLING":
                job["status"] = "CANCELLED"
            elif exit_code == 75:
                job["status"] = "PAUSED"
            elif exit_code == 0:
                job["status"] = "COMPLETE"
            else:
                job["status"] = "FAILED"
                job["error"] = job["error"] or f"Worker exited with code {exit_code}"
            self._save(job)
            self._processes.pop(job_id, None)

    @staticmethod
    def _pause_marker(job: dict) -> Path:
        marker = ROOT / "data" / "checkpoints" / job["source_sha256"] / "pause.request"
        marker.parent.mkdir(parents=True, exist_ok=True)
        return marker

    def status(self, job_id: str) -> dict:
        with self._lock:
            job = self._read(job_id)
            interrupted_download = job["status"] == "PAUSED" and not job.get("source_sha256")
            if (job["status"] in ACTIVE or interrupted_download) and job_id not in self._processes and job_id not in self._downloads:
                checkpoint = (ROOT / "data" / "checkpoints" / job["source_sha256"] / "manifest.json"
                              if job.get("source_sha256") else None)
                if checkpoint and checkpoint.is_file() and json.loads(checkpoint.read_text(encoding="utf-8")).get("state") == "COMPLETE":
                    job["status"] = "COMPLETE"
                else:
                    job["status"] = "FAILED"
                    job["error"] = ("MCP server restarted during ingestion; resume from checkpoint if available"
                                    if job.get("source_sha256") else
                                    "MCP server restarted before receiving the complete attachment; submit a fresh file reference")
                self._save(job)
            return self._public(job)

    def has_active_collection(self, collection: str) -> bool:
        with self._lock:
            return any(self._read(job_id)["collection"] == collection
                       and (self._read(job_id)["status"] in ACTIVE or job_id in self._downloads)
                       for job_id in self._processes.keys() | self._downloads.keys())

    def pause(self, job_id: str) -> dict:
        with self._lock:
            job = self._read(job_id)
            if job_id in self._downloads:
                if job["status"] != "DOWNLOADING":
                    raise ValueError("attachment download is not running")
                self._downloads[job_id].running.clear()
                job.update(status="PAUSED", stage="Attachment download paused")
                self._save(job)
                return self._public(job)
            process = self._processes.get(job_id)
            if job["status"] not in {"RUNNING", "RESUMING"} or not process or process.poll() is not None:
                raise ValueError("job is not running")
            job["status"] = "PAUSING"
            if job["stage"] == "Starting":
                job["_pause_pending"] = True
            else:
                self._pause_marker(job).write_text("pause\n", encoding="utf-8")
            self._save(job)
            return self._public(job)

    def resume(self, job_id: str) -> dict:
        with self._lock:
            job = self._read(job_id)
            if job_id in self._downloads:
                if job["status"] != "PAUSED":
                    raise ValueError("attachment download is not paused")
                job.update(status="DOWNLOADING", stage="Receiving original attachment")
                self._save(job)
                self._downloads[job_id].running.set()
                return self._public(job)
            if job["status"] not in {"PAUSED", "CANCELLED", "FAILED"}:
                raise ValueError("job is not paused, cancelled, or failed")
            if not job.get("source_sha256"):
                raise ValueError("attachment download did not finish; submit a fresh file reference")
            checkpoint = ROOT / "data" / "checkpoints" / job["source_sha256"] / "snapshot.json"
            if not checkpoint.is_file():
                raise ValueError("no checkpoint is available for this job")
            self._launch(job, "resume")
            return self._public(job)

    def cancel(self, job_id: str) -> dict:
        with self._lock:
            job = self._read(job_id)
            if job_id in self._downloads:
                self._downloads[job_id].cancelled.set()
                self._downloads[job_id].running.set()
                job.update(status="CANCELLING", stage="Cancelling attachment download")
                self._save(job)
                return self._public(job)
            if job["status"] == "PAUSED":
                job["status"] = "CANCELLED"
                self._save(job)
                return self._public(job)
            process = self._processes.get(job_id)
            if job["status"] not in ACTIVE or not process or process.poll() is not None:
                raise ValueError("job is not active")
            job["status"] = "CANCELLING"
            self._save(job)
            process.kill()
            return self._public(job)
