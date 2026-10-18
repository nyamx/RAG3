"""Verify original attachment transport, large files, and download job controls."""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import runpy
import sys
import tempfile
import threading
import time
import unittest
import zipfile
from email.message import Message
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import rag3.attachments as attachments
import rag3.mcp_jobs as jobs


class Response:
    status = 200

    def __init__(self, source: Path, *, expected: int | None = None):
        self.stream = source.open("rb")
        self.headers = Message()
        self.headers["Content-Length"] = str(source.stat().st_size if expected is None else expected)
        self.headers["Content-Type"] = "application/pdf"

    def getheader(self, name, default=None):
        return self.headers.get(name, default)

    def read(self, count):
        assert 0 < count <= 1024 * 1024, "attachments must be streamed"
        return self.stream.read(count)


def connection_factory(source: Path, *, expected=None, release=None, entered=None, failure=None):
    class Connection:
        def __init__(self, *args, **kwargs):
            self.response = None

        def request(self, *args, **kwargs):
            if failure:
                raise OSError(failure)

        def getresponse(self):
            if entered:
                entered.set()
            if release:
                assert release.wait(10), "test did not release the download"
            self.response = Response(source, expected=expected)
            return self.response

        def close(self):
            if self.response:
                self.response.stream.close()
    return Connection


class AttachmentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="rag3-attachments-")
        self.root = Path(self.temp.name)
        self.job_root = self.root / "jobs"
        self.patches = [patch.object(jobs, "ROOT", self.root),
                        patch.object(jobs, "JOB_ROOT", self.job_root),
                        patch.object(jobs, "list_collections", return_value=["test"])]
        for item in self.patches:
            item.start()
        self.manager = jobs.JobManager()
        self.launched = []

        def launch(job, mode):
            source = Path(job["source"])
            with source.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            self.launched.append((source.name, source.stat().st_size, digest))
            job["status"] = "COMPLETE"
            self.manager._save(job)
        self.manager._launch = launch

    def tearDown(self):
        for control in list(self.manager._downloads.values()):
            control.cancelled.set()
            control.running.set()
        for item in reversed(self.patches):
            item.stop()
        self.temp.cleanup()

    def fixture(self, name="original.pdf", content=b"%PDF-1.7\noriginal bytes\x00\xff"):
        path = self.root / name
        path.write_bytes(content)
        return path

    def submit(self, filename="original.pdf"):
        return self.manager.submit_attachment("test", {
            "download_url": "https://files.example.com/attachment?token=PRIVATE_TEST_TOKEN",
            "file_id": "file_test", "file_name": filename,
        })

    def finish(self, job_id):
        for _ in range(200):
            result = self.manager.status(job_id)
            if result["status"] in {"COMPLETE", "FAILED", "CANCELLED"} and job_id not in self.manager._downloads:
                return result
            time.sleep(0.02)
        self.fail("download job did not finish")

    def test_large_original_is_streamed_without_base64_or_size_cap(self):
        source = self.fixture()
        with source.open("ab") as stream:
            for _ in range(27):
                stream.write(b"x" * 1024 * 1024)
        with source.open("rb") as stream:
            expected = hashlib.file_digest(stream, "sha256").hexdigest()
        with patch.object(attachments, "_PublicHTTPSConnection", connection_factory(source)):
            job = self.submit()
            result = self.finish(job["job_id"])
        self.assertEqual(result["status"], "COMPLETE", result)
        self.assertEqual(self.launched, [(source.name, source.stat().st_size, expected)])
        self.assertEqual(result["source_sha256"], expected)
        self.assertEqual(result["file_size"], source.stat().st_size)
        self.assertNotIn("PRIVATE_TEST_TOKEN", self.manager._path(job["job_id"]).read_text())

    def test_original_formats_and_legacy_local_input_are_preserved(self):
        docx = io.BytesIO()
        with zipfile.ZipFile(docx, "w") as archive:
            archive.writestr("word/document.xml", "<document/>")
        files = {"original.pdf": b"%PDF-1.7\n\x00\xff",
                 "original.doc": bytes.fromhex("d0cf11e0a1b11ae1") + b"original",
                 "original.docx": docx.getvalue(), "original.md": "한글 원문\n".encode()}
        for name, content in files.items():
            with self.subTest(name=name):
                source = self.fixture(name, content)
                with patch.object(attachments, "_PublicHTTPSConnection", connection_factory(source)):
                    job = self.submit(name)
                    self.assertEqual(self.finish(job["job_id"])["status"], "COMPLETE")
                self.assertEqual(self.launched[-1], (name, len(content), hashlib.sha256(content).hexdigest()))
        large = b"%PDF-1.7\n" + b"x" * (26 * 1024 * 1024)
        job = self.manager.submit_file("test", "legacy.pdf", large)
        self.assertEqual(job["status"], "COMPLETE")
        self.assertEqual(self.launched[-1][1], len(large))

    def test_download_returns_job_before_receiving_file_and_can_pause_resume(self):
        source = self.fixture()
        release, entered = threading.Event(), threading.Event()
        with patch.object(attachments, "_PublicHTTPSConnection",
                          connection_factory(source, release=release, entered=entered)):
            job = self.submit()
            self.assertEqual(job["status"], "DOWNLOADING")
            self.assertTrue(entered.wait(2))
            self.assertTrue(self.manager.has_active_collection("test"))
            self.assertEqual(self.manager.pause(job["job_id"])["status"], "PAUSED")
            release.set()
            time.sleep(0.05)
            self.assertFalse(self.launched)
            self.assertTrue(self.manager.has_active_collection("test"))
            self.assertEqual(self.manager.resume(job["job_id"])["status"], "DOWNLOADING")
            self.assertEqual(self.finish(job["job_id"])["status"], "COMPLETE")

    def test_cancel_paused_download_never_launches_ingestion(self):
        source = self.fixture()
        release, entered = threading.Event(), threading.Event()
        with patch.object(attachments, "_PublicHTTPSConnection",
                          connection_factory(source, release=release, entered=entered)):
            job = self.submit()
            self.assertTrue(entered.wait(2))
            self.manager.pause(job["job_id"])
            self.assertEqual(self.manager.cancel(job["job_id"])["status"], "CANCELLING")
            release.set()
            self.assertEqual(self.finish(job["job_id"])["status"], "CANCELLED")
            self.assertFalse(self.launched)
            self.assertFalse((self.job_root / job["job_id"] / "attachment.download.part").exists())

    def test_truncated_download_is_not_ingested(self):
        source = self.fixture()
        with patch.object(attachments, "_PublicHTTPSConnection", connection_factory(source, expected=1000)):
            job = self.submit()
            result = self.finish(job["job_id"])
        self.assertEqual(result["status"], "FAILED")
        self.assertFalse(self.launched)
        self.assertFalse((self.job_root / job["job_id"] / "attachment.download.part").exists())

    def test_signed_url_is_not_in_failure_or_job_record(self):
        source = self.fixture()
        with patch.object(attachments, "_PublicHTTPSConnection",
                          connection_factory(source, failure="PRIVATE_TEST_TOKEN")):
            job = self.submit()
            result = self.finish(job["job_id"])
        self.assertEqual(result["status"], "FAILED")
        self.assertNotIn("PRIVATE_TEST_TOKEN", json.dumps(result))
        self.assertNotIn("PRIVATE_TEST_TOKEN", self.manager._path(job["job_id"]).read_text())

    def test_private_downloads_and_path_traversal_are_rejected(self):
        for url in ("http://files.example.com/file.pdf", "https://127.0.0.1/file.pdf",
                    "file:///private.pdf", "https://user:pass@example.com/file.pdf"):
            with self.subTest(url=url), self.assertRaises(attachments.AttachmentError):
                attachments.validate_url(url)
        for name in ("../test.pdf", "..\\test.pdf", "file.pdf:private", "CON.pdf"):
            with self.subTest(name=name), self.assertRaises(attachments.AttachmentError):
                attachments.validate_filename(name)
        with patch.object(attachments.socket, "getaddrinfo", return_value=[
                (2, 1, 6, "", ("127.0.0.1", 443))]), patch.object(attachments.socket, "create_connection") as connect:
            with self.assertRaises(attachments.AttachmentError):
                attachments._PublicHTTPSConnection("private.example.com").connect()
            connect.assert_not_called()

    def test_restart_before_download_completed_has_actionable_status(self):
        job_id = "a" * 32
        for status in ("DOWNLOADING", "PAUSED"):
            with self.subTest(status=status):
                self.manager._save({"job_id": job_id, "status": status, "source_sha256": None})
                result = self.manager.status(job_id)
                self.assertEqual(result["status"], "FAILED")
                self.assertIn("fresh file reference", result["error"])
                with self.assertRaisesRegex(ValueError, "fresh file reference"):
                    self.manager.resume(job_id)


class ToolSchemaTests(unittest.TestCase):
    def test_chatgpt_file_metadata_has_complete_file_object_schema(self):
        namespace = runpy.run_path(str(ROOT / "scripts/rag3-mcp-server.py"))
        tools = asyncio.run(namespace["mcp"].list_tools())
        self.assertEqual(len(tools), 16)
        tool = next(t for t in tools if t.name == "rag3_ingest_file")
        descriptor = tool.model_dump(by_alias=True, exclude_none=True)
        self.assertEqual(descriptor["_meta"]["openai/fileParams"], ["file"])
        schema = descriptor["inputSchema"]
        self.assertNotIn("content_base64", schema.get("required", []))
        file_schema = schema["properties"]["file"]
        if "$ref" in file_schema:
            file_schema = schema["$defs"][file_schema["$ref"].rsplit("/", 1)[1]]
        self.assertEqual(file_schema["type"], "object")
        self.assertEqual(set(file_schema["properties"]), {"download_url", "file_id", "mime_type", "file_name"})
        self.assertEqual(set(file_schema["required"]), {"download_url", "file_id"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
