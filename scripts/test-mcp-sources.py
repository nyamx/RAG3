"""Exercise source identity, bounded transport and real PDF page rendering."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import rag3.sources as sources


class SourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="rag3-source-")
        self.root = Path(self.temp.name)
        self.client = SimpleNamespace(collection_exists=lambda name: name == "test",
            close=lambda: None, scroll=self.scroll)
        self.meta = {}
        self.patches = [patch.object(sources, "DATA_ROOT", self.root / "data"),
                        patch.object(sources, "get_client", return_value=self.client)]
        for item in self.patches:
            item.start()

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.temp.cleanup()

    def scroll(self, **kwargs):
        document_id = kwargs["scroll_filter"].must[0].match.value
        if document_id != self.meta.get("document_id"):
            return [], None
        return [SimpleNamespace(payload={"meta": self.meta})], None

    def source(self, content: bytes, filename="original.pdf"):
        digest = hashlib.sha256(content).hexdigest()
        folder = self.root / "data/sources" / digest
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / filename
        path.write_bytes(content)
        self.meta = {"document_id": digest, "source_sha256": digest, "source_uri": str(path)}
        return path, digest

    def test_large_original_round_trip_has_no_file_cap_and_keeps_each_response_small(self):
        content = b"%PDF-1.7\n" + bytes(range(256)) * (27 * 1024 * 1024 // 256)
        _, digest = self.source(content, "original book.pdf")
        metadata = sources.source_metadata("test", digest)
        self.assertEqual(metadata["filename"], "original book.pdf")
        self.assertNotIn(str(self.root), json.dumps(metadata))
        restored = bytearray()
        offset = 0
        while True:
            part = sources.source_part("test", digest, offset)
            wire = json.dumps(part).encode()
            self.assertLess(len(wire), 1024 * 1024)
            data = base64.b64decode(part["data_base64"], validate=True)
            self.assertEqual(hashlib.sha256(data).hexdigest(), part["part_sha256"])
            restored.extend(data)
            if part["next_offset"] is None:
                break
            self.assertGreater(part["next_offset"], offset)
            offset = part["next_offset"]
        self.assertEqual(restored, content)
        self.assertEqual(hashlib.sha256(restored).hexdigest(), digest)

    def test_supported_original_formats_are_unchanged(self):
        for name, content in (("original.doc", b"\xd0\xcfword"), ("original.docx", b"PK\x00word"),
                              ("original.md", "# 원본 파일\n".encode())):
            with self.subTest(filename=name):
                _, digest = self.source(content, name)
                metadata = sources.source_metadata("test", digest)
                self.assertEqual(metadata["file_size"], len(content))
                part = sources.source_part("test", digest, 0)
                self.assertEqual(base64.b64decode(part["data_base64"]), content)

    def test_missing_and_wrong_collection_cannot_read_a_source(self):
        _, digest = self.source(b"%PDF-test")
        for collection, document_id in (("missing", digest), ("test", "b" * 64)):
            with self.subTest(collection=collection), self.assertRaises(sources.SourceError) as error:
                sources.source_metadata(collection, document_id)
            self.assertEqual(error.exception.code, "DOCUMENT_NOT_FOUND")

    def test_tampered_original_and_missing_source_are_rejected(self):
        path, digest = self.source(b"%PDF-original")
        path.write_bytes(b"%PDF-modified")
        with self.assertRaises(sources.SourceError) as error:
            sources.source_metadata("test", digest)
        self.assertEqual(error.exception.code, "SOURCE_CHANGED")
        path.unlink()
        with self.assertRaises(sources.SourceError) as error:
            sources.source_metadata("test", digest)
        self.assertEqual(error.exception.code, "SOURCE_UNAVAILABLE")

    def test_stored_metadata_cannot_escape_managed_source_folder(self):
        path, digest = self.source(b"%PDF-original")
        outside = self.root / "private.pdf"
        outside.write_bytes(path.read_bytes())
        self.meta["source_uri"] = str(outside)
        with self.assertRaises(sources.SourceError) as error:
            sources.source_metadata("test", digest)
        self.assertEqual(error.exception.code, "SOURCE_UNAVAILABLE")
        self.meta["source_uri"] = str(path)
        self.meta["source_sha256"] = "a" * 64
        with self.assertRaises(sources.SourceError):
            sources.source_metadata("test", digest)

    def test_invalid_offsets_and_ids_do_not_read_arbitrary_paths(self):
        _, digest = self.source(b"%PDF-original")
        for offset in (-1, 10000, True):
            with self.subTest(offset=offset), self.assertRaises(sources.SourceError):
                sources.source_part("test", digest, offset)
        for document_id in ("../private.pdf", "C:\\private.pdf", ".."):
            with self.subTest(document_id=document_id), self.assertRaises(sources.SourceError):
                sources.source_metadata("test", document_id)

    def test_real_pdf_page_image_and_physical_page_validation(self):
        make_pdf = runpy.run_path(str(ROOT / "scripts/test-mcp-files-lifecycle-e2e.py"))["make_pdf"]
        content = make_pdf("Original PDF page fixture")
        _, digest = self.source(content)
        metadata, image = sources.render_page("test", digest, 1, max_dimension=1000)
        self.assertEqual(metadata["page_count"], 1)
        self.assertEqual(metadata["source_sha256"], digest)
        with Image.open(io.BytesIO(image)) as rendered:
            self.assertEqual(rendered.format, "JPEG")
            self.assertEqual(max(rendered.size), 1000)
        self.assertLessEqual(len(image), sources.PAGE_IMAGE_BYTES)
        for page in (0, 2, True):
            with self.subTest(page=page), self.assertRaises(sources.SourceError):
                sources.render_page("test", digest, page)

    def test_non_pdf_page_request_is_rejected(self):
        _, digest = self.source(b"# Markdown", "original.md")
        with self.assertRaises(sources.SourceError):
            sources.render_page("test", digest, 1)


class ToolTests(unittest.TestCase):
    def test_source_tools_are_read_only_and_resources_are_registered(self):
        namespace = runpy.run_path(str(ROOT / "scripts/rag3-mcp-server.py"))

        async def check():
            server = namespace["mcp"]
            tools = {tool.name: tool for tool in await server.list_tools()}
            self.assertEqual(tools["rag3_read_source_part"].meta["ui"]["visibility"], ["app"])
            for name in ("rag3_get_source", "rag3_get_page"):
                self.assertTrue(tools[name].annotations.read_only_hint)
                self.assertIn("collection", tools[name].input_schema["required"])
            self.assertEqual(tools["rag3_get_source"].meta["ui"]["resourceUri"],
                             namespace["SOURCE_VIEW_URI"])
            resources = await server.list_resources()
            self.assertIn(namespace["SOURCE_VIEW_URI"], {str(item.uri) for item in resources})
            templates = await server.list_resource_templates()
            self.assertTrue(any("rag3://sources/" in str(item.uri_template) for item in templates))
        asyncio.run(check())


if __name__ == "__main__":
    unittest.main(verbosity=2)
