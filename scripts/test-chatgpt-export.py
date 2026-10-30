from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
import zipfile
from contextlib import contextmanager
from pathlib import Path
import sys
from unittest.mock import patch

from qdrant_client import QdrantClient, models

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag3.chatgpt_export import (iter_shards, normalize_conversation,
                                 split_shard_source)
from rag3.ingestion_light import normalize_chunk
from rag3.ingestion_backend import (_known_chatgpt_conversations, _stage_plain,
                                    _verified_completed_chatgpt_source)
from rag3.ingestion_manager import Scheduler


def conversation():
    def node(node_id, parent, role=None, text=None):
        message = None if role is None else {
            "id": node_id, "author": {"role": role},
            "content": {"content_type": "text", "parts": [text]},
            "create_time": 100.5,
        }
        return {"id": node_id, "parent": parent, "message": message}

    return {"id": "opaque-chat", "title": "Example", "create_time": 100.0,
            "is_starred": None, "current_node": "answer", "mapping": {
                "root": node("root", None),
                "question": node("question", "root", "user", "Find this term"),
                "old-answer": node("old-answer", "question", "assistant", "discarded branch"),
                "answer": node("answer", "question", "assistant", "This is the answer"),
            }}


class ChatGPTExportTests(unittest.TestCase):
    def test_current_path_and_message_metadata(self):
        chat = normalize_conversation(conversation())
        self.assertEqual([m["message_id"] for m in chat["messages"]], ["question", "answer"])
        self.assertEqual([m["role"] for m in chat["messages"]], ["user", "assistant"])
        self.assertEqual(chat["created_at_ms"], 100000)
        self.assertNotIn("is_starred", chat)

    def test_invalid_path(self):
        chat = conversation()
        chat["mapping"]["question"]["parent"] = "answer"
        with self.assertRaisesRegex(ValueError, "cycle"):
            normalize_conversation(chat)

    def test_internal_reasoning_is_excluded(self):
        chat = conversation()
        chat["mapping"]["answer"]["message"]["content"] = {
            "content_type": "thoughts", "parts": ["not visible text"]}
        normalized = normalize_conversation(chat)
        self.assertEqual(len(normalized["messages"]), 1)
        self.assertEqual(normalized["skipped_unsupported"], 1)

    def test_zip_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            export = Path(root) / "export.zip"
            with zipfile.ZipFile(export, "w") as archive:
                archive.writestr("../conversations-000.json", "[]")
            with self.assertRaisesRegex(ValueError, "unsafe"):
                list(iter_shards(export))

    def test_zip_and_per_message_chunks(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            export = root / "export.zip"
            second = conversation()
            second["id"] = "second-chat"
            second["conversation_id"] = "second-chat"
            with zipfile.ZipFile(export, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("conversations-000.json", json.dumps([conversation(), second]))
                archive.writestr("ads.json", "{}")
            name, raw, chats = list(iter_shards(export))[0]
            self.assertEqual(name, "conversations-000.json")
            self.assertEqual(len(chats), 2)
            source = root / name
            source.write_bytes(raw)
            docs = split_shard_source({"managed_source": source},
                                      {"source_sha256": "a" * 64,
                                       "document_id": "a" * 64, "title": "conversations-000"})
            self.assertEqual(len(docs), 4)
            self.assertEqual([d.meta["chatgpt_message_id"] for d in docs],
                             ["question", "answer", "question", "answer"])
            self.assertEqual(docs[0].meta["chatgpt_conversation_id"], "opaque-chat")
            self.assertEqual(docs[2].meta["chatgpt_conversation_id"], "second-chat")
            normalized = [normalize_chunk(doc, index) for index, doc in enumerate(docs)]
            self.assertEqual([doc.meta["chunk_index"] for doc in normalized], [0, 1, 2, 3])
            self.assertEqual(len({doc.meta["document_id"] for doc in normalized}), 1)
            self.assertEqual([doc.meta["chatgpt_role"] for doc in normalized],
                             ["user", "assistant", "user", "assistant"])

    def test_existing_conversation_is_skipped_even_after_new_messages(self):
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / "conversations-000.json"
            old = conversation()
            old["mapping"]["followup"] = {
                "parent": "answer", "message": {"id": "followup",
                "author": {"role": "user"},
                "content": {"content_type": "text", "parts": ["A new message"]}}}
            old["current_node"] = "followup"
            new = conversation()
            new["id"] = "new-chat"
            source.write_text(json.dumps([old, new]), encoding="utf-8")
            calls = []
            def lookup(ids):
                calls.append(ids)
                return {"opaque-chat"}
            chunks = split_shard_source({"managed_source": source},
                                        {"source_sha256": "a" * 64},
                                        lookup_existing=lookup)
            self.assertEqual(calls, [{"opaque-chat", "new-chat"}])
            self.assertEqual(len(chunks), 2)
            self.assertEqual({c.meta["chatgpt_conversation_id"] for c in chunks}, {"new-chat"})

    def test_all_existing_conversations_need_no_chunks(self):
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / "conversations.json"
            source.write_text(json.dumps([conversation()]), encoding="utf-8")
            self.assertEqual(split_shard_source({"managed_source": source}, {},
                             lookup_existing=lambda ids: ids), [])
            self.assertEqual([name for name, _, _ in iter_shards(source)], ["conversations.json"])

    def test_duplicate_conversation_id_in_one_shard_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / "conversations-000.json"
            source.write_text(json.dumps([conversation(), conversation()]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate conversation ID"):
                split_shard_source({"managed_source": source}, {}, lookup_existing=lambda ids: set())

    def test_qdrant_lookup_ignores_unfinished_and_unready_sources(self):
        with tempfile.TemporaryDirectory() as root:
            database = Path(root) / "jobs.sqlite"
            db = sqlite3.connect(database)
            db.execute("CREATE TABLE jobs (collection TEXT, source_sha256 TEXT, filename TEXT, status TEXT)")
            db.execute("CREATE TABLE ingestion_progress (collection TEXT, source_sha256 TEXT)")
            complete_sha, paused_sha = "a" * 64, "b" * 64
            db.executemany("INSERT INTO jobs VALUES (?,?,?,?)", [
                ("chats", complete_sha, "conversations-000.json", "COMPLETE"),
                ("chats", paused_sha, "conversations-001.json", "PAUSED")])
            db.execute("INSERT INTO ingestion_progress VALUES (?,?)", ("chats", paused_sha))
            db.commit()
            db.close()

            @contextmanager
            def journal():
                connection = sqlite3.connect(database)
                connection.row_factory = sqlite3.Row
                try:
                    yield connection
                finally:
                    connection.close()

            client = QdrantClient(":memory:")
            client.create_collection("chats", vectors_config=models.VectorParams(
                size=1, distance=models.Distance.COSINE))
            client.upsert("chats", points=[
                models.PointStruct(id=1, vector=[1.0], payload={"meta": {
                    "chatgpt_conversation_id": "old", "source_sha256": complete_sha,
                    "_rag3_ingest_state": "ready"}}),
                models.PointStruct(id=2, vector=[1.0], payload={"meta": {
                    "chatgpt_conversation_id": "partial", "source_sha256": paused_sha,
                    "_rag3_ingest_state": "ready"}}),
                models.PointStruct(id=3, vector=[1.0], payload={"meta": {
                    "chatgpt_conversation_id": "unready", "source_sha256": complete_sha,
                    "_rag3_ingest_state": "parsed"}})])
            with patch("rag3.ingestion_backend.connect", journal):
                with self.assertRaisesRegex(ValueError, "unfinished checkpoint"):
                    _known_chatgpt_conversations(client, "chats", "c" * 64,
                                                 {"old", "partial", "unready"})
                db = sqlite3.connect(database)
                try:
                    db.execute("DELETE FROM ingestion_progress")
                    db.commit()
                finally:
                    db.close()
                self.assertEqual(_known_chatgpt_conversations(client, "chats", "c" * 64,
                                 {"old", "partial", "unready"}), {"old"})
                more = {f"old-{i}" for i in range(12)}
                client.upsert("chats", points=[models.PointStruct(
                    id=i + 4, vector=[1.0], payload={"meta": {
                        "chatgpt_conversation_id": f"old-{i}",
                        "source_sha256": complete_sha, "_rag3_ingest_state": "ready"}})
                    for i in range(12)])
                self.assertEqual(_known_chatgpt_conversations(client, "chats", "c" * 64,
                                 more | {"old"}), more | {"old"})

    def test_exact_completed_shard_can_be_submitted_again(self):
        with tempfile.TemporaryDirectory() as root:
            database = Path(root) / "jobs.sqlite"
            db = sqlite3.connect(database)
            db.execute("CREATE TABLE jobs (job_id TEXT, collection TEXT, source_sha256 TEXT, status TEXT)")
            db.execute("INSERT INTO jobs VALUES (?,?,?,?)", ("old", "chats", "a" * 64, "COMPLETE"))
            db.commit()
            db.close()

            @contextmanager
            def journal():
                connection = sqlite3.connect(database)
                connection.row_factory = sqlite3.Row
                try:
                    yield connection
                finally:
                    connection.close()

            with patch("rag3.ingestion_manager.connect", journal):
                scheduler = Scheduler.__new__(Scheduler)
                scheduler.conflict("a" * 64, "chats", allow_completed=True)
                with self.assertRaisesRegex(ValueError, "same SHA"):
                    scheduler.conflict("a" * 64, "chats")

    def test_exact_completed_source_requires_all_vectors(self):
        with tempfile.TemporaryDirectory() as root:
            database = Path(root) / "jobs.sqlite"
            db = sqlite3.connect(database)
            db.execute("CREATE TABLE jobs (collection TEXT, source_sha256 TEXT, filename TEXT, status TEXT, result TEXT, created_at TEXT)")
            db.execute("INSERT INTO jobs VALUES (?,?,?,?,?,?)", (
                "chats", "a" * 64, "conversations-000.json", "COMPLETE",
                json.dumps({"documents_written": 2}), "2026-10-27"))
            db.commit()
            db.close()

            @contextmanager
            def journal():
                connection = sqlite3.connect(database)
                connection.row_factory = sqlite3.Row
                try:
                    yield connection
                finally:
                    connection.close()

            with patch("rag3.ingestion_backend.connect", journal):
                with patch("rag3.ingestion_backend._source_counts", return_value=(2, 2, 2, 2)):
                    self.assertTrue(_verified_completed_chatgpt_source(None, "chats", "a" * 64))
                with patch("rag3.ingestion_backend._source_counts", return_value=(2, 2, 1, 2)):
                    with self.assertRaisesRegex(ValueError, "incomplete Qdrant points"):
                        _verified_completed_chatgpt_source(None, "chats", "a" * 64)

    def test_all_existing_shard_stages_no_points(self):
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / "conversations-000.json"
            source.write_text(json.dumps([conversation()]), encoding="utf-8")
            preserved = {"managed_source": source, "title": "conversations-000",
                         "source_uri": str(source)}
            events = []
            with patch("rag3.ingestion_light.preserve_source", return_value=preserved), \
                 patch("rag3.ingestion_backend._client", return_value=object()), \
                 patch("rag3.ingestion_backend._known_chatgpt_conversations",
                       return_value={"opaque-chat"}):
                count = _stage_plain(source, "chats", "a" * 64,
                                     {"phase": "PARSING"}, lambda: False,
                                     events.append, "unused")
            self.assertEqual(count, 0)
            self.assertTrue(any("SKIPPED" in event for event in events))


if __name__ == "__main__":
    unittest.main()
