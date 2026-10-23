"""Isolated search deadline and GPU error checks; no live Qdrant or model."""
from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from starlette.requests import Request

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag3.backend_boundary import BackendCommandFailure, SearchCallBudget, install_routes
from rag3.gpu_model_service import GPUModelService, GPUServiceConfig, GPUServiceError

spec = importlib.util.spec_from_file_location(
    "rag3_mcp_search_budget_test", ROOT / "scripts" / "rag3-mcp-server.py")
assert spec is not None and spec.loader is not None
server = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = server
spec.loader.exec_module(server)


class SearchBudgetTests(unittest.TestCase):
    def test_gpu_deadline_is_shorter_than_tool_timeout(self):
        class RequestHandle:
            def result(self):
                return [0.1, 0.2]
            def cancel(self):
                return True

        class FakeGPU:
            def __init__(self):
                self.deadline = None
            def submit_query(self, text, *, deadline_monotonic):
                self.deadline = deadline_monotonic
                return RequestHandle()

        fake = FakeGPU()
        start = time.monotonic()
        with patch.object(server, "GPU_BGE_SERVICE", fake):
            result = asyncio.run(server._timed_call(server._encode_query_gpu, "query"))
        self.assertEqual(result, [0.1, 0.2])
        self.assertGreater(fake.deadline, start)
        self.assertLess(fake.deadline, start + server.TOOL_TIMEOUT_SECONDS)

    def test_cancelled_tool_removes_waiting_gpu_query(self):
        # Importing retrieval is intentionally expensive; finish it before
        # blocking the fake GPU owner on the document request.
        import rag3.retrieval
        active = threading.Event()
        release = threading.Event()

        def encode(_model, batch, *, query, **_):
            if not query:
                active.set()
                if not release.wait(30):
                    raise RuntimeError("test document batch did not release")
            return [[0.1, 0.2] for _ in batch]

        gpu = GPUModelService(
            GPUServiceConfig(default_deadline_seconds=10),
            model_factory=object, encode_fn=encode)
        document = gpu.submit_documents(["document"])
        self.assertTrue(active.wait(2))

        def fake_search(_query, *, dense_encoder, **_):
            dense_encoder("query")
            return {"hits": []}

        async def run():
            task = asyncio.create_task(server._timed_call(
                server._locked_search, "query", None, "scratch"))
            for _ in range(200):
                if gpu.snapshot()["query_waiting"] == 1:
                    break
                await asyncio.sleep(.005)
            self.assertEqual(gpu.snapshot()["query_waiting"], 1)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertEqual(gpu.snapshot()["query_waiting"], 0)

        try:
            with patch.object(server, "GPU_BGE_SERVICE", gpu), patch(
                    "rag3.retrieval.search", side_effect=fake_search):
                asyncio.run(run())
        finally:
            release.set()
            document.result(timeout=2)
            gpu.close(timeout=2)
        self.assertEqual(gpu.metrics()["query_batches"], 0)

    def test_gpu_failure_code_and_safe_message(self):
        class FailedGPU:
            def submit_query(self, _text, *, deadline_monotonic):
                raise GPUServiceError("QUEUE_FULL", "private queue details")

        def fake_search(_query, *, dense_encoder, **_):
            dense_encoder("query")

        with patch.object(server, "GPU_BGE_SERVICE", FailedGPU()), patch(
                "rag3.retrieval.search", side_effect=fake_search):
            token = server._CALL_COLLECTION.set("scratch")
            try:
                with self.assertRaises(server.ToolFailure) as caught:
                    server._search_impl("query")
            finally:
                server._CALL_COLLECTION.reset(token)
        self.assertEqual(caught.exception.code, "QUEUE_FULL")
        self.assertTrue(caught.exception.retryable)
        self.assertNotIn("private", caught.exception.message)

    def test_ui_search_passes_http_budget_into_gpu_query(self):
        class RequestHandle:
            def result(self):
                return [0.1, 0.2]
            def cancel(self):
                return True

        class FakeGPU:
            def __init__(self):
                self.deadline = None
            def submit_query(self, _text, *, deadline_monotonic):
                self.deadline = deadline_monotonic
                return RequestHandle()

        def fake_search(_query, *, dense_encoder, **_):
            dense_encoder("query")
            return {"hits": []}

        gpu = FakeGPU()
        budget = SearchCallBudget(90)
        with patch.object(server, "GPU_BGE_SERVICE", gpu), patch(
                "rag3.retrieval.search", side_effect=fake_search):
            self.assertEqual(server._ui_command(
                "search", {"query": "query", "collection": "scratch"}, budget),
                {"results": []})
        self.assertEqual(gpu.deadline, budget.deadline_monotonic)

    def test_http_search_preserves_structured_gpu_error(self):
        class FakeMCP:
            def __init__(self):
                self.routes = {}
            def custom_route(self, path, **_):
                def register(fn):
                    self.routes[path] = fn
                    return fn
                return register

        fake_mcp = FakeMCP()
        observed = {}

        def ui_commands(action, params, budget):
            observed.update(action=action, params=params, budget=budget)
            raise BackendCommandFailure("QUEUE_FULL", "GPU search queue is full.", True)

        install_routes(fake_mcp, root=ROOT, qdrant_url="http://127.0.0.1:1",
                       ui_commands=ui_commands, ui_token="test-token")
        body = json.dumps({"action": "search", "params": {"query": "q"}}).encode()
        scope = {"type": "http", "method": "POST", "path": "/api/v1/command",
                 "headers": [(b"x-rag3-token", b"test-token"),
                             (b"content-length", str(len(body)).encode())]}

        async def receive():
            return {"type": "http.request", "body": body, "more_body": False}

        response = asyncio.run(fake_mcp.routes["/api/v1/command"](Request(scope, receive)))
        self.assertEqual(response.status_code, 503)
        self.assertEqual(json.loads(response.body)["error"]["code"], "QUEUE_FULL")
        self.assertIsNotNone(observed["budget"])
        self.assertLess(observed["budget"].deadline_monotonic, time.monotonic() + 90)


if __name__ == "__main__":
    unittest.main(verbosity=2)
