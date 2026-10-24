"""Cold real-HTTP Stage B health smoke with isolated fake Qdrant.

Starts the actual MCP backend on a random loopback port. It never calls live
Qdrant, searches, ingestion, or SQLite. The test-scoped token and lock are
removed after the launched process exits, including on assertion failure.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

import psutil


ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "scripts" / "rag3-mcp-server.py"


def port_owner(port: int, pid: int) -> bool:
    try:
        return any(conn.pid == pid and conn.laddr and conn.laddr.port == port
                   and conn.status == "LISTEN"
                   for conn in psutil.net_connections(kind="tcp"))
    except (psutil.Error, OSError):
        return False


def safe_server_process(pid: int) -> psutil.Process | None:
    try:
        process = psutil.Process(pid)
        args = process.cmdline()
        if len(args) >= 2 and Path(args[-1]).resolve() == SERVER.resolve():
            return process
    except (psutil.Error, OSError, ValueError):
        pass
    return None


def free_loopback_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def main() -> None:
    port = free_loopback_port()
    qdrant_port = free_loopback_port()
    while port == 8000 or qdrant_port in {6333, port}:
        port = free_loopback_port()
        qdrant_port = free_loopback_port()

    # Health only opens a TCP connection; no Qdrant API is used.
    fake_qdrant = socket.socket()
    fake_qdrant.bind(("127.0.0.1", qdrant_port))
    fake_qdrant.listen(8)
    scope = f"bhealth{port}"
    token_path = ROOT / "data" / f"rag3-backend-ui-{port}.token"
    lock_path = ROOT / "data" / f"backend-test-{scope}.lock"
    assert not token_path.exists() and not lock_path.exists(), "Test scope collision"
    test_data = tempfile.TemporaryDirectory(prefix="rag3-b-health-data-")
    test_data_root = Path(test_data.name).resolve()
    assert test_data_root.is_relative_to(Path(tempfile.gettempdir()).resolve())

    env = os.environ.copy()
    env.update({
        "RAG3_MCP_TRANSPORT": "streamable-http",
        "RAG3_MCP_PORT": str(port),
        "RAG3_QDRANT_URL": f"http://127.0.0.1:{qdrant_port}",
        "RAG3_BACKEND_TEST_INSTANCE": scope,
        "RAG3_TEST_DATA_ROOT": str(test_data_root),
        "RAG3_MCP_WRITE_ENABLED": "0",
        "RAG3_MCP_DELETE_ENABLED": "0",
    })
    proc = None
    launcher_created_at = None
    actual_pid = None
    print(f"TEST_SCOPE port={port} fake_qdrant_port={qdrant_port}", flush=True)
    try:
        # A Windows child can hold the log open until finally terminates it.
        with tempfile.TemporaryDirectory(
            prefix="rag3-b-http-health-", ignore_cleanup_errors=True,
        ) as tmp:
            log_path = Path(tmp) / "server.log"
            with log_path.open("wb") as output:
                proc = subprocess.Popen(
                    [sys.executable, str(SERVER)], cwd=ROOT, env=env,
                    stdout=output, stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)
                    if os.name == "nt" else 0,
                )
                launcher_created_at = psutil.Process(proc.pid).create_time()
                health = None
                start = time.monotonic()
                while time.monotonic() - start < 120:
                    if proc.poll() is not None:
                        raise RuntimeError(f"Backend exited {proc.returncode}")
                    try:
                        with urllib.request.urlopen(
                            f"http://127.0.0.1:{port}/health", timeout=2,
                        ) as response:
                            assert response.status == 200
                            health = json.load(response)
                        break
                    except (OSError, ValueError):
                        time.sleep(.25)
                if health is None:
                    raise RuntimeError("Cold backend /health did not respond within 120s")

                service = health["gpu_model_service"]
                assert health["backend"] == "ready", health
                assert health["qdrant_transport"] == "reachable", health
                assert health["gpu_model_scope"] == \
                    "shared_backend_single_instance", health
                assert service["state"] == health["gpu_model"] == "not_initialized", health
                assert service["model_load_count"] == 0, service
                assert service["query_waiting"] == service["document_waiting"] == 0, service
                assert service["active_kind"] is None, service
                actual_pid = int(health["pid"])
                assert safe_server_process(actual_pid) is not None, actual_pid
                assert port_owner(port, actual_pid), "Health PID does not own HTTP listener"
                print("REAL_HTTP_HEALTH_PASS " + json.dumps({
                    "launcher_pid": proc.pid, "server_pid": actual_pid, "port": port,
                    "cold_start_seconds": round(time.monotonic() - start, 2),
                    "gpu_model_service": service,
                    "rss_mib": round(psutil.Process(actual_pid).memory_info().rss / 1048576, 1),
                }, ensure_ascii=False), flush=True)

    except BaseException:
        if "log_path" in locals() and log_path.exists():
            print("SERVER_LOG_TAIL " +
                  log_path.read_text(encoding="utf8", errors="replace")[-2000:],
                  flush=True)
        raise
    finally:
        processes = []
        if proc is not None and proc.poll() is None:
            try:
                launcher = psutil.Process(proc.pid)
                if launcher.create_time() == launcher_created_at:
                    processes = launcher.children(recursive=True) + [launcher]
            except psutil.Error:
                pass
        if actual_pid is not None and port_owner(port, actual_pid):
            server = safe_server_process(actual_pid)
            if (server is not None and launcher_created_at is not None
                    and server.create_time() >= launcher_created_at
                    and all(p.pid != actual_pid for p in processes)):
                processes.insert(0, server)
        for process in processes:
            try:
                process.terminate()
            except psutil.Error:
                pass
        _, alive = psutil.wait_procs(processes, timeout=8)
        for process in alive:
            try:
                process.kill()
            except psutil.Error:
                pass
        psutil.wait_procs(alive, timeout=5)
        if proc is not None:
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=3)
        fake_qdrant.close()
        stopped = False
        for _ in range(20):
            with socket.socket() as probe:
                probe.settimeout(.5)
                stopped = probe.connect_ex(("127.0.0.1", port)) != 0
            if stopped:
                break
            time.sleep(.1)
        if stopped and proc is not None and proc.poll() is not None:
            token_path.unlink(missing_ok=True)
            lock_path.unlink(missing_ok=True)
        if "log_path" in locals() and log_path.exists():
            # TempDirectory may have left the log while the child was alive.
            log_path.unlink()
            log_path.parent.rmdir()
        test_data.cleanup()
        print("CLEANUP " + json.dumps({
            "server_exited": proc is None or proc.poll() is not None,
            "http_listener_stopped": stopped,
            "test_token_removed": not token_path.exists(),
            "test_lock_removed": not lock_path.exists(),
        }), flush=True)


if __name__ == "__main__":
    main()
