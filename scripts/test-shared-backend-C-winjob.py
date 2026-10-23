"""Isolated Windows Job Object smoke test; starts no RAG3 services."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag3.windows_job import WindowsParserJob


@unittest.skipUnless(os.name == "nt", "Windows Job Object test")
class WindowsParserJobTests(unittest.TestCase):
    def test_close_terminates_assigned_child(self) -> None:
        job = WindowsParserJob()
        child = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        try:
            job.assign(child.pid)
            self.assertIsNone(child.poll())
            job.close()
            # Windows job close may use exit code zero; timely exit is the proof.
            child.wait(timeout=5)
        finally:
            job.close()
            if child.poll() is None:
                child.kill()
            child.wait(timeout=5)

    def test_close_terminates_gated_child_and_grandchild(self) -> None:
        import ctypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = (ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32)
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.WaitForSingleObject.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
        kernel.WaitForSingleObject.restype = ctypes.c_uint32
        kernel.CloseHandle.argtypes = (ctypes.c_void_p,)
        kernel.CloseHandle.restype = ctypes.c_int
        child_code = (
            "import pathlib,subprocess,sys,time\n"
            "if sys.stdin.buffer.read(1) != b'G': sys.exit(0)\n"
            "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],"
            "stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,"
            "stderr=subprocess.DEVNULL)\n"
            "pathlib.Path(sys.argv[1]).write_text(str(p.pid))\n"
            "time.sleep(60)\n"
        )
        with tempfile.TemporaryDirectory() as td:
            pid_file = Path(td) / "grandchild.pid"
            job = WindowsParserJob()
            child = subprocess.Popen(
                [sys.executable, "-c", child_code, str(pid_file)],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            grandchild_handle = None
            try:
                job.assign(child.pid)
                assert child.stdin is not None
                child.stdin.write(b"G")
                child.stdin.flush()
                deadline = time.monotonic() + 5
                while not pid_file.exists() and time.monotonic() < deadline:
                    self.assertIsNone(child.poll())
                    time.sleep(.01)
                self.assertTrue(pid_file.exists(), "gated child did not spawn")
                grandchild_pid = int(pid_file.read_text())
                grandchild_handle = kernel.OpenProcess(0x00100000, False, grandchild_pid)
                self.assertTrue(grandchild_handle, "grandchild exited before job close")
                self.assertIsNone(child.poll())
                job.close()
                child.wait(timeout=5)
                self.assertEqual(kernel.WaitForSingleObject(grandchild_handle, 5000), 0)
            finally:
                job.close()
                if child.poll() is None:
                    child.kill()
                child.wait(timeout=5)
                if grandchild_handle:
                    kernel.CloseHandle(grandchild_handle)
                if child.stdin is not None:
                    child.stdin.close()

    def test_failed_assignment_keeps_gate_closed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            marker = Path(td) / "should-not-exist"
            child_code = (
                "import pathlib,sys\n"
                "if sys.stdin.buffer.read(1) == b'G': "
                "pathlib.Path(sys.argv[1]).write_text('ran')\n"
            )
            job = WindowsParserJob()
            job.close()
            child = subprocess.Popen(
                [sys.executable, "-c", child_code, str(marker)],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            try:
                with self.assertRaises(OSError):
                    job.assign(child.pid)
                assert child.stdin is not None
                child.stdin.close()  # EOF, never release the parser gate.
                self.assertEqual(child.wait(timeout=5), 0)
                self.assertFalse(marker.exists())
            finally:
                if child.poll() is None:
                    child.kill()
                child.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
