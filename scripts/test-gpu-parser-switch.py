"""Isolated checks for the PDF parser's exclusive GPU handoff."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag3.ingestion_backend import BackendScheduler, _use_cuda_parser


class FakeGPU:
    def __init__(self):
        self.events = []
        self.reserved = False

    def reserve_for_parsing(self):
        self.events.append('reserve')
        self.reserved = True

    def release_after_parsing(self):
        self.events.append('release')
        self.reserved = False


class SwitchTest(unittest.TestCase):
    def test_cuda_parser_stops_before_bge_is_released(self):
        gpu = FakeGPU()
        scheduler = BackendScheduler(gpu)
        scheduler._stage = lambda *_: None
        scheduler._stopping = lambda *_: False

        class Parser:
            def __init__(self, _callback, *, device):
                self.assert_device = device
                self.gpu = gpu

            def start(self):
                self.gpu.events.append('start')
                assert self.gpu.reserved and self.assert_device == 'cuda'

            def parse(self, *_args, **_kwargs):
                self.gpu.events.append('parse')
                return {'status': 'PARSED'}

            def stop(self):
                self.gpu.events.append('stop')
                assert self.gpu.reserved

        with patch('rag3.persistent_parser.PersistentCPUParser', Parser):
            self.assertEqual(scheduler._parse_cuda({'job_id': 'test'})['status'], 'PARSED')
        self.assertEqual(gpu.events, ['reserve', 'start', 'parse', 'stop', 'release'])
        self.assertIsNone(scheduler._gpu_parser)

    def test_parser_error_still_releases_gpu(self):
        gpu = FakeGPU()
        scheduler = BackendScheduler(gpu)
        scheduler._stage = lambda *_: None
        scheduler._stopping = lambda *_: False

        class Parser:
            def __init__(self, *_args, **_kwargs): pass
            def start(self): gpu.events.append('start')
            def parse(self, *_args, **_kwargs): raise RuntimeError('parser failed')
            def stop(self): gpu.events.append('stop')

        with patch('rag3.persistent_parser.PersistentCPUParser', Parser):
            with self.assertRaisesRegex(RuntimeError, 'parser failed'):
                scheduler._parse_cuda({'job_id': 'test'})
        self.assertEqual(gpu.events, ['reserve', 'start', 'stop', 'release'])
        self.assertIsNone(scheduler._gpu_parser)

    def test_pdf_page_threshold(self):
        self.assertFalse(_use_cuda_parser('example.docx'))
        self.assertFalse(_use_cuda_parser(Path(__file__).resolve().parents[1] /
                                          'artifacts' / 'e2e-formats' / 'e2e.pdf'))
        with patch('pypdfium2.PdfDocument') as pdf_type:
            pdf_type.return_value.__len__.return_value = 8
            self.assertTrue(_use_cuda_parser('example.pdf'))
            pdf_type.return_value.close.assert_called_once()
        with patch.dict('os.environ', {'RAG3_CUDA_PARSE_MIN_PAGES': '1'}):
            self.assertTrue(_use_cuda_parser('example.pdf'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
