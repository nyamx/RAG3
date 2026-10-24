"""Offscreen Stage D UI response ordering and write-timeout regression."""
from __future__ import annotations
import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

class UIResilience(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Must be set by caller before importing RAG3 modules.
        from rag3.constants import DATA_ROOT
        assert DATA_ROOT != ROOT/'data', 'production data root forbidden'
        from PySide6.QtWidgets import QApplication
        cls.app=QApplication.instance() or QApplication([])
        spec=importlib.util.spec_from_file_location('stage_d_ui_resilience',ROOT/'scripts'/'ingest-ui-stage-d.py')
        cls.mod=importlib.util.module_from_spec(spec)
        sys.modules[spec.name]=cls.mod
        spec.loader.exec_module(cls.mod)

    def setUp(self):
        self.window=self.mod.IngestionWindow()
        self.calls=[]

    def tearDown(self):
        self.window.close()

    def _fake(self,responses, *, synchronous=True):
        def command(action,params,done,*,read_only=True):
            self.calls.append((action,dict(params),read_only,done))
            if synchronous:
                response=responses.get(action,({},None))
                done(*response)
            return True
        self.window.backend.command=command

    def test_missing_token_releases_collection_busy_state(self):
        def no_token(action,params,done,*,read_only=True):
            self.calls.append((action,params,read_only,done))
            done(None,'backend token unavailable')
            return False
        self.window.backend.command=no_token
        self.assertFalse(self.window.refresh_collections())
        self.assertFalse(self.window._collection_pending)
        self.assertEqual(self.window.ui_state,self.mod.UiState.READY)
        self.assertEqual(len(self.calls),1)

    def test_inline_collections_callback_has_correct_version(self):
        self.window._latest_versions['collections']=7
        self._fake({'collections.list':({'collections':['tmp_col']},None)})
        self.assertTrue(self.window.refresh_collections())
        self.assertEqual(self.window._collections_seen_version,7)
        self.assertEqual(self.window.current_collection(),'tmp_col')
        self.assertFalse(self.window._collection_pending)

    def test_inline_contents_callback_has_correct_version(self):
        self.window.collection.setText('tmp_col')
        self.window._latest_versions['contents:tmp_col']=12
        self._fake({'collections.contents':({'collection':'tmp_col','documents':[],
                          'document_count':0,'chunk_count':0},None)})
        self.assertTrue(self.window.refresh_contents())
        self.assertEqual(self.window._contents_seen.get('tmp_col'),12)
        self.assertFalse(self.window._collection_pending)

    def test_late_response_from_prior_collection_is_discarded(self):
        self.window.collection.setText('older')
        self._fake({},synchronous=False)
        self.assertTrue(self.window.refresh_contents())
        cb=self.calls[-1][3]
        self.window.collection.setText('newer')
        cb({'collection':'older','documents':[{'title':'OLD'}]},None)
        self.assertNotEqual(self.window._displayed_contents_collection,'older')
        self.assertEqual(self.window.contents_table.rowCount(),0)

    def test_write_timeout_does_not_replay_job_submission(self):
        from rag3.constants import DATA_ROOT
        p=DATA_ROOT/'sample.txt'
        p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text('sample',encoding='utf-8')
        self.window.collection.setText('tmp_col')
        self._fake({'jobs.submit':(None,'TIMEOUT: completion unknown'),
                    'jobs.list':({'jobs':[],'total':0},None)})
        self.window.add_path(str(p))
        self.assertEqual(len([a for a,*_ in self.calls if a=='jobs.submit']),1)
        self.assertFalse(self.window._job_write_pending)
        self.assertIn('unknown',self.window.errors.toPlainText())

    def test_ui_jobs_list_excludes_hidden_and_writes_are_not_replayed(self):
        self._fake({'jobs.list':({'jobs':[],'total':0},None),
                    'jobs.action':(None,'TIMEOUT: completion unknown')})
        self.window.load_jobs()
        args=[x for x in self.calls if x[0]=='jobs.list']
        self.assertEqual(args[0][1]['include_hidden'],False)
        self.window._manager_action('cancel','job-test-1')
        self.assertEqual(len([x for x in self.calls if x[0]=='jobs.action']),1)
        self.assertEqual([x[2] for x in self.calls if x[0]=='jobs.action'],[False])

if __name__=='__main__':
    unittest.main(verbosity=2)
