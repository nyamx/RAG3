"""Stage D isolated search/API/UI contract regression: no production data writes."""
from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class StageDContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix='rag3-stage-d-')
        os.environ['RAG3_TEST_DATA_ROOT'] = cls.tmp.name
        os.environ['RAG3_BACKEND_TEST_INSTANCE'] = 'stage_d_contract'
        os.environ['QT_QPA_PLATFORM'] = 'offscreen'
        from rag3.constants import DATA_ROOT
        from rag3.ingestion_jobs import DB_PATH
        assert DB_PATH.parent == DATA_ROOT and not DB_PATH.is_file()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_01_nofusion_deduplicates_and_retains_method_scores(self):
        from rag3 import retrieval as r
        a='00000000-0000-0000-0000-000000000001'
        b='00000000-0000-0000-0000-000000000002'
        c='00000000-0000-0000-0000-000000000003'
        def point(id,score):
            return SimpleNamespace(id=id,score=score,payload={'content':'sample text',
                'meta':{'document_id':'d1','title':'Example','section_paths':['Section 1'],
                        'source_uri':'file://private','chunk_index':0}})
        class FakeQ:
            def __init__(self,*args,**kwargs): self.n=0
            def collection_exists(self,*a):return True
            def query_points(self,**kw):
                self.n+=1
                pts=([point(a,0.9),point(b,0.8)] if self.n==1 else
                     [point(a,0.7),point(c,0.6)])
                return SimpleNamespace(points=pts)
        with patch.object(r,'QdrantClient',FakeQ), patch.object(r,'collection_bm25_config',return_value={"k":1.2,"b":0.75}):
            out=r.search('example',semantic=True,bm25=True,rank_fusion=False,top_k=2,
                         collection='test_collection',dense_encoder=lambda _: [0.1,0.2])
        self.assertEqual([p['point_id'] for p in out['hits']],[a,b,c])
        self.assertEqual([p['rank'] for p in out['hits']],[1,2,3])
        self.assertEqual(len(out['hits'][0]['method_details']),2)
        self.assertEqual([x['score'] for x in out['hits'][0]['method_details']],[0.9,0.7])

    def test_02_backend_ui_search_once_with_stable_mcp_ids_and_versions(self):
        from rag3 import retrieval as r
        spec = importlib.util.spec_from_file_location('stage_d_api_server', ROOT/'scripts'/'rag3-mcp-server.py')
        mod=importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod  # Required for Pydantic forward-ref resolution.
        spec.loader.exec_module(mod)
        calls=[]
        def fake_search(query,**kwargs):
            calls.append((query,kwargs))
            return {'query':query,'collection':kwargs['collection'],
                    'methods':['bge_m3_semantic'],'rank_fusion':False,'top_k':2,
                    'hits':[{'point_id':'00000000-0000-0000-0000-000000000004',
                             'document_id':'doc','title':'Example','chunk_index':0,
                             'document_version':1,'content':'Source text',
                             'score':0.5,'rank':1,'retriever':'bge_m3_semantic',
                             'page_start':1,'page_end':1,
                             'source_uri':'file:///hidden','section_paths':['One']} ]}
        with patch.object(r,'search',fake_search):
            output=mod._ui_command('search',{'query':'example','collection':'temp',
                  'options':{'semantic':True,'bm25':False,'rank_fusion':False,'top_k':2}})
        self.assertEqual(len(calls),1)
        self.assertEqual(output['results'][0]['id'], output['ui']['hits'][0]['id'])
        self.assertEqual(output['ui']['hits'][0]['content'],'Source text')
        self.assertEqual(output['ui']['hits'][0]['section_paths'],['One'])
        versions=mod._ui_command('ui.revisions',{'collection':'temp'})
        self.assertIn('jobs',versions['versions'])
        self.assertEqual(mod._ui_command('jobs.list',{'include_hidden':False})['total'],0)
        with self.assertRaises(ValueError):
            mod._ui_command('jobs.list',{'include_hidden':'false'})

    def test_03_staged_ui_is_async_and_does_not_own_services(self):
        source=(ROOT/'scripts'/'ingest-ui-stage-d.py').read_text(encoding='utf-8')
        for forbidden in ('QProcess', 'manager_client(', 'list_jobs(', 'change_versions(',
                          'ensure_infrastructure()', 'launch_mcp_tray()',
                          'launch_codex_plugin_check()', 'ingestion_manager.client()'):
            self.assertNotIn(forbidden,source)
        from PySide6.QtWidgets import QApplication
        spec=importlib.util.spec_from_file_location('stage_d_candidate_ui',ROOT/'scripts'/'ingest-ui-stage-d.py')
        ui=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ui)
        app=QApplication.instance() or QApplication([])
        window=ui.IngestionWindow()
        events=[]
        def fake_command(action,params,done,*,read_only=True):
            events.append((action,params,read_only))
            responses={
                'ui.revisions':{'versions':{'jobs':0,'collections':0}},
                'jobs.list':{'total':0,'jobs':[]},
                'collections.list':{'collections':['temp']},
                'search':{'ui':{'hits':[],'collection':'temp',
                    'methods':['bge_m3_semantic'],'rank_fusion':False}}}
            done(responses.get(action,{}),None)
            return True
        window.backend.command=fake_command
        app.processEvents()
        window.collection.setText('temp')
        window.retrieval_semantic.setChecked(True)
        window.retrieval_bm25.setChecked(False)
        window.retrieval_rank_fusion.setChecked(False)
        window.query.setText('example')
        window.run_retrieval()
        self.assertIn('search',[x[0] for x in events])
        self.assertEqual(next(x[1]['options'] for x in events if x[0]=='search')['bm25'],False)
        window.close()

if __name__=='__main__':
    unittest.main(verbosity=2)
