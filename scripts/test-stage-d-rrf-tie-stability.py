"""Stage D: deterministic RRF tie sorting, including final top_k boundary."""
from __future__ import annotations
import os
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

class RRFTieStability(unittest.TestCase):
    def test_full_candidates_rank_and_stable_ties(self):
        from rag3 import retrieval as r
        ids = ['00000000-0000-0000-0000-00000000000'+str(i) for i in range(1, 6)]
        def point(idx, score):
            return SimpleNamespace(id=ids[idx], score=score,
                payload={'content':'Sample', 'meta':{'document_id':'doc', 'title':'Sample'}})
        # All equal scores include a tie at top_k boundary. Qdrant switches order.
        first = [point(3,0.5), point(1,0.666666666), point(4,0.4), point(0,0.666666666), point(2,0.5)]
        second = list(reversed(first))
        class FakeQ:
            order = 0
            seen = []
            def __init__(self, *args, **kwargs): pass
            def collection_exists(self, _name): return True
            def query_points(self, **kw):
                self.__class__.seen.append(kw)
                self.__class__.order += 1
                return SimpleNamespace(points=first if self.__class__.order % 2 else second)
        with patch.object(r, 'QdrantClient', FakeQ), \
             patch.object(r, 'collection_bm25_config', return_value={'k':1.2,'b':0.75}):
            left = r.search('q', semantic=True, bm25=True, rank_fusion=True,
                            top_k=3, candidate_k=5, collection='isolation',
                            dense_encoder=lambda _q: [0.1, 0.2])
            right = r.search('q', semantic=True, bm25=True, rank_fusion=True,
                             top_k=3, candidate_k=5, collection='isolation',
                             dense_encoder=lambda _q: [0.1, 0.2])
        expected = [ids[0], ids[1], ids[2]]
        self.assertEqual([x['point_id'] for x in left['hits']], expected)
        self.assertEqual([x['point_id'] for x in right['hits']], expected)
        self.assertEqual([x['rank'] for x in left['hits']], [1,2,3])
        self.assertEqual([x['score'] for x in left['hits']], [0.666666666,0.666666666,0.5])
        self.assertEqual([x['retriever'] for x in left['hits']], ['rrf']*3)
        self.assertEqual([call['limit'] for call in FakeQ.seen], [10,10])
        self.assertEqual([[p.limit for p in call['prefetch']] for call in FakeQ.seen], [[5,5],[5,5]])

if __name__ == '__main__':
    unittest.main(verbosity=2)
