"""Verify that supported Qdrant collection mutations advance UI revisions."""
from __future__ import annotations

import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag3.collections import collection_contents, create_collection, delete_collections
from rag3.ingestion_jobs import change_versions, content_scope

name = '__rag3_change_test_' + uuid.uuid4().hex[:12]
before = change_versions('collections', content_scope(name))
created = False
try:
    create_collection(name)
    created = True
    after_create = change_versions('collections', content_scope(name))
    assert after_create['collections'] > before['collections']
    contents = collection_contents(name)
    assert contents['document_count'] == 0
finally:
    if created:
        delete_collections([name])

after_delete = change_versions('collections', content_scope(name))
assert after_delete['collections'] > after_create['collections']
assert after_delete[content_scope(name)] > before[content_scope(name)]
print('CHANGE_NOTIFICATION_QDRANT_OK')
