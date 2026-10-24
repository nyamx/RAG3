"""Process-owned reusable Qdrant clients; complete construction-argument identity."""
from __future__ import annotations

import atexit
import os
import threading
from qdrant_client import QdrantClient

_lock = threading.RLock()
_clients = {}
_pid = os.getpid()


def _freeze(value):
    if isinstance(value, dict):
        return ('dict', tuple(sorted((k, _freeze(v)) for k,v in value.items())))
    if isinstance(value, (list,tuple)):
        return ('list',tuple(_freeze(v) for v in value))
    if isinstance(value, (str,int,float,bool,type(None))):
        return value
    # Nontrivial objects cannot safely share a presumed-equivalent constructor.
    raise TypeError('Qdrant client option must have a stable, value-based key')


def get_client(*, url, **kwargs):
    global _pid, _clients
    pid = os.getpid()
    key = _freeze({'url':url, **kwargs})
    with _lock:
        if pid != _pid:
            _clients = {}  # Never inherit and reuse parent process client objects.
            _pid = pid
        client = _clients.get(key)
        if client is None:
            client = QdrantClient(url=url, **kwargs)  # Do not bypass compatibility check.
            _clients[key] = client
        return client


def close_all():
    with _lock:
        clients = list(_clients.values())
        _clients.clear()
    for client in clients:
        try: client.close()
        except Exception: pass

atexit.register(close_all)
