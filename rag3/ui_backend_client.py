"""Single asynchronous, authenticated loopback command client for the desktop UI.

Read-only operations may be retried once following token rotation.  Writes are
never replayed automatically: an HTTP timeout can mean committed-but-unobserved.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QObject, QUrl
from PySide6.QtNetwork import QNetworkAccessManager, QNetworkReply, QNetworkRequest

from rag3.backend_boundary import _token_file
from rag3.constants import ROOT


class UIBackendClient(QObject):
    def __init__(self, parent: QObject | None = None, *, port: int | None = None):
        super().__init__(parent)
        self.port = int(port if port is not None else os.environ.get('RAG3_MCP_PORT', '8000'))
        if not 1 <= self.port <= 65535:
            raise ValueError('RAG3_MCP_PORT must be 1..65535')
        self.token_file = _token_file(ROOT, self.port)
        self.url = QUrl(f'http://127.0.0.1:{self.port}/api/v1/command')
        self.network = QNetworkAccessManager(self)
        self._pending: set[QNetworkReply] = set()

    def command(self, action: str, params: dict,
                done: Callable[[dict | None, str | None], None], *, read_only: bool = True) -> bool:
        """Call done(result, error) asynchronously on Qt's event loop.

        A write failure is deliberately ambiguous: no automatic retry.
        """
        body = json.dumps({'action': action, 'params': params}, ensure_ascii=False).encode('utf-8')

        def attempt(retried: bool = False) -> bool:
            try:
                token = self.token_file.read_text(encoding='ascii').strip()
                if not token:
                    raise ValueError('token file empty')
            except (OSError, ValueError):
                done(None, 'RAG3 backend token unavailable; check backend startup.')
                return False
            request = QNetworkRequest(self.url)
            request.setRawHeader(b'X-RAG3-Token', token.encode('ascii'))
            request.setHeader(QNetworkRequest.KnownHeaders.ContentTypeHeader, 'application/json')
            request.setTransferTimeout(97000)
            reply = self.network.post(request, body)
            self._pending.add(reply)

            def finished():
                self._pending.discard(reply)
                status = reply.attribute(QNetworkRequest.Attribute.HttpStatusCodeAttribute)
                try:
                    data = json.loads(bytes(reply.readAll()).decode('utf-8'))
                except (UnicodeError, ValueError):
                    data = None
                message = reply.errorString()
                reply.deleteLater()
                if status == 401 and read_only and not retried:
                    attempt(True)
                    return
                if not isinstance(data, dict) or not data.get('ok'):
                    failure = data.get('error') if isinstance(data, dict) else None
                    code = failure.get('code') if isinstance(failure, dict) else 'NETWORK_ERROR'
                    detail = failure.get('message') if isinstance(failure, dict) else None
                    error = f'{code}: {detail or message}'
                    if not read_only:
                        error += ' (completion unknown; refresh state before retrying)'
                    done(None, error)
                    return
                done(data.get('result'), None)

            reply.finished.connect(finished)
            return True

        return attempt()
