"""MCP compatibility facade for the shared SQLite ingestion manager."""

from __future__ import annotations

import hashlib
import codecs
import json
import os
import re
import subprocess
import sys
import threading
import uuid
import zipfile
import time
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

from rag3.constants import DEFAULT_QDRANT_URL, ROOT, DATA_ROOT, MCP_FILE_SUFFIXES, UTF8_TEXT_SUFFIXES
from rag3.attachments import (AttachmentCancelled, download_attachment,
                              validate_filename, validate_url)

JOB_ROOT = DATA_ROOT / "mcp_jobs"
MAX_TEXT_CHARS = 1_000_000
ALLOWED_SUFFIXES = MCP_FILE_SUFFIXES
ACTIVE = {"QUEUED", "DOWNLOADING", "RUNNING", "RESUMING", "PAUSING", "CANCELLING"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_content(suffix: str, content: bytes) -> None:
    if not content:
        raise ValueError("file must not be empty")
    if suffix == ".pdf" and not content.startswith(b"%PDF-"):
        raise ValueError("PDF header is invalid")
    if suffix == ".doc" and not content.startswith(bytes.fromhex("d0cf11e0a1b11ae1")):
        raise ValueError("DOC header is invalid")
    if suffix == ".docx":
        try:
            with zipfile.ZipFile(BytesIO(content)) as archive:
                if "word/document.xml" not in archive.namelist():
                    raise ValueError("DOCX document part is missing")
        except zipfile.BadZipFile as exc:
            raise ValueError("DOCX archive is invalid") from exc
    if suffix == ".md" or suffix in UTF8_TEXT_SUFFIXES:
        content.decode("utf-8-sig")


def _validate_source(source: Path) -> None:
    with source.open("rb") as stream:
        header = stream.read(8)
        if not header:
            raise ValueError("file must not be empty")
        suffix = source.suffix.lower()
        if suffix == ".pdf" and not header.startswith(b"%PDF-"):
            raise ValueError("PDF header is invalid")
        if suffix == ".doc" and header != bytes.fromhex("d0cf11e0a1b11ae1"):
            raise ValueError("DOC header is invalid")
        if suffix == ".md" or suffix in UTF8_TEXT_SUFFIXES:
            stream.seek(0)
            decoder = codecs.getincrementaldecoder("utf-8-sig")()
            while chunk := stream.read(1024 * 1024):
                decoder.decode(chunk)
            decoder.decode(b"", final=True)
    if source.suffix.lower() == ".docx":
        try:
            with zipfile.ZipFile(source) as archive:
                if "word/document.xml" not in archive.namelist():
                    raise ValueError("DOCX document part is missing")
        except zipfile.BadZipFile as exc:
            raise ValueError("DOCX archive is invalid") from exc


from rag3.ingestion_manager import client


class JobManager:
    """Preserves public MCP method signatures; no process or job JSON lives here."""
    def __init__(self, *, qdrant_url: str = DEFAULT_QDRANT_URL) -> None:
        self.qdrant_url = qdrant_url

    def submit_file(self, collection: str, filename: str, content: bytes) -> dict:
        validate_filename(filename)
        suffix = Path(filename).suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            raise ValueError('supported formats: PDF, DOC, DOCX, MD, TXT, JSON, JSONL, CSV, TSV')
        _validate_content(suffix, content)
        import base64
        return client('submit', origin='MCP', collection=collection,
                      filename=filename,content_base64=base64.b64encode(content).decode(),
                      requested=True)

    def submit_attachment(self, collection: str, attachment: dict, filename: str = '') -> dict:
        validate_url(attachment['download_url'])
        if not attachment.get('file_id'):
            raise ValueError('attachment file_id is required')
        name = attachment.get('file_name') or filename
        if name: validate_filename(name)
        return client('submit',origin='MCP',collection=collection,
                      filename=filename,attachment=attachment,requested=True)

    def submit_text(self, collection: str, title: str, content: str) -> dict:
        if not isinstance(content,str) or not content.strip() or len(content)>MAX_TEXT_CHARS:
            raise ValueError(f'text must be nonempty and at most {MAX_TEXT_CHARS} characters')
        if not isinstance(title,str) or not title.strip() or len(title)>200:
            raise ValueError('title must be 1..200 characters')
        safe_title = re.sub(r'[^\w. -]','_',title.strip()).strip(' .') or 'generated'
        return self.submit_file(collection,safe_title+'.md',content.encode('utf-8'))

    def status(self, job_id: str) -> dict:
        return client('status',job_id=job_id)

    def list(self,collection: str='',status: str='',limit: int=50,offset: int=0) -> dict:
        return client('list',collection=collection,status=status,limit=limit,offset=offset)

    def has_active_collection(self, collection: str) -> bool:
        return client('has_active_collection',collection=collection)['active']

    def pause(self, job_id: str) -> dict:
        return client('action',job_id=job_id,operation='pause')

    def resume(self, job_id: str) -> dict:
        return client('action',job_id=job_id,operation='resume')

    def cancel(self, job_id: str) -> dict:
        return client('action',job_id=job_id,operation='cancel')
