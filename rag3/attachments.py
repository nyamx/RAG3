"""Stream user-authorized attachment downloads without passing file bytes in MCP."""

from __future__ import annotations

import hashlib
import http.client
import ipaddress
import socket
import ssl
from email.message import Message
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit

from rag3.constants import MCP_FILE_SUFFIXES


class AttachmentError(ValueError):
    pass


class AttachmentCancelled(Exception):
    pass


def validate_url(url: str) -> None:
    try:
        parts = urlsplit(url)
        if (parts.scheme != "https" or not parts.hostname or parts.username or parts.password
                or parts.port not in (None, 443) or any(ord(c) < 32 for c in url)):
            raise ValueError
        try:
            address = ipaddress.ip_address(parts.hostname)
        except ValueError:
            address = None
        if address is not None and not address.is_global:
            raise ValueError
    except (ValueError, TypeError):
        raise AttachmentError("Attachment download_url must be a public HTTPS URL.") from None


def validate_filename(filename: str) -> str:
    if (not filename or filename in {".", ".."} or filename != filename.strip(" .")
            or any(c in filename for c in '<>:"/\\|?*\0')
            or any(ord(c) < 32 for c in filename)):
        raise AttachmentError("filename must be a plain file name.")
    if filename.split(".")[0].upper() in {
            "CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
            *(f"LPT{i}" for i in range(1, 10))}:
        raise AttachmentError("filename is reserved by Windows.")
    if Path(filename).suffix.lower() not in MCP_FILE_SUFFIXES:
        raise AttachmentError("supported formats: PDF, DOC, DOCX, MD, TXT, JSON, JSONL, CSV, TSV")
    return filename


class _PublicHTTPSConnection(http.client.HTTPSConnection):
    def connect(self) -> None:
        # Resolve once and pin the actual connection to a checked public address.
        # Re-check every redirect; no cookies, API keys, or proxy credentials are sent.
        try:
            addresses = socket.getaddrinfo(self.host, 443, type=socket.SOCK_STREAM)
            if not addresses or any(not ipaddress.ip_address(item[4][0]).is_global
                                    for item in addresses):
                raise AttachmentError("Attachment URL resolves to a non-public address.")
            for _, _, _, _, endpoint in addresses:
                sock = None
                try:
                    sock = socket.create_connection((endpoint[0], endpoint[1]), timeout=self.timeout)
                    self.sock = self._context.wrap_socket(sock, server_hostname=self.host)
                    return
                except OSError:
                    if sock is not None:
                        sock.close()
            raise AttachmentError("Could not connect to the attachment download host.")
        except AttachmentError:
            raise
        except OSError:
            raise AttachmentError("Could not resolve the attachment download host.") from None


def _filename(supplied: str, headers, url: str) -> str:
    if supplied:
        return validate_filename(supplied)
    message = Message()
    message["Content-Disposition"] = headers.get("Content-Disposition", "")
    name = message.get_filename()
    if name:
        return validate_filename(name)
    name = unquote(urlsplit(url).path.rsplit("/", 1)[-1])
    if Path(name).suffix.lower() in MCP_FILE_SUFFIXES:
        return validate_filename(name)
    suffix = {"application/pdf": ".pdf", "application/msword": ".doc",
              "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
              "text/markdown": ".md", "text/plain": ".txt",
              "application/json": ".json", "application/x-ndjson": ".jsonl",
              "text/csv": ".csv", "text/tab-separated-values": ".tsv"}.get(
                  headers.get("Content-Type", "").split(";")[0].strip().lower())
    if not suffix:
        raise AttachmentError("The attachment needs its original file_name and extension.")
    return "attachment" + suffix


def download_attachment(url: str, folder: Path, *, filename: str = "",
                        checkpoint=lambda: None, progress=lambda count: None) -> tuple[Path, str, str, int]:
    """Keep the original bytes on disk; there is no application file-size cap."""
    partial = folder / "attachment.download.part"
    connection = None
    try:
        for _ in range(6):
            checkpoint()
            validate_url(url)
            parts = urlsplit(url)
            connection = _PublicHTTPSConnection(parts.hostname, timeout=30,
                                                 context=ssl.create_default_context())
            path = parts.path or "/"
            if parts.query:
                path += "?" + parts.query
            connection.request("GET", path, headers={"Accept-Encoding": "identity"})
            response = connection.getresponse()
            if response.status in {301, 302, 303, 307, 308}:
                location = response.getheader("Location")
                connection.close()
                connection = None
                if not location:
                    raise AttachmentError("Attachment download redirect has no destination.")
                url = urljoin(url, location)
                continue
            if response.status != 200:
                raise AttachmentError(f"Attachment download returned HTTP {response.status}; refresh the file reference.")
            if response.getheader("Content-Encoding", "identity").lower() not in {"", "identity"}:
                raise AttachmentError("Attachment download did not return original unencoded bytes.")
            name = _filename(filename, response.headers, url)
            expected = response.getheader("Content-Length")
            expected = int(expected) if expected is not None else None
            digest = hashlib.sha256()
            count = 0
            with partial.open("wb") as destination:
                while True:
                    checkpoint()
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    destination.write(chunk)
                    digest.update(chunk)
                    count += len(chunk)
                    progress(count)
            checkpoint()
            if not count or (expected is not None and count != expected):
                raise AttachmentError("Attachment download is empty or incomplete; submit the original file again.")
            source = folder / name
            partial.replace(source)
            return source, name, digest.hexdigest(), count
        raise AttachmentError("Attachment download has too many redirects.")
    except (AttachmentError, AttachmentCancelled):
        raise
    except (OSError, http.client.HTTPException, ValueError):
        # Signed URLs contain credentials and must not appear in job errors/logs.
        raise AttachmentError("Attachment download failed; submit a fresh file reference and retry.") from None
    finally:
        if connection is not None:
            connection.close()
        partial.unlink(missing_ok=True)
