"""Extract searchable chat text from one ChatGPT export JSON shard."""
from __future__ import annotations

import json
import math
import re
import zipfile
from pathlib import Path

from haystack import Document

SHARD = re.compile(r"conversations(?:-\d+)?\.json\Z")
MAX_SHARD_BYTES = 128 * 1024 * 1024
MAX_EXPORT_BYTES = 2 * 1024 * 1024 * 1024
MAX_MESSAGE_CHARS = 3000


def _milliseconds(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return int(value * 1000) if value >= 0 else None


def normalize_conversation(raw: dict) -> dict:
    """Keep only the current text path; IDs remain opaque strings."""
    if not isinstance(raw, dict):
        raise ValueError("conversation must be an object")
    conversation_id = raw.get("conversation_id") or raw.get("id")
    mapping, node_id = raw.get("mapping"), raw.get("current_node")
    if not isinstance(conversation_id, str) or not conversation_id or not isinstance(mapping, dict):
        raise ValueError("conversation ID or mapping is missing")
    if not isinstance(node_id, str) or not node_id:
        raise ValueError("current_node is missing")
    path, seen = [], set()
    while node_id is not None:
        if node_id in seen:
            raise ValueError("conversation path contains a cycle")
        seen.add(node_id)
        node = mapping.get(node_id)
        if not isinstance(node, dict):
            raise ValueError("conversation path has a missing node")
        path.append((node_id, node.get("message")))
        node_id = node.get("parent")
        if node_id is not None and (not isinstance(node_id, str) or not node_id):
            raise ValueError("conversation parent is invalid")

    messages = []
    skipped_empty = skipped_unsupported = 0
    for turn_index, (node_id, message) in enumerate(reversed(path)):
        if message is None:
            continue
        if not isinstance(message, dict):
            skipped_unsupported += 1
            continue
        role = (message.get("author") or {}).get("role")
        content = message.get("content") or {}
        content_type = content.get("content_type") if isinstance(content, dict) else None
        parts = content.get("parts") if isinstance(content, dict) else None
        if (role not in {"user", "assistant"} or content_type not in {"text", "multimodal_text"}
                or not isinstance(parts, list)):
            skipped_unsupported += 1
            continue
        text_parts = [part for part in parts if isinstance(part, str)]
        if len(text_parts) != len(parts):
            skipped_unsupported += len(parts) - len(text_parts)
        text = "\n".join(text_parts).strip()
        if not text:
            skipped_empty += 1
            continue
        item = {"message_id": message.get("id") or node_id, "role": role,
                "turn_index": turn_index, "text": text}
        created = _milliseconds(message.get("create_time"))
        if created is not None:
            item["created_at_ms"] = created
        metadata = message.get("metadata")
        model = metadata.get("model_slug") if isinstance(metadata, dict) else None
        if isinstance(model, str) and model:
            item["model_slug"] = model
        messages.append(item)
    title = raw.get("title")
    result = {"conversation_id": conversation_id,
              "title": title.strip() if isinstance(title, str) and title.strip() else "Untitled chat",
              "messages": messages}
    for source, target in (("create_time", "created_at_ms"), ("update_time", "updated_at_ms")):
        value = _milliseconds(raw.get(source))
        if value is not None:
            result[target] = value
    for field in ("is_archived", "is_starred"):
        if isinstance(raw.get(field), bool):
            result[field] = raw[field]
    result["skipped_empty"] = skipped_empty
    result["skipped_unsupported"] = skipped_unsupported
    return result


def _shards(path: Path):
    if path.is_dir():
        files = sorted(p for p in path.glob("conversations*.json") if SHARD.fullmatch(p.name))
        if not files:
            raise ValueError("no ChatGPT conversation JSON files found")
        if sum(p.stat().st_size for p in files) > MAX_EXPORT_BYTES:
            raise ValueError("export exceeds size limit")
        for file in files:
            if file.is_symlink() or file.stat().st_size > MAX_SHARD_BYTES:
                raise ValueError("unsafe or oversized export shard")
            yield file.name, file.open("rb")
    elif path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            entries = [item for item in archive.infolist() if SHARD.fullmatch(Path(item.filename).name)]
            if not entries:
                raise ValueError("no ChatGPT conversation JSON files found")
            names = [Path(item.filename).name for item in entries]
            if len(names) != len(set(names)) or sum(item.file_size for item in entries) > MAX_EXPORT_BYTES:
                raise ValueError("duplicate or oversized export shards")
            for item in sorted(entries, key=lambda entry: entry.filename):
                parts = Path(item.filename.replace("\\", "/")).parts
                if ".." in parts or item.file_size > MAX_SHARD_BYTES or item.flag_bits & 1:
                    raise ValueError("unsafe or oversized export shard")
                if item.file_size > max(item.compress_size, 1) * 200:
                    raise ValueError("export shard compression ratio is too high")
                yield Path(item.filename).name, archive.open(item)
    elif path.is_file() and SHARD.fullmatch(path.name):
        if path.stat().st_size > MAX_SHARD_BYTES:
            raise ValueError("export shard exceeds size limit")
        yield path.name, path.open("rb")
    else:
        raise ValueError("provide an export ZIP, extracted folder, or conversation JSON file")


def iter_shards(path: str | Path):
    """Yield each unchanged JSON file and its conversations, one file at a time."""
    for name, stream in _shards(Path(path).expanduser().resolve()):
        with stream:
            raw = stream.read()
        conversations = json.loads(raw.decode("utf-8-sig"))
        if not isinstance(conversations, list):
            raise ValueError(f"{name} must contain a list of conversations")
        yield name, raw, conversations


def split_shard_source(preserved: dict, base_meta: dict, *, lookup_existing=None) -> list[Document]:
    source = Path(preserved["managed_source"])
    conversations = json.loads(source.read_text(encoding="utf-8-sig"))
    if not isinstance(conversations, list):
        raise ValueError("ChatGPT export shard must contain a list")
    ids = []
    for raw in conversations:
        if not isinstance(raw, dict):
            raise ValueError("conversation must be an object")
        conversation_id = raw.get("conversation_id") or raw.get("id")
        if not isinstance(conversation_id, str) or not conversation_id:
            raise ValueError("conversation ID is missing")
        ids.append(conversation_id)
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate conversation ID in export shard")
    existing = set(lookup_existing(set(ids))) if lookup_existing else set()
    if not existing.issubset(ids):
        raise ValueError("conversation lookup returned an unexpected ID")
    chunks = []
    for raw, conversation_id in zip(conversations, ids, strict=True):
        if conversation_id in existing:
            continue
        chat = normalize_conversation(raw)
        common = {**base_meta, "content_type": "chat_message", "source_kind": "chatgpt_export",
                  "chatgpt_export_file": source.name,
                  "chatgpt_conversation_id": chat["conversation_id"],
                  "chatgpt_conversation_title": chat["title"]}
        for source_key, meta_key in (("created_at_ms", "chatgpt_conversation_created_at_ms"),
                                     ("updated_at_ms", "chatgpt_conversation_updated_at_ms"),
                                     ("is_archived", "chatgpt_is_archived"),
                                     ("is_starred", "chatgpt_is_starred")):
            if source_key in chat:
                common[meta_key] = chat[source_key]
        for message in chat["messages"]:
            meta = {**common, "chatgpt_message_id": message["message_id"],
                    "chatgpt_role": message["role"], "chatgpt_turn_index": message["turn_index"]}
            if "created_at_ms" in message:
                meta["chatgpt_message_created_at_ms"] = message["created_at_ms"]
            if "model_slug" in message:
                meta["chatgpt_model_slug"] = message["model_slug"]
            text = message["text"]
            for part_index, offset in enumerate(range(0, len(text), MAX_MESSAGE_CHARS)):
                chunks.append(Document(content=text[offset:offset + MAX_MESSAGE_CHARS],
                                       meta={**meta, "chatgpt_message_chunk_index": part_index}))
    return chunks
