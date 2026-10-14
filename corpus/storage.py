"""Local corpus persistence with stable IDs and duplicate detection."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from api import config
from corpus.schema import Passage, StoredDocument


@dataclass(frozen=True)
class StoredUpload:
    document_id: str
    checksum: str
    path: Path
    duplicate: bool


def ensure_data_directories() -> None:
    for path in (
        config.RAW_DOCUMENT_DIR,
        config.NORMALIZED_DOCUMENT_DIR,
        config.PASSAGE_DIR,
        config.REPORT_DIR,
        config.REVIEW_DIR,
        config.MANIFEST_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def _safe_filename(filename: str) -> str:
    name = Path(filename).name
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def save_upload(filename: str, content: bytes) -> StoredUpload:
    ensure_data_directories()
    suffix = Path(filename).suffix.lower()
    if suffix not in config.SUPPORTED_UPLOAD_SUFFIXES:
        raise ValueError(f"Unsupported file type: {suffix}")
    if not content:
        raise ValueError("Uploaded document is empty")
    if len(content) > config.MAX_UPLOAD_BYTES:
        raise ValueError("Uploaded document exceeds the configured size limit")

    checksum = hashlib.sha256(content).hexdigest()
    document_id = checksum[:16]
    destination = config.RAW_DOCUMENT_DIR / f"{document_id}-{_safe_filename(filename)}"
    duplicate = destination.exists()
    if not duplicate:
        destination.write_bytes(content)
    return StoredUpload(document_id, checksum, destination, duplicate)


def save_normalized_text(document_id: str, text: str) -> Path:
    ensure_data_directories()
    path = config.NORMALIZED_DOCUMENT_DIR / f"{document_id}.txt"
    path.write_text(text, encoding="utf-8")
    return path


def save_passages(document_id: str, passages: list[Passage]) -> Path:
    ensure_data_directories()
    path = config.PASSAGE_DIR / f"{document_id}.jsonl"
    body = "\n".join(passage.model_dump_json() for passage in passages)
    path.write_text(f"{body}\n" if body else "", encoding="utf-8")
    return path


def append_manifest(document: StoredDocument) -> None:
    ensure_data_directories()
    existing_ids: set[str] = set()
    if config.MANIFEST_PATH.exists():
        for line in config.MANIFEST_PATH.read_text(encoding="utf-8").splitlines():
            if line.strip():
                existing_ids.add(json.loads(line)["document_id"])
    if document.document_id in existing_ids:
        return
    with config.MANIFEST_PATH.open("a", encoding="utf-8") as manifest:
        manifest.write(document.model_dump_json() + "\n")
