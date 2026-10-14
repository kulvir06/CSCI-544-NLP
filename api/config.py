"""Application configuration with safe, project-relative defaults."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.getenv("NLP_DATA_DIR", PROJECT_ROOT / "data")).resolve()
RAW_DOCUMENT_DIR = DATA_DIR / "raw"
NORMALIZED_DOCUMENT_DIR = DATA_DIR / "normalized"
PASSAGE_DIR = DATA_DIR / "passages"
REPORT_DIR = DATA_DIR / "reports"
REVIEW_DIR = DATA_DIR / "reviews"
INDEX_DIR = DATA_DIR / "index"
RETRIEVAL_INDEX_PATH = INDEX_DIR / "approved_passages.jsonl"
MANIFEST_DIR = DATA_DIR / "manifests"
MANIFEST_PATH = MANIFEST_DIR / "documents.jsonl"
REVIEW_LOG_PATH = REVIEW_DIR / "document_reviews.jsonl"

MAX_UPLOAD_BYTES = int(os.getenv("NLP_MAX_UPLOAD_BYTES", 25 * 1024 * 1024))
SUPPORTED_UPLOAD_SUFFIXES = {".pdf", ".html", ".htm", ".txt"}

# Corpus creation is intentionally upload-only. Source URLs are provenance
# strings and are never fetched by this application.
NETWORK_INGESTION_ENABLED = False
