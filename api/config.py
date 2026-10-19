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
RETRIEVAL_INDEX_PATH = INDEX_DIR / "temporal_claims.jsonl"
SEMANTIC_PASSAGE_INDEX_PATH = INDEX_DIR / "semantic_passages.jsonl"
TEMPORAL_CACHE_DIR = DATA_DIR / "temporal_cache"
EMBEDDING_CACHE_DIR = DATA_DIR / "embedding_cache"
MANIFEST_DIR = DATA_DIR / "manifests"
MANIFEST_PATH = MANIFEST_DIR / "documents.jsonl"
REVIEW_LOG_PATH = REVIEW_DIR / "document_reviews.jsonl"

MAX_UPLOAD_BYTES = int(os.getenv("NLP_MAX_UPLOAD_BYTES", 25 * 1024 * 1024))
SUPPORTED_UPLOAD_SUFFIXES = {".pdf", ".html", ".htm", ".txt"}

# Corpus creation is intentionally upload-only. Source URLs are provenance
# strings and are never fetched by this application.
NETWORK_INGESTION_ENABLED = False

# Ollama is used only on local, already-uploaded text. Nothing in the corpus
# pipeline downloads source documents. Models must be installed separately.
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_TEMPORAL_MODEL = os.getenv("OLLAMA_TEMPORAL_MODEL", "qwen2.5:3b")
OLLAMA_FALLBACK_TEMPORAL_MODEL = os.getenv(
    "OLLAMA_FALLBACK_TEMPORAL_MODEL", "llama3.1:8b"
)
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
OLLAMA_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "300"))
OLLAMA_PRIMARY_NUM_PREDICT = int(os.getenv("OLLAMA_PRIMARY_NUM_PREDICT", "600"))
OLLAMA_FALLBACK_NUM_PREDICT = int(os.getenv("OLLAMA_FALLBACK_NUM_PREDICT", "900"))
OLLAMA_FALLBACK_CONFIDENCE_THRESHOLD = float(
    os.getenv("OLLAMA_FALLBACK_CONFIDENCE_THRESHOLD", "0.70")
)
TEMPORAL_CONFIDENCE_THRESHOLD = float(
    os.getenv("TEMPORAL_CONFIDENCE_THRESHOLD", "0.65")
)
TEMPORAL_RELEVANCE_RATIO = float(os.getenv("TEMPORAL_RELEVANCE_RATIO", "0.80"))
