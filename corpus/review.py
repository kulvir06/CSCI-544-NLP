"""Auditable human review gate for document metadata.

Source URLs are treated as provenance strings. This module never opens them.
"""

from __future__ import annotations

import json
from pathlib import Path

from api import config
from corpus.schema import (
    CorpusUse,
    DocumentMetadata,
    DocumentReview,
    DocumentReviewRequest,
    DocumentStatus,
    ReviewDecision,
    ReviewQueueItem,
    SourceTier,
    StoredDocument,
)
from corpus.source_classification import classify_source
from corpus.storage import ensure_data_directories


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_documents() -> dict[str, StoredDocument]:
    return {
        item["document_id"]: StoredDocument.model_validate(item)
        for item in _read_jsonl(config.MANIFEST_PATH)
    }


def load_latest_reviews() -> dict[str, DocumentReview]:
    latest: dict[str, DocumentReview] = {}
    for item in _read_jsonl(config.REVIEW_LOG_PATH):
        review = DocumentReview.model_validate(item)
        latest[review.document_id] = review
    return latest


def _blocking_issues(
    metadata: DocumentMetadata,
    source_tier: SourceTier,
) -> list[str]:
    issues: list[str] = []
    if not metadata.source_url:
        issues.append("missing_source_url")
    if metadata.status == DocumentStatus.UNKNOWN:
        issues.append("unknown_document_status")
    if source_tier == SourceTier.UNKNOWN:
        issues.append("unknown_source_authority")
    if not metadata.effective_from and not metadata.valid_as_of:
        issues.append("missing_temporal_basis")
    if (metadata.effective_from or metadata.valid_as_of) and not metadata.date_evidence:
        issues.append("missing_date_evidence")
    return issues


def _approval_blockers(
    metadata: DocumentMetadata,
    source_tier: SourceTier,
    corpus_use: CorpusUse,
) -> list[str]:
    issues = _blocking_issues(metadata, source_tier)
    if corpus_use == CorpusUse.MEANING_ONLY:
        return [
            issue
            for issue in issues
            if issue
            not in {
                "missing_temporal_basis",
                "missing_date_evidence",
                "unknown_document_status",
            }
        ]
    return issues


def _merge_metadata(
    original: DocumentMetadata,
    request: DocumentReviewRequest,
) -> DocumentMetadata:
    updates = request.corrections.model_dump(exclude_unset=True)
    return DocumentMetadata.model_validate({**original.model_dump(), **updates})


def build_review_queue(include_resolved: bool = False) -> list[ReviewQueueItem]:
    documents = load_documents()
    latest_reviews = load_latest_reviews()
    queue: list[ReviewQueueItem] = []

    for document_id, document in documents.items():
        latest = latest_reviews.get(document_id)
        metadata = latest.metadata if latest else document.metadata
        source_tier = latest.source_tier if latest else document.source_tier
        decision = latest.decision if latest else None
        if not include_resolved and decision in {
            ReviewDecision.APPROVED,
            ReviewDecision.REJECTED,
        }:
            continue
        queue.append(
            ReviewQueueItem(
                document=document,
                current_metadata=metadata,
                current_source_tier=source_tier,
                decision=decision,
                corpus_use=latest.corpus_use if latest else None,
                blocking_issues=_blocking_issues(metadata, source_tier),
                latest_review=latest,
            )
        )
    return queue


def submit_review(
    document_id: str,
    request: DocumentReviewRequest,
) -> DocumentReview:
    documents = load_documents()
    if document_id not in documents:
        raise ValueError(f"Unknown document_id: {document_id}")
    if not request.reviewer.strip():
        raise ValueError("reviewer must not be blank")

    latest = load_latest_reviews().get(document_id)
    base_metadata = latest.metadata if latest else documents[document_id].metadata
    metadata = _merge_metadata(base_metadata, request)
    source_tier = classify_source(metadata)
    blockers = _approval_blockers(metadata, source_tier, request.corpus_use)

    if (
        request.decision == ReviewDecision.APPROVED
        and request.corpus_use == CorpusUse.EXCLUDED
    ):
        raise ValueError("Approved documents cannot use corpus_use=excluded")
    if (
        request.decision == ReviewDecision.REJECTED
        and request.corpus_use != CorpusUse.EXCLUDED
    ):
        raise ValueError("Rejected documents must use corpus_use=excluded")
    if request.decision == ReviewDecision.APPROVED and blockers:
        raise ValueError(
            "Cannot approve document while metadata issues remain: "
            + ", ".join(blockers)
        )

    review = DocumentReview(
        document_id=document_id,
        decision=request.decision,
        corpus_use=request.corpus_use,
        reviewer=request.reviewer.strip(),
        metadata=metadata,
        source_tier=source_tier,
        notes=request.notes,
    )
    ensure_data_directories()
    with config.REVIEW_LOG_PATH.open("a", encoding="utf-8") as review_log:
        review_log.write(review.model_dump_json() + "\n")
    return review


def load_retrieval_ready_documents(
    corpus_use: CorpusUse | None = None,
) -> list[DocumentReview]:
    """Return the latest approved versions, optionally filtered by use."""
    return [
        review
        for review in load_latest_reviews().values()
        if review.decision == ReviewDecision.APPROVED
        and (corpus_use is None or review.corpus_use == corpus_use)
    ]
