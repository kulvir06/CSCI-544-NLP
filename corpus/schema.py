"""Typed contracts for uploaded documents and selected passages."""

from __future__ import annotations

from datetime import date, datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator


class SourceTier(int, Enum):
    PRIMARY_LAW = 1
    FEDERAL_GUIDANCE = 2
    UNIVERSITY_ADVISORY = 3
    UNKNOWN = 4


class DocumentStatus(str, Enum):
    CURRENT = "current"
    HISTORICAL = "historical"
    SUPERSEDED = "superseded"
    PROPOSED = "proposed"
    UNKNOWN = "unknown"


class ReviewDecision(str, Enum):
    APPROVED = "approved"
    REJECTED = "rejected"


class CorpusUse(str, Enum):
    TEMPORAL_AND_MEANING = "temporal_and_meaning"
    MEANING_ONLY = "meaning_only"
    EXCLUDED = "excluded"


class DocumentMetadata(BaseModel):
    original_filename: str
    title: Optional[str] = None
    issuer: Optional[str] = None
    source_url: Optional[str] = None
    published_at: Optional[date] = None
    effective_from: Optional[date] = None
    effective_to: Optional[date] = None
    valid_as_of: Optional[date] = None
    retrieved_at: Optional[date] = None
    date_evidence: Optional[str] = None
    rule_family: Optional[str] = None
    status: DocumentStatus = DocumentStatus.UNKNOWN

    @model_validator(mode="after")
    def validate_interval(self) -> "DocumentMetadata":
        if self.effective_to and not self.effective_from:
            raise ValueError("effective_to requires effective_from")
        if (
            self.effective_from
            and self.effective_to
            and self.effective_to < self.effective_from
        ):
            raise ValueError("effective_to must be on or after effective_from")
        return self


class MetadataCandidates(BaseModel):
    date_mentions: list[str] = Field(default_factory=list)
    effective_date_candidates: list[str] = Field(default_factory=list)
    valid_as_of_candidates: list[str] = Field(default_factory=list)
    inferred_fields: list[str] = Field(default_factory=list)


class StoredDocument(BaseModel):
    document_id: str
    checksum_sha256: str
    stored_filename: str
    media_type: str
    byte_size: int
    metadata: DocumentMetadata
    metadata_candidates: MetadataCandidates = Field(default_factory=MetadataCandidates)
    source_tier: SourceTier
    ingested_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    review_flags: list[str] = Field(default_factory=list)


class ParsedBlock(BaseModel):
    text: str
    section: Optional[str] = None
    page_number: Optional[int] = None
    block_index: int


class Passage(BaseModel):
    passage_id: str
    document_id: str
    text: str
    section: Optional[str] = None
    page_number: Optional[int] = None
    block_index: int
    relevance_score: float
    selection_reasons: list[str] = Field(default_factory=list)
    requires_review: bool = False


class IngestionResult(BaseModel):
    document: StoredDocument
    total_blocks: int
    selected_passages: int
    duplicate_upload: bool
    passage_file: str
    normalized_text_file: str


class BatchDocumentResult(BaseModel):
    filename: str
    outcome: str
    document_id: Optional[str] = None
    selected_passages: int = 0
    review_flags: list[str] = Field(default_factory=list)
    error: Optional[str] = None


class BatchIngestionReport(BaseModel):
    input_directory: str
    discovered_files: int
    processed_files: int
    failed_files: int
    total_selected_passages: int
    results: list[BatchDocumentResult]
    report_file: Optional[str] = None


class MetadataCorrections(BaseModel):
    title: Optional[str] = None
    issuer: Optional[str] = None
    source_url: Optional[str] = None
    published_at: Optional[date] = None
    effective_from: Optional[date] = None
    effective_to: Optional[date] = None
    valid_as_of: Optional[date] = None
    retrieved_at: Optional[date] = None
    date_evidence: Optional[str] = None
    rule_family: Optional[str] = None
    status: Optional[DocumentStatus] = None


class DocumentReviewRequest(BaseModel):
    decision: ReviewDecision
    corpus_use: CorpusUse
    reviewer: str = Field(min_length=1)
    corrections: MetadataCorrections = Field(default_factory=MetadataCorrections)
    notes: Optional[str] = None


class DocumentReview(BaseModel):
    document_id: str
    decision: ReviewDecision
    corpus_use: CorpusUse
    reviewer: str
    reviewed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    metadata: DocumentMetadata
    source_tier: SourceTier
    notes: Optional[str] = None


class ReviewQueueItem(BaseModel):
    document: StoredDocument
    current_metadata: DocumentMetadata
    current_source_tier: SourceTier
    decision: Optional[ReviewDecision] = None
    corpus_use: Optional[CorpusUse] = None
    blocking_issues: list[str] = Field(default_factory=list)
    meaning_blocking_issues: list[str] = Field(default_factory=list)
    temporal_blocking_issues: list[str] = Field(default_factory=list)
    latest_review: Optional[DocumentReview] = None
