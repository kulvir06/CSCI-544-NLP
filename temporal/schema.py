"""Typed contracts for model-extracted, temporally scoped legal claims."""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator

from corpus.schema import DocumentStatus, SourceTier


class TemporalRole(str, Enum):
    OPERATIVE = "operative"
    HISTORICAL = "historical"
    TRANSITIONAL = "transitional"
    PROPOSED = "proposed"
    REJECTED = "rejected"
    PUBLIC_COMMENT = "public_comment"
    EXAMPLE = "example"
    UNKNOWN = "unknown"


class ExtractedClaim(BaseModel):
    claim_text: str = Field(min_length=1)
    evidence_quote: str = Field(min_length=1)
    temporal_role: TemporalRole
    valid_from: Optional[date] = None
    valid_to: Optional[date] = None
    date_basis: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_interval(self) -> "ExtractedClaim":
        if self.valid_to and not self.valid_from:
            raise ValueError("valid_to requires valid_from")
        if self.valid_from and self.valid_to and self.valid_to < self.valid_from:
            raise ValueError("valid_to must be on or after valid_from")
        return self


class ClaimExtraction(BaseModel):
    claims: list[ExtractedClaim] = Field(default_factory=list, max_length=8)


class TemporalClaim(BaseModel):
    claim_id: str
    document_id: str
    source_passage_id: str
    text: str
    evidence_quote: str
    context: str
    section: Optional[str] = None
    page_number: Optional[int] = None
    passage_relevance_score: float
    title: Optional[str] = None
    issuer: Optional[str] = None
    source_url: Optional[str] = None
    source_tier: SourceTier
    document_status: DocumentStatus
    temporal_role: TemporalRole
    valid_from: Optional[date] = None
    valid_to: Optional[date] = None
    valid_as_of: Optional[date] = None
    date_basis: str
    temporal_confidence: float = Field(ge=0.0, le=1.0)
    extraction_model: str
    embedding_model: str
    embedding: list[float] = Field(default_factory=list)


class TemporalIndexReport(BaseModel):
    source_documents: int
    source_passages: int
    indexed_claims: int
    skipped_passages: int
    failed_passages: int
    extraction_model: str
    embedding_model: str
    index_file: str
    failures: list[str] = Field(default_factory=list)
