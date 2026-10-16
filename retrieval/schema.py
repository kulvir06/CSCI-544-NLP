from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from corpus.schema import CorpusUse, DocumentStatus, SourceTier


class PassageTemporalStatus(str, Enum):
    APPLICABLE = "applicable"
    NOT_APPLICABLE = "not_applicable"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    NOT_REQUESTED = "not_requested"


class TemporalVerdict(str, Enum):
    VERIFIED_APPLICABLE = "verified_applicable"
    VERIFIED_NOT_APPLICABLE = "verified_not_applicable"
    INSUFFICIENT_TEMPORAL_EVIDENCE = "insufficient_temporal_evidence"
    NOT_REQUESTED = "not_requested"


class RetrievalStatus(str, Enum):
    FOUND = "found"
    NO_RELEVANT_PASSAGES = "no_relevant_passages"


class IndexedPassage(BaseModel):
    passage_id: str
    document_id: str
    text: str
    section: Optional[str] = None
    page_number: Optional[int] = None
    passage_relevance_score: float
    title: Optional[str] = None
    issuer: Optional[str] = None
    source_url: Optional[str] = None
    source_tier: SourceTier
    corpus_use: CorpusUse
    published_at: Optional[date] = None
    effective_from: Optional[date] = None
    effective_to: Optional[date] = None
    valid_as_of: Optional[date] = None
    date_evidence: Optional[str] = None
    document_status: DocumentStatus
    source_passage_id: Optional[str] = None
    temporal_role: Optional[str] = None
    temporal_confidence: Optional[float] = None
    temporal_evidence: Optional[str] = None
    extraction_method: Optional[str] = None
    block_index: Optional[int] = None
    previous_passage_id: Optional[str] = None
    next_passage_id: Optional[str] = None
    temporal_scan_required: Optional[bool] = None
    temporal_scan_status: Optional[str] = None
    temporal_routing_reasons: list[str] = Field(default_factory=list)


class IndexBuildReport(BaseModel):
    approved_documents: int
    indexed_documents: int
    indexed_passages: int
    skipped_documents: list[str] = Field(default_factory=list)
    index_file: str


class RetrievalRequest(BaseModel):
    query: str = Field(min_length=1)
    relevant_date: Optional[date] = None
    top_k: int = Field(default=5, ge=1, le=20)


class RetrievalHit(BaseModel):
    passage: IndexedPassage
    score: float
    lexical_score: Optional[float] = None
    semantic_score: Optional[float] = None
    temporal_status: PassageTemporalStatus


class RetrievalResponse(BaseModel):
    query: str
    relevant_date: Optional[date] = None
    retrieval_status: RetrievalStatus
    temporal_verdict: TemporalVerdict
    ranking_method: str = "temporal_filter_then_hybrid"
    hits: list[RetrievalHit] = Field(default_factory=list)
