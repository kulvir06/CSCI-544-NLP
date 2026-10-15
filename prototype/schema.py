from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator

from retrieval.schema import RetrievalResponse
from verification.schema import SafetyVerificationResponse


class GeneratorMode(str, Enum):
    EXTRACTIVE_BASELINE = "extractive_baseline"
    PROVIDED_CANDIDATE = "provided_candidate"


class PrototypeStatus(str, Enum):
    ANSWERED = "answered"
    UNSAFE_BLOCKED = "unsafe_blocked"
    ABSTAINED = "abstained"
    NO_EVIDENCE = "no_evidence"


class PrototypeRequest(BaseModel):
    question: str = Field(min_length=1)
    relevant_date: Optional[date] = None
    top_k: int = Field(default=5, ge=1, le=20)
    generator_mode: GeneratorMode = GeneratorMode.EXTRACTIVE_BASELINE
    candidate_explanation: Optional[str] = None

    @model_validator(mode="after")
    def candidate_required_for_provided_mode(self) -> "PrototypeRequest":
        if (
            self.generator_mode == GeneratorMode.PROVIDED_CANDIDATE
            and not (self.candidate_explanation or "").strip()
        ):
            raise ValueError(
                "candidate_explanation is required for provided_candidate mode"
            )
        return self


class Citation(BaseModel):
    document_id: str
    passage_id: str
    title: Optional[str] = None
    issuer: Optional[str] = None
    source_url: Optional[str] = None
    page_number: Optional[int] = None


class PrototypeResponse(BaseModel):
    status: PrototypeStatus
    question: str
    relevant_date: Optional[date] = None
    generator_mode: GeneratorMode
    answer: Optional[str] = None
    draft_explanation: Optional[str] = None
    retrieval: RetrievalResponse
    verification: Optional[SafetyVerificationResponse] = None
    citations: list[Citation] = Field(default_factory=list)
    message: str

