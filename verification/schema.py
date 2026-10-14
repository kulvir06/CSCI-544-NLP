from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from retrieval.schema import TemporalVerdict


class FactType(str, Enum):
    QUANTITY = "quantity"
    DATE = "date"
    FORM = "form"
    NEGATION = "negation"
    CONDITION = "condition"
    EXCEPTION = "exception"
    OBLIGATION = "obligation"


class FactStatus(str, Enum):
    PRESERVED = "preserved"
    MISSING = "missing"
    CONTRADICTED = "contradicted"


class MeaningVerdict(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    REVIEW = "review"


class SafetyVerdict(str, Enum):
    SAFE = "safe"
    UNSAFE = "unsafe"
    ABSTAIN = "abstain"


class AtomicFact(BaseModel):
    fact_id: str
    fact_type: FactType
    source_text: str
    normalized_value: Optional[str] = None


class FactCheck(BaseModel):
    fact: AtomicFact
    status: FactStatus
    lexical_coverage: float = Field(ge=0.0, le=1.0)
    explanation: str


class MeaningVerificationRequest(BaseModel):
    source_text: str = Field(min_length=1)
    explanation: str = Field(min_length=1)


class MeaningVerificationResponse(BaseModel):
    verdict: MeaningVerdict
    critical_fact_preservation: float = Field(ge=0.0, le=1.0)
    extracted_fact_count: int
    preserved_fact_count: int
    missing_fact_count: int
    contradicted_fact_count: int
    checks: list[FactCheck] = Field(default_factory=list)


class SafetyVerificationRequest(MeaningVerificationRequest):
    temporal_verdict: TemporalVerdict = TemporalVerdict.NOT_REQUESTED
    temporal_verification_required: bool = False


class SafetyVerificationResponse(BaseModel):
    verdict: SafetyVerdict
    temporal_verdict: TemporalVerdict
    meaning: MeaningVerificationResponse
    reasons: list[str] = Field(default_factory=list)
