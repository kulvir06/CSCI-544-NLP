from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator

from retrieval.schema import TemporalVerdict
from verification.schema import MeaningVerdict, SafetyVerdict


class ErrorType(str, Enum):
    SAFE = "safe"
    WRONG_VERSION = "wrong_version"
    TEMPORAL_UNCERTAINTY = "temporal_uncertainty"
    QUANTITY_CHANGED = "quantity_changed"
    DATE_CHANGED = "date_changed"
    NEGATION_REVERSED = "negation_reversed"
    CONDITION_OMITTED = "condition_omitted"
    EXCEPTION_OMITTED = "exception_omitted"
    FORM_CHANGED = "form_changed"


class GoldLabel(str, Enum):
    SAFE = "safe"
    UNSAFE = "unsafe"
    ABSTAIN = "abstain"


class SystemConfiguration(str, Enum):
    STATIC_BASELINE = "static_baseline"
    TEMPORAL_ONLY = "temporal_only"
    MEANING_ONLY = "meaning_only"
    COMBINED = "combined"


class EvaluationCase(BaseModel):
    case_id: str
    source_text: str = Field(min_length=1)
    explanation: str = Field(min_length=1)
    error_type: ErrorType
    gold_label: GoldLabel
    temporal_verification_required: bool = False
    observed_temporal_verdict: TemporalVerdict = TemporalVerdict.NOT_REQUESTED
    expected_temporal_verdict: TemporalVerdict = TemporalVerdict.NOT_REQUESTED
    relevant_date: Optional[date] = None
    document_id: Optional[str] = None
    passage_id: Optional[str] = None
    page_number: Optional[int] = None
    source_url: Optional[str] = None
    reviewer: Optional[str] = None
    mutation_description: Optional[str] = None
    review_notes: Optional[str] = None
    development_fixture: bool = True

    @model_validator(mode="after")
    def reviewed_cases_require_provenance(self) -> "EvaluationCase":
        if not self.development_fixture:
            missing = [
                name
                for name, value in (
                    ("document_id", self.document_id),
                    ("passage_id", self.passage_id),
                    ("source_url", self.source_url),
                    ("reviewer", self.reviewer),
                )
                if not value
            ]
            if missing:
                raise ValueError(
                    "Reviewed benchmark case is missing: " + ", ".join(missing)
                )
        return self


class CaseEvaluationResult(BaseModel):
    case_id: str
    error_type: ErrorType
    gold_label: GoldLabel
    meaning_verdict: MeaningVerdict
    critical_fact_preservation: float
    predictions: dict[SystemConfiguration, SafetyVerdict]


class ErrorMetrics(BaseModel):
    count: int
    correct: int
    unsafe_detected: int
    false_safe: int
    abstained: int


class SystemMetrics(BaseModel):
    total_cases: int
    exact_accuracy: float
    false_safe_rate: float
    unsafe_detection_rate: float
    safe_acceptance_rate: float
    abstention_rate: float
    per_error: dict[ErrorType, ErrorMetrics]


class EvaluationReport(BaseModel):
    dataset_status: str
    total_cases: int
    temporal_verdict_accuracy: float
    mean_critical_fact_preservation: float
    systems: dict[SystemConfiguration, SystemMetrics]
    cases: list[CaseEvaluationResult]
