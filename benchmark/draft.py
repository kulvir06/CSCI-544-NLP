"""Automatically draft targeted benchmark cases from approved passages."""

from __future__ import annotations

from datetime import date, timedelta
import re

from corpus.schema import CorpusUse
from evaluation.mutations import (
    change_first_date,
    change_first_form,
    change_first_quantity,
    remove_clause_from_cue,
    reverse_first_negation,
)
from evaluation.schema import ErrorType, EvaluationCase, GoldLabel
from retrieval.index import load_index
from retrieval.schema import IndexedPassage, TemporalVerdict
from verification.meaning import extract_atomic_facts
from verification.schema import FactType


FACT_ERROR_MAP = {
    FactType.QUANTITY: ErrorType.QUANTITY_CHANGED,
    FactType.DATE: ErrorType.DATE_CHANGED,
    FactType.FORM: ErrorType.FORM_CHANGED,
    FactType.NEGATION: ErrorType.NEGATION_REVERSED,
    FactType.CONDITION: ErrorType.CONDITION_OMITTED,
    FactType.EXCEPTION: ErrorType.EXCEPTION_OMITTED,
}
CONDITION_CUES = ("only if", "provided that", "subject to", "as long as", "if", "when")
EXCEPTION_CUES = ("unless", "except", "however", "notwithstanding")


def _temporal_context(
    passage: IndexedPassage,
) -> tuple[bool, TemporalVerdict, date | None]:
    if passage.corpus_use != CorpusUse.TEMPORAL_AND_MEANING:
        return False, TemporalVerdict.NOT_REQUESTED, None
    relevant_date = passage.effective_from or passage.valid_as_of
    return True, TemporalVerdict.VERIFIED_APPLICABLE, relevant_date


def _remove_first_matching_cue(text: str, cues: tuple[str, ...]) -> str:
    lower = text.lower()
    for cue in cues:
        if re.search(rf"\b{re.escape(cue)}\b", lower):
            return remove_clause_from_cue(text, cue)
    raise ValueError("No supported clause cue found")


def _mutate(text: str, fact_type: FactType) -> tuple[str, str]:
    if fact_type == FactType.QUANTITY:
        return change_first_quantity(text), "Changed the first quantity by one unit."
    if fact_type == FactType.DATE:
        return change_first_date(text), "Moved the first calendar date by one day."
    if fact_type == FactType.FORM:
        return change_first_form(text), "Substituted the first form identifier."
    if fact_type == FactType.NEGATION:
        return reverse_first_negation(text), "Reversed the first negative constraint."
    if fact_type == FactType.CONDITION:
        return (
            _remove_first_matching_cue(text, CONDITION_CUES),
            "Removed the first conditional clause.",
        )
    if fact_type == FactType.EXCEPTION:
        return (
            _remove_first_matching_cue(text, EXCEPTION_CUES),
            "Removed the first exception clause.",
        )
    raise ValueError(f"Unsupported fact type: {fact_type}")


def _base_case(
    passage: IndexedPassage,
    case_id: str,
    explanation: str,
    error_type: ErrorType,
    gold_label: GoldLabel,
    mutation_description: str,
) -> EvaluationCase:
    temporal_required, temporal_verdict, relevant_date = _temporal_context(passage)
    return EvaluationCase(
        case_id=case_id,
        source_text=passage.text,
        explanation=explanation,
        error_type=error_type,
        gold_label=gold_label,
        temporal_verification_required=temporal_required,
        observed_temporal_verdict=temporal_verdict,
        expected_temporal_verdict=temporal_verdict,
        relevant_date=relevant_date,
        document_id=passage.document_id,
        passage_id=passage.passage_id,
        page_number=passage.page_number,
        source_url=passage.source_url,
        mutation_description=mutation_description,
        development_fixture=True,
    )


def draft_cases(
    max_per_error: int = 2,
    safe_controls: int = 2,
) -> list[EvaluationCase]:
    passages = sorted(
        load_index(),
        key=lambda item: (
            len(item.text),
            -item.passage_relevance_score,
            item.passage_id,
        ),
    )
    counts = {error_type: 0 for error_type in ErrorType}
    cases: list[EvaluationCase] = []

    for passage in passages:
        if counts[ErrorType.SAFE] < safe_controls:
            cases.append(
                _base_case(
                    passage,
                    f"draft-safe-{passage.passage_id}",
                    passage.text,
                    ErrorType.SAFE,
                    GoldLabel.SAFE,
                    "Unmodified source control.",
                )
            )
            counts[ErrorType.SAFE] += 1

        seen_fact_types: set[FactType] = set()
        for fact in extract_atomic_facts(passage.text):
            if fact.fact_type not in FACT_ERROR_MAP or fact.fact_type in seen_fact_types:
                continue
            seen_fact_types.add(fact.fact_type)
            error_type = FACT_ERROR_MAP[fact.fact_type]
            if counts[error_type] >= max_per_error:
                continue
            try:
                explanation, description = _mutate(passage.text, fact.fact_type)
            except ValueError:
                continue
            cases.append(
                _base_case(
                    passage,
                    f"draft-{error_type.value}-{passage.passage_id}",
                    explanation,
                    error_type,
                    GoldLabel.UNSAFE,
                    description,
                )
            )
            counts[error_type] += 1

        if passage.corpus_use == CorpusUse.TEMPORAL_AND_MEANING:
            if counts[ErrorType.WRONG_VERSION] < max_per_error and passage.effective_from:
                case = _base_case(
                    passage,
                    f"draft-wrong-version-{passage.passage_id}",
                    passage.text,
                    ErrorType.WRONG_VERSION,
                    GoldLabel.UNSAFE,
                    "Kept the meaning intact but selected a date before this version applies.",
                )
                case.relevant_date = passage.effective_from - timedelta(days=1)
                case.observed_temporal_verdict = TemporalVerdict.VERIFIED_NOT_APPLICABLE
                case.expected_temporal_verdict = TemporalVerdict.VERIFIED_NOT_APPLICABLE
                cases.append(case)
                counts[ErrorType.WRONG_VERSION] += 1
            elif (
                counts[ErrorType.TEMPORAL_UNCERTAINTY] < max_per_error
                and passage.valid_as_of
                and not passage.effective_from
            ):
                case = _base_case(
                    passage,
                    f"draft-temporal-uncertainty-{passage.passage_id}",
                    passage.text,
                    ErrorType.TEMPORAL_UNCERTAINTY,
                    GoldLabel.ABSTAIN,
                    "Queried one day beyond a single-date snapshot.",
                )
                case.relevant_date = passage.valid_as_of + timedelta(days=1)
                case.observed_temporal_verdict = (
                    TemporalVerdict.INSUFFICIENT_TEMPORAL_EVIDENCE
                )
                case.expected_temporal_verdict = (
                    TemporalVerdict.INSUFFICIENT_TEMPORAL_EVIDENCE
                )
                cases.append(case)
                counts[ErrorType.TEMPORAL_UNCERTAINTY] += 1

    return cases
