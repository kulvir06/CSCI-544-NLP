"""Conservative atomic-fact checks for plain-language legal simplification.

This is a deterministic baseline. It is intentionally strict and sends
ambiguous paraphrases to review instead of treating them as safe.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
import re

from verification.schema import (
    AtomicFact,
    FactCheck,
    FactStatus,
    FactType,
    MeaningVerificationRequest,
    MeaningVerificationResponse,
    MeaningVerdict,
)


SENTENCE_PATTERN = re.compile(r"(?<=[.!?;])\s+")
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)?")
QUANTITY_PATTERN = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*[- ]?"
    r"(hours?|days?|weeks?|months?|years?|percent|%)\b",
    re.I,
)
FORM_PATTERN = re.compile(r"\bForm\s+(I-\d+[A-Z]?)\b", re.I)
DATE_PATTERN = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\s+\d{1,2},\s+\d{4}\b|"
    r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{4}\b",
    re.I,
)
NEGATION_PATTERN = re.compile(
    r"\b(must not|may not|cannot|can't|shall not|prohibited|ineligible|"
    r"no more than|not more than|only|up to|at most)\b",
    re.I,
)
CONDITION_PATTERN = re.compile(
    r"\b(if|only if|provided that|subject to|eligible|qualif(?:y|ies|ied)|"
    r"as long as|when)\b",
    re.I,
)
EXCEPTION_PATTERN = re.compile(
    r"\b(unless|except|exception|however|notwithstanding)\b",
    re.I,
)
OBLIGATION_PATTERN = re.compile(
    r"\b(must|required|shall|need(?:s)? to|has to)\b",
    re.I,
)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "for",
    "from", "has", "have", "in", "is", "it", "of", "on", "or", "that",
    "the", "their", "this", "to", "with",
}


def _sentences(text: str) -> list[str]:
    return [sentence.strip() for sentence in SENTENCE_PATTERN.split(text) if sentence.strip()]


def _tokens(text: str) -> set[str]:
    return {
        token.lower()
        for token in TOKEN_PATTERN.findall(text)
        if token.lower() not in STOPWORDS
    }


def _coverage(source: str, explanation: str) -> float:
    source_tokens = _tokens(source)
    if not source_tokens:
        return 1.0
    return len(source_tokens & _tokens(explanation)) / len(source_tokens)


def _normalize_date(value: str) -> str:
    for date_format in ("%B %d, %Y", "%Y-%m-%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(value.title(), date_format).date().isoformat()
        except ValueError:
            continue
    return value.lower()


def extract_atomic_facts(source_text: str) -> list[AtomicFact]:
    facts: list[AtomicFact] = []

    def add_fact(
        fact_type: FactType,
        sentence: str,
        normalized_value: str | None = None,
    ) -> None:
        facts.append(
            AtomicFact(
                fact_id=f"fact-{len(facts) + 1:03d}",
                fact_type=fact_type,
                source_text=sentence,
                normalized_value=normalized_value,
            )
        )

    for sentence in _sentences(source_text):
        for match in DATE_PATTERN.finditer(sentence):
            add_fact(FactType.DATE, sentence, _normalize_date(match.group(0)))
        for match in QUANTITY_PATTERN.finditer(sentence):
            number, unit = match.groups()
            normalized_unit = "percent" if unit == "%" else unit.lower().rstrip("s")
            add_fact(FactType.QUANTITY, sentence, f"{number} {normalized_unit}")
        for match in FORM_PATTERN.finditer(sentence):
            add_fact(FactType.FORM, sentence, match.group(1).upper())
        for fact_type, pattern in (
            (FactType.NEGATION, NEGATION_PATTERN),
            (FactType.CONDITION, CONDITION_PATTERN),
            (FactType.EXCEPTION, EXCEPTION_PATTERN),
            (FactType.OBLIGATION, OBLIGATION_PATTERN),
        ):
            if pattern.search(sentence):
                add_fact(fact_type, sentence)
    return facts


def _quantity_values(text: str) -> dict[str, set[str]]:
    values: dict[str, set[str]] = defaultdict(set)
    for number, unit in QUANTITY_PATTERN.findall(text):
        normalized_unit = "percent" if unit == "%" else unit.lower().rstrip("s")
        values[normalized_unit].add(number)
    return values


def _check_fact(fact: AtomicFact, explanation: str) -> FactCheck:
    coverage = _coverage(fact.source_text, explanation)
    status = FactStatus.MISSING
    reason = "The explanation does not preserve enough of this source fact."

    if fact.fact_type == FactType.QUANTITY and fact.normalized_value:
        number, unit = fact.normalized_value.split(" ", 1)
        explanation_values = _quantity_values(explanation)
        if number in explanation_values.get(unit, set()):
            status = FactStatus.PRESERVED
            reason = "The same quantity and unit appear in the explanation."
        elif explanation_values.get(unit):
            status = FactStatus.CONTRADICTED
            reason = "The explanation uses a different value for the same unit."
    elif fact.fact_type == FactType.DATE and fact.normalized_value:
        explanation_dates = {
            _normalize_date(value) for value in DATE_PATTERN.findall(explanation)
        }
        if fact.normalized_value in explanation_dates:
            status = FactStatus.PRESERVED
            reason = "The same calendar date appears in the explanation."
        elif explanation_dates:
            status = FactStatus.CONTRADICTED
            reason = "The explanation substitutes a different calendar date."
    elif fact.fact_type == FactType.FORM and fact.normalized_value:
        explanation_forms = {value.upper() for value in FORM_PATTERN.findall(explanation)}
        if fact.normalized_value in explanation_forms:
            status = FactStatus.PRESERVED
            reason = "The required form identifier is preserved."
        elif explanation_forms:
            status = FactStatus.CONTRADICTED
            reason = "The explanation substitutes a different form identifier."
    elif fact.fact_type == FactType.NEGATION:
        if NEGATION_PATTERN.search(explanation) and coverage >= 0.35:
            status = FactStatus.PRESERVED
            reason = "The limiting or negative meaning is preserved."
        elif coverage >= 0.35:
            status = FactStatus.CONTRADICTED
            reason = "Related content appears, but the limiting or negative meaning is lost."
    else:
        cue_pattern = {
            FactType.CONDITION: CONDITION_PATTERN,
            FactType.EXCEPTION: EXCEPTION_PATTERN,
            FactType.OBLIGATION: OBLIGATION_PATTERN,
        }[fact.fact_type]
        if cue_pattern.search(explanation) and coverage >= 0.35:
            status = FactStatus.PRESERVED
            reason = f"The {fact.fact_type.value} and its key content are preserved."
        elif coverage >= 0.55:
            status = FactStatus.MISSING
            reason = f"Related content appears, but the {fact.fact_type.value} cue is missing."

    return FactCheck(
        fact=fact,
        status=status,
        lexical_coverage=round(coverage, 4),
        explanation=reason,
    )


def verify_meaning(
    request: MeaningVerificationRequest,
) -> MeaningVerificationResponse:
    facts = extract_atomic_facts(request.source_text)
    checks = [_check_fact(fact, request.explanation) for fact in facts]
    preserved = sum(check.status == FactStatus.PRESERVED for check in checks)
    missing = sum(check.status == FactStatus.MISSING for check in checks)
    contradicted = sum(check.status == FactStatus.CONTRADICTED for check in checks)
    preservation = preserved / len(checks) if checks else 0.0

    if not checks:
        verdict = MeaningVerdict.REVIEW
    elif contradicted or missing:
        verdict = MeaningVerdict.FAIL
    else:
        verdict = MeaningVerdict.PASS

    return MeaningVerificationResponse(
        verdict=verdict,
        critical_fact_preservation=round(preservation, 4),
        extracted_fact_count=len(checks),
        preserved_fact_count=preserved,
        missing_fact_count=missing,
        contradicted_fact_count=contradicted,
        checks=checks,
    )
