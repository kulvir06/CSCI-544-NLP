"""Run four verification configurations over a targeted case set."""

from __future__ import annotations

import json
from pathlib import Path
import math

from evaluation.schema import (
    CaseEvaluationResult,
    ErrorMetrics,
    EvaluationCase,
    EvaluationReport,
    GoldLabel,
    SystemConfiguration,
    SystemMetrics,
)
from retrieval.schema import TemporalVerdict
from retrieval.search import verify_source_temporality
from verification.meaning import verify_meaning
from verification.safety import verify_safety
from verification.schema import (
    MeaningVerificationRequest,
    MeaningVerdict,
    SafetyVerificationRequest,
    SafetyVerdict,
)


def load_cases(path: Path) -> list[EvaluationCase]:
    return [
        EvaluationCase.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _meaning_prediction(verdict: MeaningVerdict) -> SafetyVerdict:
    if verdict == MeaningVerdict.PASS:
        return SafetyVerdict.SAFE
    if verdict == MeaningVerdict.FAIL:
        return SafetyVerdict.UNSAFE
    return SafetyVerdict.ABSTAIN


def _temporal_prediction(
    case: EvaluationCase, temporal_verdict: TemporalVerdict
) -> SafetyVerdict:
    if not case.temporal_verification_required:
        return SafetyVerdict.SAFE
    if temporal_verdict == TemporalVerdict.VERIFIED_APPLICABLE:
        return SafetyVerdict.SAFE
    if temporal_verdict == TemporalVerdict.INSUFFICIENT_TEMPORAL_EVIDENCE:
        return SafetyVerdict.ABSTAIN
    return SafetyVerdict.UNSAFE


def _temporal_verdict(case: EvaluationCase) -> tuple[TemporalVerdict, str | None]:
    if not case.automatic_temporal_verification:
        return case.observed_temporal_verdict, None
    if not case.passage_id or not case.relevant_date:
        return TemporalVerdict.INSUFFICIENT_TEMPORAL_EVIDENCE, None
    return verify_source_temporality(
        case.passage_id,
        case.source_text,
        case.relevant_date,
    )


def evaluate_case(case: EvaluationCase) -> CaseEvaluationResult:
    temporal_verdict, matched_claim_id = _temporal_verdict(case)
    meaning = verify_meaning(
        MeaningVerificationRequest(
            source_text=case.source_text,
            explanation=case.explanation,
            critical_facts=case.atomic_fact_checklist,
        )
    )
    combined = verify_safety(
        SafetyVerificationRequest(
            source_text=case.source_text,
            explanation=case.explanation,
            temporal_verdict=temporal_verdict,
            temporal_verification_required=case.temporal_verification_required,
            critical_facts=case.atomic_fact_checklist,
        )
    )
    return CaseEvaluationResult(
        case_id=case.case_id,
        error_type=case.error_type,
        gold_label=case.gold_label,
        meaning_verdict=meaning.verdict,
        critical_fact_preservation=meaning.critical_fact_preservation,
        observed_temporal_verdict=temporal_verdict,
        expected_temporal_verdict=case.expected_temporal_verdict,
        matched_temporal_claim_id=matched_claim_id,
        predictions={
            SystemConfiguration.STATIC_BASELINE: SafetyVerdict.SAFE,
            SystemConfiguration.TEMPORAL_ONLY: _temporal_prediction(
                case, temporal_verdict
            ),
            SystemConfiguration.MEANING_ONLY: _meaning_prediction(meaning.verdict),
            SystemConfiguration.COMBINED: combined.verdict,
        },
    )


def _metrics_for_system(
    results: list[CaseEvaluationResult],
    system: SystemConfiguration,
) -> SystemMetrics:
    total = len(results)
    unsafe_cases = [result for result in results if result.gold_label == GoldLabel.UNSAFE]
    safe_cases = [result for result in results if result.gold_label == GoldLabel.SAFE]
    exact = sum(result.predictions[system].value == result.gold_label.value for result in results)
    false_safe = sum(
        result.predictions[system] == SafetyVerdict.SAFE for result in unsafe_cases
    )
    unsafe_detected = sum(
        result.predictions[system] == SafetyVerdict.UNSAFE for result in unsafe_cases
    )
    safe_accepted = sum(
        result.predictions[system] == SafetyVerdict.SAFE for result in safe_cases
    )
    abstained = sum(
        result.predictions[system] == SafetyVerdict.ABSTAIN for result in results
    )

    grouped: dict = {}
    for error_type in sorted({result.error_type for result in results}, key=lambda item: item.value):
        group = [result for result in results if result.error_type == error_type]
        grouped[error_type] = ErrorMetrics(
            count=len(group),
            correct=sum(
                result.predictions[system].value == result.gold_label.value
                for result in group
            ),
            unsafe_detected=sum(
                result.predictions[system] == SafetyVerdict.UNSAFE for result in group
            ),
            false_safe=sum(
                result.gold_label == GoldLabel.UNSAFE
                and result.predictions[system] == SafetyVerdict.SAFE
                for result in group
            ),
            abstained=sum(
                result.predictions[system] == SafetyVerdict.ABSTAIN for result in group
            ),
        )

    def wilson(successes: int, sample_size: int) -> tuple[float, float]:
        if sample_size == 0:
            return (0.0, 0.0)
        z = 1.959963984540054
        proportion = successes / sample_size
        denominator = 1 + z * z / sample_size
        center = (proportion + z * z / (2 * sample_size)) / denominator
        margin = (
            z
            * math.sqrt(
                proportion * (1 - proportion) / sample_size
                + z * z / (4 * sample_size * sample_size)
            )
            / denominator
        )
        return (round(max(0.0, center - margin), 4), round(min(1.0, center + margin), 4))

    return SystemMetrics(
        total_cases=total,
        unsafe_cases=len(unsafe_cases),
        safe_cases=len(safe_cases),
        false_safe_count=false_safe,
        unsafe_detected_count=unsafe_detected,
        safe_accepted_count=safe_accepted,
        abstained_count=abstained,
        exact_accuracy=round(exact / total, 4) if total else 0.0,
        false_safe_rate=round(false_safe / len(unsafe_cases), 4) if unsafe_cases else 0.0,
        false_safe_wilson_95=wilson(false_safe, len(unsafe_cases)),
        unsafe_detection_rate=(
            round(unsafe_detected / len(unsafe_cases), 4) if unsafe_cases else 0.0
        ),
        safe_acceptance_rate=(
            round(safe_accepted / len(safe_cases), 4) if safe_cases else 0.0
        ),
        safe_acceptance_wilson_95=wilson(safe_accepted, len(safe_cases)),
        abstention_rate=round(abstained / total, 4) if total else 0.0,
        per_error=grouped,
    )


def run_evaluation(cases: list[EvaluationCase]) -> EvaluationReport:
    results = [evaluate_case(case) for case in cases]
    temporal_results = [
        result
        for case, result in zip(cases, results)
        if case.temporal_verification_required
    ]
    temporal_correct = sum(
        result.observed_temporal_verdict == result.expected_temporal_verdict
        for result in temporal_results
    )
    mean_preservation = (
        sum(result.critical_fact_preservation for result in results) / len(results)
        if results
        else 0.0
    )
    if not any(case.development_fixture for case in cases):
        dataset_status = "reviewed_benchmark"
    elif any(case.automatic_temporal_verification for case in cases):
        dataset_status = "benchmark_candidates"
    else:
        dataset_status = "development_fixtures"
    return EvaluationReport(
        dataset_status=dataset_status,
        total_cases=len(cases),
        temporal_verdict_accuracy=(
            round(temporal_correct / len(temporal_results), 4)
            if temporal_results
            else 0.0
        ),
        mean_critical_fact_preservation=round(mean_preservation, 4),
        systems={
            system: _metrics_for_system(results, system)
            for system in SystemConfiguration
        },
        cases=results,
    )


def save_report(report: EvaluationReport, output_path: Path) -> Path:
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return output_path


def run_file(input_path: Path, output_path: Path) -> EvaluationReport:
    report = run_evaluation(load_cases(input_path))
    save_report(report, output_path)
    return report
