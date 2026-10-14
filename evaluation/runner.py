"""Run four verification configurations over a targeted case set."""

from __future__ import annotations

import json
from pathlib import Path

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


def _temporal_prediction(case: EvaluationCase) -> SafetyVerdict:
    if not case.temporal_verification_required:
        return SafetyVerdict.SAFE
    if case.observed_temporal_verdict == TemporalVerdict.VERIFIED_APPLICABLE:
        return SafetyVerdict.SAFE
    if case.observed_temporal_verdict == TemporalVerdict.INSUFFICIENT_TEMPORAL_EVIDENCE:
        return SafetyVerdict.ABSTAIN
    return SafetyVerdict.UNSAFE


def evaluate_case(case: EvaluationCase) -> CaseEvaluationResult:
    meaning = verify_meaning(
        MeaningVerificationRequest(
            source_text=case.source_text,
            explanation=case.explanation,
        )
    )
    combined = verify_safety(
        SafetyVerificationRequest(
            source_text=case.source_text,
            explanation=case.explanation,
            temporal_verdict=case.observed_temporal_verdict,
            temporal_verification_required=case.temporal_verification_required,
        )
    )
    return CaseEvaluationResult(
        case_id=case.case_id,
        error_type=case.error_type,
        gold_label=case.gold_label,
        meaning_verdict=meaning.verdict,
        critical_fact_preservation=meaning.critical_fact_preservation,
        predictions={
            SystemConfiguration.STATIC_BASELINE: SafetyVerdict.SAFE,
            SystemConfiguration.TEMPORAL_ONLY: _temporal_prediction(case),
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

    return SystemMetrics(
        total_cases=total,
        exact_accuracy=round(exact / total, 4) if total else 0.0,
        false_safe_rate=round(false_safe / len(unsafe_cases), 4) if unsafe_cases else 0.0,
        unsafe_detection_rate=(
            round(unsafe_detected / len(unsafe_cases), 4) if unsafe_cases else 0.0
        ),
        safe_acceptance_rate=(
            round(safe_accepted / len(safe_cases), 4) if safe_cases else 0.0
        ),
        abstention_rate=round(abstained / total, 4) if total else 0.0,
        per_error=grouped,
    )


def run_evaluation(cases: list[EvaluationCase]) -> EvaluationReport:
    results = [evaluate_case(case) for case in cases]
    temporal_cases = [case for case in cases if case.temporal_verification_required]
    temporal_correct = sum(
        case.observed_temporal_verdict == case.expected_temporal_verdict
        for case in temporal_cases
    )
    mean_preservation = (
        sum(result.critical_fact_preservation for result in results) / len(results)
        if results
        else 0.0
    )
    dataset_status = (
        "development_fixtures"
        if any(case.development_fixture for case in cases)
        else "reviewed_benchmark"
    )
    return EvaluationReport(
        dataset_status=dataset_status,
        total_cases=len(cases),
        temporal_verdict_accuracy=(
            round(temporal_correct / len(temporal_cases), 4)
            if temporal_cases
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

