"""Export benchmark drafts for review and finalize only signed-off cases."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from evaluation.runner import load_cases
from evaluation.schema import EvaluationCase


REVIEW_FIELDS = (
    "case_id",
    "benchmark_split",
    "question",
    "document_id",
    "passage_id",
    "page_number",
    "source_url",
    "error_type",
    "gold_label",
    "expected_temporal_verdict",
    "relevant_date",
    "mutation_description",
    "source_text",
    "explanation",
    "approve",
    "reviewer",
    "review_notes",
    "atomic_fact_checklist",
)


def write_drafts(
    cases: list[EvaluationCase],
    jsonl_path: Path,
    review_csv_path: Path,
) -> tuple[Path, Path]:
    jsonl_path = jsonl_path.expanduser().resolve()
    review_csv_path = review_csv_path.expanduser().resolve()
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    review_csv_path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(case.model_dump_json() for case in cases)
    jsonl_path.write_text(f"{body}\n" if body else "", encoding="utf-8")

    with review_csv_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        for case in cases:
            writer.writerow(
                {
                    "case_id": case.case_id,
                    "benchmark_split": (
                        case.benchmark_split.value if case.benchmark_split else ""
                    ),
                    "question": case.question or "",
                    "document_id": case.document_id or "",
                    "passage_id": case.passage_id or "",
                    "page_number": case.page_number or "",
                    "source_url": case.source_url or "",
                    "error_type": case.error_type.value,
                    "gold_label": case.gold_label.value,
                    "expected_temporal_verdict": case.expected_temporal_verdict.value,
                    "relevant_date": case.relevant_date or "",
                    "mutation_description": case.mutation_description or "",
                    "source_text": case.source_text,
                    "explanation": case.explanation,
                    "approve": "",
                    "reviewer": "",
                    "review_notes": "",
                    "atomic_fact_checklist": " | ".join(
                        case.atomic_fact_checklist
                    ),
                }
            )
    return jsonl_path, review_csv_path


def _read_review_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as input_file:
        rows = list(csv.DictReader(input_file))
    return {
        row.get("case_id", "").strip(): row
        for row in rows
        if row.get("case_id", "").strip()
    }


def _approved(row: dict[str, str]) -> bool:
    return row.get("approve", "").strip().lower() in {
        "yes", "y", "true", "approved"
    }


def _cohen_kappa(left: list[str], right: list[str]) -> float | None:
    if not left or len(left) != len(right):
        return None
    observed = sum(a == b for a, b in zip(left, right)) / len(left)
    labels = set(left) | set(right)
    left_counts = Counter(left)
    right_counts = Counter(right)
    expected = sum(
        (left_counts[label] / len(left)) * (right_counts[label] / len(right))
        for label in labels
    )
    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return round((observed - expected) / (1 - expected), 4)


def finalize_independently_reviewed_cases(
    draft_jsonl_path: Path,
    review_a_path: Path,
    review_b_path: Path,
    output_jsonl_path: Path,
    disagreement_csv_path: Path,
) -> dict:
    """Finalize only cases independently approved with matching labels."""
    drafts = {case.case_id: case for case in load_cases(draft_jsonl_path)}
    rows_a = _read_review_rows(review_a_path)
    rows_b = _read_review_rows(review_b_path)
    finalized: list[EvaluationCase] = []
    disagreements: list[dict[str, str]] = []
    gold_a: list[str] = []
    gold_b: list[str] = []
    jointly_reviewed = 0

    compared_fields = (
        "approve", "gold_label", "error_type", "expected_temporal_verdict"
    )
    for case_id, draft in drafts.items():
        row_a = rows_a.get(case_id)
        row_b = rows_b.get(case_id)
        if not row_a or not row_b:
            disagreements.append(
                {"case_id": case_id, "reason": "missing_reviewer_row"}
            )
            continue
        reviewer_a = row_a.get("reviewer", "").strip()
        reviewer_b = row_b.get("reviewer", "").strip()
        if not reviewer_a or not reviewer_b or reviewer_a == reviewer_b:
            disagreements.append(
                {"case_id": case_id, "reason": "reviewers_missing_or_not_independent"}
            )
            continue
        jointly_reviewed += 1
        gold_a.append(row_a.get("gold_label", "").strip())
        gold_b.append(row_b.get("gold_label", "").strip())
        mismatches = [
            field for field in compared_fields
            if (
                (_approved(row_a) if field == "approve" else row_a.get(field, "").strip())
                != (_approved(row_b) if field == "approve" else row_b.get(field, "").strip())
            )
        ]
        if mismatches or not (_approved(row_a) and _approved(row_b)):
            disagreements.append(
                {
                    "case_id": case_id,
                    "reason": "label_disagreement" if mismatches else "not_jointly_approved",
                    "mismatched_fields": " | ".join(mismatches),
                    "reviewer_a": reviewer_a,
                    "reviewer_b": reviewer_b,
                    "a_gold_label": row_a.get("gold_label", ""),
                    "b_gold_label": row_b.get("gold_label", ""),
                    "a_error_type": row_a.get("error_type", ""),
                    "b_error_type": row_b.get("error_type", ""),
                    "a_expected_temporal_verdict": row_a.get("expected_temporal_verdict", ""),
                    "b_expected_temporal_verdict": row_b.get("expected_temporal_verdict", ""),
                }
            )
            continue
        finalized.append(
            EvaluationCase.model_validate(
                {
                    **draft.model_dump(),
                    "gold_label": row_a["gold_label"].strip(),
                    "error_type": row_a["error_type"].strip(),
                    "expected_temporal_verdict": row_a[
                        "expected_temporal_verdict"
                    ].strip(),
                    "reviewer": f"{reviewer_a}; {reviewer_b}",
                    "review_notes": " | ".join(
                        value for value in (
                            row_a.get("review_notes", "").strip(),
                            row_b.get("review_notes", "").strip(),
                        ) if value
                    ) or None,
                    "development_fixture": False,
                }
            )
        )

    disagreement_csv_path.parent.mkdir(parents=True, exist_ok=True)
    disagreement_fields = sorted(
        {key for row in disagreements for key in row} | {"case_id", "reason"}
    )
    with disagreement_csv_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=disagreement_fields)
        writer.writeheader()
        writer.writerows(disagreements)

    output_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(case.model_dump_json() for case in finalized)
    output_jsonl_path.write_text(f"{body}\n" if body else "", encoding="utf-8")
    case_agreement = (
        (jointly_reviewed - sum(row.get("reason") == "label_disagreement" for row in disagreements))
        / jointly_reviewed
        if jointly_reviewed else 0.0
    )
    return {
        "draft_cases": len(drafts),
        "jointly_reviewed": jointly_reviewed,
        "finalized_cases": len(finalized),
        "disagreements": len(disagreements),
        "raw_case_agreement": round(case_agreement, 4),
        "gold_label_cohen_kappa": _cohen_kappa(gold_a, gold_b),
        "output_file": str(output_jsonl_path),
        "disagreement_file": str(disagreement_csv_path),
    }


def finalize_reviewed_cases(
    draft_jsonl_path: Path,
    review_csv_path: Path,
    output_jsonl_path: Path,
) -> dict:
    drafts = {case.case_id: case for case in load_cases(draft_jsonl_path)}
    finalized: list[EvaluationCase] = []
    failures: list[dict[str, str]] = []
    reviewed_rows = 0

    with review_csv_path.open(encoding="utf-8-sig", newline="") as input_file:
        for row_number, row in enumerate(csv.DictReader(input_file), start=2):
            approval = row.get("approve", "").strip().lower()
            if approval not in {"yes", "y", "true", "approved"}:
                continue
            reviewed_rows += 1
            case_id = row.get("case_id", "").strip()
            try:
                if case_id not in drafts:
                    raise ValueError("Case ID does not exist in the draft JSONL")
                reviewer = row.get("reviewer", "").strip()
                if not reviewer:
                    raise ValueError("Approved case requires a reviewer")
                finalized.append(
                    EvaluationCase.model_validate(
                        {
                            **drafts[case_id].model_dump(),
                            "reviewer": reviewer,
                            "review_notes": row.get("review_notes", "").strip() or None,
                            "development_fixture": False,
                        }
                    )
                )
            except ValueError as error:
                failures.append(
                    {
                        "row": str(row_number),
                        "case_id": case_id,
                        "error": str(error),
                    }
                )

    output_jsonl_path = output_jsonl_path.expanduser().resolve()
    output_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(case.model_dump_json() for case in finalized)
    output_jsonl_path.write_text(f"{body}\n" if body else "", encoding="utf-8")
    return {
        "reviewed_rows": reviewed_rows,
        "finalized_cases": len(finalized),
        "failed_rows": len(failures),
        "failures": failures,
        "output_file": str(output_jsonl_path),
    }
