"""Export benchmark drafts for review and finalize only signed-off cases."""

from __future__ import annotations

import csv
from pathlib import Path

from evaluation.runner import load_cases
from evaluation.schema import EvaluationCase


REVIEW_FIELDS = (
    "case_id",
    "document_id",
    "passage_id",
    "page_number",
    "source_url",
    "error_type",
    "gold_label",
    "relevant_date",
    "mutation_description",
    "source_text",
    "explanation",
    "approve",
    "reviewer",
    "review_notes",
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
                    "document_id": case.document_id or "",
                    "passage_id": case.passage_id or "",
                    "page_number": case.page_number or "",
                    "source_url": case.source_url or "",
                    "error_type": case.error_type.value,
                    "gold_label": case.gold_label.value,
                    "relevant_date": case.relevant_date or "",
                    "mutation_description": case.mutation_description or "",
                    "source_text": case.source_text,
                    "explanation": case.explanation,
                    "approve": "",
                    "reviewer": "",
                    "review_notes": "",
                }
            )
    return jsonl_path, review_csv_path


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

