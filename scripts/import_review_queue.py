"""Import team decisions from an exported metadata-review CSV."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import date
from pathlib import Path

from corpus.review import submit_review
from corpus.schema import (
    DocumentReviewRequest,
    DocumentStatus,
    MetadataCorrections,
    ReviewDecision,
)


TEXT_CORRECTIONS = ("title", "issuer", "source_url", "rule_family")
DATE_CORRECTIONS = ("published_at", "effective_from", "effective_to")


def _request_from_row(row: dict[str, str]) -> DocumentReviewRequest:
    corrections: dict[str, object] = {
        field: row[field].strip()
        for field in TEXT_CORRECTIONS
        if row.get(field, "").strip()
    }
    for field in DATE_CORRECTIONS:
        value = row.get(field, "").strip()
        if value:
            corrections[field] = date.fromisoformat(value)
    status = row.get("status", "").strip()
    if status:
        corrections["status"] = DocumentStatus(status)

    return DocumentReviewRequest(
        decision=ReviewDecision(row.get("decision", "").strip().lower()),
        reviewer=row.get("reviewer", "").strip(),
        corrections=MetadataCorrections.model_validate(corrections),
        notes=row.get("notes", "").strip() or None,
    )


def import_review_queue(input_path: Path) -> dict:
    input_path = input_path.expanduser().resolve()
    results: list[dict[str, str]] = []
    skipped_rows = 0

    with input_path.open(encoding="utf-8-sig", newline="") as input_file:
        for row_number, row in enumerate(csv.DictReader(input_file), start=2):
            decision = row.get("decision", "").strip()
            if not decision:
                skipped_rows += 1
                continue
            document_id = row.get("document_id", "").strip()
            try:
                review = submit_review(document_id, _request_from_row(row))
                results.append(
                    {
                        "row": str(row_number),
                        "document_id": document_id,
                        "outcome": review.decision.value,
                    }
                )
            except (ValueError, TypeError) as error:
                results.append(
                    {
                        "row": str(row_number),
                        "document_id": document_id,
                        "outcome": "failed",
                        "error": str(error),
                    }
                )

    return {
        "input_file": str(input_path),
        "processed_rows": len(results),
        "skipped_rows": skipped_rows,
        "failed_rows": sum(item["outcome"] == "failed" for item in results),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Import metadata-review decisions.")
    parser.add_argument("--input", required=True, type=Path)
    arguments = parser.parse_args()
    print(json.dumps(import_review_queue(arguments.input), indent=2))


if __name__ == "__main__":
    main()
