"""Export the local metadata review queue as a team-editable CSV file."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from api import config
from corpus.review import build_review_queue
from corpus.storage import ensure_data_directories


FIELDNAMES = (
    "document_id",
    "filename",
    "title",
    "issuer",
    "source_url",
    "published_at",
    "effective_from",
    "effective_to",
    "valid_as_of",
    "retrieved_at",
    "date_evidence",
    "rule_family",
    "status",
    "source_tier",
    "date_mentions",
    "suggested_effective_dates",
    "suggested_valid_as_of_dates",
    "meaning_blocking_issues",
    "temporal_blocking_issues",
    "decision",
    "corpus_use",
    "reviewer",
    "notes",
)


def export_review_queue(output_path: Path, include_resolved: bool = False) -> Path:
    ensure_data_directories()
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    queue = build_review_queue(include_resolved=include_resolved)

    with output_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDNAMES)
        writer.writeheader()
        for item in queue:
            metadata = item.current_metadata
            writer.writerow(
                {
                    "document_id": item.document.document_id,
                    "filename": metadata.original_filename,
                    "title": metadata.title or "",
                    "issuer": metadata.issuer or "",
                    "source_url": metadata.source_url or "",
                    "published_at": metadata.published_at or "",
                    "effective_from": metadata.effective_from or "",
                    "effective_to": metadata.effective_to or "",
                    "valid_as_of": metadata.valid_as_of or "",
                    "retrieved_at": metadata.retrieved_at or "",
                    "date_evidence": metadata.date_evidence or "",
                    "rule_family": metadata.rule_family or "",
                    "status": metadata.status.value,
                    "source_tier": item.current_source_tier.value,
                    "date_mentions": ";".join(
                        item.document.metadata_candidates.date_mentions
                    ),
                    "suggested_effective_dates": ";".join(
                        item.document.metadata_candidates.effective_date_candidates
                    ),
                    "suggested_valid_as_of_dates": ";".join(
                        item.document.metadata_candidates.valid_as_of_candidates
                    ),
                    "meaning_blocking_issues": ";".join(
                        item.meaning_blocking_issues
                    ),
                    "temporal_blocking_issues": ";".join(
                        item.temporal_blocking_issues
                    ),
                    "decision": item.decision.value if item.decision else "",
                    "corpus_use": item.corpus_use.value if item.corpus_use else "",
                    "reviewer": (
                        item.latest_review.reviewer if item.latest_review else ""
                    ),
                    "notes": item.latest_review.notes if item.latest_review else "",
                }
            )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export pending metadata reviews.")
    parser.add_argument(
        "--output",
        type=Path,
        default=config.REPORT_DIR / "metadata-review-queue.csv",
    )
    parser.add_argument("--include-resolved", action="store_true")
    arguments = parser.parse_args()
    output_path = export_review_queue(arguments.output, arguments.include_resolved)
    print(output_path)


if __name__ == "__main__":
    main()
