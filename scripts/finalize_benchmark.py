"""Finalize only benchmark cases explicitly approved in the review CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark.review import (
    finalize_independently_reviewed_cases,
    finalize_reviewed_cases,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Finalize reviewed benchmark cases.")
    parser.add_argument("--draft-jsonl", required=True, type=Path)
    parser.add_argument("--review-csv", required=True, type=Path)
    parser.add_argument("--review-csv-b", type=Path)
    parser.add_argument("--disagreements", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    if arguments.review_csv_b:
        report = finalize_independently_reviewed_cases(
            arguments.draft_jsonl,
            arguments.review_csv,
            arguments.review_csv_b,
            arguments.output,
            arguments.disagreements
            or arguments.output.with_name("benchmark-disagreements.csv"),
        )
    else:
        report = finalize_reviewed_cases(
            arguments.draft_jsonl,
            arguments.review_csv,
            arguments.output,
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
