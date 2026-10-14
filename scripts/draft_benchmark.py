"""Draft a small, diverse benchmark from the approved passage index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from api import config
from benchmark.draft import draft_cases
from benchmark.review import write_drafts


def main() -> None:
    parser = argparse.ArgumentParser(description="Draft benchmark cases for review.")
    parser.add_argument("--max-per-error", type=int, default=2)
    parser.add_argument("--safe-controls", type=int, default=2)
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=config.REPORT_DIR / "benchmark-draft.jsonl",
    )
    parser.add_argument(
        "--review-csv",
        type=Path,
        default=config.REPORT_DIR / "benchmark-review.csv",
    )
    arguments = parser.parse_args()
    cases = draft_cases(arguments.max_per_error, arguments.safe_controls)
    jsonl_path, csv_path = write_drafts(
        cases,
        arguments.output_jsonl,
        arguments.review_csv,
    )
    print(
        json.dumps(
            {
                "drafted_cases": len(cases),
                "draft_jsonl": str(jsonl_path),
                "review_csv": str(csv_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

