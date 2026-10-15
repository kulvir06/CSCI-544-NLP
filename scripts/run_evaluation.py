"""Run the four-configuration targeted safety evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

from api import config
from evaluation.runner import run_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Run targeted verifier evaluation.")
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=config.REPORT_DIR / "evaluation-report.json",
    )
    arguments = parser.parse_args()
    report = run_file(arguments.cases, arguments.output)
    print(report.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
