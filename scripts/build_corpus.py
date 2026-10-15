"""Build the corpus from a directory of user-supplied local documents."""

from __future__ import annotations

import argparse
from pathlib import Path

from corpus.batch import ingest_directory


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest a local directory without downloading or scraping anything."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="Directory containing user-supplied PDF, HTML, or text files.",
    )
    arguments = parser.parse_args()
    report = ingest_directory(arguments.input_dir)
    print(report.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
