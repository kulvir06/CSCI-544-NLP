"""Run the end-to-end reviewed retrieval and verification prototype."""

from __future__ import annotations

import argparse
from datetime import date

from prototype.pipeline import run_prototype
from prototype.schema import GeneratorMode, PrototypeRequest


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the safe F-1 prototype.")
    parser.add_argument("--question", required=True)
    parser.add_argument("--date", type=date.fromisoformat, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-explanation", default=None)
    arguments = parser.parse_args()
    mode = (
        GeneratorMode.PROVIDED_CANDIDATE
        if arguments.candidate_explanation
        else GeneratorMode.EXTRACTIVE_BASELINE
    )
    response = run_prototype(
        PrototypeRequest(
            question=arguments.question,
            relevant_date=arguments.date,
            top_k=arguments.top_k,
            generator_mode=mode,
            candidate_explanation=arguments.candidate_explanation,
        )
    )
    print(response.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
