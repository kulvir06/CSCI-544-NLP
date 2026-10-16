"""Build the automatic claim-level temporal and semantic retrieval index."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from api import config
from temporal.enrichment import build_temporal_index, document_ids_for_directory
from temporal.ollama import OllamaClient


def _installed(requested: str, available: list[str]) -> bool:
    return requested in available or f"{requested}:latest" in available


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Use local Ollama models to atomize passages, infer temporal scope, "
            "and create semantic embeddings. No review log is consulted."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Optional local corpus directory used to scope the index.",
    )
    parser.add_argument(
        "--max-passages",
        type=int,
        default=None,
        help="Development limit; omit to process the full corpus.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help=(
            "Concurrent Ollama requests (default: 2). Use 1 to disable "
            "parallel processing."
        ),
    )
    parser.add_argument(
        "--primary-model",
        default=None,
        help="Fast primary temporal model (default: configured qwen2.5:3b).",
    )
    parser.add_argument(
        "--fallback-model",
        default=None,
        help="Fallback model for uncertain/invalid results (default: llama3.1:8b).",
    )
    arguments = parser.parse_args()
    if arguments.workers < 1:
        parser.error("--workers must be at least 1")
    document_ids = (
        document_ids_for_directory(arguments.input_dir)
        if arguments.input_dir else None
    )
    primary_model = arguments.primary_model or config.OLLAMA_TEMPORAL_MODEL
    fallback_model = (
        arguments.fallback_model or config.OLLAMA_FALLBACK_TEMPORAL_MODEL
    )
    client = OllamaClient(
        temporal_model=primary_model,
        num_predict=config.OLLAMA_PRIMARY_NUM_PREDICT,
    )
    fallback_client = None
    if fallback_model and fallback_model != primary_model:
        fallback_client = OllamaClient(
            temporal_model=fallback_model,
            num_predict=config.OLLAMA_FALLBACK_NUM_PREDICT,
        )
    available = client.available_models()
    required = {client.temporal_model, client.embedding_model}
    if fallback_client:
        required.add(fallback_client.temporal_model)
    missing = [model for model in required if not _installed(model, available)]
    if missing:
        raise SystemExit(
            "Missing Ollama model(s): " + ", ".join(missing)
            + ". Install them with `ollama pull <model>` and run again."
        )
    try:
        report = build_temporal_index(
            client,
            fallback_model=fallback_client,
            document_ids=document_ids,
            max_passages=arguments.max_passages,
            workers=arguments.workers,
            progress=lambda message: print(message, flush=True),
        )
    except KeyboardInterrupt:
        print(
            "\nStopped. Completed passage analyses are cached; run the same "
            "command again to resume.",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(130)
    print(report.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
