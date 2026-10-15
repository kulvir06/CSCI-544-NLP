"""Batch ingestion for a local, user-supplied directory."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from api import config
from corpus.pipeline import ingest_upload
from corpus.schema import BatchDocumentResult, BatchIngestionReport, DocumentMetadata
from corpus.storage import ensure_data_directories


def ingest_directory(input_directory: Path) -> BatchIngestionReport:
    input_directory = input_directory.expanduser().resolve()
    if not input_directory.is_dir():
        raise ValueError(f"Input directory does not exist: {input_directory}")

    files = sorted(
        path
        for path in input_directory.iterdir()
        if path.is_file() and path.suffix.lower() in config.SUPPORTED_UPLOAD_SUFFIXES
    )
    results: list[BatchDocumentResult] = []
    total_passages = 0

    for path in files:
        try:
            result = ingest_upload(
                path.name,
                path.read_bytes(),
                DocumentMetadata(original_filename=path.name),
            )
            total_passages += result.selected_passages
            results.append(
                BatchDocumentResult(
                    filename=path.name,
                    outcome="duplicate" if result.duplicate_upload else "processed",
                    document_id=result.document.document_id,
                    selected_passages=result.selected_passages,
                    review_flags=result.document.review_flags,
                )
            )
        except Exception as error:  # Report one bad file and continue the batch.
            results.append(
                BatchDocumentResult(
                    filename=path.name,
                    outcome="failed",
                    error=str(error),
                )
            )

    report = BatchIngestionReport(
        input_directory=str(input_directory),
        discovered_files=len(files),
        processed_files=sum(result.outcome != "failed" for result in results),
        failed_files=sum(result.outcome == "failed" for result in results),
        total_selected_passages=total_passages,
        results=results,
    )

    ensure_data_directories()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = config.REPORT_DIR / f"batch-{timestamp}.json"
    report.report_file = str(report_path)
    report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return report
