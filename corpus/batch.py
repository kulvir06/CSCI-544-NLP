"""Batch ingestion for a local, user-supplied directory."""

from __future__ import annotations

import csv
from datetime import date, datetime, timezone
from pathlib import Path

from api import config
from corpus.pipeline import ingest_upload
from corpus.schema import (
    BatchDocumentResult,
    BatchIngestionReport,
    DocumentMetadata,
    DocumentStatus,
)
from corpus.storage import ensure_data_directories


CATALOG_FILENAME = "source-catalog.csv"


def _catalog_status(row: dict[str, str]) -> DocumentStatus:
    text = f"{row.get('Title', '')} {row.get('Status', '')}".lower()
    if "proposed" in text:
        return DocumentStatus.PROPOSED
    if "superseded" in text:
        return DocumentStatus.SUPERSEDED
    if "historical" in text:
        return DocumentStatus.HISTORICAL
    if "current" in text:
        return DocumentStatus.CURRENT
    return DocumentStatus.UNKNOWN


def _catalog_rule_family(topics: str) -> str | None:
    lowered = topics.lower()
    if "cap-gap" in lowered or "cap gap" in lowered:
        return "cap_gap"
    if "stem opt" in lowered:
        return "stem_opt"
    if "cpt" in lowered or "curricular practical training" in lowered:
        return "cpt"
    if "opt" in lowered or "optional practical training" in lowered:
        return "opt"
    if "f-1" in lowered:
        return "general_f1"
    return None


def _load_source_catalog(input_directory: Path) -> dict[str, DocumentMetadata]:
    """Load supplied provenance without opening any source URL."""
    path = input_directory / CATALOG_FILENAME
    if not path.exists():
        return {}

    catalog: dict[str, DocumentMetadata] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row_number, row in enumerate(csv.DictReader(handle), start=2):
            filename = (row.get("Filename") or "").strip()
            if not filename:
                raise ValueError(f"{CATALOG_FILENAME} row {row_number} has no Filename")
            if filename in catalog:
                raise ValueError(
                    f"{CATALOG_FILENAME} contains duplicate Filename: {filename}"
                )
            date_evidence = (row.get("Publication / effective date") or "").strip()
            catalog[filename] = DocumentMetadata(
                original_filename=filename,
                title=(row.get("Title") or "").strip() or None,
                issuer=(row.get("Source organization") or "").strip() or None,
                source_url=(row.get("Original URL") or "").strip() or None,
                retrieved_at=date.today(),
                date_evidence=date_evidence or None,
                rule_family=_catalog_rule_family(row.get("Topics") or ""),
                status=_catalog_status(row),
            )
    return catalog


def ingest_directory(input_directory: Path) -> BatchIngestionReport:
    input_directory = input_directory.expanduser().resolve()
    if not input_directory.is_dir():
        raise ValueError(f"Input directory does not exist: {input_directory}")

    files = sorted(
        path
        for path in input_directory.iterdir()
        if path.is_file() and path.suffix.lower() in config.SUPPORTED_UPLOAD_SUFFIXES
    )
    catalog = _load_source_catalog(input_directory)
    results: list[BatchDocumentResult] = []
    total_passages = 0

    for path in files:
        try:
            result = ingest_upload(
                path.name,
                path.read_bytes(),
                catalog.get(
                    path.name,
                    DocumentMetadata(original_filename=path.name),
                ),
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
