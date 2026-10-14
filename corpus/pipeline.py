"""End-to-end local ingestion for user-supplied documents."""

from __future__ import annotations

import mimetypes

from corpus.metadata_extraction import enrich_metadata
from corpus.parser import parse_document
from corpus.schema import DocumentMetadata, IngestionResult, SourceTier, StoredDocument
from corpus.selector import select_passages
from corpus.source_classification import classify_source
from corpus.storage import (
    append_manifest,
    save_normalized_text,
    save_passages,
    save_upload,
)


def ingest_upload(
    filename: str,
    content: bytes,
    metadata: DocumentMetadata,
) -> IngestionResult:
    stored = save_upload(filename, content)
    blocks = parse_document(stored.path)
    metadata, metadata_candidates = enrich_metadata(metadata, blocks)
    source_tier = classify_source(metadata)
    passages = select_passages(stored.document_id, blocks)

    review_flags: list[str] = []
    if not metadata.source_url:
        review_flags.append("missing_source_url")
    if not metadata.effective_from:
        review_flags.append("missing_effective_date")
    if source_tier == SourceTier.UNKNOWN:
        review_flags.append("unknown_source_authority")
    if not passages:
        review_flags.append("no_rule_passages_selected")

    document = StoredDocument(
        document_id=stored.document_id,
        checksum_sha256=stored.checksum,
        stored_filename=stored.path.name,
        media_type=mimetypes.guess_type(filename)[0] or "application/octet-stream",
        byte_size=len(content),
        metadata=metadata,
        metadata_candidates=metadata_candidates,
        source_tier=source_tier,
        review_flags=review_flags,
    )

    normalized_path = save_normalized_text(
        stored.document_id,
        "\n\n".join(block.text for block in blocks),
    )
    passage_path = save_passages(stored.document_id, passages)
    append_manifest(document)

    return IngestionResult(
        document=document,
        total_blocks=len(blocks),
        selected_passages=len(passages),
        duplicate_upload=stored.duplicate,
        passage_file=str(passage_path),
        normalized_text_file=str(normalized_path),
    )
