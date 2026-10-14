"""Build a local retrieval index from explicitly approved passage files."""

from __future__ import annotations

import json

from api import config
from corpus.review import load_retrieval_ready_documents
from corpus.schema import Passage
from corpus.storage import ensure_data_directories
from retrieval.schema import IndexedPassage, IndexBuildReport


def build_index() -> IndexBuildReport:
    approved = load_retrieval_ready_documents()
    indexed: list[IndexedPassage] = []
    indexed_document_ids: set[str] = set()
    skipped_documents: list[str] = []

    for review in approved:
        passage_path = config.PASSAGE_DIR / f"{review.document_id}.jsonl"
        if not passage_path.exists():
            skipped_documents.append(review.document_id)
            continue

        for line in passage_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            passage = Passage.model_validate_json(line)
            metadata = review.metadata
            indexed.append(
                IndexedPassage(
                    passage_id=passage.passage_id,
                    document_id=passage.document_id,
                    text=passage.text,
                    section=passage.section,
                    page_number=passage.page_number,
                    passage_relevance_score=passage.relevance_score,
                    title=metadata.title,
                    issuer=metadata.issuer,
                    source_url=metadata.source_url,
                    source_tier=review.source_tier,
                    corpus_use=review.corpus_use,
                    published_at=metadata.published_at,
                    effective_from=metadata.effective_from,
                    effective_to=metadata.effective_to,
                    valid_as_of=metadata.valid_as_of,
                    date_evidence=metadata.date_evidence,
                    document_status=metadata.status,
                )
            )
            indexed_document_ids.add(review.document_id)

    ensure_data_directories()
    body = "\n".join(item.model_dump_json() for item in indexed)
    config.RETRIEVAL_INDEX_PATH.write_text(
        f"{body}\n" if body else "",
        encoding="utf-8",
    )
    return IndexBuildReport(
        approved_documents=len(approved),
        indexed_documents=len(indexed_document_ids),
        indexed_passages=len(indexed),
        skipped_documents=skipped_documents,
        index_file=str(config.RETRIEVAL_INDEX_PATH),
    )


def load_index() -> list[IndexedPassage]:
    if not config.RETRIEVAL_INDEX_PATH.exists():
        return []
    return [
        IndexedPassage.model_validate(json.loads(line))
        for line in config.RETRIEVAL_INDEX_PATH.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

