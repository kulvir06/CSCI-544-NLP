"""Convert selected passages into automatically inferred temporal claims."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Callable, Protocol

from api import config
from corpus.schema import Passage, StoredDocument
from corpus.storage import ensure_data_directories
from temporal.schema import (
    ClaimExtraction,
    ExtractedClaim,
    TemporalClaim,
    TemporalIndexReport,
    TemporalRole,
)


class TemporalModel(Protocol):
    temporal_model: str
    embedding_model: str

    def extract_claims(self, prompt: str) -> ClaimExtraction: ...

    def embed(self, texts: list[str]) -> list[list[float]]: ...


def _load_passages(document_id: str) -> list[Passage]:
    path = config.PASSAGE_DIR / f"{document_id}.jsonl"
    if not path.exists():
        return []
    return [
        Passage.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_manifest_documents() -> dict[str, StoredDocument]:
    """Load automatic ingestion records without reading the human review log."""
    if not config.MANIFEST_PATH.exists():
        return {}
    documents = [
        StoredDocument.model_validate_json(line)
        for line in config.MANIFEST_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {document.document_id: document for document in documents}


def _context(passages: list[Passage], index: int) -> str:
    parts: list[str] = []
    for neighbor_index in range(max(0, index - 1), min(len(passages), index + 2)):
        label = "SOURCE PASSAGE" if neighbor_index == index else "SURROUNDING CONTEXT"
        parts.append(f"[{label}]\n{passages[neighbor_index].text}")
    return "\n\n".join(parts)


def _prompt(document: StoredDocument, passage: Passage, context: str) -> str:
    metadata = document.metadata
    candidates = document.metadata_candidates
    return f"""Analyze the SOURCE PASSAGE as part of an F-1 employment-rule corpus.

Automatically inferred document context (may be incomplete; do not treat it as verified):
- filename: {metadata.original_filename}
- title: {metadata.title or 'unknown'}
- issuer: {metadata.issuer or 'unknown'}
- publication date: {metadata.published_at or 'unknown'}
- dates detected elsewhere in the document: {', '.join(candidates.date_mentions) or 'none'}
- section: {passage.section or 'unknown'}

{context}

Tasks:
1. Split the SOURCE PASSAGE into atomic legal or regulatory claims. Return no claim for navigation, boilerplate, or a fragment with no substantive proposition.
2. Copy an exact evidence_quote for each claim from the SOURCE PASSAGE only.
3. Classify each claim as operative, historical, transitional, proposed, rejected, public_comment, example, or unknown.
4. Infer valid_from and valid_to only when the supplied text and context support them. Normalize supported dates as YYYY-MM-DD. Never use today's date and never guess a missing boundary.
5. In date_basis, state what wording supports the interval or why it is unknown.
6. Confidence measures confidence in both the role and interval. Use low confidence when either is ambiguous.

Important distinctions:
- A document can be currently valid while a sentence inside it describes an old rule.
- A final rule can quote proposals, public comments, rejected alternatives, and superseded requirements.
- 'Previous', 'formerly', 'proposed', 'commenter', and 'we decline' are not operative-current language.
"""


def _cache_path(passage: Passage, model_name: str) -> Path:
    digest = hashlib.sha256(
        f"v2\0{model_name}\0{passage.text}".encode("utf-8")
    ).hexdigest()[:24]
    return config.TEMPORAL_CACHE_DIR / f"{passage.passage_id}-{digest}.json"


def _extract_with_cache(
    model: TemporalModel,
    document: StoredDocument,
    passage: Passage,
    context: str,
) -> tuple[ClaimExtraction, bool]:
    path = _cache_path(passage, model.temporal_model)
    if path.exists():
        return (
            ClaimExtraction.model_validate_json(path.read_text(encoding="utf-8")),
            True,
        )
    extraction = model.extract_claims(_prompt(document, passage, context))
    path.write_text(extraction.model_dump_json(indent=2), encoding="utf-8")
    return extraction, False


def _valid_claim(claim: ExtractedClaim, passage: Passage) -> bool:
    normalized_quote = " ".join(claim.evidence_quote.split()).casefold()
    normalized_source = " ".join(passage.text.split()).casefold()
    return bool(normalized_quote) and normalized_quote in normalized_source


def build_temporal_index(
    model: TemporalModel,
    document_ids: set[str] | None = None,
    max_passages: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> TemporalIndexReport:
    """Build an automatic claim index without consulting the human review log."""
    ensure_data_directories()
    documents = _load_manifest_documents()
    if document_ids is not None:
        documents = {
            key: value for key, value in documents.items() if key in document_ids
        }

    work: list[tuple[StoredDocument, list[Passage], int, Passage]] = []
    for document in documents.values():
        passages = _load_passages(document.document_id)
        for index, passage in enumerate(passages):
            work.append((document, passages, index, passage))
            if max_passages is not None and len(work) >= max_passages:
                break
        if max_passages is not None and len(work) >= max_passages:
            break

    pending: list[tuple[StoredDocument, Passage, str, ExtractedClaim]] = []
    source_passages = 0
    skipped_passages = 0
    failed_passages = 0
    failures: list[str] = []

    total_passages = len(work)
    started_at = time.monotonic()
    if progress:
        progress(
            f"Found {len(documents)} documents and {total_passages} passages to analyze."
        )

    for position, (document, passages, index, passage) in enumerate(work, start=1):
        source_passages += 1
        context = _context(passages, index)
        if progress:
            progress(f"[{position}/{total_passages}] Analyzing {passage.passage_id} ...")
        try:
            extraction, cache_hit = _extract_with_cache(
                model, document, passage, context
            )
        except Exception as error:
            failed_passages += 1
            failures.append(f"{passage.passage_id}: {error}")
            if progress:
                progress(f"[{position}/{total_passages}] Failed: {error}")
            continue
        accepted = [claim for claim in extraction.claims if _valid_claim(claim, passage)]
        if not accepted:
            skipped_passages += 1
        else:
            pending.extend(
                (document, passage, context, claim) for claim in accepted
            )
        if progress:
            source = "cache" if cache_hit else "model"
            elapsed = time.monotonic() - started_at
            progress(
                f"[{position}/{total_passages}] Done via {source}: "
                f"{len(accepted)} accepted claim(s); elapsed {elapsed / 60:.1f} min"
            )

    embeddings: list[list[float]] = []
    for start in range(0, len(pending), 32):
        end = min(start + 32, len(pending))
        if progress:
            progress(f"Embedding claims {start + 1}-{end} of {len(pending)} ...")
        embeddings.extend(
            model.embed([item[3].claim_text for item in pending[start:end]])
        )

    claims: list[TemporalClaim] = []
    for position, ((document, passage, context, extracted), embedding) in enumerate(
        zip(pending, embeddings), start=1
    ):
        metadata = document.metadata
        claim_digest = hashlib.sha256(
            f"{passage.passage_id}\0{extracted.claim_text}".encode("utf-8")
        ).hexdigest()[:12]
        claims.append(
            TemporalClaim(
                claim_id=f"{passage.passage_id}-c{position:05d}-{claim_digest}",
                document_id=document.document_id,
                source_passage_id=passage.passage_id,
                text=extracted.claim_text,
                evidence_quote=extracted.evidence_quote,
                context=context,
                section=passage.section,
                page_number=passage.page_number,
                passage_relevance_score=passage.relevance_score,
                title=metadata.title,
                issuer=metadata.issuer,
                source_url=metadata.source_url,
                source_tier=document.source_tier,
                document_status=metadata.status,
                temporal_role=extracted.temporal_role,
                valid_from=extracted.valid_from,
                valid_to=extracted.valid_to,
                date_basis=extracted.date_basis,
                temporal_confidence=extracted.confidence,
                extraction_model=model.temporal_model,
                embedding_model=model.embedding_model,
                embedding=embedding,
            )
        )

    body = "\n".join(claim.model_dump_json() for claim in claims)
    config.RETRIEVAL_INDEX_PATH.write_text(
        f"{body}\n" if body else "", encoding="utf-8"
    )
    if progress:
        progress(f"Index written to {config.RETRIEVAL_INDEX_PATH}")
    return TemporalIndexReport(
        source_documents=len(documents),
        source_passages=source_passages,
        indexed_claims=len(claims),
        skipped_passages=skipped_passages,
        failed_passages=failed_passages,
        extraction_model=model.temporal_model,
        embedding_model=model.embedding_model,
        index_file=str(config.RETRIEVAL_INDEX_PATH),
        failures=failures[:100],
    )


def document_ids_for_directory(input_directory: Path) -> set[str]:
    """Map explicitly supplied local files to their stable ingestion IDs."""
    ids: set[str] = set()
    for path in input_directory.resolve().iterdir():
        if path.is_file() and path.suffix.lower() in config.SUPPORTED_UPLOAD_SUFFIXES:
            ids.add(hashlib.sha256(path.read_bytes()).hexdigest()[:16])
    return ids


def load_temporal_claims() -> list[TemporalClaim]:
    if not config.RETRIEVAL_INDEX_PATH.exists():
        return []
    claims: list[TemporalClaim] = []
    for line in config.RETRIEVAL_INDEX_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        try:
            claims.append(TemporalClaim.model_validate(payload))
            continue
        except ValueError:
            # Compatibility for old test fixtures. New production indexes are
            # always written by build_temporal_index above.
            from corpus.schema import CorpusUse
            from retrieval.schema import IndexedPassage

            legacy = IndexedPassage.model_validate(payload)
            role = (
                TemporalRole.OPERATIVE
                if legacy.corpus_use == CorpusUse.TEMPORAL_AND_MEANING
                else TemporalRole.UNKNOWN
            )
            claims.append(
                TemporalClaim(
                    claim_id=legacy.passage_id,
                    document_id=legacy.document_id,
                    source_passage_id=legacy.passage_id,
                    text=legacy.text,
                    evidence_quote=legacy.text,
                    context=legacy.text,
                    section=legacy.section,
                    page_number=legacy.page_number,
                    passage_relevance_score=legacy.passage_relevance_score,
                    title=legacy.title,
                    issuer=legacy.issuer,
                    source_url=legacy.source_url,
                    source_tier=legacy.source_tier,
                    document_status=legacy.document_status,
                    temporal_role=role,
                    valid_from=legacy.effective_from,
                    valid_to=legacy.effective_to,
                    valid_as_of=legacy.valid_as_of,
                    date_basis=legacy.date_evidence or "legacy index compatibility",
                    temporal_confidence=(
                        1.0 if legacy.effective_from or legacy.valid_as_of else 0.0
                    ),
                    extraction_model="legacy_document_metadata",
                    embedding_model="none",
                )
            )
    return claims
