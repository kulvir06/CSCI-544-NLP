"""Convert selected passages into automatically inferred temporal claims."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Callable, Protocol

from api import config
from corpus.schema import Passage, StoredDocument
from corpus.storage import ensure_data_directories
from temporal.schema import (
    ClaimExtraction,
    ExtractedClaim,
    SemanticPassage,
    TemporalClaim,
    TemporalIndexReport,
    TemporalRole,
)


ROLE_CHANGE_PATTERN = re.compile(
    r"\b(?:prior to|previous(?:ly)?|formerly|supersed(?:e|ed|ing)|"
    r"replac(?:e|ed|ing)|no longer|interim rule|final rule|proposed rule|"
    r"transition(?:al)?|expire[sd]?|expiration)\b",
    re.IGNORECASE,
)
HIGH_RISK_PASSAGE_REASONS = {
    "temporal_expression",
    "rule:quantity",
    "rule:exception",
}
SUBSTANTIVE_RULE_REASONS = {
    "rule:obligation",
    "rule:condition",
    "rule:quantity",
    "rule:exception",
    "rule:form",
}


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


def classify_temporal_route(
    passages: list[Passage], index: int
) -> tuple[bool, list[str], float]:
    """High-recall, non-generative routing for expensive temporal analysis."""
    passage = passages[index]
    reasons: list[str] = []
    selected_reasons = set(passage.selection_reasons)
    matched_selection = sorted(selected_reasons & HIGH_RISK_PASSAGE_REASONS)
    reasons.extend(matched_selection)
    if (
        ROLE_CHANGE_PATTERN.search(passage.text)
        and selected_reasons & SUBSTANTIVE_RULE_REASONS
    ):
        reasons.append("temporal_role_or_change_language")

    reasons = list(dict.fromkeys(reasons))
    if not reasons:
        return False, [], 0.75
    confidence = 0.95 if "temporal_expression" in reasons else 0.85
    return True, reasons, confidence


def _embedding_cache_path(text: str, model_name: str) -> Path:
    digest = hashlib.sha256(
        f"v1\0{model_name}\0{text}".encode("utf-8")
    ).hexdigest()
    return config.EMBEDDING_CACHE_DIR / f"{digest}.json"


def _embed_with_cache(
    model: TemporalModel,
    texts: list[str],
    progress: Callable[[str], None] | None = None,
    label: str = "passages",
) -> list[list[float]]:
    """Embed texts in batches and persist vectors by content hash."""
    vectors: list[list[float] | None] = [None] * len(texts)
    misses: dict[Path, tuple[str, list[int]]] = {}
    for position, text in enumerate(texts):
        path = _embedding_cache_path(text, model.embedding_model)
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(payload, list) and payload:
                    vectors[position] = [float(value) for value in payload]
                    continue
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                pass
        if path in misses:
            misses[path][1].append(position)
        else:
            misses[path] = (text, [position])

    unique_misses = list(misses.items())
    cached_count = len(texts) - sum(len(item[1][1]) for item in unique_misses)
    if progress:
        progress(
            f"Embedding {len(texts)} {label}: {cached_count} cached, "
            f"{len(unique_misses)} new unique text(s)."
        )
    for start in range(0, len(unique_misses), 32):
        batch = unique_misses[start:start + 32]
        if progress:
            progress(
                f"Embedding new {label} {start + 1}-"
                f"{start + len(batch)} of {len(unique_misses)} ..."
            )
        embedded = model.embed([item[1][0] for item in batch])
        for (path, (_, positions)), vector in zip(batch, embedded):
            path.write_text(json.dumps(vector, separators=(",", ":")), encoding="utf-8")
            for position in positions:
                vectors[position] = vector
    if any(vector is None for vector in vectors):
        raise ValueError("Embedding cache did not produce a vector for every text")
    return [vector for vector in vectors if vector is not None]


def _analyze_passage(
    model: TemporalModel,
    item: tuple[StoredDocument, list[Passage], int, Passage],
) -> tuple[str, list[ExtractedClaim], bool]:
    """Analyze one passage; safe to execute in a worker thread."""
    document, passages, index, passage = item
    context = _context(passages, index)
    extraction, cache_hit = _extract_with_cache(
        model, document, passage, context
    )
    accepted = [
        claim for claim in extraction.claims if _valid_claim(claim, passage)
    ]
    return context, accepted, cache_hit


def build_temporal_index(
    model: TemporalModel,
    document_ids: set[str] | None = None,
    max_passages: int | None = None,
    workers: int = 1,
    progress: Callable[[str], None] | None = None,
) -> TemporalIndexReport:
    """Build an automatic claim index without consulting the human review log."""
    if workers < 1:
        raise ValueError("workers must be at least 1")
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
    source_passages = len(work)
    failed_passages = 0
    failures: list[str] = []

    total_passages = len(work)
    routing: dict[str, tuple[bool, list[str], float]] = {}
    candidate_work: list[
        tuple[int, tuple[StoredDocument, list[Passage], int, Passage]]
    ] = []
    for position, item in enumerate(work, start=1):
        _, passages, index, passage = item
        decision = classify_temporal_route(passages, index)
        routing[passage.passage_id] = decision
        if decision[0]:
            candidate_work.append((position, item))
    temporal_candidates = len(candidate_work)
    skipped_passages = total_passages - temporal_candidates
    scan_status = {
        item[3].passage_id: (
            "pending" if routing[item[3].passage_id][0] else "not_required"
        )
        for item in work
    }

    started_at = time.monotonic()
    if progress:
        progress(
            f"Found {len(documents)} documents and {total_passages} passages."
        )
        progress(
            f"Temporal router selected {temporal_candidates} passages for Ollama "
            f"and skipped {total_passages - temporal_candidates}."
        )
    effective_workers = min(workers, temporal_candidates) if temporal_candidates else 1
    if progress:
        label = "worker" if effective_workers == 1 else "parallel workers"
        progress(f"Using {effective_workers} {label}.")

    # Keep only `effective_workers` tasks in flight. This bounds memory usage,
    # makes Ctrl+C responsive, and avoids flooding Ollama's request queue.
    results: dict[
        int, tuple[StoredDocument, Passage, str, list[ExtractedClaim]]
    ] = {}
    executor = ThreadPoolExecutor(
        max_workers=effective_workers,
        thread_name_prefix="temporal-extraction",
    )
    in_flight: dict[
        Future[tuple[str, list[ExtractedClaim], bool]],
        tuple[int, StoredDocument, Passage],
    ] = {}
    next_offset = 0

    def submit_next() -> None:
        nonlocal next_offset
        if next_offset >= temporal_candidates:
            return
        position, item = candidate_work[next_offset]
        document, _, _, passage = item
        if progress:
            progress(
                f"[{next_offset + 1}/{temporal_candidates}] "
                f"Analyzing {passage.passage_id} ..."
            )
        future = executor.submit(_analyze_passage, model, item)
        in_flight[future] = (position, document, passage)
        next_offset += 1

    for _ in range(effective_workers):
        submit_next()

    try:
        while in_flight:
            completed, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in sorted(completed, key=lambda item: in_flight[item][0]):
                position, document, passage = in_flight.pop(future)
                try:
                    context, accepted, cache_hit = future.result()
                except Exception as error:
                    failed_passages += 1
                    scan_status[passage.passage_id] = "failed"
                    failures.append(f"{passage.passage_id}: {error}")
                    if progress:
                        progress(
                            f"[{len(results) + failed_passages}/{temporal_candidates} complete] "
                            f"{passage.passage_id} failed: {error}"
                        )
                else:
                    if not accepted:
                        skipped_passages += 1
                        scan_status[passage.passage_id] = "analyzed_no_claim"
                    else:
                        scan_status[passage.passage_id] = "completed"
                    results[position] = (document, passage, context, accepted)
                    if progress:
                        source = "cache" if cache_hit else "model"
                        elapsed = time.monotonic() - started_at
                        progress(
                            f"[{len(results) + failed_passages}/{temporal_candidates} complete] "
                            f"{passage.passage_id}: Done via {source}: "
                            f"{len(accepted)} accepted claim(s); "
                            f"elapsed {elapsed / 60:.1f} min"
                        )
                submit_next()
    except BaseException:
        for future in in_flight:
            future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)

    # Completion order is nondeterministic, so restore corpus order before
    # assigning stable claim IDs and writing the index.
    for position in sorted(results):
        document, passage, context, accepted = results[position]
        pending.extend((document, passage, context, claim) for claim in accepted)

    passage_embeddings = _embed_with_cache(
        model,
        [item[3].text for item in work],
        progress=progress,
        label="passages",
    )
    semantic_passages: list[SemanticPassage] = []
    for item, embedding in zip(work, passage_embeddings):
        document, passages, index, passage = item
        metadata = document.metadata
        required, reasons, routing_confidence = routing[passage.passage_id]
        semantic_passages.append(
            SemanticPassage(
                passage_id=passage.passage_id,
                claim_id=passage.passage_id,
                document_id=document.document_id,
                text=passage.text,
                block_index=passage.block_index,
                previous_passage_id=(
                    passages[index - 1].passage_id if index > 0 else None
                ),
                next_passage_id=(
                    passages[index + 1].passage_id
                    if index + 1 < len(passages) else None
                ),
                section=passage.section,
                page_number=passage.page_number,
                passage_relevance_score=passage.relevance_score,
                title=metadata.title,
                issuer=metadata.issuer,
                source_url=metadata.source_url,
                source_tier=document.source_tier,
                document_status=metadata.status,
                temporal_scan_required=required,
                temporal_scan_status=scan_status[passage.passage_id],
                temporal_routing_reasons=reasons,
                temporal_routing_confidence=routing_confidence,
                embedding_model=model.embedding_model,
                embedding=embedding,
            )
        )
    semantic_body = "\n".join(
        passage.model_dump_json() for passage in semantic_passages
    )
    config.SEMANTIC_PASSAGE_INDEX_PATH.write_text(
        f"{semantic_body}\n" if semantic_body else "", encoding="utf-8"
    )
    if progress:
        progress(f"Semantic index written to {config.SEMANTIC_PASSAGE_INDEX_PATH}")

    embeddings = _embed_with_cache(
        model,
        [item[3].claim_text for item in pending],
        progress=progress,
        label="temporal claims",
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
        semantic_passages=len(semantic_passages),
        temporal_candidates=temporal_candidates,
        semantic_index_file=str(config.SEMANTIC_PASSAGE_INDEX_PATH),
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


def load_semantic_passages() -> list[SemanticPassage]:
    if not config.SEMANTIC_PASSAGE_INDEX_PATH.exists():
        return []
    return [
        SemanticPassage.model_validate_json(line)
        for line in config.SEMANTIC_PASSAGE_INDEX_PATH.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
