"""Temporal-filter-first hybrid retrieval over automatically enriched claims."""

from __future__ import annotations

from collections import Counter
from datetime import date
import math
import re

from api import config
from corpus.schema import CorpusUse
from retrieval.schema import (
    IndexedPassage,
    PassageTemporalStatus,
    RetrievalHit,
    RetrievalRequest,
    RetrievalResponse,
    RetrievalStatus,
    TemporalVerdict,
)
from temporal.enrichment import load_temporal_claims
from temporal.ollama import OllamaClient, OllamaUnavailableError
from temporal.schema import TemporalClaim, TemporalRole


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)?")
DURATION_QUERY_PATTERN = re.compile(r"\b(how long|duration|length)\b", re.I)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "for",
    "from", "how", "i", "in", "is", "it", "may", "my", "of", "on", "or",
    "the", "to", "what", "when", "with",
}
NON_NORMATIVE_ROLES = {
    TemporalRole.PROPOSED,
    TemporalRole.REJECTED,
    TemporalRole.PUBLIC_COMMENT,
    TemporalRole.EXAMPLE,
}


def _tokens(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_PATTERN.findall(text)
        if token.lower() not in STOPWORDS
    ]


def _query_tokens(query: str) -> list[str]:
    tokens = _tokens(query)
    if DURATION_QUERY_PATTERN.search(query):
        tokens.extend(["month", "months", "day", "days", "year", "years"])
    return tokens


def _has_substantive_overlap(query: str, claim: TemporalClaim) -> bool:
    """Keep generic words from making an unrelated temporal claim look valid."""
    query_terms = set(_tokens(query))
    claim_terms = set(_tokens(claim.text))
    required = 1 if len(query_terms) <= 1 else 2
    return len(query_terms & claim_terms) >= required


def _bm25_scores(query: str, claims: list[TemporalClaim]) -> list[float]:
    if not claims:
        return []
    query_terms = set(_query_tokens(query))
    if not query_terms:
        return [0.0] * len(claims)
    tokenized = [_tokens(claim.text) for claim in claims]
    document_frequency = {
        term: sum(term in set(tokens) for tokens in tokenized)
        for term in query_terms
    }
    average_length = sum(len(tokens) for tokens in tokenized) / len(tokenized) or 1.0
    scores: list[float] = []
    for claim, tokens in zip(claims, tokenized):
        frequencies = Counter(tokens)
        score = 0.0
        for term in query_terms:
            frequency = frequencies[term]
            if not frequency:
                continue
            frequency_in_documents = document_frequency[term]
            inverse_document_frequency = math.log(
                1 + (len(claims) - frequency_in_documents + 0.5)
                / (frequency_in_documents + 0.5)
            )
            denominator = frequency + 1.5 * (
                1 - 0.75 + 0.75 * len(tokens) / average_length
            )
            score += inverse_document_frequency * frequency * 2.5 / denominator
        if score > 0:
            score += min(claim.passage_relevance_score, 10.0) * 0.001
        scores.append(score)
    return scores


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or len(left) != len(right):
        return 0.0
    denominator = math.sqrt(sum(value * value for value in left)) * math.sqrt(
        sum(value * value for value in right)
    )
    if denominator == 0:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / denominator


def _temporal_status(
    claim: TemporalClaim,
    relevant_date: date,
) -> PassageTemporalStatus:
    if claim.temporal_role in NON_NORMATIVE_ROLES:
        return PassageTemporalStatus.NOT_APPLICABLE
    if (
        claim.temporal_role == TemporalRole.UNKNOWN
        or claim.temporal_confidence < config.TEMPORAL_CONFIDENCE_THRESHOLD
    ):
        return PassageTemporalStatus.INSUFFICIENT_EVIDENCE
    if claim.valid_as_of:
        return (
            PassageTemporalStatus.APPLICABLE
            if relevant_date == claim.valid_as_of
            else PassageTemporalStatus.INSUFFICIENT_EVIDENCE
        )
    if not claim.valid_from:
        return PassageTemporalStatus.INSUFFICIENT_EVIDENCE
    if relevant_date < claim.valid_from:
        return PassageTemporalStatus.NOT_APPLICABLE
    if claim.valid_to and relevant_date > claim.valid_to:
        return PassageTemporalStatus.NOT_APPLICABLE
    return PassageTemporalStatus.APPLICABLE


def _query_embedding(query: str, claims: list[TemporalClaim]) -> list[float] | None:
    embedded_claim = next((claim for claim in claims if claim.embedding), None)
    if not embedded_claim:
        return None
    try:
        return OllamaClient(
            embedding_model=embedded_claim.embedding_model
        ).embed([query])[0]
    except (OllamaUnavailableError, ValueError, IndexError):
        return None


def _rank(
    query: str,
    claims: list[TemporalClaim],
) -> tuple[list[tuple[TemporalClaim, float, float, float]], bool]:
    lexical = _bm25_scores(query, claims)
    query_embedding = _query_embedding(query, claims)
    semantic = [
        _cosine(query_embedding, claim.embedding) if query_embedding else 0.0
        for claim in claims
    ]
    maximum_lexical = max(lexical, default=0.0)
    ranked: list[tuple[TemporalClaim, float, float, float]] = []
    for claim, lexical_score, semantic_score in zip(claims, lexical, semantic):
        if lexical_score <= 0 and semantic_score <= 0:
            continue
        normalized_lexical = lexical_score / maximum_lexical if maximum_lexical else 0.0
        if query_embedding:
            normalized_semantic = (semantic_score + 1.0) / 2.0
            combined = 0.45 * normalized_lexical + 0.55 * normalized_semantic
        else:
            combined = normalized_lexical
        ranked.append((claim, combined, lexical_score, semantic_score))
    ranked.sort(key=lambda item: (-item[1], item[0].claim_id))
    return ranked, query_embedding is not None


def _as_passage(claim: TemporalClaim) -> IndexedPassage:
    corpus_use = (
        CorpusUse.TEMPORAL_AND_MEANING
        if claim.valid_from or claim.valid_as_of
        else CorpusUse.MEANING_ONLY
    )
    return IndexedPassage(
        passage_id=claim.claim_id,
        source_passage_id=claim.source_passage_id,
        document_id=claim.document_id,
        text=claim.text,
        section=claim.section,
        page_number=claim.page_number,
        passage_relevance_score=claim.passage_relevance_score,
        title=claim.title,
        issuer=claim.issuer,
        source_url=claim.source_url,
        source_tier=claim.source_tier,
        corpus_use=corpus_use,
        effective_from=claim.valid_from,
        effective_to=claim.valid_to,
        valid_as_of=claim.valid_as_of,
        date_evidence=claim.date_basis,
        document_status=claim.document_status,
        temporal_role=claim.temporal_role.value,
        temporal_confidence=claim.temporal_confidence,
        temporal_evidence=claim.evidence_quote,
        extraction_method=claim.extraction_model,
    )


def retrieve(request: RetrievalRequest) -> RetrievalResponse:
    claims = load_temporal_claims()
    if not claims:
        return RetrievalResponse(
            query=request.query,
            relevant_date=request.relevant_date,
            retrieval_status=RetrievalStatus.NO_RELEVANT_PASSAGES,
            temporal_verdict=(
                TemporalVerdict.INSUFFICIENT_TEMPORAL_EVIDENCE
                if request.relevant_date else TemporalVerdict.NOT_REQUESTED
            ),
        )

    # This is only a broad lexical candidate gate. Temporal scopes were inferred
    # offline, and the actual semantic/lexical ranking happens after filtering.
    claims = [claim for claim in claims if _has_substantive_overlap(request.query, claim)]
    if not claims:
        return RetrievalResponse(
            query=request.query,
            relevant_date=request.relevant_date,
            retrieval_status=RetrievalStatus.NO_RELEVANT_PASSAGES,
            temporal_verdict=(
                TemporalVerdict.INSUFFICIENT_TEMPORAL_EVIDENCE
                if request.relevant_date else TemporalVerdict.NOT_REQUESTED
            ),
        )

    statuses: dict[str, PassageTemporalStatus] = {}
    if request.relevant_date:
        statuses = {
            claim.claim_id: _temporal_status(claim, request.relevant_date)
            for claim in claims
        }
        applicable = [
            claim for claim in claims
            if statuses[claim.claim_id] == PassageTemporalStatus.APPLICABLE
        ]
        uncertain = [
            claim for claim in claims
            if statuses[claim.claim_id] == PassageTemporalStatus.INSUFFICIENT_EVIDENCE
        ]
        inapplicable = [
            claim for claim in claims
            if statuses[claim.claim_id] == PassageTemporalStatus.NOT_APPLICABLE
        ]
        if applicable:
            candidate_claims = applicable
            verdict = TemporalVerdict.VERIFIED_APPLICABLE
        elif uncertain:
            candidate_claims = uncertain
            verdict = TemporalVerdict.INSUFFICIENT_TEMPORAL_EVIDENCE
        else:
            candidate_claims = inapplicable
            verdict = TemporalVerdict.VERIFIED_NOT_APPLICABLE
    else:
        candidate_claims = [
            claim for claim in claims if claim.temporal_role not in NON_NORMATIVE_ROLES
        ]
        verdict = TemporalVerdict.NOT_REQUESTED
        statuses = {
            claim.claim_id: PassageTemporalStatus.NOT_REQUESTED
            for claim in candidate_claims
        }

    ranked, used_embeddings = _rank(request.query, candidate_claims)
    ranking_method = (
        "temporal_filter_then_hybrid"
        if used_embeddings else "temporal_filter_then_bm25"
    )
    if not ranked:
        return RetrievalResponse(
            query=request.query,
            relevant_date=request.relevant_date,
            retrieval_status=RetrievalStatus.NO_RELEVANT_PASSAGES,
            temporal_verdict=(
                TemporalVerdict.INSUFFICIENT_TEMPORAL_EVIDENCE
                if request.relevant_date else TemporalVerdict.NOT_REQUESTED
            ),
            ranking_method=ranking_method,
        )

    selected = ranked[:request.top_k]
    return RetrievalResponse(
        query=request.query,
        relevant_date=request.relevant_date,
        retrieval_status=RetrievalStatus.FOUND,
        temporal_verdict=verdict,
        ranking_method=ranking_method,
        hits=[
            RetrievalHit(
                passage=_as_passage(claim),
                score=round(combined, 6),
                lexical_score=round(lexical, 6),
                semantic_score=round(semantic, 6) if used_embeddings else None,
                temporal_status=statuses[claim.claim_id],
            )
            for claim, combined, lexical, semantic in selected
        ],
    )
