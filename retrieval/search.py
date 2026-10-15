"""Deterministic BM25 retrieval with an explicit temporal applicability gate."""

from __future__ import annotations

from collections import Counter
from datetime import date
import math
import re

from corpus.schema import CorpusUse
from retrieval.index import load_index
from retrieval.schema import (
    IndexedPassage,
    PassageTemporalStatus,
    RetrievalHit,
    RetrievalRequest,
    RetrievalResponse,
    RetrievalStatus,
    TemporalVerdict,
)


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)?")
DURATION_QUERY_PATTERN = re.compile(r"\b(how long|duration|length)\b", re.I)
NON_OPERATIVE_DISCUSSION_PATTERN = re.compile(
    r"\b(commenters?|recommended|recommendation|alternative considered)\b|"
    r"^(?:comment|response)\.",
    re.I,
)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "for",
    "from", "how", "i", "in", "is", "it", "may", "my", "of", "on", "or",
    "the", "to", "what", "when", "with",
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
        # Regulations normally state durations as numbered time units rather
        # than repeating a user's phrase such as "how long".
        tokens.extend(["month", "months", "day", "days", "year", "years"])
    return tokens


def _has_substantive_query_overlap(query: str, passage_text: str) -> bool:
    """Reject dated candidates that share only one generic query term."""
    query_terms = set(_tokens(query))
    passage_terms = set(_tokens(passage_text))
    required_matches = 1 if len(query_terms) <= 1 else 2
    return len(query_terms & passage_terms) >= required_matches


def _bm25_scores(query: str, passages: list[IndexedPassage]) -> list[float]:
    if not passages:
        return []
    query_terms = set(_query_tokens(query))
    if not query_terms:
        return [0.0] * len(passages)

    tokenized = [_tokens(passage.text) for passage in passages]
    document_frequency = {
        term: sum(term in set(tokens) for tokens in tokenized)
        for term in query_terms
    }
    average_length = sum(len(tokens) for tokens in tokenized) / len(tokenized)
    average_length = average_length or 1.0
    k1 = 1.5
    b = 0.75
    scores: list[float] = []

    for passage, tokens in zip(passages, tokenized):
        frequencies = Counter(tokens)
        score = 0.0
        for term in query_terms:
            frequency = frequencies[term]
            if not frequency:
                continue
            frequency_in_documents = document_frequency[term]
            inverse_document_frequency = math.log(
                1 + (len(passages) - frequency_in_documents + 0.5)
                / (frequency_in_documents + 0.5)
            )
            denominator = frequency + k1 * (
                1 - b + b * len(tokens) / average_length
            )
            score += inverse_document_frequency * frequency * (k1 + 1) / denominator

        # Preserve the ingestion signal only as a small tie-breaker.
        if score > 0:
            # Preambles contain proposed alternatives and public comments that
            # mention attractive query terms but do not state the operative
            # rule. Apply this conservative penalty only for duration questions.
            if (
                DURATION_QUERY_PATTERN.search(query)
                and NON_OPERATIVE_DISCUSSION_PATTERN.search(passage.text)
            ):
                score *= 0.6
            score += min(passage.passage_relevance_score, 10.0) * 0.001
        scores.append(score)
    return scores


def _temporal_status(
    passage: IndexedPassage,
    relevant_date: date,
) -> PassageTemporalStatus:
    if passage.corpus_use != CorpusUse.TEMPORAL_AND_MEANING:
        return PassageTemporalStatus.INSUFFICIENT_EVIDENCE
    if passage.effective_from:
        if relevant_date < passage.effective_from:
            return PassageTemporalStatus.NOT_APPLICABLE
        if passage.effective_to and relevant_date > passage.effective_to:
            return PassageTemporalStatus.NOT_APPLICABLE
        return PassageTemporalStatus.APPLICABLE
    if passage.valid_as_of:
        if relevant_date == passage.valid_as_of:
            return PassageTemporalStatus.APPLICABLE
        return PassageTemporalStatus.INSUFFICIENT_EVIDENCE
    return PassageTemporalStatus.INSUFFICIENT_EVIDENCE


def retrieve(request: RetrievalRequest) -> RetrievalResponse:
    passages = load_index()
    scores = _bm25_scores(request.query, passages)
    ranked = sorted(
        (
            (passage, score)
            for passage, score in zip(passages, scores)
            if score > 0
        ),
        key=lambda item: (-item[1], item[0].passage_id),
    )

    if not ranked:
        temporal_verdict = (
            TemporalVerdict.INSUFFICIENT_TEMPORAL_EVIDENCE
            if request.relevant_date
            else TemporalVerdict.NOT_REQUESTED
        )
        return RetrievalResponse(
            query=request.query,
            relevant_date=request.relevant_date,
            retrieval_status=RetrievalStatus.NO_RELEVANT_PASSAGES,
            temporal_verdict=temporal_verdict,
        )

    if not request.relevant_date:
        # For undated questions, keep a tight lexical cohort so passages sharing
        # only a generic word do not appear as evidence.
        relevance_floor = ranked[0][1] * 0.5
        ranked = [item for item in ranked[:50] if item[1] >= relevance_floor]
        hits = [
            RetrievalHit(
                passage=passage,
                score=round(score, 6),
                temporal_status=PassageTemporalStatus.NOT_REQUESTED,
            )
            for passage, score in ranked[: request.top_k]
        ]
        return RetrievalResponse(
            query=request.query,
            retrieval_status=RetrievalStatus.FOUND,
            temporal_verdict=TemporalVerdict.NOT_REQUESTED,
            hits=hits,
        )

    # A dated question must not discard an applicable historical version merely
    # because a newer or undated passage has the highest raw lexical score.
    # Require direct overlap with multiple substantive query terms rather than
    # comparing old and new documents through a brittle global score ratio.
    temporal_candidates = [
        item
        for item in ranked[:100]
        if _has_substantive_query_overlap(request.query, item[0].text)
    ]
    classified = [
        (passage, score, _temporal_status(passage, request.relevant_date))
        for passage, score in temporal_candidates
    ]
    applicable = [item for item in classified if item[2] == PassageTemporalStatus.APPLICABLE]
    # Undated evidence remains a fallback, but only from the tighter relevance
    # cohort used by ordinary retrieval.
    fallback_floor = ranked[0][1] * 0.5
    insufficient = [
        item for item in classified
        if item[2] == PassageTemporalStatus.INSUFFICIENT_EVIDENCE
        and item[1] >= fallback_floor
    ]

    if applicable:
        selected = applicable[: request.top_k]
        verdict = TemporalVerdict.VERIFIED_APPLICABLE
    elif insufficient:
        selected = insufficient[: request.top_k]
        verdict = TemporalVerdict.INSUFFICIENT_TEMPORAL_EVIDENCE
    else:
        selected = classified[: request.top_k]
        verdict = TemporalVerdict.VERIFIED_NOT_APPLICABLE

    return RetrievalResponse(
        query=request.query,
        relevant_date=request.relevant_date,
        retrieval_status=RetrievalStatus.FOUND,
        temporal_verdict=verdict,
        hits=[
            RetrievalHit(
                passage=passage,
                score=round(score, 6),
                temporal_status=status,
            )
            for passage, score, status in selected
        ],
    )
