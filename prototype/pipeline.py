"""Orchestrate reviewed retrieval, explanation, and two-stage verification."""

from retrieval.schema import RetrievalRequest, RetrievalStatus
from retrieval.search import retrieve
from verification.safety import verify_safety
from verification.schema import SafetyVerificationRequest, SafetyVerdict

from prototype.schema import (
    Citation,
    GeneratorMode,
    PrototypeRequest,
    PrototypeResponse,
    PrototypeStatus,
)


def _citations(retrieval) -> list[Citation]:
    return [
        Citation(
            document_id=hit.passage.document_id,
            passage_id=hit.passage.passage_id,
            title=hit.passage.title,
            issuer=hit.passage.issuer,
            source_url=hit.passage.source_url,
            page_number=hit.passage.page_number,
        )
        for hit in retrieval.hits[:1]
    ]


def _generate_explanation(request: PrototypeRequest, source_text: str) -> str:
    if request.generator_mode == GeneratorMode.PROVIDED_CANDIDATE:
        return (request.candidate_explanation or "").strip()
    # A deterministic control condition: it preserves the source wording so the
    # verifier and temporal gate can be tested without an external LLM.
    return source_text.strip()


def run_prototype(request: PrototypeRequest) -> PrototypeResponse:
    retrieval = retrieve(
        RetrievalRequest(
            query=request.question,
            relevant_date=request.relevant_date,
            top_k=request.top_k,
        )
    )
    citations = _citations(retrieval)

    if retrieval.retrieval_status == RetrievalStatus.NO_RELEVANT_PASSAGES:
        return PrototypeResponse(
            status=PrototypeStatus.NO_EVIDENCE,
            question=request.question,
            relevant_date=request.relevant_date,
            generator_mode=request.generator_mode,
            retrieval=retrieval,
            citations=citations,
            message="No reviewed passage was sufficiently relevant to the question.",
        )

    source_text = retrieval.hits[0].passage.text
    draft = _generate_explanation(request, source_text)
    verification = verify_safety(
        SafetyVerificationRequest(
            source_text=source_text,
            explanation=draft,
            temporal_verdict=retrieval.temporal_verdict,
            temporal_verification_required=request.relevant_date is not None,
        )
    )

    if verification.verdict == SafetyVerdict.SAFE:
        status = PrototypeStatus.ANSWERED
        answer = draft
        message = "The explanation passed the required temporal and meaning checks."
    elif verification.verdict == SafetyVerdict.UNSAFE:
        status = PrototypeStatus.UNSAFE_BLOCKED
        answer = None
        message = "The draft was blocked because a required safety check failed."
    else:
        status = PrototypeStatus.ABSTAINED
        answer = None
        message = "The system abstained because the available evidence was insufficient."

    return PrototypeResponse(
        status=status,
        question=request.question,
        relevant_date=request.relevant_date,
        generator_mode=request.generator_mode,
        answer=answer,
        draft_explanation=draft,
        retrieval=retrieval,
        verification=verification,
        citations=citations,
        message=message,
    )
