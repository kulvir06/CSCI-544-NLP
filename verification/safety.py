"""Combine temporal applicability and meaning preservation conservatively."""

from retrieval.schema import TemporalVerdict
from verification.meaning import verify_meaning
from verification.schema import (
    MeaningVerificationRequest,
    MeaningVerdict,
    SafetyVerificationRequest,
    SafetyVerificationResponse,
    SafetyVerdict,
)


def verify_safety(request: SafetyVerificationRequest) -> SafetyVerificationResponse:
    meaning = verify_meaning(
        MeaningVerificationRequest(
            source_text=request.source_text,
            explanation=request.explanation,
        )
    )
    reasons: list[str] = []

    if request.temporal_verification_required:
        if request.temporal_verdict == TemporalVerdict.INSUFFICIENT_TEMPORAL_EVIDENCE:
            reasons.append("Temporal applicability could not be established.")
            return SafetyVerificationResponse(
                verdict=SafetyVerdict.ABSTAIN,
                temporal_verdict=request.temporal_verdict,
                meaning=meaning,
                reasons=reasons,
            )
        if request.temporal_verdict != TemporalVerdict.VERIFIED_APPLICABLE:
            reasons.append("The supplied source version is not verified for the relevant date.")
            return SafetyVerificationResponse(
                verdict=SafetyVerdict.UNSAFE,
                temporal_verdict=request.temporal_verdict,
                meaning=meaning,
                reasons=reasons,
            )

    if meaning.verdict == MeaningVerdict.FAIL:
        reasons.append("At least one legally important fact was omitted or contradicted.")
        verdict = SafetyVerdict.UNSAFE
    elif meaning.verdict == MeaningVerdict.REVIEW:
        reasons.append("No checkable atomic facts were found; human review is required.")
        verdict = SafetyVerdict.ABSTAIN
    else:
        reasons.append("Temporal and meaning checks required for this request passed.")
        verdict = SafetyVerdict.SAFE

    return SafetyVerificationResponse(
        verdict=verdict,
        temporal_verdict=request.temporal_verdict,
        meaning=meaning,
        reasons=reasons,
    )
