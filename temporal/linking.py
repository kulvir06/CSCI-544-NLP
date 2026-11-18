"""Conservative, local temporal linking over cached atomic claims."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

from temporal.schema import TemporalClaim, TemporalRole


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)?")
NUMBER_PATTERN = re.compile(r"\b\d+(?:\.\d+)?\b")
FORM_PATTERN = re.compile(r"\bForm\s+(I-\d+[A-Z]?)\b", re.I)
CFR_PATTERN = re.compile(r"\b(?:8\s+)?CFR\s+\d+(?:\.\d+)?(?:\([^)]*\))*", re.I)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "in", "is", "it", "of", "on", "or", "that", "the", "this",
    "to", "was", "will", "with",
}


def _tokens(text: str) -> set[str]:
    return {
        token.casefold()
        for token in TOKEN_PATTERN.findall(text)
        if token.casefold() not in STOPWORDS
    }


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or len(left) != len(right):
        return 0.0
    denominator = math.sqrt(sum(x * x for x in left)) * math.sqrt(
        sum(x * x for x in right)
    )
    if denominator == 0:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / denominator


def _signature(text: str) -> tuple[set[str], set[str], set[str]]:
    forms = {value.upper() for value in FORM_PATTERN.findall(text)}
    cfr = {" ".join(value.casefold().split()) for value in CFR_PATTERN.findall(text)}
    quantity_text = FORM_PATTERN.sub(" ", CFR_PATTERN.sub(" ", text))
    return (
        set(NUMBER_PATTERN.findall(quantity_text)),
        forms,
        cfr,
    )


def _compatible(target: TemporalClaim, donor: TemporalClaim) -> tuple[bool, float]:
    target_numbers, target_forms, target_cfr = _signature(target.text)
    donor_numbers, donor_forms, donor_cfr = _signature(donor.text)
    if target_numbers != donor_numbers:
        return False, 0.0
    if target_forms and donor_forms and target_forms != donor_forms:
        return False, 0.0
    if target_cfr and donor_cfr and not (target_cfr & donor_cfr):
        return False, 0.0

    target_tokens = _tokens(target.text)
    donor_tokens = _tokens(donor.text)
    overlap = len(target_tokens & donor_tokens) / len(target_tokens) if target_tokens else 0.0
    semantic = _cosine(target.embedding, donor.embedding)
    exact_reference = bool(target_cfr & donor_cfr)
    accepted = (
        exact_reference and semantic >= 0.72
    ) or (
        semantic >= 0.82 and overlap >= 0.30 and len(target_tokens & donor_tokens) >= 3
    )
    confidence = min(0.99, 0.65 * semantic + 0.35 * overlap)
    return accepted, confidence


def link_cached_claims(
    claims: list[TemporalClaim],
    minimum_confidence: float = 0.78,
) -> tuple[list[TemporalClaim], dict]:
    """Propagate supported start dates between near-equivalent cached claims.

    Only operative claims can donate or receive dates. Numeric, form, and CFR
    signatures prevent superficially similar but legally different provisions
    from linking. Existing intervals are never overwritten.
    """
    donors = [
        claim for claim in claims
        if claim.temporal_role == TemporalRole.OPERATIVE
        and (claim.valid_from or claim.valid_as_of)
        and claim.temporal_confidence >= 0.65
    ]
    linked: list[TemporalClaim] = []
    linked_count = 0
    method_counts: Counter[str] = Counter()

    for claim in claims:
        payload = claim.model_dump()
        payload.update(
            {
                "base_valid_from": claim.valid_from,
                "base_valid_to": claim.valid_to,
                "base_valid_as_of": claim.valid_as_of,
            }
        )
        if (
            claim.temporal_role != TemporalRole.OPERATIVE
            or claim.valid_from
            or claim.valid_as_of
        ):
            linked.append(TemporalClaim.model_validate(payload))
            continue

        candidates: list[tuple[float, TemporalClaim]] = []
        for donor in donors:
            if donor.claim_id == claim.claim_id:
                continue
            compatible, confidence = _compatible(claim, donor)
            if compatible and confidence >= minimum_confidence:
                candidates.append((confidence, donor))
        if not candidates:
            linked.append(TemporalClaim.model_validate(payload))
            continue

        candidates.sort(key=lambda item: (-item[0], item[1].claim_id))
        confidence, donor = candidates[0]
        if donor.valid_from:
            payload["valid_from"] = donor.valid_from
            payload["linked_valid_from"] = donor.valid_from
        elif donor.valid_as_of:
            payload["valid_as_of"] = donor.valid_as_of
            payload["linked_valid_as_of"] = donor.valid_as_of
        payload["temporal_link_sources"] = [donor.claim_id]
        payload["temporal_link_method"] = "cached_claim_semantic_equivalence"
        payload["temporal_link_confidence"] = round(confidence, 4)
        payload["date_basis"] = (
            f"{claim.date_basis} | Linked from equivalent cached claim "
            f"{donor.claim_id}: {donor.date_basis}"
        )
        linked.append(TemporalClaim.model_validate(payload))
        linked_count += 1
        method_counts["cached_claim_semantic_equivalence"] += 1

    return linked, {
        "total_claims": len(claims),
        "dated_donors": len(donors),
        "newly_linked_claims": linked_count,
        "still_without_temporal_scope": sum(
            not (claim.valid_from or claim.valid_as_of) for claim in linked
        ),
        "methods": dict(method_counts),
        "minimum_confidence": minimum_confidence,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_linked_index(
    base_path: Path,
    output_path: Path,
    metadata_path: Path,
    minimum_confidence: float = 0.78,
) -> dict:
    claims = [
        TemporalClaim.model_validate_json(line)
        for line in base_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    linked, report = link_cached_claims(claims, minimum_confidence)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(claim.model_dump_json() for claim in linked) + "\n",
        encoding="utf-8",
    )
    report.update(
        {
            "base_index": str(base_path.resolve()),
            "base_sha256": _sha256(base_path),
            "linked_index": str(output_path.resolve()),
        }
    )
    metadata_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def linked_index_is_current(
    base_path: Path, linked_path: Path, metadata_path: Path
) -> bool:
    if not base_path.exists() or not linked_path.exists() or not metadata_path.exists():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return metadata.get("base_sha256") == _sha256(base_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
