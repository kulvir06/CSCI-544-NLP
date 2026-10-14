"""Conservative metadata inference from uploaded content.

Inferred fields are candidates, not legal conclusions. URLs are never opened.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from corpus.schema import (
    DocumentMetadata,
    DocumentStatus,
    MetadataCandidates,
    ParsedBlock,
)


MONTHS = (
    "January|February|March|April|May|June|July|August|"
    "September|October|November|December"
)
DATE_PATTERN = re.compile(
    rf"\b(?:{MONTHS})\s+\d{{1,2}},\s+\d{{4}}\b|"
    r"\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b"
)
EFFECTIVE_PATTERN = re.compile(
    rf"\b(?:effective|takes? effect|effective beginning|effective from)\s*"
    rf"(?:on\s*)?((?:{MONTHS})\s+\d{{1,2}},\s+\d{{4}}|\d{{1,2}}[/-]\d{{1,2}}[/-]\d{{4}})",
    re.I,
)
VALID_AS_OF_PATTERN = re.compile(
    rf"\b(?:up to date as of|current as of|valid as of|as of)\s*"
    rf"((?:{MONTHS})\s+\d{{1,2}},\s+\d{{4}}|\d{{1,2}}[/-]\d{{1,2}}[/-]\d{{4}})",
    re.I,
)


def _infer_issuer(filename: str) -> Optional[str]:
    """Infer only from a source-identifying filename, not cited body text."""
    lower = filename.lower()
    if (
        "federal register" in lower
        or "code of federal regulations" in lower
        or re.search(r"\b8\s+cfr\b", lower)
    ):
        return "U.S. Federal Regulation"
    if "u.s. citizenship and immigration services" in lower or "uscis" in lower:
        return "USCIS"
    if "student and exchange visitor program" in lower or "sevp" in lower:
        return "SEVP"
    if "immigration and customs enforcement" in lower or re.search(r"\bice\b", lower):
        return "ICE"
    return None


def _infer_rule_family(text: str) -> Optional[str]:
    lower = text.lower()
    if "cap-gap" in lower or "cap gap" in lower:
        return "cap_gap"
    if "stem opt" in lower:
        return "stem_opt"
    if "curricular practical training" in lower or re.search(r"\bcpt\b", lower):
        return "cpt"
    if "optional practical training" in lower or re.search(r"\bopt\b", lower):
        return "opt"
    if "f-1" in lower or "f1" in lower:
        return "general_f1"
    return None


def _infer_status(text: str) -> DocumentStatus:
    lower = text.lower()
    if any(term in lower for term in ("superseded", "obsolete", "no longer in effect")):
        return DocumentStatus.SUPERSEDED
    if any(
        term in lower
        for term in ("proposed rule", "notice of proposed rulemaking", "draft")
    ):
        return DocumentStatus.PROPOSED
    return DocumentStatus.UNKNOWN


def enrich_metadata(
    metadata: DocumentMetadata,
    blocks: list[ParsedBlock],
) -> tuple[DocumentMetadata, MetadataCandidates]:
    sample = "\n".join(block.text for block in blocks[:20])
    context = f"{metadata.original_filename}\n{sample}"
    inferred_fields: list[str] = []

    if not metadata.title:
        metadata.title = Path(metadata.original_filename).stem.replace("_", " ")
        inferred_fields.append("title")
    if not metadata.issuer:
        metadata.issuer = _infer_issuer(metadata.original_filename)
        if metadata.issuer:
            inferred_fields.append("issuer")
    if not metadata.rule_family:
        metadata.rule_family = _infer_rule_family(context)
        if metadata.rule_family:
            inferred_fields.append("rule_family")
    if metadata.status == DocumentStatus.UNKNOWN:
        inferred_status = _infer_status(context)
        if inferred_status != DocumentStatus.UNKNOWN:
            metadata.status = inferred_status
            inferred_fields.append("status")

    date_mentions = list(dict.fromkeys(DATE_PATTERN.findall(context)))[:20]
    effective_candidates = list(
        dict.fromkeys(match.group(1) for match in EFFECTIVE_PATTERN.finditer(context))
    )[:10]
    valid_as_of_candidates = list(
        dict.fromkeys(match.group(1) for match in VALID_AS_OF_PATTERN.finditer(context))
    )[:10]

    return metadata, MetadataCandidates(
        date_mentions=date_mentions,
        effective_date_candidates=effective_candidates,
        valid_as_of_candidates=valid_as_of_candidates,
        inferred_fields=inferred_fields,
    )
