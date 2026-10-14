"""Rule-aware passage scoring for F-1 employment and extension material."""

from __future__ import annotations

import re

from corpus.schema import ParsedBlock, Passage


TOPIC_PATTERNS = {
    "f1_status": re.compile(r"\bF-?1\b", re.I),
    "employment": re.compile(r"\b(employ(?:ment|er|ed)?|work authorization)\b", re.I),
    "opt": re.compile(r"\b(?:STEM\s+)?OPT\b|optional practical training", re.I),
    "cpt": re.compile(r"\bCPT\b|curricular practical training", re.I),
    "cap_gap": re.compile(r"cap[- ]gap", re.I),
    "extension": re.compile(r"\bextend(?:ed|s|ing)?|extension\b", re.I),
}
RULE_PATTERNS = {
    "obligation": re.compile(r"\b(must|required|shall|may not|cannot|prohibited)\b", re.I),
    "condition": re.compile(r"\b(if|only if|provided that|subject to|eligible|qualif(?:y|ies|ied))\b", re.I),
    "exception": re.compile(r"\b(unless|except|however|notwithstanding)\b", re.I),
    "quantity": re.compile(r"\b\d+(?:\.\d+)?\s*(?:days?|months?|years?|hours?|percent|%)\b", re.I),
    "form": re.compile(r"\bForm\s+I-\d+[A-Z]?\b", re.I),
}
TEMPORAL_PATTERN = re.compile(
    r"\b(effective|beginning|ending|before|after|until|on or after|as of)\b|"
    r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b",
    re.I,
)


def score_block(block: ParsedBlock) -> tuple[float, list[str]]:
    reasons: list[str] = []
    topic_hits = [name for name, pattern in TOPIC_PATTERNS.items() if pattern.search(block.text)]
    rule_hits = [name for name, pattern in RULE_PATTERNS.items() if pattern.search(block.text)]

    score = 1.25 * len(topic_hits) + 0.75 * len(rule_hits)
    reasons.extend(f"topic:{name}" for name in topic_hits)
    reasons.extend(f"rule:{name}" for name in rule_hits)

    if TEMPORAL_PATTERN.search(block.text):
        score += 1.5
        reasons.append("temporal_expression")
    if block.section and any(pattern.search(block.section) for pattern in TOPIC_PATTERNS.values()):
        score += 1.0
        reasons.append("relevant_section")
    return score, reasons


def select_passages(
    document_id: str,
    blocks: list[ParsedBlock],
    minimum_score: float = 2.0,
) -> list[Passage]:
    passages: list[Passage] = []
    for block in blocks:
        score, reasons = score_block(block)
        if score < minimum_score:
            continue
        passages.append(
            Passage(
                passage_id=f"{document_id}-p{len(passages) + 1:04d}",
                document_id=document_id,
                text=block.text,
                section=block.section,
                page_number=block.page_number,
                block_index=block.block_index,
                relevance_score=round(score, 3),
                selection_reasons=reasons,
                requires_review=score < 3.5,
            )
        )
    return passages
