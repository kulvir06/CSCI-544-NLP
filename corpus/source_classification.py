"""Offline source-authority classification.

This module only inspects metadata supplied with a file. It never opens URLs.
"""

from __future__ import annotations

from corpus.schema import DocumentMetadata, SourceTier


PRIMARY_MARKERS = (
    "ecfr.gov",
    "federalregister.gov",
    "govinfo.gov",
    "federal register",
    "electronic code of federal regulations",
)
FEDERAL_GUIDANCE_MARKERS = (
    "uscis.gov",
    "ice.gov",
    "studyinthestates.dhs.gov",
    "department of homeland security",
    "uscis",
    "student and exchange visitor program",
    "sevp",
)
UNIVERSITY_MARKERS = (
    ".edu",
    "university",
    "college",
    "international students office",
    "office of international services",
)


def classify_source(metadata: DocumentMetadata) -> SourceTier:
    haystack = " ".join(
        value
        for value in (metadata.source_url, metadata.issuer, metadata.title)
        if value
    ).lower()

    if any(marker in haystack for marker in PRIMARY_MARKERS):
        return SourceTier.PRIMARY_LAW
    if any(marker in haystack for marker in FEDERAL_GUIDANCE_MARKERS):
        return SourceTier.FEDERAL_GUIDANCE
    if any(marker in haystack for marker in UNIVERSITY_MARKERS):
        return SourceTier.UNIVERSITY_ADVISORY
    return SourceTier.UNKNOWN
