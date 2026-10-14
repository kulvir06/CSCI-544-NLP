"""Deterministic helpers for drafting subtle negative cases.

Generated cases are candidates only and must be reviewed before benchmark use.
"""

from __future__ import annotations

from datetime import datetime, timedelta
import re

from verification.meaning import DATE_PATTERN, FORM_PATTERN, QUANTITY_PATTERN


def change_first_quantity(text: str) -> str:
    match = QUANTITY_PATTERN.search(text)
    if not match:
        raise ValueError("No quantity found")
    replacement = str(int(float(match.group(1))) + 1)
    return text[: match.start(1)] + replacement + text[match.end(1) :]


def change_first_form(text: str) -> str:
    match = FORM_PATTERN.search(text)
    if not match:
        raise ValueError("No form identifier found")
    replacement = "I-765" if match.group(1).upper() != "I-765" else "I-983"
    return text[: match.start(1)] + replacement + text[match.end(1) :]


def reverse_first_negation(text: str) -> str:
    replacements = (
        (r"\bmust not\b", "may"),
        (r"\bmay not\b", "may"),
        (r"\bcannot\b", "can"),
        (r"\bno more than\b", "more than"),
        (r"\bnot more than\b", "more than"),
    )
    for pattern, replacement in replacements:
        changed, count = re.subn(pattern, replacement, text, count=1, flags=re.I)
        if count:
            return changed
    raise ValueError("No supported negative expression found")


def change_first_date(text: str) -> str:
    match = DATE_PATTERN.search(text)
    if not match:
        raise ValueError("No supported calendar date found")
    value = match.group(0)
    for date_format in ("%B %d, %Y", "%Y-%m-%d", "%m/%d/%Y"):
        try:
            parsed = datetime.strptime(value.title(), date_format)
            replacement = (parsed + timedelta(days=1)).strftime(date_format)
            return text[: match.start()] + replacement + text[match.end() :]
        except ValueError:
            continue
    raise ValueError("Date could not be normalized")


def remove_clause_from_cue(text: str, cue: str) -> str:
    pattern = re.compile(rf"\s*\b{re.escape(cue)}\b[^.;]*", re.I)
    changed, count = pattern.subn("", text, count=1)
    if not count:
        raise ValueError(f"Cue not found: {cue}")
    return changed.strip()

