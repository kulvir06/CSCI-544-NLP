"""Local file parsers. No network access is performed here."""

from __future__ import annotations

import re
from pathlib import Path

from bs4 import BeautifulSoup
from pypdf import PdfReader

from corpus.schema import ParsedBlock


class UnsupportedDocumentError(ValueError):
    pass


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _paragraph_blocks(text: str, page_number: int | None = None) -> list[ParsedBlock]:
    raw_blocks = re.split(r"\n\s*\n|(?<=\.)\s*\n", text)
    blocks: list[ParsedBlock] = []
    for raw in raw_blocks:
        cleaned = _clean(raw)
        if len(cleaned) < 20:
            continue
        blocks.append(
            ParsedBlock(
                text=cleaned,
                page_number=page_number,
                block_index=len(blocks),
            )
        )
    return blocks


def parse_pdf(path: Path) -> list[ParsedBlock]:
    blocks: list[ParsedBlock] = []
    reader = PdfReader(str(path))
    for page_index, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        for block in _paragraph_blocks(page_text, page_index):
            block.block_index = len(blocks)
            blocks.append(block)
    return blocks


def parse_html(path: Path) -> list[ParsedBlock]:
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"), "html.parser")
    for node in soup(["script", "style", "nav", "footer"]):
        node.decompose()

    blocks: list[ParsedBlock] = []
    current_section: str | None = None
    for node in soup.find_all(["h1", "h2", "h3", "h4", "p", "li"]):
        text = _clean(node.get_text(" ", strip=True))
        if not text:
            continue
        if node.name in {"h1", "h2", "h3", "h4"}:
            current_section = text
            continue
        if len(text) < 20:
            continue
        blocks.append(
            ParsedBlock(
                text=text,
                section=current_section,
                block_index=len(blocks),
            )
        )
    return blocks


def parse_text(path: Path) -> list[ParsedBlock]:
    return _paragraph_blocks(path.read_text(encoding="utf-8", errors="replace"))


def parse_document(path: Path) -> list[ParsedBlock]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return parse_pdf(path)
    if suffix in {".html", ".htm"}:
        return parse_html(path)
    if suffix == ".txt":
        return parse_text(path)
    raise UnsupportedDocumentError(f"Unsupported file type: {suffix}")
