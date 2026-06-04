"""Document loaders. Each yields ordered ``Paragraph`` objects with page + paragraph_idx.

paragraph_idx is per-page and 1-based so citations read naturally ('p4 ¶2'). Markdown has
no pages, so it is treated as a single page 1.
"""

from __future__ import annotations

import re
from pathlib import Path

from .chunking import Paragraph

SUPPORTED_SUFFIXES = {".pdf", ".md", ".markdown", ".txt"}

_PARA_SPLIT = re.compile(r"\n\s*\n")


def _split_paragraphs(text: str, page: int) -> list[Paragraph]:
    paras: list[Paragraph] = []
    idx = 1
    for block in _PARA_SPLIT.split(text):
        cleaned = " ".join(block.split()).strip()
        if cleaned:
            paras.append(Paragraph(text=cleaned, page=page, paragraph_idx=idx))
            idx += 1
    return paras


def load_pdf(path: Path) -> list[Paragraph]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    out: list[Paragraph] = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        out.extend(_split_paragraphs(text, page_num))
    return out


def load_markdown(path: Path) -> list[Paragraph]:
    text = path.read_text(encoding="utf-8")
    return _split_paragraphs(text, page=1)


def load_document(path: Path) -> list[Paragraph]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf(path)
    if suffix in {".md", ".markdown", ".txt"}:
        return load_markdown(path)
    raise ValueError(f"unsupported file type: {path}")


def discover(data_dir: Path) -> list[Path]:
    return sorted(
        p for p in data_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    )
