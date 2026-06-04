"""Load versioned prompt templates by name from this directory."""

from __future__ import annotations

from functools import cache
from pathlib import Path

_PROMPT_DIR = Path(__file__).resolve().parent


@cache
def load_prompt(name: str) -> str:
    """Return the raw template text for ``name`` (e.g. 'answer_v1')."""
    path = _PROMPT_DIR / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"prompt template not found: {path}")
    return path.read_text(encoding="utf-8")
