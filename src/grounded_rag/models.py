"""Shared data types passed between ingest, retrieval, and answer layers."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """A unit of text plus the metadata needed to cite it."""

    chunk_id: str
    text: str
    source_path: str
    page: int
    paragraph_idx: int

    def citation_label(self) -> str:
        """Human-facing citation tag, e.g. '[handbook.pdf p4 ¶2]'."""
        from pathlib import Path

        name = Path(self.source_path).name
        return f"[{name} p{self.page} ¶{self.paragraph_idx}]"


class RetrievedChunk(BaseModel):
    """A chunk returned by retrieval, with its score and how it was found."""

    chunk: Chunk
    score: float
    retriever: str = "dense"  # dense | bm25 | rrf | rerank


class Citation(BaseModel):
    chunk_id: str
    label: str
    source_path: str
    page: int
    paragraph_idx: int


class Answer(BaseModel):
    answer: str
    refused: bool = False
    citations: list[Citation] = Field(default_factory=list)
    contexts: list[RetrievedChunk] = Field(default_factory=list)
