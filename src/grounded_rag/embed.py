"""Embedding behind a swappable interface.

Default impl wraps ``sentence-transformers`` with bge-small. bge models want a query
prefix for retrieval; documents are embedded plain.
"""

from __future__ import annotations

from typing import Protocol

from .config import settings
from .log import get_logger

log = get_logger(__name__)

# bge-* retrieval instruction; applied to queries only, not documents.
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...
    @property
    def dim(self) -> int: ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name or settings.embed_model
        log.info("loading_embedder", model=self.model_name)
        self._model = SentenceTransformer(self.model_name)
        self._is_bge = "bge" in self.model_name.lower()

    @property
    def dim(self) -> int:
        return int(self._model.get_sentence_embedding_dimension())

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vecs = self._model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True
        )
        return vecs.tolist()

    def embed_query(self, text: str) -> list[float]:
        q = f"{_BGE_QUERY_PREFIX}{text}" if self._is_bge else text
        vec = self._model.encode(
            [q], normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True
        )
        return vec[0].tolist()


def get_embedder() -> Embedder:
    return SentenceTransformerEmbedder()
