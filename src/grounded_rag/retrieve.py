"""Retrieval.

Phase 1: dense top-k only. The ``Retriever`` shape (``retrieve(query, top_n) ->
list[RetrievedChunk]``) is the seam Phase 2 extends with BM25 + RRF fusion + a
cross-encoder re-ranker, without changing the answer layer's call site.
"""

from __future__ import annotations

from .config import settings
from .embed import Embedder, get_embedder
from .log import get_logger
from .models import RetrievedChunk
from .stores.vector import VectorStore, get_vector_store

log = get_logger(__name__)


class DenseRetriever:
    def __init__(self, embedder: Embedder | None = None, store: VectorStore | None = None) -> None:
        self._embedder = embedder or get_embedder()
        self._store = store or get_vector_store()

    def retrieve(self, query: str, top_n: int | None = None) -> list[RetrievedChunk]:
        top_n = top_n or settings.top_n
        qvec = self._embedder.embed_query(query)
        results = self._store.query(qvec, top_k=settings.top_k)
        log.info("dense_retrieve", query=query, candidates=len(results), top_n=top_n)
        return results[:top_n]


def get_retriever() -> DenseRetriever:
    return DenseRetriever()
