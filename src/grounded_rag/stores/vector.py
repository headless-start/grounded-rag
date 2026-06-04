"""Vector store behind a thin interface so Weaviate/Qdrant can drop in later.

Only the Chroma implementation exists today; the ``VectorStore`` Protocol is the seam.
We pass precomputed embeddings in (Chroma's default embedder is bypassed) so the
embedder stays swappable independently of the store.
"""

from __future__ import annotations

from typing import Any, Protocol, cast

from ..config import settings
from ..log import get_logger
from ..models import Chunk, RetrievedChunk

log = get_logger(__name__)


class VectorStore(Protocol):
    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None: ...
    def delete_sources(self, source_paths: list[str]) -> None: ...
    def query(self, embedding: list[float], top_k: int) -> list[RetrievedChunk]: ...
    def all_chunks(self) -> list[Chunk]: ...
    def count(self) -> int: ...


def _chunk_to_metadata(c: Chunk) -> dict[str, str | int]:
    return {
        "source_path": c.source_path,
        "page": c.page,
        "paragraph_idx": c.paragraph_idx,
    }


def _metadata_to_chunk(chunk_id: str, text: str, md: dict) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        source_path=str(md["source_path"]),
        page=int(md["page"]),
        paragraph_idx=int(md["paragraph_idx"]),
    )


class ChromaVectorStore:
    def __init__(self, persist_dir: str | None = None, collection: str | None = None) -> None:
        import chromadb

        path = persist_dir or str(settings.resolved_chroma_dir)
        self._client = chromadb.PersistentClient(path=path)
        self._collection_name = collection or settings.collection
        # cosine space matches our normalized embeddings
        self._col = self._client.get_or_create_collection(
            name=self._collection_name, metadata={"hnsw:space": "cosine"}
        )
        log.info("chroma_ready", path=path, collection=self._collection_name)

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        # upsert => idempotent: re-ingesting the same chunk_id overwrites, never duplicates.
        self._col.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=cast(Any, embeddings),
            documents=[c.text for c in chunks],
            metadatas=cast(Any, [_chunk_to_metadata(c) for c in chunks]),
        )

    def delete_sources(self, source_paths: list[str]) -> None:
        """Remove all chunks belonging to the given source files.

        Keeps re-ingest idempotent across edits: a document whose text changed yields new
        content-hash ids, so without this its old chunks would linger as stale duplicates.
        """
        if not source_paths:
            return
        self._col.delete(where=cast(Any, {"source_path": {"$in": source_paths}}))

    def query(self, embedding: list[float], top_k: int) -> list[RetrievedChunk]:
        res = self._col.query(
            query_embeddings=cast(Any, [embedding]),
            n_results=top_k,
            include=cast(Any, ["documents", "metadatas", "distances"]),
        )
        ids = res["ids"][0]
        docs = (res["documents"] or [[]])[0]
        metas = (res["metadatas"] or [[]])[0]
        dists = (res["distances"] or [[]])[0]
        out: list[RetrievedChunk] = []
        for cid, text, md, dist in zip(ids, docs, metas, dists, strict=True):
            # cosine distance -> similarity in [0, 1]-ish
            out.append(
                RetrievedChunk(
                    chunk=_metadata_to_chunk(cid, text, dict(md)),
                    score=1.0 - float(dist),
                    retriever="dense",
                )
            )
        return out

    def all_chunks(self) -> list[Chunk]:
        res = self._col.get(include=cast(Any, ["documents", "metadatas"]))
        ids = res["ids"]
        docs = res["documents"] or []
        metas = res["metadatas"] or []
        return [
            _metadata_to_chunk(cid, text, dict(md))
            for cid, text, md in zip(ids, docs, metas, strict=True)
        ]

    def count(self) -> int:
        return int(self._col.count())


def get_vector_store() -> VectorStore:
    return ChromaVectorStore()
