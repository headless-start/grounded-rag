"""Ingestion: load -> chunk -> embed -> persist.

Idempotent by construction: chunk_ids are content-hashed and the vector store upserts,
so re-running over an unchanged corpus updates in place and never duplicates. Run as a
module: ``python -m grounded_rag.ingest``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .chunking import chunk_paragraphs
from .config import settings
from .embed import Embedder, get_embedder
from .loaders import discover, load_document
from .log import get_logger
from .models import Chunk
from .stores.vector import VectorStore, get_vector_store

log = get_logger(__name__)


def build_chunks(data_dir: Path) -> list[Chunk]:
    files = discover(data_dir)
    if not files:
        log.warning("no_documents_found", data_dir=str(data_dir))
        return []
    chunks: list[Chunk] = []
    for path in files:
        paras = load_document(path)
        file_chunks = chunk_paragraphs(paras, source_path=str(path))
        log.info("chunked_file", file=str(path), paragraphs=len(paras), chunks=len(file_chunks))
        chunks.extend(file_chunks)
    return chunks


def ingest(
    data_dir: Path | None = None,
    *,
    embedder: Embedder | None = None,
    store: VectorStore | None = None,
    batch_size: int = 64,
) -> int:
    data_dir = data_dir or settings.resolved_data_dir
    embedder = embedder or get_embedder()
    store = store or get_vector_store()

    chunks = build_chunks(data_dir)
    if not chunks:
        return 0

    # Idempotent across edits: drop any existing chunks for the files we're about to
    # (re)ingest, then upsert fresh. Re-running on an unchanged corpus is a no-op net.
    touched_sources = sorted({c.source_path for c in chunks})
    store.delete_sources(touched_sources)

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        vectors = embedder.embed_documents([c.text for c in batch])
        store.upsert(batch, vectors)
        log.info("upserted_batch", start=start, size=len(batch))

    total = store.count()
    log.info("ingest_complete", chunks_ingested=len(chunks), collection_total=total)
    return len(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest the corpus into the vector store.")
    parser.add_argument("--data-dir", type=Path, default=None)
    args = parser.parse_args()
    n = ingest(args.data_dir)
    print(f"ingested {n} chunks")


if __name__ == "__main__":
    main()
