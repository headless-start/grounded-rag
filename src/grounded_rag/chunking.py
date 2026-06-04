"""Token-aware chunking.

Splits documents into 500-800 token chunks with 100-token overlap. Overlap matters:
a fact that straddles a chunk boundary is otherwise unretrievable. Each chunk carries
``source_path``, ``page``, ``paragraph_idx`` and a deterministic ``chunk_id`` so the
answer layer can cite it precisely and ingest can be idempotent.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import tiktoken

from .config import settings
from .models import Chunk

# A page is delivered as ordered paragraphs; chunking respects paragraph order and
# tracks which paragraph a chunk started in so citations point at the right ¶.


@dataclass
class Paragraph:
    text: str
    page: int
    paragraph_idx: int


_ENCODER = tiktoken.get_encoding("cl100k_base")


def _tokens(text: str) -> list[int]:
    return _ENCODER.encode(text)


def _detok(tokens: list[int]) -> str:
    return _ENCODER.decode(tokens).strip()


def _chunk_id(source_path: str, page: int, paragraph_idx: int, text: str) -> str:
    h = hashlib.sha1(f"{source_path}|{page}|{paragraph_idx}|{text}".encode()).hexdigest()
    return h[:16]


def chunk_paragraphs(
    paragraphs: list[Paragraph],
    source_path: str,
    *,
    max_tokens: int | None = None,
    min_tokens: int | None = None,
    overlap_tokens: int | None = None,
) -> list[Chunk]:
    """Pack ordered paragraphs into token-bounded chunks with overlap.

    Paragraphs are accumulated until adding the next would exceed ``max_tokens``; the
    chunk is then emitted and the next chunk seeded with a ``overlap_tokens`` tail of the
    previous one. A single oversized paragraph is split on token windows.
    """
    max_tokens = max_tokens or settings.chunk_max_tokens
    min_tokens = min_tokens or settings.chunk_min_tokens
    overlap_tokens = overlap_tokens if overlap_tokens is not None else settings.chunk_overlap_tokens

    chunks: list[Chunk] = []
    # buffer of (token_ids, page, paragraph_idx) we are currently packing
    buf_tokens: list[int] = []
    buf_page: int | None = None
    buf_para: int | None = None

    def flush() -> None:
        nonlocal buf_tokens, buf_page, buf_para
        if not buf_tokens or buf_page is None or buf_para is None:
            return
        text = _detok(buf_tokens)
        if not text:
            buf_tokens = []
            return
        chunks.append(
            Chunk(
                chunk_id=_chunk_id(source_path, buf_page, buf_para, text),
                text=text,
                source_path=source_path,
                page=buf_page,
                paragraph_idx=buf_para,
            )
        )
        # seed next buffer with the overlap tail
        tail = buf_tokens[-overlap_tokens:] if overlap_tokens else []
        buf_tokens = list(tail)

    for para in paragraphs:
        ptoks = _tokens(para.text)
        if not ptoks:
            continue
        # oversized single paragraph -> window it
        if len(ptoks) > max_tokens:
            flush()
            buf_tokens, buf_page, buf_para = [], None, None
            start = 0
            step = max_tokens - overlap_tokens
            while start < len(ptoks):
                window = ptoks[start : start + max_tokens]
                text = _detok(window)
                if text:
                    chunks.append(
                        Chunk(
                            chunk_id=_chunk_id(source_path, para.page, para.paragraph_idx, text),
                            text=text,
                            source_path=source_path,
                            page=para.page,
                            paragraph_idx=para.paragraph_idx,
                        )
                    )
                start += step
            continue

        if buf_page is None:
            buf_page, buf_para = para.page, para.paragraph_idx

        if len(buf_tokens) + len(ptoks) > max_tokens and len(buf_tokens) >= min_tokens:
            flush()
            # buffer now holds overlap tail from the previous chunk; attach this para to it
            buf_page, buf_para = para.page, para.paragraph_idx
        buf_tokens.extend(ptoks)

    flush()
    return chunks
