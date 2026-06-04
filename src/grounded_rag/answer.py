"""Answer assembly + citation enforcement + refusal.

Flow: retrieve contexts -> if too few, refuse before calling the LLM -> assemble a prompt
whose context blocks are numbered ``[1] [2] ...`` -> generate -> resolve the bracketed
indices the model cited back to the chunks they point at. A claim that grounds in nothing
real is refused rather than surfaced as a hallucination.

Citing by index (the model only has to copy a digit) is far more reliable than asking a
small model to reproduce a full ``[file pX ¶Y]`` label verbatim — with coarse chunks the
model tends to invent a sub-paragraph number, which enforcement then rejects. We resolve
the indices ourselves and rewrite them into the human ``[file pX ¶Y]`` form in the answer.
"""

from __future__ import annotations

import re

from .config import settings
from .llm import LLMClient, get_llm
from .log import get_logger
from .models import Answer, Citation, RetrievedChunk
from .prompts import load_prompt
from .retrieve import DenseRetriever, get_retriever

log = get_logger(__name__)

REFUSAL_TEXT = "insufficient evidence in the provided documents"

# matches numeric context citations the model emits, e.g. [1] or [1, 3] or [2][4]
_INDEX_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def _format_context(contexts: list[RetrievedChunk]) -> str:
    # Number each block; the model cites these numbers, we map them back to chunks.
    return "\n\n".join(f"[{i}] {rc.chunk.text}" for i, rc in enumerate(contexts, start=1))


def _cited_indices(text: str) -> list[int]:
    """Distinct 1-based indices cited in the answer, in first-seen order."""
    out: list[int] = []
    for group in _INDEX_RE.findall(text):
        for part in group.split(","):
            n = int(part.strip())
            if n not in out:
                out.append(n)
    return out


class AnswerService:
    def __init__(
        self,
        retriever: DenseRetriever | None = None,
        llm: LLMClient | None = None,
        prompt_version: str | None = None,
    ) -> None:
        self._retriever = retriever or get_retriever()
        self._llm = llm or get_llm()
        self._template = load_prompt(prompt_version or settings.answer_prompt_version)

    def ask(self, question: str) -> Answer:
        contexts = self._retriever.retrieve(question)

        # Pre-generation refusal: not enough evidence to even attempt grounding.
        if len(contexts) < settings.refusal_min_contexts:
            log.info("refuse_no_context", question=question, n=len(contexts))
            return Answer(answer=REFUSAL_TEXT, refused=True, citations=[], contexts=contexts)

        prompt = self._template.format(context=_format_context(contexts), question=question)
        raw = self._llm.complete(system="Answer strictly from the context.", user=prompt)

        # Explicit refusal from the model.
        if REFUSAL_TEXT in raw.lower():
            log.info("model_refused", question=question)
            return Answer(answer=REFUSAL_TEXT, refused=True, citations=[], contexts=contexts)

        text, citations = self._resolve_citations(raw, contexts)

        # Post-generation enforcement: an answer that grounds in nothing real is a refusal.
        if not citations:
            log.warning("ungrounded_answer_refused", question=question, raw=raw[:200])
            return Answer(answer=REFUSAL_TEXT, refused=True, citations=[], contexts=contexts)

        return Answer(answer=text, refused=False, citations=citations, contexts=contexts)

    def _resolve_citations(
        self, text: str, contexts: list[RetrievedChunk]
    ) -> tuple[str, list[Citation]]:
        """Map the model's numeric citations to real chunks and rewrite them as labels.

        Returns the answer text with every valid ``[n]`` replaced by the chunk's human
        ``[file pX ¶Y]`` label (invalid indices are dropped), plus the deduped citation
        list. An index outside ``1..len(contexts)`` cannot be grounded, so it is discarded.
        """
        citations: list[Citation] = []
        seen: set[str] = set()

        def replace(match: re.Match[str]) -> str:
            labels: list[str] = []
            for part in match.group(1).split(","):
                n = int(part.strip())
                if not (1 <= n <= len(contexts)):
                    continue  # fabricated index -> not grounded, drop it
                chunk = contexts[n - 1].chunk
                label = chunk.citation_label()
                labels.append(label)
                if chunk.chunk_id not in seen:
                    seen.add(chunk.chunk_id)
                    citations.append(
                        Citation(
                            chunk_id=chunk.chunk_id,
                            label=label,
                            source_path=chunk.source_path,
                            page=chunk.page,
                            paragraph_idx=chunk.paragraph_idx,
                        )
                    )
            return " ".join(labels)

        rewritten = _INDEX_RE.sub(replace, text)
        # collapse any double spaces left where an invalid index was dropped
        rewritten = re.sub(r" {2,}", " ", rewritten).strip()
        return rewritten, citations


def get_answer_service() -> AnswerService:
    return AnswerService()
