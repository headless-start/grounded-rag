"""Answer assembly + citation enforcement + refusal.

Flow: retrieve contexts -> if too few, refuse before calling the LLM -> assemble a
prompt that demands inline citations -> generate -> verify every cited label maps to a
chunk we actually retrieved. A claim that cites nothing real is not grounded, so we
refuse rather than surface a hallucination.
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

# matches inline citation labels like [handbook.pdf p4 ¶2]
_CITATION_RE = re.compile(r"\[[^\[\]]+?\sp\d+\s¶\d+\]")


def _format_context(contexts: list[RetrievedChunk]) -> str:
    blocks = []
    for rc in contexts:
        label = rc.chunk.citation_label()
        blocks.append(f"{label}\n{rc.chunk.text}")
    return "\n\n".join(blocks)


def _split_citation(label: str) -> tuple[str, int, int] | None:
    # '[name p4 ¶2]' -> (name, 4, 2)
    m = re.match(r"\[(.+)\sp(\d+)\s¶(\d+)\]$", label)
    if not m:
        return None
    return m.group(1), int(m.group(2)), int(m.group(3))


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

        citations = self._verify_citations(raw, contexts)

        # Post-generation enforcement: an answer that grounds in nothing real is a refusal.
        if not citations:
            log.warning("ungrounded_answer_refused", question=question, raw=raw[:200])
            return Answer(answer=REFUSAL_TEXT, refused=True, citations=[], contexts=contexts)

        return Answer(answer=raw, refused=False, citations=citations, contexts=contexts)

    def _verify_citations(self, text: str, contexts: list[RetrievedChunk]) -> list[Citation]:
        """Keep only citations whose label matches a chunk that was actually retrieved."""
        by_label = {rc.chunk.citation_label(): rc.chunk for rc in contexts}
        seen: set[str] = set()
        out: list[Citation] = []
        for label in _CITATION_RE.findall(text):
            if label in by_label and label not in seen:
                seen.add(label)
                c = by_label[label]
                out.append(
                    Citation(
                        chunk_id=c.chunk_id,
                        label=label,
                        source_path=c.source_path,
                        page=c.page,
                        paragraph_idx=c.paragraph_idx,
                    )
                )
        return out


def get_answer_service() -> AnswerService:
    return AnswerService()
