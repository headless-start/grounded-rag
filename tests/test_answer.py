"""Answer-layer tests using a fake retriever + fake LLM (no network, no models)."""

from grounded_rag.answer import REFUSAL_TEXT, AnswerService
from grounded_rag.models import Chunk, RetrievedChunk


class FakeRetriever:
    def __init__(self, contexts):
        self._contexts = contexts

    def retrieve(self, query, top_n=None):
        return self._contexts


class FakeLLM:
    def __init__(self, reply):
        self.reply = reply

    def complete(self, system, user):
        return self.reply


def _ctx():
    chunk = Chunk(
        chunk_id="abc123",
        text="PTO accrues at 1.67 days per month.",
        source_path="data/acme_handbook.md",
        page=1,
        paragraph_idx=2,
    )
    return [RetrievedChunk(chunk=chunk, score=0.9, retriever="dense")]


def _service(contexts, reply):
    svc = AnswerService.__new__(AnswerService)
    svc._retriever = FakeRetriever(contexts)
    svc._llm = FakeLLM(reply)
    from grounded_rag.prompts import load_prompt

    svc._template = load_prompt("answer_v1")
    return svc


def test_grounded_answer_keeps_valid_citation():
    contexts = _ctx()
    label = contexts[0].chunk.citation_label()
    svc = _service(contexts, f"Employees accrue 1.67 days per month {label}.")
    ans = svc.ask("how much PTO?")
    assert not ans.refused
    assert len(ans.citations) == 1
    assert ans.citations[0].chunk_id == "abc123"


def test_ungrounded_answer_is_refused():
    # model answers but cites no real label -> not grounded -> refuse
    svc = _service(_ctx(), "Employees accrue 1.67 days per month.")
    ans = svc.ask("how much PTO?")
    assert ans.refused
    assert ans.answer == REFUSAL_TEXT


def test_no_context_refuses_before_llm():
    svc = _service([], "should never be used")
    ans = svc.ask("anything?")
    assert ans.refused


def test_explicit_model_refusal():
    svc = _service(_ctx(), REFUSAL_TEXT)
    ans = svc.ask("unrelated question")
    assert ans.refused


def test_fabricated_citation_is_dropped():
    # model cites a label that was never retrieved -> dropped -> refusal
    svc = _service(_ctx(), "Made up fact [ghost.pdf p9 ¶9].")
    ans = svc.ask("q")
    assert ans.refused
