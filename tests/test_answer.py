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

    svc._template = load_prompt("answer_v3")
    return svc


def test_grounded_answer_resolves_index_to_label():
    contexts = _ctx()
    svc = _service(contexts, "Employees accrue 1.67 days per month [1].")
    ans = svc.ask("how much PTO?")
    assert not ans.refused
    assert len(ans.citations) == 1
    assert ans.citations[0].chunk_id == "abc123"
    # the numeric [1] is rewritten into the human citation label
    assert contexts[0].chunk.citation_label() in ans.answer
    assert "[1]" not in ans.answer


def test_ungrounded_answer_is_refused():
    # model answers but cites no source -> not grounded -> refuse
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


def test_out_of_range_index_is_dropped():
    # only one context exists; citing [9] is fabricated -> dropped -> refusal
    svc = _service(_ctx(), "Made up fact [9].")
    ans = svc.ask("q")
    assert ans.refused
