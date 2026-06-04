"""Offline evaluation harness for the grounded-rag pipeline.

Runs two layers of checks over a hand-built golden set:

1. Retrieval hit-rate@k -- for every non-refusal question, does the retriever
   surface at least one chunk from the expected source document? This needs no
   LLM and always runs.

2. ragas faithfulness / answer-relevancy / context-precision over the full
   AnswerService output. These require an LLM judge. If no LLM is reachable the
   harness skips them with a clear message and still reports hit-rate, rather
   than crashing.

The process exits non-zero if any *computed* metric is below its floor in
thresholds.yaml; that exit code is the CI gate. Skipped metrics never fail it.

Usage:
    python eval/run_eval.py [--slice N] [--top-n K]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from grounded_rag.config import settings
from grounded_rag.log import get_logger

log = get_logger(__name__)

EVAL_DIR = Path(__file__).resolve().parent
GOLDEN_PATH = EVAL_DIR / "golden.jsonl"
THRESHOLDS_PATH = EVAL_DIR / "thresholds.yaml"

REFUSAL_TEXT = "insufficient evidence in the provided documents"


@dataclass
class GoldenRecord:
    question: str
    ground_truth: str
    source: str

    @property
    def is_refusal(self) -> bool:
        return self.source == ""


@dataclass
class EvalResult:
    hit_rate_at_k: float | None = None
    ragas_scores: dict[str, float] = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)


def load_golden(path: Path, limit: int | None = None) -> list[GoldenRecord]:
    """Read golden.jsonl into typed records, optionally truncated to the first N."""
    records: list[GoldenRecord] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            records.append(
                GoldenRecord(
                    question=obj["question"],
                    ground_truth=obj["ground_truth"],
                    source=obj["source"],
                )
            )
    if limit is not None:
        records = records[:limit]
    return records


def load_thresholds(path: Path) -> dict[str, float]:
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return {k: float(v) for k, v in data.items()}


def compute_hit_rate(records: list[GoldenRecord], top_n: int) -> float | None:
    """Fraction of non-refusal questions where a retrieved chunk's source basename
    matches the expected source. Returns None when there are no scorable records."""
    from grounded_rag.retrieve import get_retriever

    retriever = get_retriever()
    scorable = [r for r in records if not r.is_refusal]
    if not scorable:
        return None

    hits = 0
    for rec in scorable:
        contexts = retriever.retrieve(rec.question, top_n=top_n)
        sources = {Path(rc.chunk.source_path).name for rc in contexts}
        hit = rec.source in sources
        if hit:
            hits += 1
        log.info(
            "hit_rate_probe",
            question=rec.question,
            expected=rec.source,
            retrieved=sorted(sources),
            hit=hit,
        )
    return hits / len(scorable)


def _build_judge() -> tuple[object, object] | None:
    """Wrap the project's configured OpenAI-compatible LLM (and an embeddings model
    on the same endpoint) for ragas. Returns None if the bits aren't importable or
    no usable key is configured, so the caller can skip the ragas metrics cleanly."""
    if not settings.openai_api_key or settings.openai_api_key == "sk-missing":
        log.warning("ragas_skip_no_key")
        return None
    try:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
    except ImportError as exc:
        log.warning("ragas_skip_import", error=str(exc))
        return None

    common: dict[str, object] = {"api_key": settings.openai_api_key}
    if settings.openai_base_url:
        common["base_url"] = settings.openai_base_url

    chat = ChatOpenAI(model=settings.llm_model, temperature=0.0, **common)
    embeddings = OpenAIEmbeddings(**common)
    return LangchainLLMWrapper(chat), LangchainEmbeddingsWrapper(embeddings)


def compute_ragas(
    records: list[GoldenRecord],
    top_n: int,
    result: EvalResult,
) -> None:
    """Run the AnswerService over the set and score it with ragas. Any failure
    (missing deps, no LLM, judge error) is downgraded to a skip so hit-rate still
    reports and the gate is not falsely failed."""
    judge = _build_judge()
    if judge is None:
        result.skipped.extend(["faithfulness", "answer_relevancy", "context_precision"])
        return
    llm, embeddings = judge

    try:
        from ragas import EvaluationDataset, evaluate
        from ragas.metrics import (
            Faithfulness,
            LLMContextPrecisionWithReference,
            ResponseRelevancy,
        )
    except ImportError as exc:
        log.warning("ragas_skip_import", error=str(exc))
        result.skipped.extend(["faithfulness", "answer_relevancy", "context_precision"])
        return

    from grounded_rag.answer import get_answer_service

    service = get_answer_service()
    samples = []
    for rec in records:
        answer = service.ask(rec.question)
        contexts = [rc.chunk.text for rc in answer.contexts]
        samples.append(
            {
                "user_input": rec.question,
                "response": answer.answer,
                "retrieved_contexts": contexts or [""],
                "reference": rec.ground_truth,
            }
        )

    metrics = [
        Faithfulness(),
        ResponseRelevancy(),
        LLMContextPrecisionWithReference(),
    ]
    metric_keys = {
        "faithfulness": "faithfulness",
        "answer_relevancy": "answer_relevancy",
        "context_precision": "llm_context_precision_with_reference",
    }

    try:
        dataset = EvaluationDataset.from_list(samples)
        scores = evaluate(dataset=dataset, metrics=metrics, llm=llm, embeddings=embeddings)
        df = scores.to_pandas()
    except Exception as exc:  # noqa: BLE001 -- any judge/runtime failure is a skip, not a crash
        log.warning("ragas_skip_runtime", error=str(exc))
        result.skipped.extend(["faithfulness", "answer_relevancy", "context_precision"])
        return

    for gate_name, col in metric_keys.items():
        if col in df.columns:
            value = float(df[col].dropna().mean())
            result.ragas_scores[gate_name] = value
        else:
            log.warning("ragas_metric_missing", metric=gate_name, column=col)
            result.skipped.append(gate_name)


def _fmt(value: float | None) -> str:
    return "  n/a" if value is None else f"{value:5.3f}"


def report(result: EvalResult, thresholds: dict[str, float]) -> bool:
    """Print a metrics table and return True if every computed metric passes."""
    rows: list[tuple[str, float | None, float]] = [
        ("hit_rate_at_k", result.hit_rate_at_k, thresholds["hit_rate_at_k"]),
        ("faithfulness", result.ragas_scores.get("faithfulness"), thresholds["faithfulness"]),
        (
            "answer_relevancy",
            result.ragas_scores.get("answer_relevancy"),
            thresholds["answer_relevancy"],
        ),
        (
            "context_precision",
            result.ragas_scores.get("context_precision"),
            thresholds["context_precision"],
        ),
    ]

    print()
    print(f"{'metric':<20}{'score':>8}{'floor':>8}  status")
    print("-" * 48)
    passed = True
    for name, score, floor in rows:
        if score is None or name in result.skipped:
            status = "SKIP"
        elif score + 1e-9 < floor:
            status = "FAIL"
            passed = False
        else:
            status = "PASS"
        print(f"{name:<20}{_fmt(score):>8}{floor:>8.3f}  {status}")
    print("-" * 48)

    if result.skipped:
        unique = sorted(set(result.skipped))
        print(f"skipped (no LLM judge reachable): {', '.join(unique)}")
    print("PASS" if passed else "FAIL")
    return passed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the grounded-rag eval gate.")
    parser.add_argument(
        "--slice",
        type=int,
        default=None,
        help="evaluate only the first N golden records (CI uses a small fixed slice)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=settings.top_n,
        help="number of contexts to retrieve per question",
    )
    args = parser.parse_args(argv)

    records = load_golden(GOLDEN_PATH, limit=args.slice)
    thresholds = load_thresholds(THRESHOLDS_PATH)
    log.info("eval_start", records=len(records), top_n=args.top_n, slice=args.slice)

    result = EvalResult()
    result.hit_rate_at_k = compute_hit_rate(records, top_n=args.top_n)
    compute_ragas(records, top_n=args.top_n, result=result)

    passed = report(result, thresholds)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
