"""Validate that eval/golden.jsonl is well-formed and grounded in the real corpus.

These checks guard the eval set itself: a malformed or fabricated golden record
would silently weaken the CI quality gate, so we fail fast on it here.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_PATH = REPO_ROOT / "eval" / "golden.jsonl"
DATA_DIR = REPO_ROOT / "data"

REFUSAL_TEXT = "insufficient evidence in the provided documents"
REQUIRED_KEYS = {"question", "ground_truth", "source"}


def _load_records() -> list[dict]:
    records = []
    with GOLDEN_PATH.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            records.append((lineno, json.loads(line)))
    return records


def test_golden_file_exists() -> None:
    assert GOLDEN_PATH.is_file(), f"missing golden set at {GOLDEN_PATH}"


def test_every_line_parses_and_has_required_keys() -> None:
    for lineno, rec in _load_records():
        assert isinstance(rec, dict), f"line {lineno} is not a JSON object"
        assert set(rec) == REQUIRED_KEYS, (
            f"line {lineno} keys are {set(rec)}, expected {REQUIRED_KEYS}"
        )
        for key in REQUIRED_KEYS:
            assert isinstance(rec[key], str), f"line {lineno} key {key!r} must be a string"
        assert rec["question"].strip(), f"line {lineno} has an empty question"
        assert rec["ground_truth"].strip(), f"line {lineno} has an empty ground_truth"


def test_record_count_in_expected_band() -> None:
    records = _load_records()
    assert 25 <= len(records) <= 40, f"expected 25-40 records, found {len(records)}"


def test_refusal_records_are_well_formed() -> None:
    refusals = [rec for _, rec in _load_records() if rec["source"] == ""]
    assert 5 <= len(refusals) <= 8, f"expected 5-8 refusal records, found {len(refusals)}"
    for rec in refusals:
        assert rec["ground_truth"] == REFUSAL_TEXT, (
            f"refusal ground_truth must be exactly {REFUSAL_TEXT!r}"
        )


def test_non_refusal_sources_reference_real_files() -> None:
    for lineno, rec in _load_records():
        if rec["source"] == "":
            continue
        path = DATA_DIR / rec["source"]
        assert path.is_file(), f"line {lineno} source {rec['source']!r} is not a file in data/"


def test_non_refusal_ground_truth_is_not_the_refusal_text() -> None:
    for lineno, rec in _load_records():
        if rec["source"] != "":
            assert rec["ground_truth"] != REFUSAL_TEXT, (
                f"line {lineno} is grounded but uses the refusal text"
            )


def test_all_three_documents_are_covered() -> None:
    sources = {rec["source"] for _, rec in _load_records() if rec["source"]}
    expected = {"acme_handbook.md", "acme_security_policy.md", "atlas_api_guide.md"}
    assert expected <= sources, f"golden set must cover {expected}, covers {sources}"


def test_no_duplicate_questions() -> None:
    questions = [rec["question"] for _, rec in _load_records()]
    assert len(questions) == len(set(questions)), "golden set has duplicate questions"
