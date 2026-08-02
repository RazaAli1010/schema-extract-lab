"""The gates that stop a bad input from becoming a good-looking number.

Each of these is a silent way to inflate a score: dropping documents the model
failed on, scoring one arm's predictions as another's, trusting a hand-written
`schema_valid`, or measuring against gold the model was trained on. All of them
raise (SPEC §5.5) rather than warning.
"""

from __future__ import annotations

import pytest

from _fakes import doc, gold, gold_row, prediction_row
from sxl.io import write_jsonl
from sxl.metrics import (
    MetricsError,
    MetricsPaths,
    assert_no_leakage,
    score_arm,
    score_teacher_arm,
)
from sxl.verify import LeakageDetected

DOCS = [doc(i) for i in range(10)]
GOLD_ROWS = [gold_row(d, title=f"role {i}", required_skills=["python"]) for i, d in enumerate(DOCS)]


def _preds(rows, *, arm="base_fewshot"):
    return [prediction_row(r["doc_id"], gold(**r["gold"]), arm=arm) for r in rows]


def _score(gold_rows, pred_rows, arm="base_fewshot", **kw):
    return score_arm(gold_rows, pred_rows, arm, check_leakage=False, **kw)


# --- coverage ------------------------------------------------------------------


def test_missing_predictions_are_counted_and_scored_as_invalid():
    """5 of 10 documents absent from the file: N stays 10, and the 5 score as misses."""
    preds = _preds(GOLD_ROWS[:5])
    result = _score(GOLD_ROWS, preds)

    assert result["n"] == 10
    assert result["n_missing_predictions"] == 5
    assert result["schema_valid_rate"] == 0.5  # the missing 5 are invalid, not excluded

    # `title` is present in all 10 golds; only 5 were predicted, all correctly.
    title = result["per_field"]["title"]
    assert title["precision"] == 1.0
    assert title["recall"] == 0.5
    assert title["support"] == 10
    assert title["em"] == 0.5


def test_a_complete_prediction_file_reports_zero_missing():
    result = _score(GOLD_ROWS, _preds(GOLD_ROWS))
    assert result["n_missing_predictions"] == 0
    assert result["macro_f1"] > 0.0


def test_an_extra_prediction_doc_id_raises():
    preds = _preds(GOLD_ROWS) + [prediction_row("jp_notinthegold", gold())]
    with pytest.raises(RuntimeError, match="not in the gold split"):
        _score(GOLD_ROWS, preds)


def test_a_duplicate_prediction_doc_id_raises():
    preds = _preds(GOLD_ROWS)
    with pytest.raises(MetricsError, match="duplicate doc_id"):
        _score(GOLD_ROWS, [*preds, preds[0]])


# --- arm and validity labels ---------------------------------------------------


def test_an_arm_mismatch_raises():
    """Scoring `lora_ft` predictions as `base_fewshot` is the copy-paste error to catch."""
    preds = _preds(GOLD_ROWS, arm="lora_ft")
    with pytest.raises(RuntimeError, match="lora_ft"):
        _score(GOLD_ROWS, preds, "base_fewshot")


def test_an_unknown_arm_raises_value_error():
    with pytest.raises(ValueError, match="unknown arm"):
        _score(GOLD_ROWS, _preds(GOLD_ROWS), "not_an_arm")


def test_schema_valid_true_with_an_invalid_parsed_raises():
    """`schema_valid` must come from `validate_prediction`, never be written by hand."""
    preds = _preds(GOLD_ROWS)
    preds[3]["parsed"] = {"title": "Engineer"}  # missing 15 required keys
    with pytest.raises(RuntimeError, match="validate_prediction"):
        _score(GOLD_ROWS, preds)


def test_schema_valid_false_with_a_valid_parsed_also_raises():
    """The disagreement is checked in both directions — this one hides a good prediction."""
    preds = _preds(GOLD_ROWS)
    preds[2]["schema_valid"] = False
    with pytest.raises(RuntimeError, match="validate_prediction"):
        _score(GOLD_ROWS, preds)


def test_schema_valid_false_with_parsed_null_is_accepted():
    preds = _preds(GOLD_ROWS)
    preds[1] = prediction_row(GOLD_ROWS[1]["doc_id"], None)
    result = _score(GOLD_ROWS, preds)
    assert result["schema_valid_rate"] == 0.9


# --- split isolation -----------------------------------------------------------


def test_assert_no_leakage_raises_when_a_gold_id_is_planted_in_train(tmp_path):
    paths = MetricsPaths.in_dir(tmp_path)
    write_jsonl(paths.train, [{"doc_id": GOLD_ROWS[4]["doc_id"], "text": "leaked"}])

    with pytest.raises(LeakageDetected, match=GOLD_ROWS[4]["doc_id"]):
        assert_no_leakage(GOLD_ROWS, paths=paths)


def test_assert_no_leakage_also_checks_dev(tmp_path):
    paths = MetricsPaths.in_dir(tmp_path)
    write_jsonl(paths.dev, [{"doc_id": GOLD_ROWS[0]["doc_id"], "text": "leaked"}])

    with pytest.raises(LeakageDetected):
        assert_no_leakage(GOLD_ROWS, paths=paths)


def test_assert_no_leakage_passes_on_disjoint_splits(tmp_path):
    paths = MetricsPaths.in_dir(tmp_path)
    write_jsonl(paths.train, [{"doc_id": "jp_somethingelse", "text": "fine"}])
    assert_no_leakage(GOLD_ROWS, paths=paths)  # does not raise


def test_score_arm_runs_the_leakage_gate_by_default(tmp_path):
    paths = MetricsPaths.in_dir(tmp_path)
    write_jsonl(paths.train, [{"doc_id": GOLD_ROWS[6]["doc_id"], "text": "leaked"}])

    with pytest.raises(LeakageDetected):
        score_arm(GOLD_ROWS, _preds(GOLD_ROWS), "base_fewshot", paths=paths)


# --- the teacher arm -----------------------------------------------------------


def test_score_teacher_arm_goes_through_the_same_code_path(tmp_path):
    """`eval_pool` is a superset of the gold set; only the sampled rows are scored."""
    paths = MetricsPaths.in_dir(tmp_path)
    write_jsonl(paths.gold, GOLD_ROWS[:6])
    # The teacher got `title` wrong on one document a human later corrected.
    pool = [
        gold_row(d, title="wrong title" if i == 0 else f"role {i}", required_skills=["python"])
        for i, d in enumerate(DOCS)
    ]
    write_jsonl(paths.eval_pool, pool)

    result = score_teacher_arm(paths=paths, check_leakage=False)

    assert result["arm"] == "teacher"
    assert result["n"] == 6  # the gold split, not the 10-row pool
    assert result["n_missing_predictions"] == 0
    assert result["schema_valid_rate"] == 1.0
    assert result["per_field"]["title"]["f1"] == round(2 * 5 / (2 * 5 + 1 + 1), 4)
    assert 0.0 < result["macro_f1"] < 1.0


def test_score_teacher_arm_counts_a_gold_doc_missing_from_the_pool(tmp_path):
    paths = MetricsPaths.in_dir(tmp_path)
    write_jsonl(paths.gold, GOLD_ROWS[:4])
    write_jsonl(paths.eval_pool, GOLD_ROWS[:3])  # one gold document is not in the pool

    result = score_teacher_arm(paths=paths, check_leakage=False)
    assert result["n"] == 4
    assert result["n_missing_predictions"] == 1
    assert result["schema_valid_rate"] == 0.75
