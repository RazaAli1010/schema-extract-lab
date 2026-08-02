"""Degenerate inputs for SPEC §3.5 — the cases where a metric usually grows a `nan`.

Every ratio in `sxl.metrics` divides integers exactly once, with an explicit
zero-denominator branch, so no input may produce `ZeroDivisionError`, `nan`, or a
1.0 handed out for a field nobody predicted.
"""

from __future__ import annotations

import math

from _fakes import doc, gold, gold_row, prediction_row
from sxl.metrics import Counts, score_arm, score_field
from sxl.schema import FIELD_NAMES, empty_posting

DOCS = [doc(i) for i in range(6)]


def _score(gold_rows, pred_rows):
    return score_arm(gold_rows, pred_rows, "base_fewshot", check_leakage=False)


# --- absent on both sides ------------------------------------------------------


def test_field_absent_in_both_gold_and_prediction():
    """An EM hit, no TP/FP/FN, support 0, and f1 exactly 0.0 — not nan, not 1.0.

    This is the asymmetry SPEC §3.5 is explicit about: both-absent is a success
    for EM and invisible to F1.
    """
    rows = [gold_row(DOCS[0])]  # every field absent
    preds = [prediction_row(DOCS[0]["doc_id"], gold())]  # every field absent
    result = _score(rows, preds)

    for field in FIELD_NAMES:
        m = result["per_field"][field]
        assert m["em"] == 1.0, field
        assert m["support"] == 0, field
        assert m["precision"] == m["recall"] == m["f1"] == 0.0, field
        assert not math.isnan(m["f1"]), field
    assert result["macro_f1"] == 0.0


def test_all_absent_gold_never_raises_zero_division():
    gold_rows = [gold_row(d) for d in DOCS[:3]]
    preds = [prediction_row(d["doc_id"], gold(title="Engineer")) for d in DOCS[:3]]
    result = _score(gold_rows, preds)

    title = result["per_field"]["title"]
    assert title["support"] == 0
    assert title["precision"] == 0.0  # 3 FP, 0 TP
    assert title["recall"] == 0.0
    assert title["f1"] == 0.0
    assert title["em"] == 0.0


def test_no_nan_anywhere_in_the_output():
    gold_rows = [gold_row(d) for d in DOCS[:3]]
    preds = [prediction_row(DOCS[0]["doc_id"], None)]  # one invalid, two missing
    result = _score(gold_rows, preds)

    floats = [result["macro_f1"], result["schema_valid_rate"], result["macro_f1_null_baseline"]]
    floats += [v for m in result["per_field"].values() for v in (m["em"], m["precision"], m["f1"])]
    assert not any(math.isnan(v) for v in floats)


# --- the degenerate floor ------------------------------------------------------


def test_predicting_empty_posting_everywhere_equals_the_null_baseline():
    """An arm that answers `null` to everything must land exactly on its own floor."""
    gold_rows = [
        gold_row(DOCS[0], title="Engineer", seniority="senior", required_skills=["python"]),
        gold_row(DOCS[1], company="Acme", salary_min=1000.0),
        gold_row(DOCS[2]),  # all absent
    ]
    empty = empty_posting().model_dump(mode="json")
    preds = [prediction_row(r["doc_id"], dict(empty)) for r in gold_rows]

    result = _score(gold_rows, preds)
    assert result["macro_f1"] == result["macro_f1_null_baseline"]


def test_the_null_baseline_is_independent_of_the_arm_being_scored():
    """The floor describes the gold split, not the predictions, so two arms agree."""
    gold_rows = [gold_row(DOCS[0], title="Engineer"), gold_row(DOCS[1], company="Acme")]
    good = [prediction_row(r["doc_id"], gold(**r["gold"])) for r in gold_rows]
    bad = [prediction_row(r["doc_id"], None) for r in gold_rows]

    assert (
        _score(gold_rows, good)["macro_f1_null_baseline"]
        == (_score(gold_rows, bad)["macro_f1_null_baseline"])
    )


# --- normalization is wired in -------------------------------------------------


def test_case_and_whitespace_differences_count_as_a_match():
    counts = score_field("title", "Senior Engineer ", "senior  engineer", True)
    assert counts == Counts(tp=1, em_hits=1, support=1)


def test_int_gold_matches_float_prediction():
    """`salary_min` 120000 vs 120000.0 — both go through `norm_number`."""
    assert score_field("salary_min", 120000, 120000.0, True) == Counts(tp=1, em_hits=1, support=1)


def test_country_and_currency_compare_case_insensitively():
    assert score_field("location_country", "us", "US", True) == Counts(tp=1, em_hits=1, support=1)
    assert score_field("salary_currency", "usd", "USD", True) == Counts(tp=1, em_hits=1, support=1)


def test_skills_compare_as_a_set_not_a_sequence():
    """Order and casing are irrelevant; membership is not."""
    assert score_field("required_skills", ["SQL", "Python"], ["python", "sql"], True) == Counts(
        tp=2, em_hits=1, support=2
    )


def test_education_none_is_present_but_unknown_is_absent():
    """`none` means "states no formal education required"; `unknown` means "silent"."""
    assert score_field("education_level", "none", "none", True) == Counts(
        tp=1, em_hits=1, support=1
    )
    assert score_field("education_level", "unknown", "unknown", True) == Counts(em_hits=1)
    # Conflating the two must cost something in both directions.
    assert score_field("education_level", "none", "unknown", True) == Counts(fn=1, support=1)
    assert score_field("education_level", "unknown", "none", True) == Counts(fp=1)


# --- invalid predictions -------------------------------------------------------


def test_an_invalid_prediction_is_denied_the_both_absent_em_credit():
    """SPEC §3.5d: an unparseable output is not credited with omitting a field."""
    assert score_field("title", None, None, False) == Counts()
    assert score_field("required_skills", [], None, False) == Counts()


def test_an_invalid_prediction_produces_fn_on_every_present_gold_field():
    counts = score_field("title", "Engineer", None, False)
    assert counts == Counts(fn=1, support=1)
    assert counts.tp == 0
