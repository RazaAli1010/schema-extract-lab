"""The golden fixture for SPEC §3.5 — the most important test in the repo.

Four hand-built documents with every TP/FP/FN written out below as a literal. If
a future session "improves" `sxl.metrics` and this file fails, **the metric
changed** — that is the point of the test, not an inconvenience. Re-derive the
arithmetic by hand before touching a single expected number here.

The whole fixture is scored with `check_leakage=False`: these doc_ids are
synthetic and there is no train/dev file to check against. Leakage has its own
test in `test_metrics_integrity.py`.
"""

from __future__ import annotations

from _fakes import doc, gold, gold_row, prediction_row
from sxl.metrics import score_arm

# --- the fixture --------------------------------------------------------------
#
# A fully-populated posting. `company` and `posting_date` are deliberately absent
# so doc 1 exercises "absent on both sides" (an EM hit, invisible to F1).
FULL = {
    "title": "Senior Engineer",
    "company": None,
    "employment_type": "full_time",
    "seniority": "senior",
    "remote_mode": "remote",
    "location_city": "Austin",
    "location_region": "Texas",
    "location_country": "US",
    "salary_min": 120000.0,
    "salary_max": 150000.0,
    "salary_currency": "USD",
    "salary_period": "yearly",
    "years_experience_min": 6,
    "education_level": "bachelor",
    "required_skills": ["python", "sql"],
    "posting_date": None,
}

DOCS = [doc(i) for i in range(4)]
IDS = [d["doc_id"] for d in DOCS]

GOLD_ROWS = [
    gold_row(DOCS[0], **FULL),  # doc 1: perfect prediction
    gold_row(DOCS[1], **FULL),  # doc 2: 3 fields wrong
    gold_row(DOCS[2], **FULL),  # doc 3: invalid prediction
    gold_row(DOCS[3], required_skills=["python", "sql", "aws"]),  # doc 4: set field only
]

PRED_ROWS = [
    # doc 1 — every field matches, including the two absent on both sides.
    prediction_row(IDS[0], gold(**FULL)),
    # doc 2 — valid, but three fields wrong: one string, one enum, one numeric
    # off by exactly 1.0 (proving numbers are compared by value, not by epsilon).
    prediction_row(
        IDS[1],
        gold(**FULL) | {"title": "Staff Engineer", "seniority": "mid", "salary_min": 120001.0},
    ),
    # doc 3 — the model emitted something unparseable. SPEC §3.5b: wrong, not excluded.
    prediction_row(IDS[2], None),
    # doc 4 — set arithmetic: {python, sql, aws} vs {python, sql, java}.
    prediction_row(IDS[3], gold(required_skills=["python", "sql", "java"])),
]

# --- the hand-computed expectations -------------------------------------------
#
# N = 4. Three of the four predictions are schema-valid -> schema_valid_rate 0.75.
#
# Per field, aggregated over the four documents (d1 correct / d2 wrong-or-right /
# d3 invalid -> FN on every present gold / d4 all-absent except skills):
#
#   field                 TP  FP  FN  em  support   f1
#   title                  1   1   2   2      3     2*1/(2*1+1+2) = 0.4
#   company                0   0   0   3      0     no TP anywhere -> 0.0
#   employment_type        2   0   1   3      3     2*2/(2*2+0+1) = 0.8
#   seniority              1   1   2   2      3     0.4   (d2 enum wrong)
#   remote_mode            2   0   1   3      3     0.8
#   location_city          2   0   1   3      3     0.8
#   location_region        2   0   1   3      3     0.8
#   location_country       2   0   1   3      3     0.8
#   salary_min             1   1   2   2      3     0.4   (d2 off by 1.0)
#   salary_max             2   0   1   3      3     0.8
#   salary_currency        2   0   1   3      3     0.8
#   salary_period          2   0   1   3      3     0.8
#   years_experience_min   2   0   1   3      3     0.8
#   education_level        2   0   1   3      3     0.8
#   required_skills        6   1   3   2      9     2*6/(2*6+1+3) = 12/16 = 0.75
#   posting_date           0   0   0   3      0     gold absent everywhere -> 0.0
#
# `required_skills` support counts gold *elements*: 2 + 2 + 2 + 3 = 9.
# `company`/`posting_date` are absent in all four golds: EM 3/4 (doc 3's invalid
# prediction is denied the both-absent credit), F1 0.0, support 0.
#
# macro_f1 = (10 x 0.8 + 3 x 0.4 + 0.75 + 2 x 0.0) / 16
#          = (8.0 + 1.2 + 0.75) / 16 = 9.95 / 16 = 0.621875 -> 0.6219
EXPECTED_F1 = {
    "title": 0.4,
    "company": 0.0,
    "employment_type": 0.8,
    "seniority": 0.4,
    "remote_mode": 0.8,
    "location_city": 0.8,
    "location_region": 0.8,
    "location_country": 0.8,
    "salary_min": 0.4,
    "salary_max": 0.8,
    "salary_currency": 0.8,
    "salary_period": 0.8,
    "years_experience_min": 0.8,
    "education_level": 0.8,
    "required_skills": 0.75,
    "posting_date": 0.0,
}

EXPECTED_SUPPORT = {
    "title": 3,
    "company": 0,
    "employment_type": 3,
    "seniority": 3,
    "remote_mode": 3,
    "location_city": 3,
    "location_region": 3,
    "location_country": 3,
    "salary_min": 3,
    "salary_max": 3,
    "salary_currency": 3,
    "salary_period": 3,
    "years_experience_min": 3,
    "education_level": 3,
    "required_skills": 9,
    "posting_date": 0,
}

# EM counts documents, including both-absent ones; doc 3 scores 0 on every field.
EXPECTED_EM = {
    "title": 0.5,
    "company": 0.75,
    "employment_type": 0.75,
    "seniority": 0.5,
    "remote_mode": 0.75,
    "location_city": 0.75,
    "location_region": 0.75,
    "location_country": 0.75,
    "salary_min": 0.5,
    "salary_max": 0.75,
    "salary_currency": 0.75,
    "salary_period": 0.75,
    "years_experience_min": 0.75,
    "education_level": 0.75,
    "required_skills": 0.5,
    "posting_date": 0.75,
}

EXPECTED_MACRO_F1 = 0.6219  # 9.95 / 16 = 0.621875, rounded to 4dp


def _result():
    return score_arm(GOLD_ROWS, PRED_ROWS, "base_fewshot", check_leakage=False)


def test_macro_f1_matches_the_hand_computed_value():
    assert _result()["macro_f1"] == EXPECTED_MACRO_F1


def test_every_per_field_f1_matches_the_hand_computed_value():
    per_field = _result()["per_field"]
    assert {f: m["f1"] for f, m in per_field.items()} == EXPECTED_F1


def test_every_per_field_support_matches_the_hand_computed_value():
    per_field = _result()["per_field"]
    assert {f: m["support"] for f, m in per_field.items()} == EXPECTED_SUPPORT


def test_every_per_field_em_matches_the_hand_computed_value():
    per_field = _result()["per_field"]
    assert {f: m["em"] for f, m in per_field.items()} == EXPECTED_EM


def test_schema_valid_rate_counts_the_invalid_document():
    """Three of four parsed. N stays 4 — the invalid one is not dropped (SPEC §3.5a/b)."""
    result = _result()
    assert result["n"] == 4
    assert result["schema_valid_rate"] == 0.75
    assert result["n_missing_predictions"] == 0


def test_precision_and_recall_of_a_partially_correct_field():
    """`title`: TP=1, FP=1, FN=2 -> P 0.5, R 1/3, F1 0.4."""
    title = _result()["per_field"]["title"]
    assert title["precision"] == 0.5
    assert title["recall"] == round(1 / 3, 4)
    assert title["f1"] == 0.4


def test_set_field_arithmetic():
    """{python, sql, aws} vs {python, sql, java} contributes TP=2, FP=1, FN=1 (doc 4).

    Across the split that totals TP=6, FP=1, FN=3 -> F1 = 12/16 = 0.75.
    """
    skills = _result()["per_field"]["required_skills"]
    assert skills["precision"] == round(6 / 7, 4)
    assert skills["recall"] == round(6 / 9, 4)
    assert skills["f1"] == 0.75
    assert skills["support"] == 9  # gold *elements*, not documents


def test_the_null_baseline_is_zero_for_this_fixture():
    """An arm predicting `empty_posting()` everywhere earns no TP, so macro-F1 is 0.0.

    F1 ignores true negatives by construction (SPEC §3.5c), which is exactly why
    the EM column has to be reported alongside it.
    """
    assert _result()["macro_f1_null_baseline"] == 0.0


def test_output_shape_matches_the_spec_section_3_3_contract():
    result = _result()
    assert tuple(result) == (
        "arm",
        "split",
        "n",
        "schema_valid_rate",
        "macro_f1",
        "macro_f1_null_baseline",
        "n_missing_predictions",
        "per_field",
        "generated_at",
        "git_sha",
    )
    from sxl.schema import FIELD_NAMES

    assert tuple(result["per_field"]) == FIELD_NAMES
    for metrics in result["per_field"].values():
        assert tuple(metrics) == ("em", "precision", "recall", "f1", "support")
