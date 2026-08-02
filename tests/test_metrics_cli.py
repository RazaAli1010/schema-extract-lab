"""`sxl metrics score` / `sxl metrics compare` end to end (F4 Verify).

Everything runs against `tmp_path` files passed explicitly on the command line,
so no test reads the real `data/` or writes the real `results/`.
"""

from __future__ import annotations

import json

from typer.testing import CliRunner

from _fakes import doc, gold, gold_row, prediction_row
from sxl.cli import app
from sxl.io import write_jsonl

runner = CliRunner()

DOCS = [doc(i) for i in range(8)]
GOLD_ROWS = [
    gold_row(d, title=f"role {i}", seniority="senior", required_skills=["python", "sql"])
    for i, d in enumerate(DOCS)
]


def _write(tmp_path, pred_rows, gold_rows=GOLD_ROWS):
    gold_path = tmp_path / "eval_gold.jsonl"
    pred_path = tmp_path / "preds.jsonl"
    write_jsonl(gold_path, gold_rows)
    write_jsonl(pred_path, pred_rows)
    return gold_path, pred_path, tmp_path / "out.json"


def _run(gold_path, pred_path, out, *, arm="base_fewshot", expect_n="0"):
    return runner.invoke(
        app,
        [
            "metrics", "score",
            "--arm", arm,
            "--pred", str(pred_path),
            "--gold", str(gold_path),
            "--out", str(out),
            "--expect-n", expect_n,
        ],
    )  # fmt: skip


def test_the_oracle_scores_a_perfect_macro_f1(tmp_path):
    """Predicting the gold back must yield schema_valid_rate 1.0 and macro_f1 1.0.

    Every one of the 16 fields has gold support here, so nothing is dragged to 0.0
    for lack of a populated field — see the zero-support test below for that case.
    """
    populated = [
        gold_row(
            d,
            title=f"role {i}",
            company="Acme",
            employment_type="full_time",
            seniority="senior",
            remote_mode="remote",
            location_city="Austin",
            location_region="Texas",
            location_country="US",
            salary_min=120000.0,
            salary_max=150000.0,
            salary_currency="USD",
            salary_period="yearly",
            years_experience_min=6,
            education_level="bachelor",
            required_skills=["python", "sql"],
            posting_date="2026-06-14",
        )
        for i, d in enumerate(DOCS)
    ]
    preds = [prediction_row(r["doc_id"], gold(**r["gold"])) for r in populated]
    gold_path, pred_path, out = _write(tmp_path, preds, populated)

    result = _run(gold_path, pred_path, out)
    assert result.exit_code == 0, result.output

    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["schema_valid_rate"] == 1.0
    assert written["macro_f1"] == 1.0
    assert written["n_missing_predictions"] == 0
    assert not list(tmp_path.glob("*.tmp")), "atomic write left its tmp file behind"


def test_fields_with_zero_gold_support_are_named_in_the_output(tmp_path):
    """A field absent for every gold document scores f1 0.0 — say so, never hide it."""
    preds = [prediction_row(r["doc_id"], gold(**r["gold"])) for r in GOLD_ROWS]
    gold_path, pred_path, out = _write(tmp_path, preds)

    result = _run(gold_path, pred_path, out)
    assert result.exit_code == 0, result.output
    assert "zero gold support" in result.output
    assert "posting_date" in result.output  # absent in every fixture gold row


def test_per_field_rows_are_printed_worst_first(tmp_path):
    """Ascending f1: the weakest fields are the ones a reader must not scroll past."""
    preds = [prediction_row(r["doc_id"], gold(**r["gold"])) for r in GOLD_ROWS]
    gold_path, pred_path, out = _write(tmp_path, preds)

    result = _run(gold_path, pred_path, out)
    written = json.loads(out.read_text(encoding="utf-8"))

    rows = [ln for ln in result.output.splitlines() if ln.startswith("[metrics]   ")]
    printed = [ln.split()[1] for ln in rows[1:]]  # skip the header row
    f1s = [written["per_field"][name]["f1"] for name in printed]
    assert f1s == sorted(f1s)


def test_the_output_json_preserves_field_name_order(tmp_path):
    """`per_field` must read in SPEC §3.2 order, not alphabetized by the JSON writer."""
    from sxl.schema import FIELD_NAMES

    preds = [prediction_row(r["doc_id"], gold(**r["gold"])) for r in GOLD_ROWS]
    gold_path, pred_path, out = _write(tmp_path, preds)

    assert _run(gold_path, pred_path, out).exit_code == 0
    written = json.loads(out.read_text(encoding="utf-8"))
    assert tuple(written["per_field"]) == FIELD_NAMES


def test_the_expect_n_gate_rejects_a_short_gold_file(tmp_path):
    preds = [prediction_row(r["doc_id"], gold(**r["gold"])) for r in GOLD_ROWS]
    gold_path, pred_path, out = _write(tmp_path, preds)

    result = _run(gold_path, pred_path, out, expect_n="300")
    assert result.exit_code == 1
    assert "8 rows, expected 300" in result.output
    assert not out.exists(), "a rejected run must not write a metrics file"


def test_a_missing_gold_file_exits_1(tmp_path):
    result = _run(tmp_path / "nope.jsonl", tmp_path / "p.jsonl", tmp_path / "o.json")
    assert result.exit_code == 1
    assert "does not exist" in result.output


def test_an_arm_mismatch_exits_1_rather_than_scoring(tmp_path):
    preds = [prediction_row(r["doc_id"], gold(**r["gold"]), arm="lora_ft") for r in GOLD_ROWS]
    gold_path, pred_path, out = _write(tmp_path, preds)

    result = _run(gold_path, pred_path, out, arm="base_fewshot")
    assert result.exit_code == 1
    assert "lora_ft" in result.output
    assert not out.exists()


def test_the_machine_readable_summary_line_carries_both_new_keys(tmp_path):
    """F8 consumes `macro_f1_null_baseline` and `n_missing_predictions` (F4 deltas)."""
    preds = [prediction_row(r["doc_id"], gold(**r["gold"])) for r in GOLD_ROWS[:5]]
    gold_path, pred_path, out = _write(tmp_path, preds)

    result = _run(gold_path, pred_path, out)
    summary = next(json.loads(ln) for ln in result.output.splitlines() if ln.startswith('{"arm"'))
    assert summary["n_missing_predictions"] == 3
    assert "macro_f1_null_baseline" in summary


def test_metrics_compare_sorts_by_macro_f1_descending(tmp_path):
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    for arm, f1 in (("base_fewshot", 0.4), ("lora_ft", 0.8), ("teacher", 0.6)):
        payload = {"arm": arm, "n": 300, "schema_valid_rate": 0.9, "macro_f1": f1}
        (metrics_dir / f"{arm}.json").write_text(json.dumps(payload), encoding="utf-8")

    result = runner.invoke(app, ["metrics", "compare", "--metrics-dir", str(metrics_dir)])
    assert result.exit_code == 0
    order = [ln.split()[0] for ln in result.output.splitlines()[1:] if ln.strip()]
    assert order == ["lora_ft", "teacher", "base_fewshot"]


def test_metrics_compare_on_an_empty_directory_exits_1(tmp_path):
    result = runner.invoke(app, ["metrics", "compare", "--metrics-dir", str(tmp_path)])
    assert result.exit_code == 1
    assert "no metrics found" in result.output
