"""F8 — the generated tables (SPEC §1: every cell traces to a committed file).

The fixture numbers are chosen so each expected value is arithmetic a reader can
check by eye: the teacher sits at macro_f1 0.900 and `lora_ft` at 0.850, so
`Δ vs teacher` must be exactly -0.050.
"""

from __future__ import annotations

from _fakes import write_results_tree
from sxl.config import ARMS
from sxl.report import (
    HEADLINE_HEADERS,
    NULL_BASELINE_ROW,
    best_sweep_entry,
    build_headline,
    build_per_field,
    build_sweep,
    load_results,
)


def _rows(table: str) -> list[list[str]]:
    """The data rows of the first markdown table, split into cells."""
    rows = []
    for line in table.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if set("".join(cells)) <= set("-: "):  # the alignment rule
            continue
        rows.append(cells)
    return rows


def test_the_headline_has_one_row_per_arm_plus_the_null_baseline(tmp_path):
    results = load_results(write_results_tree(tmp_path))
    rows = _rows(build_headline(results))

    assert tuple(rows[0]) == HEADLINE_HEADERS
    data = rows[1:]
    assert len(data) == len(ARMS) + 1, "five arms and the null-baseline floor"
    assert [c[0] for c in data[:-1]] == [f"`{a}`" for a in ARMS], "fixed order, SPEC §3.6"
    assert data[-1][0] == NULL_BASELINE_ROW


def test_delta_vs_teacher_is_computed_signed_and_to_three_decimals(tmp_path):
    """The "within N F1" claim. Computed from the metrics files, never asserted."""
    results = load_results(write_results_tree(tmp_path))
    rows = {r[0]: r for r in _rows(build_headline(results))}

    # lora_ft 0.850 - teacher 0.900
    assert rows["`lora_ft`"][3] == "-0.050"
    # base_fewshot 0.500 - teacher 0.900
    assert rows["`base_fewshot`"][3] == "-0.400"
    assert rows["`teacher`"][3] == "+0.000", "the teacher is its own reference"


def test_the_teacher_row_reports_api_not_a_latency(tmp_path):
    """Its measurement is `api_wall_clock` — not comparable with local generate()."""
    results = load_results(write_results_tree(tmp_path))
    row = {r[0]: r for r in _rows(build_headline(results))}["`teacher`"]

    assert row[4] == "API" and row[5] == "API"
    assert "api_wall_clock" in build_headline(results)


def test_single_stream_and_amortized_are_different_columns(tmp_path):
    """SPEC §1.1's whole point: these are two measurements, never one.

    The fixture's `lora_ft` is 8000 ms single-stream and 1250 ms amortized at its
    fastest batch. If a refactor ever sourced both cells from one field, this is
    the test that notices.
    """
    results = load_results(write_results_tree(tmp_path))
    rows = {r[0]: r for r in _rows(build_headline(results))}

    p50, amortized = rows["`lora_ft`"][4], rows["`lora_ft`"][5]
    assert p50 == "8000"
    assert amortized.startswith("1250"), amortized
    assert p50 not in amortized


def test_best_batch_is_the_fastest_row_not_the_largest(tmp_path):
    """The sweep files record `best_batch_size` as the largest non-OOM size.

    In the fixture that is batch 8 at 0.50 docs/s, while batch 4 runs at 0.80.
    Reading the recorded field would publish a figure 60% worse than the same
    sweep measured, and would do it to `base_fewshot` — the arm SPEC §3.6 calls
    the competitor that matters.
    """
    results = load_results(write_results_tree(tmp_path))
    sweep = results["sweep"]["base_fewshot"]

    assert sweep["best_batch_size"] == 8, "the fixture reproduces the recorded field"
    assert best_sweep_entry(sweep)["batch_size"] == 4, "F8 selects the fastest instead"

    row = {r[0]: r for r in _rows(build_headline(results))}["`base_fewshot`"]
    assert "(bs 4)" in row[5]


def test_the_cost_column_matches_the_batch_it_reports(tmp_path):
    """$/1k must be the cost at the batch size in the column beside it.

    Sweep entries carry throughput but no cost, so the batch-1 cost is the easy
    wrong answer here: it is present in the bench file and would render happily.
    At 0.80 docs/s and $0.36/h, 1000 documents take 1250 s = $0.125.
    """
    results = load_results(write_results_tree(tmp_path))
    row = {r[0]: r for r in _rows(build_headline(results))}["`lora_ft`"]

    assert row[6] == "0.12500"
    assert row[6] != "1.00000", "that is the batch-1 figure from the bench file"


def test_a_missing_arm_renders_as_a_dash_and_warns_instead_of_crashing(tmp_path):
    """The report must build mid-project — that is when it is most useful."""
    results = load_results(write_results_tree(tmp_path, arms=("base_fewshot",)))

    table = build_headline(results)  # must not raise
    rows = {r[0]: r for r in _rows(table)}
    assert rows["`base_fewshot`"][2] == "0.500"
    for cell in rows["`lora_ft`"][1:]:
        assert cell == "—"

    # Δ vs teacher is unknowable without the teacher, and says so rather than 0.
    assert rows["`base_fewshot`"][3] == "—"

    missing = " ".join(results["missing"])
    for arm in ("lora_ft", "lora_ft_constrained", "teacher", "base_fewshot_constrained"):
        assert arm in missing
    assert "base_fewshot:" not in missing


def test_the_teacher_is_never_reported_as_missing_a_sweep(tmp_path):
    """It is an API arm; a batch sweep is not a thing it could have."""
    results = load_results(write_results_tree(tmp_path))
    assert results["missing"] == []


def test_per_field_is_sorted_weakest_first_and_flags_low_support(tmp_path):
    results = load_results(write_results_tree(tmp_path))
    table = build_per_field(results)
    rows = _rows(table)

    header, data = rows[0], rows[1:]
    assert header[0] == "field" and header[1] == "support" and header[2] == "note"
    assert len(data) == 16

    f1s = [float(r[header.index("lora_ft f1")]) for r in data]
    assert f1s == sorted(f1s), "ascending by lora_ft f1: read the failures first"

    by_field = {r[0]: r for r in data}
    assert by_field["`posting_date`"][1] == "12"
    assert by_field["`posting_date`"][2] == "low-n", "an F1 from 12 documents is noise"
    assert by_field["`title`"][2] == ""


def test_per_field_reports_support_once_because_it_is_a_property_of_the_gold_set(tmp_path):
    """Support is the count of populated gold values, identical for every arm.

    Five identical columns would imply it varies by arm, which it cannot.
    """
    results = load_results(write_results_tree(tmp_path))
    header = _rows(build_per_field(results))[0]
    assert header.count("support") == 1


def test_the_sweep_table_keeps_oom_rows_and_marks_the_fastest(tmp_path):
    """Dropping the OOM row would imply headroom the measurement disproved."""
    results = load_results(write_results_tree(tmp_path))
    rows = _rows(build_sweep(results))[1:]

    lora = [r for r in rows if r[0] == "`lora_ft`"]
    assert [r[1] for r in lora] == ["1", "4", "8", "16"]
    assert lora[-1][-1] == "OOM"
    assert lora[-1][4] == "—", "an OOM row has no cost"
    assert lora[1][-1] == "**fastest**", "batch 4, not the largest non-OOM size"


def test_every_table_carries_the_git_sha(tmp_path):
    """So a screenshot of a table traces back to the commit that produced it."""
    results = load_results(write_results_tree(tmp_path))
    for table in (build_headline(results), build_per_field(results), build_sweep(results)):
        assert "generated by `sxl report build`" in table
        assert "from git" in table
