"""`results/teacher_stats.json` — the label-quality smoke test F3 reads first.

F2 §6: if `remote_mode` comes back 99% `"unknown"` or `required_skills` is empty for
half the corpus, the prompt is broken and F3's human verification will waste hours
discovering it. That is what `quality_warnings` exists to catch — and why it is
gated on a sample large enough to support the conclusion.
"""

from __future__ import annotations

import json
import re

import pytest
from typer.testing import CliRunner

from _fakes import (
    FakeOpenAI,
    doc,
    docs_spanning_splits,
    gold,
    gold_json,
    ok_line,
    patch_cli,
    write_docs,
)
from sxl.cli import app
from sxl.schema import ENUM_TYPES, FIELD_NAMES
from sxl.teacher import (
    TeacherPaths,
    enum_distribution,
    field_null_rate,
    is_absent,
    label,
    quality_warnings,
    report_stats,
)

runner = CliRunner()

EXPECTED_KEYS = {
    "teacher_model",
    "prompt_sha",
    "prompt_version",
    "split",
    "limit",
    "n_requested",
    "n_cached",
    "n_new_requests",
    "n_ok",
    "n_parse_failed",
    "n_schema_invalid",
    "n_api_error",
    "n_batches",
    "spend_usd",
    "spend_usd_cached",
    "tokens",
    "split_counts",
    "field_null_rate",
    "enum_distribution",
    "quality_warnings",
    "generated_at",
    "git_sha",
}


def rows(n: int, **over):
    return [{"doc_id": f"jp_{i:06d}", "gold": gold(**over)} for i in range(n)]


def run(tmp_path, docs=None, client=None):
    paths = TeacherPaths.in_dir(tmp_path)
    stats = label(
        "all",
        client=client or FakeOpenAI(),
        docs=docs or docs_spanning_splits(2),
        paths=paths,
        sleep=lambda _: None,
    )
    return paths, stats


# --- the artifact -------------------------------------------------------------


def test_stats_has_exactly_the_documented_keys(tmp_path):
    paths, stats = run(tmp_path)
    out = report_stats(stats, paths=paths)
    assert set(out) == EXPECTED_KEYS
    assert set(json.loads(paths.stats.read_text(encoding="utf-8"))) == EXPECTED_KEYS


def test_stats_records_git_sha_and_generated_at(tmp_path):
    paths, stats = run(tmp_path)
    out = report_stats(stats, paths=paths)
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", out["generated_at"])
    assert out["git_sha"]


def test_report_stats_raises_when_the_buckets_do_not_partition(tmp_path):
    """F2 §6's "every document ends in exactly one bucket" is a contract, so it is checked."""
    paths, stats = run(tmp_path)
    stats["n_ok"] -= 1
    with pytest.raises(ValueError, match="exactly one bucket"):
        report_stats(stats, paths=paths)


def test_tokens_are_reported_under_provider_neutral_names(tmp_path):
    _, stats = run(tmp_path)
    assert set(stats["tokens"]) == {"input", "output"}
    assert stats["tokens"]["input"] > 0 and stats["tokens"]["output"] > 0


# --- field_null_rate ----------------------------------------------------------


def test_field_null_rate_covers_all_sixteen_fields(tmp_path):
    _, stats = run(tmp_path)
    assert set(stats["field_null_rate"]) == set(FIELD_NAMES)
    assert all(0.0 <= v <= 1.0 for v in stats["field_null_rate"].values())


def test_field_null_rate_treats_unknown_as_absent_for_enums():
    rate = field_null_rate(rows(10))  # empty_posting(): every enum is "unknown"
    for field in ENUM_TYPES:
        assert rate[field] == 1.0, field


def test_a_populated_enum_is_not_counted_as_absent():
    rate = field_null_rate(rows(10, seniority="senior"))
    assert rate["seniority"] == 0.0
    assert rate["remote_mode"] == 1.0


def test_field_null_rate_treats_an_empty_list_as_absent_for_required_skills():
    assert field_null_rate(rows(4))["required_skills"] == 1.0
    assert field_null_rate(rows(4, required_skills=["python"]))["required_skills"] == 0.0


def test_field_null_rate_is_zero_everywhere_for_no_rows():
    rate = field_null_rate([])
    assert set(rate) == set(FIELD_NAMES)
    assert set(rate.values()) == {0.0}


def test_is_absent_distinguishes_none_from_unknown():
    """SPEC §3.2: `none` means "no degree required"; `unknown` means "the posting is silent"."""
    assert is_absent("education_level", "unknown") is True
    assert is_absent("education_level", "none") is False
    assert is_absent("title", None) is True
    assert is_absent("title", "Engineer") is False


# --- enum_distribution --------------------------------------------------------


def test_enum_distribution_includes_zero_count_members():
    """`principal: 0` across 5,000 postings is signal; a plain Counter would omit it."""
    dist = enum_distribution(rows(10, seniority="senior"))
    assert dist["seniority"]["senior"] == 10
    assert dist["seniority"]["principal"] == 0
    for field, enum_type in ENUM_TYPES.items():
        assert set(dist[field]) == {m.value for m in enum_type}, field


def test_enum_distribution_counts_sum_to_the_row_total():
    dist = enum_distribution(rows(7))
    for field, counts in dist.items():
        assert sum(counts.values()) == 7, field


def test_enum_distribution_covers_only_the_five_enum_fields():
    assert set(enum_distribution(rows(3))) == set(ENUM_TYPES)


# --- the quality guard --------------------------------------------------------


def test_the_guard_fires_above_95_percent_a_single_value():
    n = 300
    mixed = rows(n - 3) + [{"doc_id": f"x{i}", "gold": gold(seniority="senior")} for i in range(3)]
    warnings = quality_warnings(
        n_ok=n,
        enum_dist=enum_distribution(mixed),
        null_rate={"required_skills": 0.0},
    )
    assert any("seniority" in w for w in warnings)
    assert any("fix it before running F3" in w for w in warnings)


def test_the_guard_is_suppressed_below_the_minimum_sample():
    """A 20-document smoke run really is all "unknown"; firing there trains people to ignore it."""
    small = rows(20)
    assert (
        quality_warnings(
            n_ok=20, enum_dist=enum_distribution(small), null_rate={"required_skills": 1.0}
        )
        == []
    )


def test_the_guard_fires_for_mostly_empty_required_skills():
    populated = rows(300, seniority="senior", required_skills=["python"])
    warnings = quality_warnings(
        n_ok=300, enum_dist=enum_distribution(populated), null_rate={"required_skills": 0.8}
    )
    assert any("required_skills" in w for w in warnings)


def test_healthy_labels_produce_no_warnings():
    healthy = []
    for i, seniority in enumerate(["junior", "mid", "senior", "lead"] * 75):
        healthy.append(
            {
                "doc_id": f"jp_{i:06d}",
                "gold": gold(
                    seniority=seniority,
                    employment_type="full_time" if i % 2 else "contract",
                    remote_mode="remote" if i % 3 else "onsite",
                    salary_period="yearly" if i % 2 else "hourly",
                    education_level="bachelor" if i % 2 else "none",
                    required_skills=["python"],
                ),
            }
        )
    assert (
        quality_warnings(
            n_ok=len(healthy),
            enum_dist=enum_distribution(healthy),
            null_rate=field_null_rate(healthy),
        )
        == []
    )


# --- the CLI ------------------------------------------------------------------


def test_cli_exits_1_on_a_quality_warning_but_still_writes_every_file(tmp_path, monkeypatch):
    """The money is already spent and the data is worth inspecting — write, then fail loudly."""
    monkeypatch.setattr("sxl.teacher.TEACHER_QUALITY_MIN_N", 4)
    client = FakeOpenAI()  # every label is all-"unknown" -> trips the >95% guard
    paths = patch_cli(monkeypatch, tmp_path, client)
    write_docs(paths, docs_spanning_splits(3))

    result = runner.invoke(app, ["teacher", "label", "--split", "all"])

    assert result.exit_code == 1
    assert "fix it before running F3" in result.output
    for path in (paths.train, paths.dev, paths.eval_pool, paths.stats):
        assert path.exists(), path
    assert json.loads(paths.stats.read_text(encoding="utf-8"))["quality_warnings"]


def test_cli_prints_the_enum_distribution_and_absence_rates(tmp_path, monkeypatch):
    docs = [doc(i) for i in range(8)]
    client = FakeOpenAI(
        lambda r: ok_line(
            r["custom_id"],
            gold_json(
                seniority="senior",
                employment_type="full_time",
                remote_mode="remote",
                salary_period="yearly",
                education_level="bachelor",
                required_skills=["python"],
            ),
        )
    )
    paths = patch_cli(monkeypatch, tmp_path, client)
    write_docs(paths, docs)

    # --limit suppresses the "is this split big enough for F3" floors, which a
    # fixture of 8 documents obviously fails.
    result = runner.invoke(app, ["teacher", "label", "--split", "all", "--limit", "8"])

    assert result.exit_code == 0, result.output
    assert "enum distribution" in result.output
    assert "highest absence rates" in result.output
    assert "seniority" in result.output
    assert f"wrote {paths.stats}" in result.output
