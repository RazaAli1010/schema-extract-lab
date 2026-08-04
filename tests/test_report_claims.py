"""F8 — the guardrail, and the most important test in this feature.

SPEC §1.1 says the pitch's illustrative numbers ("within 3 F1", "45ms on a T4",
"200x larger model") are hypotheses to be replaced by measurement, not targets to
hit. `check_claims` is what makes that mechanical instead of aspirational: it
re-reads the generated README and rejects any unit-bearing figure that does not
trace to `README_FACTS.json`.
"""

from __future__ import annotations

from typer.testing import CliRunner

from _fakes import write_results_tree
from sxl.cli import app
from sxl.report import (
    BEGIN_MARKER,
    END_MARKER,
    build_headline,
    build_readme,
    check_claims,
    facts,
    load_results,
    splice_readme,
)

runner = CliRunner()


def _facts(tmp_path):
    return facts(load_results(write_results_tree(tmp_path)))


def _readme(body: str) -> str:
    return f"# Project\n\n{BEGIN_MARKER}\n{body}\n{END_MARKER}\n"


def test_the_generated_readme_passes_its_own_guardrail(tmp_path):
    results = load_results(write_results_tree(tmp_path))
    f = facts(results)
    readme = build_readme(f, build_headline(results, footer=False))
    assert check_claims(readme, f) == []


def test_a_parameter_ratio_claim_is_rejected(tmp_path):
    """The teacher's parameter count is not public, so no ratio may be claimed."""
    f = _facts(tmp_path)
    for claim in ("a 200x larger model", "a 200× larger model", "a 50x larger teacher"):
        violations = check_claims(_readme(claim), f)
        assert violations, claim
        assert any("ratio" in v for v in violations)


def test_calling_the_teacher_frontier_is_rejected(tmp_path):
    """SPEC §1.1: `gpt-4o-mini` is a small, cheap hosted model.

    `specs/F8-report.md` itself says "frontier teacher model" — SPEC.md is the
    source of truth and the feature spec is the bug, so this rejects it.
    """
    violations = check_claims(_readme("our frontier teacher model"), _facts(tmp_path))
    assert any("frontier" in v for v in violations)


def test_45ms_is_rejected_when_it_is_not_a_measured_p50(tmp_path):
    """The pitch's latency. Physically unreachable single-stream on a T4."""
    f = _facts(tmp_path)
    assert f["lora_ft"]["p50_ms"] == 8000.0
    violations = check_claims(_readme("p50 latency is 45ms per document"), f)
    assert any("45" in v for v in violations)

    # And the same figure spelled with a space.
    assert check_claims(_readme("45 ms per document"), f)


def test_a_number_matching_the_facts_to_the_printed_precision_is_accepted(tmp_path):
    f = _facts(tmp_path)
    assert check_claims(_readme("macro-F1 of 0.850 macro-F1"), f) == []
    assert check_claims(_readme("costs $0.12500 per 1k documents"), f) == []
    assert check_claims(_readme("p50 is 8000 ms at batch 1"), f) == []


def test_a_number_that_contradicts_the_facts_is_rejected(tmp_path):
    """The `0.812` vs `0.798` case: plausible, well-formed, and simply not true."""
    violations = check_claims(_readme("we reached 0.812 macro-F1"), _facts(tmp_path))
    assert violations == ["'0.812' is quoted with a unit but is not a value in README_FACTS.json"]


def test_the_amortized_figure_under_a_p50_label_is_rejected(tmp_path):
    """The exact conflation SPEC §1.1 exists to prevent.

    1250 ms is a real measured number — it is just not a latency. Quoting it as a
    p50 is how "45 ms on a T4" gets written in the first place.
    """
    f = _facts(tmp_path)
    assert f["lora_ft"]["amortized_ms_per_doc"] == 1250.0
    violations = check_claims(_readme("p50 latency: 1250 ms per document"), f)
    assert any("amortized" in v for v in violations)


def test_a_line_that_names_both_measurements_is_not_a_conflation(tmp_path):
    """Drawing the distinction is the opposite of hiding it."""
    line = "p50 is 8000 ms single-stream; amortized at best batch it is 1250 ms."
    assert check_claims(_readme(line), _facts(tmp_path)) == []


def test_a_readme_with_no_generated_block_is_itself_a_violation(tmp_path):
    """Otherwise a README that lost its markers would silently pass every check."""
    violations = check_claims("# Project\n\nno markers here\n", _facts(tmp_path))
    assert any("markers not found" in v for v in violations)


def test_splice_preserves_hand_written_prose_on_both_sides():
    existing = (
        f"# Title\n\nhand-written intro.\n\n{BEGIN_MARKER}\nold\n{END_MARKER}\n\ntail prose.\n"
    )
    out = splice_readme(existing, f"{BEGIN_MARKER}\nnew\n{END_MARKER}")

    assert "hand-written intro." in out
    assert "tail prose." in out
    assert "new" in out and "old" not in out


def test_splice_appends_the_block_to_a_readme_that_has_no_markers_yet():
    """Every README before F8 first runs. Nothing a human wrote may be discarded."""
    out = splice_readme("# Title\n\nplaceholder prose.\n", f"{BEGIN_MARKER}\nbody\n{END_MARKER}")
    assert "placeholder prose." in out
    assert out.count(BEGIN_MARKER) == 1


# --- the CLI contract ---------------------------------------------------------


def test_report_build_writes_every_artifact(tmp_path):
    paths = write_results_tree(tmp_path)
    result = runner.invoke(
        app,
        [
            "report",
            "build",
            "--results-dir",
            str(tmp_path / "results"),
            "--out-dir",
            str(tmp_path / "tables"),
            "--no-readme",
        ],
    )

    assert result.exit_code == 0, result.output
    for name in ("headline.md", "per_field.md", "sweep.md"):
        assert (tmp_path / "tables" / name).exists(), name
    assert paths.facts.exists()
    assert not list((tmp_path / "tables").glob("*.tmp")), "atomic write left a tmp file"


def test_report_build_exits_1_when_a_claim_is_unsupported(tmp_path, monkeypatch):
    """`--strict` is what `make report` runs, and it must be able to fail."""
    import sxl.report as report_mod

    write_results_tree(tmp_path)
    monkeypatch.setattr(
        report_mod, "check_claims", lambda readme, facts_dict: ["a fabricated claim"]
    )

    readme = tmp_path / "README.md"
    readme.write_text("# Project\n", encoding="utf-8")
    monkeypatch.setattr(
        report_mod.ReportPaths,
        "default",
        classmethod(lambda cls: cls.in_dir(tmp_path)),
    )

    result = runner.invoke(app, ["report", "build"])
    assert result.exit_code == 1, result.output
    assert "a fabricated claim" in result.output

    result = runner.invoke(app, ["report", "build", "--no-strict"])
    assert result.exit_code == 0, "--no-strict reports the violation but does not fail"
    assert "a fabricated claim" in result.output


def test_report_build_succeeds_with_only_one_arm_and_names_what_is_missing(tmp_path):
    write_results_tree(tmp_path, arms=("base_fewshot",))
    result = runner.invoke(
        app,
        [
            "report",
            "build",
            "--results-dir",
            str(tmp_path / "results"),
            "--out-dir",
            str(tmp_path / "tables"),
            "--no-readme",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "lora_ft" in result.output
    assert "—" in (tmp_path / "tables" / "headline.md").read_text(encoding="utf-8")
