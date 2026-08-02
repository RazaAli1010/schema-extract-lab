"""The CLI contract from SPEC §4. Command names never change."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from sxl.cli import app

runner = CliRunner()

# command -> the feature that owns it (SPEC §8). A feature deletes its entry when it
# lands and replaces it with a `--help`-only test below.
UNIMPLEMENTED = {
    ("gpu", "predict"): "F5",
    ("gpu", "train"): "F6",
    ("gpu", "bench"): "F7",
    ("report", "build"): "F8",
}


def test_help_exits_zero_and_lists_every_group():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for group in ("corpus", "teacher", "gold", "metrics", "gpu", "report", "schema"):
        assert group in result.stdout


@pytest.mark.parametrize(("group", "command"), sorted(UNIMPLEMENTED))
def test_every_command_in_spec_section_4_is_registered(group, command):
    result = runner.invoke(app, [group, "--help"])
    assert result.exit_code == 0
    assert command in result.stdout


@pytest.mark.parametrize(
    ("group", "command", "feature"), sorted((g, c, f) for (g, c), f in UNIMPLEMENTED.items())
)
def test_unimplemented_commands_exit_2_naming_their_feature(group, command, feature):
    result = runner.invoke(app, [group, command])
    assert result.exit_code == 2
    assert feature in result.output


def test_corpus_build_is_implemented_and_exposes_its_options():
    """F1 owns `corpus build`; it must no longer exit 2. Never invoked here — network."""
    result = runner.invoke(app, ["corpus", "build", "--help"])
    assert result.exit_code == 0
    for option in ("--target-n", "--source", "--force"):
        assert option in result.output


def test_teacher_label_is_implemented_and_exposes_its_options():
    """F2 owns `teacher label`; it must no longer exit 2. Never invoked here — it spends money."""
    result = runner.invoke(app, ["teacher", "label", "--help"])
    assert result.exit_code == 0
    for option in ("--split", "--limit", "--resume", "--model", "--dry-run"):
        assert option in result.output


def test_gold_commands_are_implemented_and_expose_their_options():
    """F3 owns `gold sample|verify|finalize`; none may exit 2 any more."""
    result = runner.invoke(app, ["gold", "--help"])
    assert result.exit_code == 0
    for command in ("sample", "verify", "finalize"):
        assert command in result.output

    result = runner.invoke(app, ["gold", "sample", "--help"])
    assert result.exit_code == 0
    for option in ("--n", "--seed", "--force"):
        assert option in result.output


def test_metrics_commands_are_implemented_and_expose_their_options():
    """F4 owns `metrics score|compare`; neither may exit 2 any more."""
    result = runner.invoke(app, ["metrics", "--help"])
    assert result.exit_code == 0
    for command in ("score", "compare"):
        assert command in result.output

    result = runner.invoke(app, ["metrics", "score", "--help"])
    assert result.exit_code == 0
    for option in ("--arm", "--pred", "--gold", "--out", "--expect-n"):
        assert option in result.output


def test_metrics_score_rejects_an_unknown_arm_with_exit_1():
    """Exit 1, not 2: exit 2 means "not implemented" throughout this CLI."""
    result = runner.invoke(app, ["metrics", "score", "--arm", "nope"])
    assert result.exit_code == 1
    assert "base_fewshot" in result.output  # the valid arms are listed


def test_schema_dump_to_stdout():
    result = runner.invoke(app, ["schema", "dump"])
    assert result.exit_code == 0
    schema = json.loads(result.stdout)
    assert len(schema["required"]) == 16


def test_schema_dump_to_file(tmp_path):
    out = tmp_path / "s.json"
    result = runner.invoke(app, ["schema", "dump", "--out", str(out)])
    assert result.exit_code == 0

    schema = json.loads(out.read_text(encoding="utf-8"))
    assert len(schema["required"]) == 16
    assert schema["additionalProperties"] is False
    assert not list(tmp_path.glob("*.tmp")), "atomic write left its tmp file behind"
