"""The CLI contract from SPEC §4. Command names never change."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from sxl.cli import app

runner = CliRunner()

# command -> the feature that owns it (SPEC §8)
UNIMPLEMENTED = {
    ("corpus", "build"): "F1",
    ("teacher", "label"): "F2",
    ("gold", "sample"): "F3",
    ("gold", "verify"): "F3",
    ("metrics", "score"): "F4",
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
