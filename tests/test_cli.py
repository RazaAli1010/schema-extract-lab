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


def test_gpu_predict_is_implemented_and_exposes_its_options():
    """F5 owns `gpu predict`; it must no longer exit 2. Never invoked here — it needs a GPU."""
    result = runner.invoke(app, ["gpu", "predict", "--help"])
    assert result.exit_code == 0
    for option in ("--arm", "--gold", "--train", "--model", "--adapter", "--batch-size"):
        assert option in result.output
    for option in ("--limit", "--out"):
        assert option in result.output


def test_gpu_predict_rejects_an_unknown_arm_with_exit_1():
    """Exit 1, not 2 — and it must fail before importing torch, which is not installed."""
    result = runner.invoke(app, ["gpu", "predict", "--arm", "nope"])
    assert result.exit_code == 1
    assert "base_fewshot" in result.output


def test_gpu_predict_rejects_the_teacher_arm():
    """`teacher` is a hosted API arm produced by F2, not something a GPU run can make."""
    result = runner.invoke(app, ["gpu", "predict", "--arm", "teacher"])
    assert result.exit_code == 1
    assert "teacher label" in result.output


def test_gpu_predict_reports_a_missing_gold_file_before_loading_a_model(tmp_path):
    # Argument validation must happen before the torch import, or the laptop gets
    # an ImportError instead of the actual problem.
    result = runner.invoke(
        app,
        ["gpu", "predict", "--arm", "base_fewshot", "--gold", str(tmp_path / "nope.jsonl")],
    )
    assert result.exit_code == 1
    assert "does not exist" in result.output


def test_gpu_predict_wires_gold_train_and_out_together(tmp_path, monkeypatch):
    """The whole command with the model faked out — argument plumbing, not generation.

    Nothing here imports torch: `load_model` and `make_generate_fn` are replaced
    before the command body reaches them, which is only possible because
    `sxl.gpu.runner` is import-safe on the laptop (SPEC §2.1).
    """
    import sxl.gpu.runner as runner_mod
    from _fakes import doc, gold_json, gold_row
    from sxl.io import write_jsonl
    from sxl.prompts import FEWSHOT_IDS

    # A train file that really contains the frozen exemplars, so `pick_fewshot`
    # takes its normal path rather than the bootstrap branch.
    train = tmp_path / "train.jsonl"
    write_jsonl(
        train, [gold_row(doc(i)) | {"doc_id": doc_id} for i, doc_id in enumerate(FEWSHOT_IDS)]
    )

    gold_path = tmp_path / "eval_gold.jsonl"
    write_jsonl(gold_path, [gold_row(doc(100 + i)) for i in range(3)])

    seen_messages = []

    def fake_generate(batch):
        seen_messages.extend(batch)
        return [
            {
                "raw_output": gold_json(title="Extracted"),
                "prompt_tokens": 1180,
                "completion_tokens": 143,
                "latency_ms": 2841.3,
            }
            for _ in batch
        ]

    monkeypatch.setattr(runner_mod, "load_model", lambda *a, **k: ("model", "tok"))
    monkeypatch.setattr(runner_mod, "make_generate_fn", lambda *a, **k: fake_generate)

    out = tmp_path / "base_fewshot.jsonl"
    result = runner.invoke(
        app,
        [
            "gpu",
            "predict",
            "--arm",
            "base_fewshot",
            "--gold",
            str(gold_path),
            "--train",
            str(train),
            "--out",
            str(out),
            "--batch-size",
            "2",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "wrote" in result.output
    assert next(iter(FEWSHOT_IDS)) in result.output  # the run log records the exemplars

    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 3
    assert all(r["arm"] == "base_fewshot" and r["schema_valid"] for r in records)

    # Each document was rendered through the few-shot builder: system + 3 shot
    # pairs + the document itself.
    assert len(seen_messages) == 3
    assert [m["role"] for m in seen_messages[0]] == (
        ["system"] + ["user", "assistant"] * len(FEWSHOT_IDS) + ["user"]
    )


def test_gpu_predict_respects_limit(tmp_path, monkeypatch):
    import sxl.gpu.runner as runner_mod
    from _fakes import doc, gold_json, gold_row
    from sxl.io import write_jsonl
    from sxl.prompts import FEWSHOT_IDS

    train = tmp_path / "train.jsonl"
    write_jsonl(
        train, [gold_row(doc(i)) | {"doc_id": doc_id} for i, doc_id in enumerate(FEWSHOT_IDS)]
    )
    gold_path = tmp_path / "eval_gold.jsonl"
    write_jsonl(gold_path, [gold_row(doc(200 + i)) for i in range(10)])

    monkeypatch.setattr(runner_mod, "load_model", lambda *a, **k: ("model", "tok"))
    monkeypatch.setattr(
        runner_mod,
        "make_generate_fn",
        lambda *a, **k: (
            lambda batch: [
                {
                    "raw_output": gold_json(),
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "latency_ms": 1.0,
                }
                for _ in batch
            ]
        ),
    )

    out = tmp_path / "smoke.jsonl"
    result = runner.invoke(
        app,
        [
            "gpu",
            "predict",
            "--arm",
            "base_fewshot",
            "--gold",
            str(gold_path),
            "--train",
            str(train),
            "--out",
            str(out),
            "--limit",
            "4",
        ],
    )
    assert result.exit_code == 0, result.output
    assert len(out.read_text(encoding="utf-8").splitlines()) == 4


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
