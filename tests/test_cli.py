"""The CLI contract from SPEC §4. Command names never change."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from sxl.cli import app

runner = CliRunner()

#: Every command SPEC §4 promises, and the feature that owns it. F8 was the last
#: entry in the old `UNIMPLEMENTED` map — with it landed, no command exits 2 any
#: more, and this list keeps the contract itself under test.
SPEC_COMMANDS = (
    ("corpus", "build"),
    ("teacher", "label"),
    ("gold", "sample"),
    ("gold", "verify"),
    ("metrics", "score"),
    ("gpu", "predict"),
    ("gpu", "train"),
    ("gpu", "bench"),
    ("report", "build"),
)


def test_help_exits_zero_and_lists_every_group():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for group in ("corpus", "teacher", "gold", "metrics", "gpu", "bench", "report", "schema"):
        assert group in result.stdout


@pytest.mark.parametrize(("group", "command"), SPEC_COMMANDS)
def test_every_command_in_spec_section_4_is_registered(group, command):
    result = runner.invoke(app, [group, "--help"])
    assert result.exit_code == 0
    assert command in result.stdout


@pytest.mark.parametrize(("group", "command"), SPEC_COMMANDS)
def test_no_command_still_exits_2(group, command):
    """Exit 2 means "not implemented" in this CLI. Every feature has landed."""
    result = runner.invoke(app, [group, command, "--help"])
    assert result.exit_code == 0, result.output


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


def test_gpu_bench_is_implemented_and_exposes_its_options():
    """F7 owns `gpu bench`; it must no longer exit 2. Never invoked here — it needs a GPU."""
    result = runner.invoke(app, ["gpu", "bench", "--help"])
    assert result.exit_code == 0
    for option in ("--arm", "--adapter", "--n-docs", "--batch-sizes", "--hourly-usd"):
        assert option in result.output
    for option in ("--repeats", "--out-dir", "--warmup"):
        assert option in result.output


def test_gpu_bench_rejects_an_unknown_arm_with_exit_1():
    result = runner.invoke(app, ["gpu", "bench", "--arm", "nope"])
    assert result.exit_code == 1
    assert "base_fewshot" in result.output


def test_gpu_bench_refuses_the_teacher_arm_and_names_the_command_that_owns_it():
    """The teacher is a network measurement, not a GPU one (F7 §Scope 6)."""
    result = runner.invoke(app, ["gpu", "bench", "--arm", "teacher"])
    assert result.exit_code == 1
    assert "sxl bench teacher" in result.output


def test_gpu_bench_requires_an_adapter_for_the_fine_tuned_arms():
    """Otherwise it would benchmark the base model and attribute the result to the fine-tune."""
    result = runner.invoke(app, ["gpu", "bench", "--arm", "lora_ft"])
    assert result.exit_code == 1
    assert "--adapter" in result.output


def test_gpu_bench_rejects_a_malformed_batch_size_list_before_loading_a_model():
    """A typo would silently shrink the sweep and move `best_batch_size`."""
    result = runner.invoke(
        app, ["gpu", "bench", "--arm", "base_fewshot", "--batch-sizes", "1,two,4"]
    )
    assert result.exit_code == 1
    assert "--batch-sizes" in result.output


def test_bench_teacher_is_implemented_and_exposes_its_options():
    """F7 owns `bench teacher`; it runs on the laptop. Never invoked here — it spends money."""
    result = runner.invoke(app, ["bench", "teacher", "--help"])
    assert result.exit_code == 0
    for option in ("--n", "--model", "--gold", "--out"):
        assert option in result.output


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


def _ft_gold_file(tmp_path, n: int = 3):
    from _fakes import doc, gold_row
    from sxl.io import write_jsonl

    path = tmp_path / "eval_gold.jsonl"
    write_jsonl(path, [gold_row(doc(300 + i)) for i in range(n)])
    return path


def _fake_runner(monkeypatch, sink: list):
    """Replace the two torch entry points of `gpu predict`. Returns nothing."""
    import sxl.gpu.runner as runner_mod
    from _fakes import gold_json

    def fake_generate(batch):
        sink.extend(batch)
        return [
            {
                "raw_output": gold_json(title="Extracted"),
                "prompt_tokens": 40,
                "completion_tokens": 143,
                "latency_ms": 12.0,
            }
            for _ in batch
        ]

    monkeypatch.setattr(runner_mod, "load_model", lambda *a, **k: ("model", "tok"))
    monkeypatch.setattr(runner_mod, "make_generate_fn", lambda *a, **k: fake_generate)


def test_gpu_predict_lora_ft_needs_no_train_file(tmp_path, monkeypatch):
    """The fine-tuned arm has no exemplars, so it must not demand `--train`.

    This is the CLI branch F6 adds, and the Kaggle notebook relies on it: the
    prediction cells pass `--gold` and `--adapter` only.
    """
    seen: list = []
    _fake_runner(monkeypatch, seen)
    gold_path = _ft_gold_file(tmp_path)
    out = tmp_path / "lora_ft.jsonl"

    result = runner.invoke(
        app,
        [
            "gpu",
            "predict",
            "--arm",
            "lora_ft",
            "--gold",
            str(gold_path),
            "--adapter",
            "someone/qwen3-1.7b-jobpost-lora",
            "--out",
            str(out),
        ],
    )

    assert result.exit_code == 0, result.output
    assert len(out.read_text(encoding="utf-8").splitlines()) == 3


def test_gpu_predict_lora_ft_uses_the_short_shot_free_prompt(tmp_path, monkeypatch):
    """System + document, and nothing else. The cost claim depends on it."""
    from sxl.prompts import FT_PROMPT_SHA, SCHEMA_BLOCK

    seen: list = []
    _fake_runner(monkeypatch, seen)

    result = runner.invoke(
        app,
        [
            "gpu",
            "predict",
            "--arm",
            "lora_ft",
            "--gold",
            str(_ft_gold_file(tmp_path)),
            "--adapter",
            "someone/adapter",
            "--out",
            str(tmp_path / "lora_ft.jsonl"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert [m["role"] for m in seen[0]] == ["system", "user"]
    assert SCHEMA_BLOCK not in "".join(m["content"] for m in seen[0])
    # The run log traces the file to the prompt that made it, as F5's does.
    assert FT_PROMPT_SHA in result.output


def test_gpu_predict_lora_ft_requires_an_adapter(tmp_path):
    """Without one this would silently measure the base model under a bare prompt."""
    result = runner.invoke(
        app,
        ["gpu", "predict", "--arm", "lora_ft", "--gold", str(_ft_gold_file(tmp_path))],
    )
    assert result.exit_code == 1
    assert "--adapter" in result.output


def test_gpu_train_is_implemented_and_exposes_its_options():
    """F6 owns `gpu train`; it must no longer exit 2. Never invoked here — needs a GPU."""
    result = runner.invoke(app, ["gpu", "train", "--help"])
    assert result.exit_code == 0
    for option in (
        "--train",
        "--dev",
        "--gold",
        "--epochs",
        "--lr",
        "--limit",
        "--max-steps",
        "--resume-from-checkpoint",
        "--load-in-4bit",
        "--out",
        "--stats-out",
        "--push-to",
    ):
        assert option in result.output


def test_gpu_train_reports_a_missing_input_before_importing_torch(tmp_path):
    result = runner.invoke(app, ["gpu", "train", "--train", str(tmp_path / "nope.jsonl")])
    assert result.exit_code == 1
    assert "does not exist" in result.output


def test_gpu_train_wires_its_options_into_train(tmp_path, monkeypatch):
    """Argument plumbing only — `train_lora.train` itself is Kaggle's problem."""
    import sxl.gpu.train_lora as train_mod
    from _fakes import train_row
    from sxl.cli import _TRAIN_SUMMARY_KEYS
    from sxl.io import write_jsonl

    paths = {}
    for name, rows in (
        ("train", [train_row(i) for i in range(4)]),
        ("dev", [train_row(i) for i in range(10, 12)]),
        ("gold", [train_row(i) for i in range(20, 22)]),
    ):
        paths[name] = tmp_path / f"{name}.jsonl"
        write_jsonl(paths[name], rows)

    captured = {}

    def fake_train(**kwargs):
        captured.update(kwargs)
        return dict.fromkeys(_TRAIN_SUMMARY_KEYS, 0)

    monkeypatch.setattr(train_mod, "train", fake_train)

    result = runner.invoke(
        app,
        [
            "gpu",
            "train",
            "--train",
            str(paths["train"]),
            "--dev",
            str(paths["dev"]),
            "--gold",
            str(paths["gold"]),
            "--out",
            str(tmp_path / "adapter"),
            "--stats-out",
            str(tmp_path / "train_stats.json"),
            "--epochs",
            "1",
            "--limit",
            "8",
            "--max-steps",
            "5",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["epochs"] == 1
    assert captured["limit"] == 8
    assert captured["max_steps"] == 5
    assert captured["train_path"] == paths["train"]
    assert captured["push_to"] == ""  # a stray run never pushes to the Hub
    assert set(json.loads(result.output.strip().splitlines()[-1])) == set(_TRAIN_SUMMARY_KEYS)

    # `--mirror-dir ""` must DISABLE the mirror. As a typer `Path` this arrived as
    # `Path(".")`, which is truthy, and checkpoints would have landed in the CWD.
    runner.invoke(
        app,
        [
            "gpu",
            "train",
            "--train",
            str(paths["train"]),
            "--dev",
            str(paths["dev"]),
            "--gold",
            str(paths["gold"]),
            "--mirror-dir",
            "",
        ],
    )
    assert captured["mirror_dir"] is None


def test_metrics_score_rejects_an_unknown_arm_with_exit_1():
    """Exit 1, not 2: exit 2 means "not implemented" throughout this CLI."""
    result = runner.invoke(app, ["metrics", "score", "--arm", "nope"])
    assert result.exit_code == 1
    assert "base_fewshot" in result.output  # the valid arms are listed


def test_report_build_is_implemented_and_exposes_its_options():
    """F8 owns `report build`; it must no longer exit 2."""
    result = runner.invoke(app, ["report", "build", "--help"])
    assert result.exit_code == 0
    for option in ("--results-dir", "--out-dir", "--readme", "--strict"):
        assert option in result.output


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
