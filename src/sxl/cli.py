"""`sxl` — the single entrypoint (SPEC §4).

Command names are part of the contract: features add commands, and never rename
them. Everything not yet implemented exits 2 naming the feature that owns it.

`sxl gpu *` imports `sxl.gpu.*` **lazily, inside the command body**, so that
`sxl --help` works on the laptop with no torch installed (SPEC §2.1).
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from sxl.config import METRICS_DIR, PREDICTIONS_DIR

ArmOpt = Annotated[str, typer.Option("--arm", help="one of config.ARMS")]

app = typer.Typer(
    name="sxl",
    help="schema-extract-lab: strict JSON extraction from unstructured documents.",
    no_args_is_help=True,
    add_completion=False,
)

corpus_app = typer.Typer(help="Corpus acquisition & normalization (F1).", no_args_is_help=True)
teacher_app = typer.Typer(help="Teacher labeling pipeline (F2).", no_args_is_help=True)
gold_app = typer.Typer(help="Eval-set sampling & human verification (F3).", no_args_is_help=True)
metrics_app = typer.Typer(help="Metric scoring (F4).", no_args_is_help=True)
gpu_app = typer.Typer(help="GPU work — Kaggle only (F5/F6/F7).", no_args_is_help=True)
report_app = typer.Typer(help="Results aggregation & README (F8).", no_args_is_help=True)
schema_app = typer.Typer(help="The JobPosting schema contract (F0).", no_args_is_help=True)

app.add_typer(corpus_app, name="corpus")
app.add_typer(teacher_app, name="teacher")
app.add_typer(gold_app, name="gold")
app.add_typer(metrics_app, name="metrics")
app.add_typer(gpu_app, name="gpu")
app.add_typer(report_app, name="report")
app.add_typer(schema_app, name="schema")


def _not_yet(command: str, feature: str) -> None:
    """Exit 2 naming the feature that owns `command`.

    Stub options all carry defaults so the body always runs — a *required* option
    would make typer exit 2 on a usage error instead, without naming the feature.
    """
    typer.secho(f"sxl {command} is implemented in {feature}.", fg=typer.colors.YELLOW, err=True)
    raise typer.Exit(code=2)


# --- F1 ----------------------------------------------------------------------
@corpus_app.command("build")
def corpus_build() -> None:
    """Fetch and normalize the job-posting corpus into data/raw/docs.jsonl."""
    _not_yet("corpus build", "F1")


# --- F2 ----------------------------------------------------------------------
@teacher_app.command("label")
def teacher_label(
    split: Annotated[str, typer.Option("--split", help="train | dev | eval_pool")] = "train",
) -> None:
    """Label a split with the teacher model via the Anthropic Batch API."""
    _not_yet("teacher label", "F2")


# --- F3 ----------------------------------------------------------------------
@gold_app.command("sample")
def gold_sample() -> None:
    """Sample eval-gold candidates from eval_pool."""
    _not_yet("gold sample", "F3")


@gold_app.command("verify")
def gold_verify() -> None:
    """Hand-verify sampled candidates into data/gold/eval_gold.jsonl."""
    _not_yet("gold verify", "F3")


# --- F4 ----------------------------------------------------------------------
@metrics_app.command("score")
def metrics_score(
    arm: ArmOpt = "",
    predictions: Annotated[
        Path, typer.Option("--predictions", help="predictions dir")
    ] = PREDICTIONS_DIR,
    out: Annotated[Path, typer.Option("--out", help="results/metrics dir")] = METRICS_DIR,
) -> None:
    """Score an arm's predictions against eval_gold."""
    _not_yet("metrics score", "F4")


# --- F5/F6/F7 — Kaggle only. Imports go INSIDE the bodies. -------------------
@gpu_app.command("predict")
def gpu_predict(arm: ArmOpt = "") -> None:
    """Run an inference arm on a GPU (Kaggle)."""
    # F5 puts its `from sxl.gpu import runner` here, inside the body.
    _not_yet("gpu predict", "F5")


@gpu_app.command("train")
def gpu_train() -> None:
    """LoRA fine-tune the student model (Kaggle)."""
    # F6 puts its `from sxl.gpu import train_lora` here, inside the body.
    _not_yet("gpu train", "F6")


@gpu_app.command("bench")
def gpu_bench(arm: ArmOpt = "") -> None:
    """Measure latency, throughput and cost for an arm (Kaggle)."""
    # F7 puts its `from sxl.gpu import bench` here, inside the body.
    _not_yet("gpu bench", "F7")


# --- F8 ----------------------------------------------------------------------
@report_app.command("build")
def report_build() -> None:
    """Aggregate results into results/tables/headline.md and the README."""
    _not_yet("report build", "F8")


# --- F0 — the one real command ----------------------------------------------
@schema_app.command("dump")
def schema_dump(
    out: Annotated[Path | None, typer.Option("--out", help="write here instead of stdout")] = None,
) -> None:
    """Print the JobPosting JSON Schema, or write it to a file."""
    import json

    from sxl.io import write_json
    from sxl.schema import JSON_SCHEMA

    if out is None:
        typer.echo(json.dumps(JSON_SCHEMA, ensure_ascii=False, indent=2, sort_keys=True))
        return
    write_json(out, JSON_SCHEMA)
    typer.echo(f"wrote {out}")


if __name__ == "__main__":  # pragma: no cover
    app()
