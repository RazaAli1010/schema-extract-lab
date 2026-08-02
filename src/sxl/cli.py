"""`sxl` — the single entrypoint (SPEC §4).

Command names are part of the contract: features add commands, and never rename
them. Everything not yet implemented exits 2 naming the feature that owns it.

`sxl gpu *` imports `sxl.gpu.*` **lazily, inside the command body**, so that
`sxl --help` works on the laptop with no torch installed (SPEC §2.1).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from sxl.config import (
    CORPUS_MIN_N,
    CORPUS_MIN_SPLIT_N,
    CORPUS_SOURCES,
    CORPUS_STATS_PATH,
    CORPUS_TARGET_N,
    DOCS_PATH,
    MAX_TEACHER_SPEND_USD,
    METRICS_DIR,
    N_GOLD_CANDIDATES,
    PREDICTIONS_DIR,
    SEED,
    TEACHER_MODEL,
    TEACHER_PRICE_USD,
    ensure_dirs,
)

ArmOpt = Annotated[str, typer.Option("--arm", help="one of config.ARMS")]

#: Echoed as one JSON line at the end of a real `teacher label` run.
_TEACHER_SUMMARY_KEYS = (
    "n_requested",
    "n_cached",
    "n_new_requests",
    "n_ok",
    "n_parse_failed",
    "n_schema_invalid",
    "n_api_error",
    "spend_usd",
    "spend_usd_cached",
)

#: F2 acceptance criteria. Only checked on a full (`--limit 0`) run of that split.
#: `eval_pool` is the tightest: F3 samples 330 candidates from it.
_TEACHER_MIN_ROWS = {"train": 4500, "dev": 280, "eval_pool": 330}

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
def corpus_build(
    target_n: Annotated[
        int, typer.Option("--target-n", help="documents to keep")
    ] = CORPUS_TARGET_N,
    source: Annotated[str, typer.Option("--source", help="HF dataset id")] = CORPUS_SOURCES[0],
    force: Annotated[bool, typer.Option("--force/--no-force", help="re-fetch")] = False,
) -> None:
    """Fetch and normalize the job-posting corpus into data/raw/docs.jsonl."""
    from sxl.corpus import build, cleanup_hf_cache, dataset_license, report, stats_from_docs

    if not force and DOCS_PATH.exists():
        existing = stats_from_docs()
        if existing["n_kept"] >= CORPUS_MIN_N:
            typer.echo(f"{DOCS_PATH} already has {existing['n_kept']} rows; --force to re-fetch")
            typer.echo(json.dumps(existing["split_counts"], sort_keys=True))
            if not CORPUS_STATS_PATH.exists():
                report(existing, dataset_license(existing["source"]))
                typer.echo(f"wrote {CORPUS_STATS_PATH}")
            raise typer.Exit(code=0)

    stats = build(target_n=target_n, source=source)
    written = report(stats, dataset_license(source))
    cleanup_hf_cache()  # after dataset_license, which also touches the cache
    typer.echo(f"wrote {DOCS_PATH} ({stats['n_kept']} rows)")
    typer.echo(f"wrote {CORPUS_STATS_PATH}")
    typer.echo(json.dumps(written["split_counts"], sort_keys=True))

    problems = []
    if stats["n_kept"] < CORPUS_MIN_N:
        problems.append(f"kept {stats['n_kept']} documents, need >= {CORPUS_MIN_N}")
    for split in ("dev", "eval_pool"):
        got = stats["split_counts"][split]
        if got < CORPUS_MIN_SPLIT_N:
            problems.append(f"{split} has {got} documents, need >= {CORPUS_MIN_SPLIT_N}")
    if problems:
        for problem in problems:
            typer.secho(f"corpus too small: {problem}", fg=typer.colors.RED, err=True)
        typer.secho(
            "F3 cannot sample 330 eval candidates from this.", fg=typer.colors.RED, err=True
        )
        raise typer.Exit(code=1)


# --- F2 ----------------------------------------------------------------------
@teacher_app.command("label")
def teacher_label(
    split: Annotated[str, typer.Option("--split", help="train | dev | eval_pool | all")] = "train",
    limit: Annotated[int, typer.Option("--limit", help="cap documents this run; 0 = no cap")] = 0,
    resume: Annotated[
        bool,
        typer.Option("--resume/--no-resume", help="harvest in-flight batches before submitting"),
    ] = True,
    model: Annotated[str, typer.Option("--model", help="teacher model id")] = TEACHER_MODEL,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run/--no-dry-run", help="count and project cost; zero API calls"),
    ] = False,
) -> None:
    """Label a split with the teacher model via the OpenAI Batch API."""
    from sxl.teacher import (
        SPLIT_CHOICES,
        TeacherError,
        TeacherPaths,
        label,
        make_client,
        report_stats,
    )

    ensure_dirs()

    # Exit 1, not typer's usual 2: in this repo exit 2 means "not implemented" and
    # must stay unambiguous (see `_not_yet`).
    if split not in SPLIT_CHOICES:
        typer.secho(
            f"unknown --split {split!r}; expected one of {' | '.join(SPLIT_CHOICES)}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)
    if model not in TEACHER_PRICE_USD:
        typer.secho(
            f"no price for {model!r}; add it to config.TEACHER_PRICE_USD before spending money",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    try:
        # No client at all on a dry run, so it works with no .env present.
        client = None if dry_run else make_client()
        stats = label(
            split, limit=limit, resume=resume, model=model, dry_run=dry_run, client=client
        )
    except TeacherError as exc:  # spend cap, missing key, timeout, cancellation
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    if dry_run:
        counts = stats["split_counts"]
        tokens = stats["projected_tokens"]
        typer.echo(f"[teacher] model={model} prompt_sha={stats['prompt_sha']} split={split}")
        typer.echo(
            f"[teacher] selected {stats['n_requested']}  cached {stats['n_cached']}  "
            f"to request {stats['n_new_requests']}   "
            + " / ".join(f"{k} {v}" for k, v in counts.items())
        )
        typer.echo(
            f"[teacher] estimated {tokens['input'] / 1e6:.2f}M input + "
            f"{tokens['output'] / 1e6:.2f}M output tokens in {stats['n_batches']} batches"
        )
        typer.echo(
            f"[teacher] projected ${stats['projected_usd']:.2f}  "
            f"(cap ${MAX_TEACHER_SPEND_USD:.2f}) — no API calls made"
        )
        raise typer.Exit(code=0)

    report_stats(stats)  # writes teacher_stats.json and prints the quality summary
    paths = TeacherPaths.default()  # report the paths actually written, not the constants
    for name, n in stats["split_counts"].items():
        typer.echo(f"wrote {paths.for_split(name)} ({n} rows)")
    typer.echo(f"wrote {paths.stats}")
    typer.echo(
        json.dumps(
            {k: stats[k] for k in _TEACHER_SUMMARY_KEYS},
            sort_keys=True,
        )
    )

    problems = list(stats["quality_warnings"])
    if limit == 0:  # a --limit run is expected to be short; do not flag it
        for name, floor in _TEACHER_MIN_ROWS.items():
            if name in stats["split_counts"] and stats["split_counts"][name] < floor:
                problems.append(f"{name} has {stats['split_counts'][name]} rows, need >= {floor}")
    if problems:
        for problem in problems:
            typer.secho(problem, fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)


# --- F3 ----------------------------------------------------------------------
@gold_app.command("sample")
def gold_sample(
    n: Annotated[int, typer.Option("--n", help="candidates to draw")] = N_GOLD_CANDIDATES,
    seed: Annotated[int, typer.Option("--seed", help="sampling seed")] = SEED,
    force: Annotated[
        bool, typer.Option("--force/--no-force", help="overwrite existing candidates")
    ] = False,
) -> None:
    """Sample eval-gold candidates from eval_pool, stratified by document length."""
    from sxl.verify import GoldError, GoldPaths, sample_candidates, stratum_of

    ensure_dirs()
    paths = GoldPaths.default()

    # Resampling after review has started would discard human work, so this is
    # opt-in and loud rather than idempotent-by-overwrite.
    if paths.candidates.exists() and not force:
        typer.secho(
            f"{paths.candidates} already exists; --force to resample "
            "(this discards any review progress against the current candidates)",
            fg=typer.colors.YELLOW,
            err=True,
        )
        raise typer.Exit(code=1)

    try:
        picked = sample_candidates(n=n, seed=seed, paths=paths)
    except GoldError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    bins: dict[str, int] = {}
    for row in picked:
        label = stratum_of(len(row["text"]))
        bins[label] = bins.get(label, 0) + 1
    typer.echo(f"wrote {paths.candidates} ({len(picked)} rows)")
    typer.echo(json.dumps({"n": len(picked), "seed": seed, "char_len_bins": bins}, sort_keys=True))


@gold_app.command("verify")
def gold_verify() -> None:
    """Hand-verify sampled candidates. Resumable — re-run to continue."""
    from sxl.verify import GoldError, GoldPaths, review

    ensure_dirs()
    paths = GoldPaths.default()
    try:
        summary = review(paths=paths)
    except GoldError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"wrote {paths.progress}")
    typer.echo(json.dumps(summary, sort_keys=True))
    if summary["n_reviewed"] >= summary["n_candidates"]:
        typer.echo("all candidates reviewed — run `sxl gold finalize`")


@gold_app.command("finalize")
def gold_finalize() -> None:
    """Replay the review log into data/gold/eval_gold.jsonl and gold_stats.json."""
    from sxl.verify import GoldError, GoldPaths, agreement_warnings, finalize

    ensure_dirs()
    paths = GoldPaths.default()
    try:
        stats = finalize(paths=paths)
    except GoldError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"wrote {paths.gold} ({stats['n_final']} rows)")
    typer.echo(f"wrote {paths.stats}")
    typer.echo(
        json.dumps(
            {k: stats[k] for k in ("n_candidates", "n_rejected", "n_final", "n_docs_edited")},
            sort_keys=True,
        )
    )

    worst = sorted(stats["teacher_field_agreement"].items(), key=lambda kv: kv[1])[:5]
    typer.echo("lowest teacher agreement: " + "  ".join(f"{k}={v:.2f}" for k, v in worst))

    problems = agreement_warnings(stats["teacher_field_agreement"])
    if problems:
        for problem in problems:
            typer.secho(problem, fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)


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
    from sxl.io import write_json
    from sxl.schema import JSON_SCHEMA

    if out is None:
        typer.echo(json.dumps(JSON_SCHEMA, ensure_ascii=False, indent=2, sort_keys=True))
        return
    write_json(out, JSON_SCHEMA)
    typer.echo(f"wrote {out}")


if __name__ == "__main__":  # pragma: no cover
    app()
