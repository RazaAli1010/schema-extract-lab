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
    ADAPTER_DIR,
    ARMS,
    BASE_MODEL,
    BENCH_BATCH_SIZES,
    BENCH_DIR,
    BENCH_N_DOCS,
    BENCH_REPEATS,
    BENCH_TEACHER_N,
    BENCH_WARMUP,
    CHECKPOINT_DIR,
    CHECKPOINT_MIRROR_DIR,
    CORPUS_MIN_N,
    CORPUS_MIN_SPLIT_N,
    CORPUS_SOURCES,
    CORPUS_STATS_PATH,
    CORPUS_TARGET_N,
    DEV_PATH,
    DOCS_PATH,
    EVAL_GOLD_PATH,
    FT_ARMS,
    GEN_BATCH_SIZE,
    MAX_NEW_TOKENS,
    MAX_TEACHER_SPEND_USD,
    METRICS_DIR,
    N_EVAL_GOLD,
    N_GOLD_CANDIDATES,
    SEED,
    T4_HOURLY_USD,
    TEACHER_MODEL,
    TEACHER_PRICE_USD,
    TRAIN_BATCH_SIZE,
    TRAIN_EPOCHS,
    TRAIN_GRAD_ACCUM,
    TRAIN_LR,
    TRAIN_MAX_LENGTH,
    TRAIN_PATH,
    TRAIN_STATS_PATH,
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

#: Echoed as one JSON line at the end of a real `gpu train` run. The full stats
#: object goes to `results/train_stats.json`; this is the at-a-glance subset a
#: Kaggle notebook cell shows without opening the file.
_TRAIN_SUMMARY_KEYS = (
    "n_train",
    "n_dev",
    "n_steps",
    "trainable_pct",
    "final_train_loss",
    "best_eval_loss",
    "peak_vram_gb",
    "train_runtime_s",
    "token_p95",
)

#: Echoed as one JSON line at the end of a bench run. Single-stream latency and
#: the cost assumption it rests on -- deliberately *not* the amortized figures,
#: which live in the sweep file under `best_*` names (SPEC §1.1).
_BENCH_SUMMARY_KEYS = (
    "arm",
    "p50_ms",
    "p95_ms",
    "mean_completion_tokens",
    "p50_spread_pct",
    "throughput_docs_per_s",
    "cost_per_1k_docs_usd",
    "gpu_hourly_usd",
    "measurement",
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
# Separate from `gpu` because the teacher benchmark is a *network* measurement that
# runs on the laptop and must never pull torch (F7 §Scope 6).
bench_app = typer.Typer(help="Non-GPU benchmarks (F7).", no_args_is_help=True)
report_app = typer.Typer(help="Results aggregation & README (F8).", no_args_is_help=True)
schema_app = typer.Typer(help="The JobPosting schema contract (F0).", no_args_is_help=True)

app.add_typer(corpus_app, name="corpus")
app.add_typer(teacher_app, name="teacher")
app.add_typer(gold_app, name="gold")
app.add_typer(metrics_app, name="metrics")
app.add_typer(gpu_app, name="gpu")
app.add_typer(bench_app, name="bench")
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
def gold_finalize(
    reviewer: Annotated[
        str, typer.Option("--reviewer", help="who reviewed: human | model_verified")
    ] = "human",
) -> None:
    """Replay the review log into data/gold/eval_gold.jsonl and gold_stats.json."""
    from sxl.verify import REVIEWERS, GoldError, GoldPaths, agreement_warnings, finalize

    ensure_dirs()
    paths = GoldPaths.default()
    if reviewer not in REVIEWERS:
        typer.secho(
            f"unknown --reviewer {reviewer!r}; expected one of {' | '.join(REVIEWERS)}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)
    try:
        stats = finalize(paths=paths, reviewer=reviewer)
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

    if reviewer != "human":
        # Loud, every run: this is the caveat that qualifies every number downstream.
        typer.secho(
            f"eval_gold was reviewed by {reviewer!r}, not a human. The `teacher` arm now "
            "measures model-vs-model agreement, and no arm is scored against human-checked "
            "ground truth. F8 must state this in the README.",
            fg=typer.colors.YELLOW,
            err=True,
        )

    problems = agreement_warnings(stats["teacher_field_agreement"])
    if problems:
        for problem in problems:
            typer.secho(problem, fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)


# --- F4 ----------------------------------------------------------------------
@metrics_app.command("score")
def metrics_score(
    arm: ArmOpt = "",
    pred: Annotated[
        Path | None,
        typer.Option(
            "--pred", help="predictions JSONL [default: artifacts/predictions/<arm>.jsonl]"
        ),
    ] = None,
    gold: Annotated[
        Path | None, typer.Option("--gold", help=f"gold JSONL [default: {EVAL_GOLD_PATH}]")
    ] = None,
    out: Annotated[
        Path | None, typer.Option("--out", help="output JSON [default: results/metrics/<arm>.json]")
    ] = None,
    expect_n: Annotated[
        int, typer.Option("--expect-n", help="required gold row count; 0 disables the check")
    ] = N_EVAL_GOLD,
) -> None:
    """Score an arm's predictions against eval_gold."""
    from sxl.io import read_jsonl
    from sxl.metrics import (
        MetricsError,
        MetricsPaths,
        report_lines,
        score_arm_files,
        write_metrics,
    )
    from sxl.verify import GoldError

    ensure_dirs()
    paths = MetricsPaths.default()

    # Exit 1, not typer's usual 2: in this repo exit 2 means "not implemented" and
    # must stay unambiguous (see `_not_yet`).
    if arm not in ARMS:
        typer.secho(
            f"unknown --arm {arm!r}; expected one of {' | '.join(ARMS)}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    gold_path = gold if gold is not None else paths.gold
    out_path = out if out is not None else paths.out_for(arm)
    if not gold_path.exists():
        typer.secho(
            f"{gold_path} does not exist; run `sxl gold finalize` first",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    if expect_n:
        n_gold = sum(1 for _ in read_jsonl(gold_path))
        if n_gold != expect_n:
            typer.secho(
                f"{gold_path} has {n_gold} rows, expected {expect_n}. "
                "Scoring a partial gold set produces a number that is not comparable "
                "to the other arms — pass --expect-n 0 if that is genuinely intended.",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1)

    try:
        result = score_arm_files(arm, pred_path=pred, gold_path=gold_path, paths=paths)
    except (MetricsError, GoldError, ValueError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    except FileNotFoundError as exc:
        typer.secho(f"missing input: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    write_metrics(result, out_path)
    typer.echo(f"wrote {out_path}")
    typer.echo(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "arm",
                    "n",
                    "schema_valid_rate",
                    "macro_f1",
                    "macro_f1_null_baseline",
                    "n_missing_predictions",
                )
            },
            sort_keys=True,
        )
    )
    for line in report_lines(result):
        typer.echo(line)


@metrics_app.command("compare")
def metrics_compare(
    metrics_dir: Annotated[
        Path, typer.Option("--metrics-dir", help="directory of <arm>.json files")
    ] = METRICS_DIR,
) -> None:
    """Print every scored arm side by side, best macro_f1 first."""
    from sxl.metrics import compare_lines, load_metrics

    results = load_metrics(metrics_dir)
    if not results:
        typer.secho(
            f"no metrics found in {metrics_dir}; run `sxl metrics score --arm <arm>` first",
            fg=typer.colors.YELLOW,
            err=True,
        )
        raise typer.Exit(code=1)
    for line in compare_lines(results):
        typer.echo(line)


# --- F5/F6/F7 — Kaggle only. Imports go INSIDE the bodies. -------------------
@gpu_app.command("predict")
def gpu_predict(
    arm: ArmOpt = "",
    gold: Annotated[
        Path | None,
        typer.Option("--gold", help=f"documents to predict [default: {EVAL_GOLD_PATH}]"),
    ] = None,
    train: Annotated[
        Path | None,
        typer.Option("--train", help=f"source of the few-shot exemplars [default: {TRAIN_PATH}]"),
    ] = None,
    model: Annotated[str, typer.Option("--model", help="HF model id")] = BASE_MODEL,
    adapter: Annotated[
        str, typer.Option("--adapter", help="LoRA adapter id or path (F6's lora_ft arms)")
    ] = "",
    batch_size: Annotated[
        int, typer.Option("--batch-size", help="prompts per generate() call")
    ] = GEN_BATCH_SIZE,
    limit: Annotated[int, typer.Option("--limit", help="cap documents; 0 = all")] = 0,
    out: Annotated[
        Path | None,
        typer.Option("--out", help="output JSONL [default: artifacts/predictions/<arm>.jsonl]"),
    ] = None,
) -> None:
    """Run an inference arm on a GPU (Kaggle).

    Resumable: a killed run leaves `<out>.partial.jsonl` behind and re-invoking
    picks up where it stopped without regenerating completed documents.
    """
    # Kaggle only. These imports pull torch, so they stay inside the body —
    # `sxl --help` must work on the laptop (SPEC §2.1).
    from sxl.gpu.runner import PredictPaths, RunnerError, fewshot_prompt_fn, predict_arm
    from sxl.io import read_jsonl
    from sxl.prompts import STUDENT_PROMPT_SHA, pick_fewshot

    # Exit 1, not typer's usual 2: in this repo exit 2 means "not implemented" and
    # must stay unambiguous (see `_not_yet`).
    if arm not in ARMS:
        typer.secho(
            f"unknown --arm {arm!r}; expected one of {' | '.join(ARMS)}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)
    if arm == "teacher":
        typer.secho(
            "the `teacher` arm is a hosted API call produced by `sxl teacher label` (F2), "
            "not a GPU run",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    gold_path = gold if gold is not None else EVAL_GOLD_PATH
    train_path = train if train is not None else TRAIN_PATH
    # Not `ensure_dirs()`: on Kaggle the package is a pip install and `config.ROOT`
    # resolves into site-packages, so creating the whole repo tree would be wrong.
    out_path = out if out is not None else PredictPaths.default().pred_for(arm)

    # The fine-tuned arms carry the schema in the weights, not the prompt (SPEC
    # §3.6), so they have no exemplars and therefore need no --train file at all.
    is_ft = arm in FT_ARMS
    required = (
        [("--gold", gold_path)] if is_ft else [("--gold", gold_path), ("--train", train_path)]
    )
    for label, path in required:
        if not path.exists():
            typer.secho(
                f"{label} {path} does not exist. On Kaggle, pass the mounted dataset paths "
                "explicitly (e.g. /kaggle/input/sxl-data/eval_gold.jsonl).",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1)

    gold_rows = list(read_jsonl(gold_path))
    if limit:
        gold_rows = gold_rows[:limit]
    if not gold_rows:
        typer.secho(f"{gold_path} has no rows", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    if is_ft:
        from sxl.prompts import FT_PROMPT_SHA, build_ft_prompt

        if not adapter:
            typer.secho(
                f"--arm {arm} needs --adapter. Without one this would measure the BASE model "
                "under the fine-tuned prompt, which is not an arm in SPEC §3.6 -- and it would "
                "score badly for a reason that has nothing to do with fine-tuning.",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1)
        # `build_ft_prompt` is already a `PromptFn`, so it goes straight through.
        shots, prompt_fn, prompt_sha = [], build_ft_prompt, FT_PROMPT_SHA
    else:
        try:
            shots = pick_fewshot(read_jsonl(train_path))
        except ValueError as exc:
            typer.secho(str(exc), fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1) from exc
        prompt_fn, prompt_sha = fewshot_prompt_fn(shots), STUDENT_PROMPT_SHA

    typer.echo(
        f"[predict] arm={arm} model={model} adapter={adapter or '-'} "
        f"prompt_sha={prompt_sha} n={len(gold_rows)} batch={batch_size}"
    )
    if shots:
        typer.echo(f"[predict] shots={[s['doc_id'] for s in shots]} from {train_path}")

    from sxl.gpu.runner import load_model, make_generate_fn

    hf_model, tok = load_model(model, adapter=adapter or None)
    if arm.endswith("_constrained"):
        from sxl.gpu.constrained import build_constrained_generator

        generate_fn = build_constrained_generator(hf_model, tok)
    else:
        generate_fn = make_generate_fn(hf_model, tok)

    try:
        summary = predict_arm(
            arm,
            gold_rows,
            generate_fn,
            out_path,
            prompt_fn=prompt_fn,
            batch_size=batch_size,
        )
    except RunnerError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"wrote {out_path} ({summary['n']} rows)")
    typer.echo(json.dumps(summary | {"prompt_sha": prompt_sha}, sort_keys=True))

    # A high truncation rate is a real cost of constrained decoding, not a
    # footnote: a grammar cannot end `required_skills` early. F8 reports it.
    if summary["n"] and summary["n_truncated"] / summary["n"] > 0.05:
        typer.secho(
            f"{summary['n_truncated']}/{summary['n']} completions hit max_new_tokens "
            "(>5%) — record this in the run log; F8 must report it.",
            fg=typer.colors.YELLOW,
            err=True,
        )


@gpu_app.command("train")
def gpu_train(
    train: Annotated[
        Path | None, typer.Option("--train", help=f"training rows [default: {TRAIN_PATH}]")
    ] = None,
    dev: Annotated[
        Path | None, typer.Option("--dev", help=f"eval rows [default: {DEV_PATH}]")
    ] = None,
    gold: Annotated[
        Path | None,
        typer.Option(
            "--gold", help=f"eval_gold, for the leakage check [default: {EVAL_GOLD_PATH}]"
        ),
    ] = None,
    model: Annotated[str, typer.Option("--model", help="base HF model id")] = BASE_MODEL,
    out: Annotated[Path, typer.Option("--out", help="where the adapter is saved")] = ADAPTER_DIR,
    stats_out: Annotated[
        Path, typer.Option("--stats-out", help="train_stats.json")
    ] = TRAIN_STATS_PATH,
    checkpoint_dir: Annotated[
        Path, typer.Option("--checkpoint-dir", help="live checkpoints (ephemeral on Kaggle)")
    ] = CHECKPOINT_DIR,
    # A str, not a Path: typer turns `--mirror-dir ""` into `Path(".")`, which is
    # truthy, so a Path here would silently mirror checkpoints into the CWD instead
    # of disabling the mirror.
    mirror_dir: Annotated[
        str, typer.Option("--mirror-dir", help="persisted checkpoint mirror; empty to disable")
    ] = str(CHECKPOINT_MIRROR_DIR),
    epochs: Annotated[int, typer.Option("--epochs")] = TRAIN_EPOCHS,
    lr: Annotated[float, typer.Option("--lr")] = TRAIN_LR,
    max_length: Annotated[
        int, typer.Option("--max-length", help="raise to 3072 if the token p95 exceeds it")
    ] = TRAIN_MAX_LENGTH,
    batch_size: Annotated[int, typer.Option("--batch-size")] = TRAIN_BATCH_SIZE,
    grad_accum: Annotated[int, typer.Option("--grad-accum")] = TRAIN_GRAD_ACCUM,
    limit: Annotated[int, typer.Option("--limit", help="cap training rows; 0 = all")] = 0,
    max_steps: Annotated[int, typer.Option("--max-steps", help="smoke runs; 0 = full")] = 0,
    resume_from_checkpoint: Annotated[
        str, typer.Option("--resume-from-checkpoint", help="'auto', a path, or empty")
    ] = "",
    load_in_4bit: Annotated[
        bool,
        typer.Option("--load-in-4bit/--no-load-in-4bit", help="the OOM fallback (SPEC §5.3)"),
    ] = False,
    push_to: Annotated[
        str, typer.Option("--push-to", help="HF repo id; empty = do not publish")
    ] = "",
    seed: Annotated[int, typer.Option("--seed")] = SEED,
) -> None:
    """LoRA fine-tune the student model (Kaggle).

    Resumable: `--resume-from-checkpoint auto` picks up the newest checkpoint,
    falling back to the persisted mirror when `/kaggle/tmp` has been wiped by a
    session ending.
    """
    train_path = train if train is not None else TRAIN_PATH
    dev_path = dev if dev is not None else DEV_PATH
    gold_path = gold if gold is not None else EVAL_GOLD_PATH

    # Checked before the import below, which pulls torch: a typo'd Kaggle path
    # should cost a second, not a model load.
    for label, path in (("--train", train_path), ("--dev", dev_path), ("--gold", gold_path)):
        if not path.exists():
            typer.secho(
                f"{label} {path} does not exist. On Kaggle, pass the mounted dataset paths "
                "explicitly (e.g. /kaggle/input/sxl-data/train.jsonl).",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1)

    from sxl.gpu.train_lora import TrainError
    from sxl.gpu.train_lora import train as run_train

    try:
        stats = run_train(
            train_path=train_path,
            dev_path=dev_path,
            gold_path=gold_path,
            model_id=model,
            output_dir=checkpoint_dir,
            adapter_dir=out,
            mirror_dir=Path(mirror_dir) if mirror_dir.strip() else None,
            stats_path=stats_out,
            epochs=epochs,
            lr=lr,
            max_length=max_length,
            batch_size=batch_size,
            grad_accum=grad_accum,
            limit=limit,
            max_steps=max_steps,
            resume_from_checkpoint=resume_from_checkpoint,
            load_in_4bit=load_in_4bit,
            push_to=push_to,
            seed=seed,
            echo=typer.echo,
        )
    except TrainError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"wrote {out}")
    typer.echo(f"wrote {stats_out}")
    typer.echo(json.dumps({k: stats[k] for k in _TRAIN_SUMMARY_KEYS}, sort_keys=True))


def _parse_batch_sizes(raw: str) -> tuple[int, ...]:
    """Parse `--batch-sizes "1,2,4"`. Loud on anything else.

    A typo here would silently shrink the sweep and change `best_batch_size`, which
    is the number the amortized cost figure is derived from -- so a bad token is a
    hard error, not a skipped entry.
    """
    sizes: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            value = int(token)
        except ValueError as exc:
            raise ValueError(f"--batch-sizes: {token!r} is not an integer") from exc
        if value < 1:
            raise ValueError(f"--batch-sizes: {value} is not a positive batch size")
        sizes.append(value)
    if not sizes:
        raise ValueError("--batch-sizes is empty")
    return tuple(sorted(set(sizes)))


@gpu_app.command("bench")
def gpu_bench(
    arm: ArmOpt = "",
    adapter: Annotated[
        str, typer.Option("--adapter", help="HF repo id or local path; required for lora_ft arms")
    ] = "",
    model: Annotated[str, typer.Option("--model", help="base HF model id")] = BASE_MODEL,
    gold: Annotated[
        Path | None,
        typer.Option("--gold", help=f"documents to benchmark [default: {EVAL_GOLD_PATH}]"),
    ] = None,
    train: Annotated[
        Path | None,
        typer.Option("--train", help=f"few-shot exemplar source [default: {TRAIN_PATH}]"),
    ] = None,
    n_docs: Annotated[
        int, typer.Option("--n-docs", help="timed documents per pass")
    ] = BENCH_N_DOCS,
    warmup: Annotated[
        int, typer.Option("--warmup", help="untimed iterations first")
    ] = BENCH_WARMUP,
    repeats: Annotated[
        int, typer.Option("--repeats", help="full repeats of the single-stream pass")
    ] = BENCH_REPEATS,
    batch_sizes: Annotated[
        str, typer.Option("--batch-sizes", help="comma-separated sweep, e.g. '1,2,4,8'")
    ] = ",".join(str(b) for b in BENCH_BATCH_SIZES),
    hourly_usd: Annotated[
        float, typer.Option("--hourly-usd", help="GPU rental rate; an assumption, echoed in output")
    ] = T4_HOURLY_USD,
    out_dir: Annotated[
        Path | None, typer.Option("--out-dir", help=f"bench output dir [default: {BENCH_DIR}]")
    ] = None,
) -> None:
    """Measure latency, throughput and cost for an arm (Kaggle).

    Writes `<out-dir>/<arm>.json` (single-stream p50/p95) and
    `<out-dir>/<arm>_sweep.json` (amortized throughput per batch size). The two
    are separate quantities and no file conflates them (SPEC §1.1).
    """
    if arm not in ARMS or arm == "teacher":
        gpu_arms = [a for a in ARMS if a != "teacher"]
        typer.secho(
            f"--arm must be one of {gpu_arms}. The teacher arm is measured over the network "
            "by `sxl bench teacher`, not on a GPU.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    try:
        sizes = _parse_batch_sizes(batch_sizes)
    except ValueError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    gold_path = gold if gold is not None else EVAL_GOLD_PATH
    train_path = train if train is not None else TRAIN_PATH
    bench_dir = out_dir if out_dir is not None else BENCH_DIR

    is_ft = arm in FT_ARMS
    if is_ft and not adapter:
        typer.secho(
            f"--arm {arm} needs --adapter. Without one this would benchmark the BASE model "
            "under the fine-tuned prompt, and the resulting latency would be attributed to "
            "the fine-tune in F8's table.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    # Checked before the torch import below: a typo'd Kaggle path should cost a
    # second, not a model load.
    needed = [("--gold", gold_path)] + ([] if is_ft else [("--train", train_path)])
    for label, path in needed:
        if not path.exists():
            typer.secho(
                f"{label} {path} does not exist. On Kaggle, pass the mounted dataset paths "
                "explicitly (e.g. /kaggle/input/sxl-data/eval_gold.jsonl).",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1)

    # `sxl.gpu.runner` is torch-free at module level (tests/test_gpu_lazy_import.py),
    # so importing `fewshot_prompt_fn` here costs nothing on the laptop.
    from sxl.gpu.runner import fewshot_prompt_fn
    from sxl.io import read_jsonl, write_json
    from sxl.prompts import FT_PROMPT_SHA, STUDENT_PROMPT_SHA, build_ft_prompt, pick_fewshot

    gold_rows = list(read_jsonl(gold_path))
    if len(gold_rows) < n_docs:
        typer.secho(
            f"--n-docs {n_docs} but {gold_path} has only {len(gold_rows)} rows.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    # The prompt builder is selected exactly as `gpu predict` selects it, so the
    # benchmark times the same prompt that produced this arm's predictions.
    # Benchmarking a different prompt would make the latency and accuracy tables
    # describe different runs (F7 §Reuse from F5).
    if is_ft:
        prompt_fn, prompt_sha = build_ft_prompt, FT_PROMPT_SHA
    else:
        try:
            shots = pick_fewshot(read_jsonl(train_path))
        except ValueError as exc:
            typer.secho(str(exc), fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1) from exc
        prompt_fn, prompt_sha = fewshot_prompt_fn(shots), STUDENT_PROMPT_SHA

    prompts = [prompt_fn(row["text"]) for row in gold_rows]
    typer.echo(
        f"[bench] arm={arm} model={model} adapter={adapter or '-'} prompt_sha={prompt_sha} "
        f"n_docs={n_docs} warmup={warmup} repeats={repeats} batch_sizes={list(sizes)}"
    )

    from sxl.gpu.bench import BenchError
    from sxl.gpu.bench import run as run_bench

    try:
        record, sweep = run_bench(
            arm,
            prompts,
            model_id=model,
            adapter=adapter or None,
            constrained=arm.endswith("_constrained"),
            n_docs=n_docs,
            warmup=warmup,
            repeats=repeats,
            batch_sizes=sizes,
            max_new_tokens=MAX_NEW_TOKENS,
            hourly_usd=hourly_usd,
            echo=typer.echo,
        )
    except BenchError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    bench_path = Path(bench_dir) / f"{arm}.json"
    sweep_path = Path(bench_dir) / f"{arm}_sweep.json"
    write_json(bench_path, record, sort_keys=False)
    write_json(sweep_path, sweep, sort_keys=False)

    typer.echo(f"wrote {bench_path}")
    typer.echo(f"wrote {sweep_path}")
    typer.echo(json.dumps({k: record[k] for k in _BENCH_SUMMARY_KEYS}, sort_keys=True))


# --- F7 (laptop) --------------------------------------------------------------
@bench_app.command("teacher")
def bench_teacher(
    n: Annotated[int, typer.Option("--n", help="sequential API calls to time")] = BENCH_TEACHER_N,
    model: Annotated[str, typer.Option("--model", help="teacher model id")] = TEACHER_MODEL,
    gold: Annotated[
        Path | None, typer.Option("--gold", help=f"documents to send [default: {EVAL_GOLD_PATH}]")
    ] = None,
    out: Annotated[
        Path | None, typer.Option("--out", help=f"output file [default: {BENCH_DIR}/teacher.json]")
    ] = None,
) -> None:
    """Time the teacher API over the network and price it at standard rates (laptop).

    Non-batch, sequential calls: a latency-sensitive deployment cannot use the
    24-hour Batch API that F2 labeled the corpus with, so this is priced at full
    rate and carries `measurement: "api_wall_clock"` to keep it from being read as
    comparable with the GPU arms (F7 §Scope 6).
    """
    gold_path = gold if gold is not None else EVAL_GOLD_PATH
    out_path = out if out is not None else BENCH_DIR / "teacher.json"

    if not gold_path.exists():
        typer.secho(f"--gold {gold_path} does not exist", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    from sxl.bench_teacher import run as run_teacher_bench
    from sxl.gpu.bench import BenchError
    from sxl.io import read_jsonl, write_json
    from sxl.teacher import MissingApiKey

    docs = list(read_jsonl(gold_path))
    typer.echo(f"[bench-teacher] {n} sequential calls to {model} from {gold_path}")

    try:
        record = run_teacher_bench(docs, n=n, model=model, echo=typer.echo)
    except (BenchError, MissingApiKey) as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    write_json(out_path, record, sort_keys=False)
    typer.echo(f"wrote {out_path}")
    typer.echo(json.dumps({k: record[k] for k in _BENCH_SUMMARY_KEYS}, sort_keys=True))


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
