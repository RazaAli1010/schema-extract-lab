"""Paths, constants and environment. No logic beyond `ensure_dirs` and `git_sha`.

Every cap in the project lives here (SPEC §5.4). Nothing in this module imports
anything heavier than the standard library.
"""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path

# --- paths -------------------------------------------------------------------
# NOTE: this resolves to the repo root for an editable install (src/sxl/config.py
# -> src/sxl -> src -> root). When the package is pip-installed from GitHub on
# Kaggle it resolves into site-packages instead, so the Kaggle notebooks (F5-F7)
# must pass explicit output paths rather than relying on these constants.
ROOT = Path(__file__).resolve().parents[2]
DATA, ARTIFACTS, RESULTS = ROOT / "data", ROOT / "artifacts", ROOT / "results"

DOCS_PATH = DATA / "raw" / "docs.jsonl"
TRAIN_PATH = DATA / "labeled" / "train.jsonl"
DEV_PATH = DATA / "labeled" / "dev.jsonl"
EVAL_POOL_PATH = DATA / "labeled" / "eval_pool.jsonl"
EVAL_GOLD_PATH = DATA / "gold" / "eval_gold.jsonl"
PREDICTIONS_DIR = ARTIFACTS / "predictions"
METRICS_DIR = RESULTS / "metrics"
BENCH_DIR = RESULTS / "bench"
TABLES_DIR = RESULTS / "tables"
CORPUS_STATS_PATH = RESULTS / "corpus_stats.json"

# Hugging Face downloads are redirected here so a build never fills ~/.cache on a
# 5 GB-free laptop (SPEC §2.1). Under DATA, so `data/**` in .gitignore covers it.
# F1 deletes this directory after a successful build.
HF_CACHE_DIR = DATA / ".hfcache"

# --- determinism -------------------------------------------------------------
SEED = 1337
DOMAIN = "job_posting"
ARMS = ("base_fewshot", "base_fewshot_constrained", "lora_ft", "lora_ft_constrained", "teacher")

# --- split targets (SPEC §3.4) ----------------------------------------------
N_TRAIN_TARGET, N_DEV_TARGET, N_EVAL_GOLD = 5000, 300, 300

# --- corpus (F1) -------------------------------------------------------------
# The original pick, `lukebarousse/data_jobs`, is metadata-only: 17 short columns
# (job_title, job_location, salary_year_avg, ...), no description field, longest
# string ~85 chars. Every row would have failed CORPUS_MIN_CHARS and the build
# would have emitted an empty corpus. Replaced 2026-07-31.
CORPUS_SOURCES = ("xanderios/linkedin-job-postings",)  # HF dataset ids, priority order
CORPUS_TARGET_N = 7500
CORPUS_MIN_N = 7000  # 7000 x 5% ~= 350 eval_pool, which must exceed F3's 330
# candidates. 6500 would leave only ~325 and starve F3 -- do not lower these.
CORPUS_MIN_CHARS = 400  # drop stubs
CORPUS_MAX_CHARS = 40000  # drop scrape artifacts / concatenated pages
CORPUS_DEDUPE_PREFIX_CHARS = 600  # near-duplicate window
CORPUS_MAX_SCAN = 200_000  # hard cap on rows read upstream (SPEC §5.4)
CORPUS_PEEK_ROWS = 20  # rows sampled to auto-detect the text column
CORPUS_MIN_FREE_BYTES = 2 * 1024**3  # refuse to start a build below this
CORPUS_MIN_SPLIT_N = 340  # required in both `dev` and `eval_pool`

# --- prompt / generation constants (SPEC §3.6), identical in every arm --------
MAX_INPUT_CHARS, MAX_NEW_TOKENS, TEMPERATURE, N_FEWSHOT = 6000, 512, 0.0, 3

# --- teacher (SPEC §6.5) -----------------------------------------------------
TEACHER_MODEL = "gpt-4o-mini"
MAX_TEACHER_SPEND_USD = 25.0  # hard stop, not a guideline
TEACHER_MAX_RETRIES = 3

# Internal artifacts (F2). Both live under DATA, so `data/**` in .gitignore covers
# them with no new rule; teacher_stats.json is under RESULTS and IS committed --
# it is the label-quality artifact F3 reads before spending human hours.
TEACHER_CACHE_PATH = DATA / "labeled" / "_teacher_cache.jsonl"  # append-only, one row per response
TEACHER_BATCHES_PATH = DATA / "labeled" / "_batches.json"  # in-flight ledger; makes --resume work
TEACHER_STATS_PATH = RESULTS / "teacher_stats.json"

TEACHER_BATCH_SIZE = 500  # requests per batch: ~5700 docs -> 12 batches, ~3 MB per input file
# OpenAI caps *enqueued* tokens per model across in-flight batches. All 12 at once
# is ~12M enqueued input tokens, which comes back as status "failed" on a low usage
# tier. Four in flight is ~4M and still ~3x faster than submitting sequentially.
TEACHER_MAX_INFLIGHT_BATCHES = 4
TEACHER_POLL_SECONDS = 60  # the Batch SLA is 24h; polling faster buys nothing
TEACHER_MAX_WAIT_S = 86_400  # past the 24h SLA, give up and tell the user to --resume

TEACHER_MAX_TOKENS = 1024  # the 16-field object is ~350 tokens; 1024 without inviting rambling
TEACHER_DOC_RETRY_ROUNDS = 1  # an unparseable document is retried at most once
TEACHER_RETRY_BACKOFF_S = (4, 16, 64)  # whole-batch backoff; len() == TEACHER_MAX_RETRIES

# USD per 1M tokens, **STANDARD** rates. Do not pre-discount: `teacher.cost_of`
# multiplies by TEACHER_BATCH_DISCOUNT, so halving the table double-applies it.
TEACHER_PRICE_USD = {"gpt-4o-mini": {"in": 0.15, "out": 0.60}, "gpt-4o": {"in": 2.50, "out": 10.00}}
TEACHER_BATCH_DISCOUNT = 0.5  # the Batch API is 50% off

# Pre-flight spend projection only. There is no tokenizer on the laptop and
# tiktoken is not in the base group -- 4.0 chars/token is the standard English
# heuristic and over-predicts slightly on bulleted job-posting text, which is the
# right direction for a spend cap.
TEACHER_CHARS_PER_TOKEN = 4.0
TEACHER_REQUEST_OVERHEAD_TOKENS = 32  # chat framing + response_format name, per request
TEACHER_EST_OUTPUT_TOKENS = 350  # the 16-field object
TEACHER_MIN_DOCS_FOR_MEAN_COST = 50  # below this, trust the char estimate over an observed mean

# Label-quality smoke test (F2 acceptance criteria).
TEACHER_MAX_ENUM_SHARE = 0.95  # no enum field may be >95% a single value
# ...but only once the sample can support that conclusion. Without this gate a
# healthy `--split dev --limit 20` smoke run trips the guard, because 20 postings
# really are all `education_level: "unknown"`.
TEACHER_QUALITY_MIN_N = 200
# >half the corpus with `required_skills: []` is a broken prompt, not a quiet corpus.
TEACHER_MAX_SKILLS_EMPTY_SHARE = 0.5

# --- gold eval set (F3) ------------------------------------------------------
# Both `_`-prefixed files live under DATA, so `data/**` in .gitignore covers them
# with no new rule -- same reasoning as TEACHER_CACHE_PATH. gold_stats.json is
# under RESULTS and IS committed: F8 quotes `teacher_field_agreement` from it.
GOLD_CANDIDATES_PATH = DATA / "gold" / "_candidates.jsonl"  # sampled, pre-verification
GOLD_PROGRESS_PATH = DATA / "gold" / "_progress.jsonl"  # append-only edit log
GOLD_STATS_PATH = RESULTS / "gold_stats.json"
# Hand-written provenance for the gold set: how it was reviewed, what a human
# audit of a 28-document sample found, and the caveats that go with it. F8 reads
# it so the README's honesty about the eval set is sourced, not composed.
GOLD_AUDIT_PATH = RESULTS / "gold_review_audit.json"

# char_len bin edges for stratified sampling. An unstratified sample
# under-represents long postings, which is exactly where extraction fails -- a
# gold set of short easy documents would inflate every arm equally and hide the
# variance the project exists to measure.
GOLD_STRATA_BINS = (400, 1500, 3000, 8000, 40000)
N_GOLD_CANDIDATES = 330  # N_EVAL_GOLD + 10% slack for documents rejected as unusable
GOLD_PROGRESS_EVERY = 25  # documents between progress reports during review
GOLD_MAX_YEARS = 40  # above this, years_experience_min is flagged `low` for review

# --- student / benchmark (SPEC §6.5) -----------------------------------------
BASE_MODEL = "Qwen/Qwen3-1.7B"
T4_HOURLY_USD = 0.35  # ~GCP on-demand T4; source recorded in results/bench/*

# Inference batch for the F5/F6 prediction runs. 1.7B fp16 is ~3.4 GB of weights;
# 8 sequences at ~1.5k prompt tokens leaves plenty of the T4's 16 GB for the KV
# cache. Halve it on OOM -- never drop to 4-bit, because the baseline and the
# fine-tuned arm must share a precision for the comparison to mean anything.
GEN_BATCH_SIZE = 8

# Kaggle volumes (SPEC §2.2). `/kaggle/working` is 20 GB and persisted as the
# notebook's output; `/kaggle/tmp` is ~60 GB of scratch that is not. Model
# weights and the HF cache go to tmp, small JSONL results to working.
KAGGLE_TMP = Path("/kaggle/tmp")
KAGGLE_WORKING = Path("/kaggle/working")

# --- benchmark (F7) ----------------------------------------------------------
# Documents per single-stream measurement. F7's own spec says 200, but 200 x
# BENCH_REPEATS x 4 arms at the p50 that same spec predicts (1.5-4 s) is 4-5
# hours against a 2.5 h acceptance cap. Halving is the honest resolution: it
# applies to every arm equally so it cannot flatter one, all three repeats stay
# real, and `n_docs` is written into every bench file so the sample size is never
# something a reader has to take on trust. Raise it if the quota allows.
BENCH_N_DOCS = 100
# Untimed iterations before the clock starts. The CUDA context, kernel autotune,
# and the first `sdpa` call are one-time costs that would otherwise all land in
# measurement #1 and drag the mean without touching the median -- which is
# exactly the kind of discrepancy that makes a benchmark untrustworthy.
BENCH_WARMUP = 10
BENCH_BATCH_SIZES = (1, 2, 4, 8, 16, 32)
BENCH_REPEATS = 3  # Kaggle GPUs are shared and thermally variable; report the spread

# transformers' default. The F7 spec suggests trying `"static"`, but
# `runner.generate_batch` takes no cache argument, so every committed number in
# results/metrics/* was produced under the dynamic cache. Benchmarking static
# would time a code path no accuracy number came from and quietly make the two
# tables incomparable. Emitted into every bench file, identical across arms.
BENCH_CACHE_IMPL = "dynamic"

# The plausibility guard. A T4 runs a 1.7B model at 40-70 tok/s single-stream, so
# anything above this is not a fast model -- it is a missing
# `torch.cuda.synchronize()` timing kernel *launches* instead of execution, or a
# generation that hit EOS after two tokens. Both must raise, because both would
# otherwise "confirm" the 45 ms/doc figure SPEC §1.1 exists to disown.
BENCH_MAX_TOK_PER_S = 500.0
BENCH_SPREAD_WARN_PCT = 15.0  # p50 spread across repeats worth calling out in F8

# Synchronous teacher calls for the network-latency probe. ~50 is enough for a
# stable p50/p95 and costs about $0.02 at standard rates.
BENCH_TEACHER_N = 50

# --- LoRA fine-tune (F6) -----------------------------------------------------
# r=16 / alpha=32 on all seven attention+MLP projections is the standard "attach
# everything" recipe. At 1.7B it lands near 1% trainable, which `train()` checks
# before the first optimizer step: 100% means `peft_config` never attached and the
# run is full-fine-tuning a 1.7B model on a T4.
LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.05
LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
TRAIN_MIN_TRAINABLE_PCT, TRAIN_MAX_TRAINABLE_PCT = 0.1, 5.0

TRAIN_EPOCHS = 3
TRAIN_LR = 2e-4  # LoRA wants ~10x a full fine-tune's LR
TRAIN_MAX_LENGTH = 2048  # 6000 chars (~1.5k tokens) + a ~350-token JSON target
TRAIN_BATCH_SIZE = 2
TRAIN_GRAD_ACCUM = 8  # effective batch 16 -> ~844 steps over 4500 rows x 3 epochs
TRAIN_WARMUP_RATIO = 0.03
TRAIN_MAX_GRAD_NORM = 0.3  # fp16 stability on a T4 (SPEC §2.3), not a default
TRAIN_LOGGING_STEPS = 10
# Equal by requirement, not coincidence: `load_best_model_at_end` needs the save
# and eval strategies to line up, and Trainer raises if they do not.
TRAIN_EVAL_STEPS = TRAIN_SAVE_STEPS = 100
TRAIN_SAVE_TOTAL_LIMIT = 2
# Rows tokenized for the sequence-length p95 check. Tokenizing all 4500 spends
# minutes of a scarce GPU session on a number a 500-row sample already settles.
TRAIN_LEN_SAMPLE = 500

# Checkpoints live in ephemeral /kaggle/tmp for the space and are mirrored into
# persisted /kaggle/working on every save, because when the session dies
# everything under /kaggle/tmp dies with it. The mirror is what
# `--resume-from-checkpoint auto` finds in a fresh session.
CHECKPOINT_DIR = KAGGLE_TMP / "ckpt"
CHECKPOINT_MIRROR_DIR = KAGGLE_WORKING / "ckpt_latest"
ADAPTER_DIR = KAGGLE_WORKING / "adapter"
# Placeholder on purpose: `train_lora.publish` refuses to push to an id that
# still contains "<". Set it here or pass `--push-to` before publishing.
ADAPTER_REPO = "<hf_user>/qwen3-1.7b-jobpost-lora"
TRAIN_STATS_PATH = RESULTS / "train_stats.json"

# The arms whose prompt is the short, shot-free, schema-free one (SPEC §3.6).
# `sxl gpu predict` branches on membership here rather than on a name prefix.
FT_ARMS = ("lora_ft", "lora_ft_constrained")
if set(FT_ARMS) - set(ARMS):  # pragma: no cover - structural guard
    raise AssertionError(f"FT_ARMS is not a subset of ARMS: {FT_ARMS}")

# --- report (F8) -------------------------------------------------------------
# The generated artifacts. All four are COMMITTED: they are the deliverable
# SPEC §1 cares about, and every cell in them must trace to a results/*.json.
HEADLINE_PATH = TABLES_DIR / "headline.md"
PER_FIELD_PATH = TABLES_DIR / "per_field.md"
SWEEP_TABLE_PATH = TABLES_DIR / "sweep.md"
README_FACTS_PATH = RESULTS / "README_FACTS.json"
README_PATH = ROOT / "README.md"

# Below this many gold values a per-field F1 is noise, not a measurement, and the
# per-field table labels it `low-n` rather than letting it be quoted.
REPORT_LOW_SUPPORT = 30
# Rule-of-thumb standard error on a per-field F1 at n=300. Differences smaller
# than this are not meaningful and the report says so instead of ranking them.
# F4's out-of-scope note defers real bootstrap CIs; this is the honest stand-in.
REPORT_NOISE_F1 = 0.03

_DIRS = (
    DATA,
    DOCS_PATH.parent,
    TRAIN_PATH.parent,
    EVAL_GOLD_PATH.parent,
    ARTIFACTS,
    PREDICTIONS_DIR,
    RESULTS,
    METRICS_DIR,
    BENCH_DIR,
    TABLES_DIR,
)


def ensure_dirs() -> None:
    """Create every directory this project writes to. Idempotent."""
    for d in _DIRS:
        d.mkdir(parents=True, exist_ok=True)


def utc_now() -> str:
    """`generated_at` for every artifact: UTC, second precision, trailing Z.

    Defined once here rather than per-feature so that `results/*.json` written by
    different stages sort and diff against each other (SPEC §5.2).
    """
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def git_sha() -> str:
    """Short git SHA of the working tree, or "unknown".

    Must never raise: this runs inside Kaggle notebooks where the package is a
    pip install with no `.git` directory and possibly no `git` binary.
    """
    try:
        out = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
    except (OSError, ValueError):
        return "unknown"
    if out.returncode != 0:
        return "unknown"
    return out.stdout.strip() or "unknown"
