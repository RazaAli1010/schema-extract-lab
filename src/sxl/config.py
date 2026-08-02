"""Paths, constants and environment. No logic beyond `ensure_dirs` and `git_sha`.

Every cap in the project lives here (SPEC §5.4). Nothing in this module imports
anything heavier than the standard library.
"""

from __future__ import annotations

import subprocess
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
    RESULTS / "tables",
)


def ensure_dirs() -> None:
    """Create every directory this project writes to. Idempotent."""
    for d in _DIRS:
        d.mkdir(parents=True, exist_ok=True)


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
