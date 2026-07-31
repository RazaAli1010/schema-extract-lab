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
