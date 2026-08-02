"""F1 — corpus acquisition and normalization.

`build()` streams a public job-postings dataset, cleans and length-filters it,
drops exact and near-duplicates, assigns content-derived `doc_id`s and writes
`data/raw/docs.jsonl` (SPEC §3.3) plus `results/corpus_stats.json`.

Two invariants carry the weight here:

* **`doc_id` is a hash of content, never a position.** Split membership derives
  from `doc_id` (SPEC §3.4), so a sequential id would silently move documents
  between train and eval the moment upstream reordered its rows.
* **Records are sorted by `doc_id` before writing.** That is what makes two
  `--force` runs byte-identical regardless of the order the stream arrived in.

`datasets` is imported lazily inside `fetch_raw` and nowhere else, so importing
this module stays free (SPEC §2.1) and so the HF cache can be redirected before
`huggingface_hub` freezes its path constants at import time.
"""

from __future__ import annotations

import hashlib
import html
import itertools
import os
import re
import shutil
import statistics
import unicodedata
from collections.abc import Iterable, Iterator
from typing import Any

from sxl.config import (
    CORPUS_DEDUPE_PREFIX_CHARS,
    CORPUS_MAX_CHARS,
    CORPUS_MAX_SCAN,
    CORPUS_MIN_CHARS,
    CORPUS_MIN_FREE_BYTES,
    CORPUS_PEEK_ROWS,
    CORPUS_SOURCES,
    CORPUS_STATS_PATH,
    DATA,
    DOCS_PATH,
    DOMAIN,
    HF_CACHE_DIR,
    ensure_dirs,
    git_sha,
    utc_now,
)
from sxl.io import read_jsonl, write_json, write_jsonl
from sxl.splits import split_for

# Columns whose value, when present, is a stable upstream identifier. Preferred
# over hashing the text because it survives the upstream fixing a typo.
SOURCE_KEY_COLUMNS = ("job_id", "id", "doc_id", "uuid", "guid", "posting_id")

# A conservative real-tag probe. Job postings contain literal "<" ("<5 years
# experience", "salary <100k"), so tags are only stripped when something that is
# unambiguously markup appears first.
_TAG_NAMES = "p|br|li|ul|ol|div|span|strong|em|b|i|h[1-6]|table|tr|td|th|a|hr|blockquote"
_HAS_HTML = re.compile(rf"<\s*/?\s*(?:{_TAG_NAMES})\b[^>]*>", re.IGNORECASE)
_ANY_TAG = re.compile(r"<[^>]+>")
_BLANK_RUN = re.compile(r"\n{3,}")


# --- cleaning ----------------------------------------------------------------


def clean_text(raw: str) -> str:
    """Normalize a raw posting into the text the model will actually see.

    Deterministic and idempotent: `clean_text(clean_text(x)) == clean_text(x)`.

    Case and punctuation are preserved on purpose — the model must see realistic
    text. Normalization *for scoring* is a separate concern owned by
    `normalize.py` and must not leak in here.
    """
    if not raw:
        return ""

    s = unicodedata.normalize("NFKC", str(raw))

    # `html.unescape` is not idempotent on its own: "&amp;lt;" -> "&lt;" -> "<".
    # Run it to a fixpoint so a second pass over cleaned text is a real no-op.
    for _ in range(3):
        unescaped = html.unescape(s)
        if unescaped == s:
            break
        s = unescaped

    if _HAS_HTML.search(s):
        s = _ANY_TAG.sub(" ", s)

    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = "\n".join(line.rstrip() for line in s.split("\n"))
    s = _BLANK_RUN.sub("\n\n", s)
    return s.strip()


# --- identity ----------------------------------------------------------------


def make_doc_id(source: str, source_key: str) -> str:
    """A stable function of (source, upstream key) — never of position.

    Two different sources yield different ids for identical text, so merging a
    second corpus later cannot collide with this one.
    """
    h = hashlib.sha256(f"{source}\x00{source_key}".encode()).hexdigest()[:10]
    return f"jp_{h}"


def _prefix_hash(text: str) -> str:
    return hashlib.sha1(text[:CORPUS_DEDUPE_PREFIX_CHARS].encode()).hexdigest()


# --- fetching ----------------------------------------------------------------


def _set_hf_cache() -> None:
    """Point every Hugging Face cache at `HF_CACHE_DIR`.

    Must run *before* `datasets`/`huggingface_hub` is imported: those modules
    freeze the cache path into module-level constants at import time, so setting
    the variables afterwards is a silent no-op that fills `~/.cache/huggingface`.

    Deliberately does not create the directory — `huggingface_hub` makes it on
    demand, and creating it here would resurrect an empty `data/.hfcache` on
    every metadata call that runs after `cleanup_hf_cache()`.
    """
    os.environ["HF_HOME"] = str(HF_CACHE_DIR)
    os.environ["HF_HUB_CACHE"] = str(HF_CACHE_DIR / "hub")
    os.environ["HF_DATASETS_CACHE"] = str(HF_CACHE_DIR / "datasets")


def cleanup_hf_cache() -> None:
    """Remove the redirected HF cache. Best-effort and safe to call twice.

    `ignore_errors` because a Windows file lock on a cached parquet shard must
    not fail an otherwise good corpus.
    """
    shutil.rmtree(HF_CACHE_DIR, ignore_errors=True)


def pick_text_column(rows: list[dict[str, Any]], source: str = "") -> str:
    """Choose the text-bearing column by median string length across sampled rows.

    Median rather than the first row's lengths: a single null `description` in
    row 1 would otherwise hand the crown to `application_url`.

    Raises when the winner is still shorter than `CORPUS_MIN_CHARS`. That is the
    `lukebarousse/data_jobs` failure mode — a metadata-only table whose longest
    column is an 85-char job title — and it must be a loud crash rather than a
    silently empty corpus (SPEC §5.5).
    """
    if not rows:
        raise ValueError(f"{source or 'source'} yielded no rows to inspect")

    medians: dict[str, float] = {}
    for column in rows[0]:
        lengths = [len(r[column]) for r in rows if isinstance(r.get(column), str)]
        if lengths:
            medians[column] = statistics.median(lengths)

    if not medians:
        raise ValueError(f"{source or 'source'} has no string columns: {sorted(rows[0])}")

    best = max(medians, key=lambda c: medians[c])
    if medians[best] < CORPUS_MIN_CHARS:
        ranked = sorted(medians, key=lambda c: medians[c], reverse=True)
        seen = ", ".join(f"{c}={medians[c]:.0f}" for c in ranked)
        raise ValueError(
            f"{source or 'source'} has no free-text column: longest is {best!r} with a median "
            f"of {medians[best]:.0f} chars, under CORPUS_MIN_CHARS={CORPUS_MIN_CHARS}. "
            f"Median string lengths seen: {seen}. This dataset is metadata-only and cannot "
            f"support job-posting extraction — pick a source with a description field."
        )
    return best


def pick_source_key_column(rows: list[dict[str, Any]], text_column: str) -> str | None:
    """The upstream's own stable id column, or None to fall back to hashing text."""
    for column in SOURCE_KEY_COLUMNS:
        if column == text_column:
            continue
        values = [r.get(column) for r in rows]
        if all(v is not None and not isinstance(v, (list, dict)) for v in values):
            return column
    return None


def fetch_raw(source: str, limit: int = CORPUS_MAX_SCAN) -> Iterator[dict[str, str]]:
    """Stream `{"source_key", "text"}` pairs from a Hugging Face dataset.

    `streaming=True` is required, not an optimization: a non-streaming load
    materializes the full arrow cache and would exhaust the laptop's disk budget
    (SPEC §2.1).

    `limit` caps rows *read* from upstream — the SPEC §5.4 "every loop has a cap"
    rule. The cap on *accepted* records lives in `build`, which stops as soon as
    it has `target_n` survivors.
    """
    _set_hf_cache()
    # Import order is load-bearing: this must follow _set_hf_cache(), and it must
    # stay inside this function. tests/test_no_torch_import.py enforces both.
    from datasets import load_dataset

    stream = load_dataset(source, split="train", streaming=True)

    peeked = list(itertools.islice(stream, CORPUS_PEEK_ROWS))
    text_column = pick_text_column(peeked, source)
    key_column = pick_source_key_column(peeked, text_column)
    print(
        f"[corpus] {source}: text column {text_column!r}, "
        f"source_key {key_column!r}" + ("" if key_column else " (hashing text)")
    )

    for row in itertools.islice(itertools.chain(peeked, stream), limit):
        text = row.get(text_column)
        if not isinstance(text, str) or not text.strip():
            continue
        if key_column is not None:
            source_key = str(row[key_column])
        else:
            source_key = hashlib.sha256(text.encode()).hexdigest()
        yield {"source_key": source_key, "text": text}


# --- dedupe ------------------------------------------------------------------


def dedupe(rows: Iterable[dict[str, Any]], stats: dict[str, int] | None = None) -> Iterator[dict]:
    """Drop exact `doc_id` repeats and near-duplicate prefixes. First seen wins.

    Reposted listings that differ only in a trailing "apply here" block are
    endemic to job-board scrapes, and without the prefix pass they would leak the
    same posting across `train` and `eval_gold`.

    Both passes are streaming and hold only two sets of short hashes.
    """
    counters = stats if stats is not None else {}
    counters.setdefault("n_dropped_dup_id", 0)
    counters.setdefault("n_dropped_dup_prefix", 0)

    seen_ids: set[str] = set()
    seen_prefixes: set[str] = set()
    for row in rows:
        if row["doc_id"] in seen_ids:
            counters["n_dropped_dup_id"] += 1
            continue
        prefix = _prefix_hash(row["text"])
        if prefix in seen_prefixes:
            counters["n_dropped_dup_prefix"] += 1
            continue
        seen_ids.add(row["doc_id"])
        seen_prefixes.add(prefix)
        yield row


# --- orchestration -----------------------------------------------------------


def _check_disk() -> None:
    free = shutil.disk_usage(DATA).free
    if free < CORPUS_MIN_FREE_BYTES:
        raise RuntimeError(
            f"only {free / 1024**3:.1f} GB free on the filesystem holding {DATA}; "
            f"need {CORPUS_MIN_FREE_BYTES / 1024**3:.1f} GB. Free some space before "
            f"starting a build that would die part-way through."
        )


def _accepted(
    rows: Iterable[dict[str, str]], source: str, stats: dict[str, int]
) -> Iterator[dict[str, Any]]:
    """clean -> length filter -> doc_id, lazily, counting every drop reason."""
    for row in rows:
        stats["n_seen"] += 1
        text = clean_text(row["text"])
        if len(text) < CORPUS_MIN_CHARS:
            stats["n_dropped_short"] += 1
            continue
        if len(text) > CORPUS_MAX_CHARS:
            stats["n_dropped_long"] += 1
            continue
        yield {"doc_id": make_doc_id(source, row["source_key"]), "text": text}


def build(
    target_n: int,
    source: str = CORPUS_SOURCES[0],
    rows: Iterable[dict[str, str]] | None = None,
    out_path: Any = None,
) -> dict[str, Any]:
    """Fetch, clean, dedupe and write the corpus. Returns the stats dict.

    `rows` is the injection seam: pass an iterable of `{"source_key", "text"}` to
    run entirely offline (every test does this), or leave it None to stream from
    `source`.
    """
    ensure_dirs()
    path = DOCS_PATH if out_path is None else out_path

    stats: dict[str, Any] = {
        "source": source,
        "n_seen": 0,
        "n_dropped_short": 0,
        "n_dropped_long": 0,
        "n_dropped_dup_id": 0,
        "n_dropped_dup_prefix": 0,
        "n_kept": 0,
    }

    fetched = rows
    if fetched is None:
        _check_disk()
        fetched = fetch_raw(source)

    kept = itertools.islice(dedupe(_accepted(fetched, source, stats), stats), target_n)
    records = [
        {
            "doc_id": r["doc_id"],
            "domain": DOMAIN,
            "text": r["text"],
            "source": source,
            "char_len": len(r["text"]),
        }
        for r in kept
    ]
    # The only sort key is doc_id (SPEC §3.4 / F1). This is what makes the output
    # byte-identical across runs whatever order the stream arrived in.
    records.sort(key=lambda r: r["doc_id"])

    stats["n_kept"] = write_jsonl(path, records)
    stats["char_len"] = _char_len_percentiles([r["char_len"] for r in records])
    stats["split_counts"] = _split_counts([r["doc_id"] for r in records])

    if rows is None:
        cleanup_hf_cache()  # only on a successful network build
    return stats


def _char_len_percentiles(lengths: list[int]) -> dict[str, int]:
    if not lengths:
        return {"p5": 0, "p50": 0, "p95": 0, "max": 0}
    if len(lengths) < 2:
        only = lengths[0]
        return {"p5": only, "p50": only, "p95": only, "max": only}
    cuts = statistics.quantiles(lengths, n=100, method="inclusive")
    return {
        "p5": int(cuts[4]),
        "p50": int(cuts[49]),
        "p95": int(cuts[94]),
        "max": max(lengths),
    }


def _split_counts(doc_ids: Iterable[str]) -> dict[str, int]:
    counts = {"train": 0, "dev": 0, "eval_pool": 0}
    for doc_id in doc_ids:
        counts[split_for(doc_id)] += 1
    return counts


def dataset_license(source: str) -> str:
    """The upstream license string, or "unknown".

    Decorative metadata: it is recorded so the README can be honest about
    provenance (F1), and it must never fail an otherwise good build. The
    exception net is deliberately broad — huggingface_hub v1 is httpx-backed, so
    the failure modes differ from the requests era (SPEC §6.2).
    """
    try:
        _set_hf_cache()
        from huggingface_hub import HfApi

        info = HfApi().dataset_info(source)
        card = getattr(info, "card_data", None) or {}
        license_ = card.get("license") if hasattr(card, "get") else None
        if isinstance(license_, list):
            license_ = ", ".join(str(x) for x in license_)
        return str(license_) if license_ else "unknown"
    except Exception:  # noqa: BLE001 — metadata only, never fatal
        return "unknown"


def report(stats: dict[str, Any], license_: str | None = None) -> dict[str, Any]:
    """Build and write `results/corpus_stats.json` (committed, small)."""
    out = {
        "source": stats["source"],
        "license": license_ if license_ is not None else "unknown",
        "n_seen": stats["n_seen"],
        "n_dropped_short": stats["n_dropped_short"],
        "n_dropped_long": stats["n_dropped_long"],
        "n_dropped_dup_id": stats["n_dropped_dup_id"],
        "n_dropped_dup_prefix": stats["n_dropped_dup_prefix"],
        "n_kept": stats["n_kept"],
        "char_len": stats["char_len"],
        "split_counts": stats["split_counts"],
        "generated_at": utc_now(),
        "git_sha": git_sha(),
    }
    write_json(CORPUS_STATS_PATH, out)
    return out


def stats_from_docs(path: Any = None) -> dict[str, Any]:
    """Recompute the reportable stats from an existing docs.jsonl. No network.

    Used by the idempotent `sxl corpus build` no-op path, where the corpus is
    already on disk and nothing should be re-downloaded.
    """
    rows = list(read_jsonl(DOCS_PATH if path is None else path))
    return {
        "source": rows[0]["source"] if rows else CORPUS_SOURCES[0],
        "n_seen": 0,
        "n_dropped_short": 0,
        "n_dropped_long": 0,
        "n_dropped_dup_id": 0,
        "n_dropped_dup_prefix": 0,
        "n_kept": len(rows),
        "char_len": _char_len_percentiles([r["char_len"] for r in rows]),
        "split_counts": _split_counts([r["doc_id"] for r in rows]),
    }
