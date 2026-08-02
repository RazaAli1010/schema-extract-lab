"""Teacher labeling over the OpenAI Batch API (F2). Laptop-only, torch-free.

Three properties matter more than speed here:

1. **No double-billing.** Every response is appended to `_teacher_cache.jsonl` the
   moment it arrives, and every submitted batch is recorded in `_batches.json`
   *before* the API call that creates it. A crash at any point re-requests only
   what was never paid for.
2. **Determinism.** Document selection is by sorted `doc_id`, never by sample, so
   `--limit 10` picks the same ten documents every time. Output files are a pure
   function of (docs, cache, caps, limit) — never appended to, never merged.
3. **Honest failure accounting.** Every requested document lands in exactly one of
   `ok` / `parse_failed` / `schema_invalid` / `api_error`. Rows that are not `ok`
   are excluded from the output files and counted in `results/teacher_stats.json`.

`openai` and `dotenv` are imported **inside function bodies**, mirroring how
`corpus.py` handles `datasets`: `tests/test_no_torch_import.py` imports this module
in a subprocess, and the whole test suite runs against a stub client.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sxl.config import (
    DEV_PATH,
    DOCS_PATH,
    DOMAIN,
    EVAL_POOL_PATH,
    MAX_TEACHER_SPEND_USD,
    N_DEV_TARGET,
    N_TRAIN_TARGET,
    ROOT,
    TEACHER_BATCH_DISCOUNT,
    TEACHER_BATCH_SIZE,
    TEACHER_BATCHES_PATH,
    TEACHER_CACHE_PATH,
    TEACHER_CHARS_PER_TOKEN,
    TEACHER_DOC_RETRY_ROUNDS,
    TEACHER_EST_OUTPUT_TOKENS,
    TEACHER_MAX_ENUM_SHARE,
    TEACHER_MAX_INFLIGHT_BATCHES,
    TEACHER_MAX_RETRIES,
    TEACHER_MAX_SKILLS_EMPTY_SHARE,
    TEACHER_MAX_TOKENS,
    TEACHER_MAX_WAIT_S,
    TEACHER_MIN_DOCS_FOR_MEAN_COST,
    TEACHER_MODEL,
    TEACHER_POLL_SECONDS,
    TEACHER_PRICE_USD,
    TEACHER_QUALITY_MIN_N,
    TEACHER_REQUEST_OVERHEAD_TOKENS,
    TEACHER_RETRY_BACKOFF_S,
    TEACHER_STATS_PATH,
    TRAIN_PATH,
    ensure_dirs,
    git_sha,
)
from sxl.config import utc_now as _now  # the one `generated_at`/`at` clock (SPEC §5.2)
from sxl.io import append_jsonl, read_jsonl, write_json, write_jsonl
from sxl.prompts import (
    PROMPT_VERSION,
    SYSTEM_TEACHER,
    TEACHER_PROMPT_SHA,
    TEMPERATURE,
    build_teacher_prompt,
    prompt_sha,
    truncate,
)
from sxl.schema import (
    ENUM_FIELDS,
    ENUM_TYPES,
    FIELD_NAMES,
    JSON_SCHEMA,
    SET_FIELDS,
    validate_prediction,
)
from sxl.splits import split_for

SPLITS = ("train", "dev", "eval_pool")
SPLIT_CHOICES = (*SPLITS, "all")
BUCKETS = ("ok", "parse_failed", "schema_invalid", "api_error")

#: SPEC §3.3 output row, keys in this exact order.
ROW_KEYS = (
    "doc_id",
    "domain",
    "text",
    "gold",
    "label_source",
    "teacher_model",
    "verified_by_human",
    "verified_at",
)

BATCH_ENDPOINT = "/v1/chat/completions"
TERMINAL_STATUSES = frozenset({"completed", "failed", "expired", "cancelled"})


# --- exceptions ---------------------------------------------------------------
# The library raises with the numbers attached; `cli.py` owns the exit codes,
# exactly as corpus.py raises and the CLI decides.


class TeacherError(RuntimeError):
    """Any contract violation in the teacher pipeline."""


class MissingApiKey(TeacherError):
    """OPENAI_API_KEY is not set."""


class SpendCapExceeded(TeacherError):
    """Projected spend would cross MAX_TEACHER_SPEND_USD. Raised before submitting."""

    def __init__(self, *, spent: float, projected: float, cap: float, n_remaining: int) -> None:
        self.spent, self.projected, self.cap, self.n_remaining = spent, projected, cap, n_remaining
        super().__init__(
            f"spend cap: would spend ${projected:.2f} on {n_remaining} more documents "
            f"on top of ${spent:.2f} already spent this run; cap is ${cap:.2f}. "
            "Nothing was submitted."
        )


class BatchTimeout(TeacherError):
    """A batch did not reach a terminal status within TEACHER_MAX_WAIT_S."""


class BatchCancelled(TeacherError):
    """A batch was cancelled out of band. Never silently resubmit a human's cancellation."""


# --- pure functions -----------------------------------------------------------


def _json_candidates(text: str) -> Iterator[str]:
    yield text

    if text.startswith("```"):
        body = text[3:]
        if body.lower().startswith("json"):
            body = body[4:]
        yield (body.rsplit("```", 1)[0] if "```" in body else body).strip()

    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        yield text[start : end + 1]


def extract_json(raw: str | None) -> dict[str, Any] | None:
    """Tolerant JSON-object parser. Returns `None` when nothing parses to a dict.

    Tried in order: the whole string, the string with ```` ``` ````/```` ```json ````
    fences stripped, then the substring from the first `{` to the last `}`.

    Malformed JSON is never repaired by regex — a document the teacher could not
    label cleanly is dropped, not guessed at (F2 scope §2).

    Every branch checks `isinstance(obj, dict)`. `json.loads` happily returns a
    list, `None` or a number for `"[1,2,3]"` / `"null"` / `"5"`, and each of those
    would otherwise reach `validate_prediction` and be misfiled as `schema_invalid`
    when the honest bucket is `parse_failed`.
    """
    if not raw or not raw.strip():
        return None

    for candidate in _json_candidates(raw.strip()):
        try:
            obj = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict):
            return obj
    return None


def cost_of(usage: Mapping[str, int], model: str = TEACHER_MODEL) -> float:
    """USD for one response at Batch API rates.

    `TEACHER_PRICE_USD` holds **standard** rates; the 50% batch discount is applied
    here. A `KeyError` on an unpriced model is deliberate and loud — the CLI
    pre-validates `--model`, because a spend cap that cannot price the model is not
    a spend cap.
    """
    p = TEACHER_PRICE_USD[model]
    return TEACHER_BATCH_DISCOUNT * (
        usage["input_tokens"] / 1e6 * p["in"] + usage["output_tokens"] / 1e6 * p["out"]
    )


def estimate_usage(docs: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Char-based token estimate for the spend guard. There is no tokenizer on the laptop."""
    system_chars = len(SYSTEM_TEACHER)
    input_tokens = sum(
        math.ceil((system_chars + len(truncate(d["text"]))) / TEACHER_CHARS_PER_TOKEN)
        + TEACHER_REQUEST_OVERHEAD_TOKENS
        for d in docs
    )
    return {"input_tokens": input_tokens, "output_tokens": TEACHER_EST_OUTPUT_TOKENS * len(docs)}


def build_request(doc: Mapping[str, Any], model: str = TEACHER_MODEL) -> dict[str, Any]:
    """One line of an OpenAI Batch input file.

    `custom_id` **is** the `doc_id` — that is how results are rejoined, and F1
    guarantees it is unique. Structured Outputs (`strict: true`) is what drives
    `n_parse_failed` and `n_schema_invalid` to ~0; `JSON_SCHEMA` is strict-compatible
    (only `$defs`/`$ref`/`anyOf`/`enum`, `additionalProperties: false`, all 16 keys
    required).

    The system message goes first so any automatic prefix caching can apply, but the
    spend projection assumes none — if it applies, the run comes in under budget,
    which is the safe direction.
    """
    system, user = build_teacher_prompt(doc["text"])
    return {
        "custom_id": doc["doc_id"],
        "method": "POST",
        "url": BATCH_ENDPOINT,
        "body": {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": TEMPERATURE,
            "max_tokens": TEACHER_MAX_TOKENS,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "job_posting", "strict": True, "schema": JSON_SCHEMA},
            },
        },
    }


def parse_result_line(
    line: Mapping[str, Any], *, model: str, prompt_sha_: str, batch_id: str
) -> dict[str, Any]:
    """Map one Batch output/error JSONL line to one cache row.

    OpenAI reports `prompt_tokens`/`completion_tokens`; the cache and `cost_of` use
    the provider-neutral `input_tokens`/`output_tokens`. That mapping happens here
    and nowhere else.

    `finish_reason == "length"` is recorded as `ok: True`. The truncated text is
    real, `extract_json` will fail on it, and it lands in `parse_failed` — the honest
    bucket, and how a too-low `TEACHER_MAX_TOKENS` becomes visible in the stats
    instead of hiding as a transport error.
    """
    row: dict[str, Any] = {
        "doc_id": line.get("custom_id"),
        "teacher_model": model,
        "prompt_sha": prompt_sha_,
        "raw_output": None,
        "usage": {"input_tokens": 0, "output_tokens": 0},  # errored requests are not billed
        "ok": False,
        "error": None,
        "finish_reason": None,
        "batch_id": batch_id,
        "at": _now(),
    }

    err = line.get("error")
    if err:
        code = err.get("code") if isinstance(err, Mapping) else None
        message = err.get("message") if isinstance(err, Mapping) else err
        row["error"] = f"request_error: {code}: {message}"
        return row

    response = line.get("response") or {}
    status = response.get("status_code")
    body = response.get("body") or {}
    if status != 200:
        detail = (body.get("error") or {}).get("message") if isinstance(body, Mapping) else None
        row["error"] = f"http_{status}: {detail}"
        return row

    usage = body.get("usage") or {}
    row["usage"] = {
        "input_tokens": int(usage.get("prompt_tokens", 0)),
        "output_tokens": int(usage.get("completion_tokens", 0)),
    }

    choices = body.get("choices") or []
    if not choices:
        row["error"] = "empty_content"
        return row

    choice = choices[0]
    row["finish_reason"] = choice.get("finish_reason")
    message = choice.get("message") or {}

    refusal = message.get("refusal")
    if refusal:
        row["error"] = f"refusal: {refusal}"
        return row

    content = message.get("content")
    if not content:
        row["error"] = "empty_content"
        return row

    row["raw_output"] = content
    row["ok"] = True
    return row


def parse_results(
    lines: Iterable[Mapping[str, Any]], *, model: str, prompt_sha_: str, batch_id: str
) -> list[dict[str, Any]]:
    """Pure, client-free mapper over a batch's result lines."""
    return [
        parse_result_line(line, model=model, prompt_sha_=prompt_sha_, batch_id=batch_id)
        for line in lines
    ]


def bucket_for_row(cache_row: Mapping[str, Any] | None) -> tuple[str, dict[str, Any] | None]:
    """Classify one cache row into exactly one of the four buckets.

    `mode="json"` on the dump is mandatory: without it `Enum` members leak into the
    dict and `json.dumps` raises at write time. It also yields the 16 keys in
    `FIELD_NAMES` order for free.
    """
    if cache_row is None or not cache_row.get("ok"):
        return "api_error", None
    obj = extract_json(cache_row.get("raw_output"))
    if obj is None:
        return "parse_failed", None
    posting = validate_prediction(obj)
    if posting is None:
        return "schema_invalid", None
    return "ok", posting.model_dump(mode="json")


def select_docs(
    docs: Iterable[Mapping[str, Any]], *, split: str = "train", limit: int = 0
) -> list[dict[str, Any]]:
    """The single definition of a run's scope. Deterministic, offline, no client.

    `train` and `dev` are capped at their targets; **`eval_pool` is never capped** —
    F3 samples 330 candidates from it and a cap below that would starve the gold set.

    Caps and `limit` are taken by **sorted `doc_id`**, never by random sample, so
    re-running selects the identical documents. Getting this wrong means paying
    twice for the same work.
    """
    if split not in SPLIT_CHOICES:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLIT_CHOICES}")

    wanted = SPLITS if split == "all" else (split,)
    caps = {"train": N_TRAIN_TARGET, "dev": N_DEV_TARGET, "eval_pool": 0}  # 0 == uncapped

    grouped: dict[str, list[dict[str, Any]]] = {s: [] for s in SPLITS}
    for doc in docs:
        s = split_for(doc["doc_id"])
        if s in wanted:
            grouped[s].append(dict(doc))

    selected: list[dict[str, Any]] = []
    for s in wanted:
        rows = sorted(grouped[s], key=lambda d: d["doc_id"])
        selected.extend(rows[: caps[s]] if caps[s] else rows)

    selected.sort(key=lambda d: d["doc_id"])
    return selected[:limit] if limit else selected


def is_absent(field: str, value: Any) -> bool:
    """SPEC §3.5's absent marker, defined once. F4 reuses this predicate."""
    if field in ENUM_FIELDS:
        return value == "unknown"
    if field in SET_FIELDS:
        return not value
    return value is None


def field_null_rate(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Per-field absence rate over the written rows.

    Keeps the SPEC §F2 key name `field_null_rate`; it is really an *absence* rate,
    because for enum fields absence is `"unknown"`, not `null`. One shared definition
    of absence is worth more than a better name.
    """
    n = len(rows)
    if not n:
        return dict.fromkeys(FIELD_NAMES, 0.0)
    return {f: round(sum(is_absent(f, r["gold"][f]) for r in rows) / n, 4) for f in FIELD_NAMES}


def enum_distribution(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    """Counts per enum member, **including members with zero occurrences**.

    `"principal": 0` across 4,500 training postings is real signal that the prompt
    never surfaces that level; a plain `Counter` would silently omit it.
    """
    out: dict[str, dict[str, int]] = {}
    for field in FIELD_NAMES:
        if field not in ENUM_FIELDS:
            continue
        counts = {member.value: 0 for member in ENUM_TYPES[field]}
        for row in rows:
            value = row["gold"][field]
            counts[value] = counts.get(value, 0) + 1
        out[field] = counts
    return out


def quality_warnings(
    *, n_ok: int, enum_dist: Mapping[str, Mapping[str, int]], null_rate: Mapping[str, float]
) -> list[str]:
    """The label-quality smoke test (F2 acceptance criteria).

    Gated on `TEACHER_QUALITY_MIN_N`: 20 postings really are all
    `education_level: "unknown"`, and firing on the smoke run would teach everyone to
    ignore the warning that matters.
    """
    warnings: list[str] = []
    if n_ok < TEACHER_QUALITY_MIN_N:
        return warnings

    for field, counts in enum_dist.items():
        if not counts:
            continue
        top, n = max(counts.items(), key=lambda kv: kv[1])
        if n / n_ok > TEACHER_MAX_ENUM_SHARE:
            warnings.append(
                f"{field} is {100 * n / n_ok:.1f}% {top!r} ({n}/{n_ok}) — "
                "the prompt is probably broken; fix it before running F3"
            )

    empty_skills = null_rate.get("required_skills", 0.0)
    if empty_skills > TEACHER_MAX_SKILLS_EMPTY_SHARE:
        warnings.append(
            f"required_skills is empty for {100 * empty_skills:.1f}% of documents — "
            "the prompt is probably broken; fix it before running F3"
        )
    return warnings


# --- paths seam ---------------------------------------------------------------


@dataclass(frozen=True)
class TeacherPaths:
    """Every file F2 touches, in one injectable object.

    `corpus.build` needed a single `out_path=`; F2 touches six files. Passing one
    `TeacherPaths.in_dir(tmp_path)` makes it structurally impossible for a test to
    write into the real `data/`.
    """

    docs: Path
    cache: Path
    batches: Path
    stats: Path
    train: Path
    dev: Path
    eval_pool: Path

    @classmethod
    def default(cls) -> TeacherPaths:
        return cls(
            docs=DOCS_PATH,
            cache=TEACHER_CACHE_PATH,
            batches=TEACHER_BATCHES_PATH,
            stats=TEACHER_STATS_PATH,
            train=TRAIN_PATH,
            dev=DEV_PATH,
            eval_pool=EVAL_POOL_PATH,
        )

    @classmethod
    def in_dir(cls, root: Path) -> TeacherPaths:
        root = Path(root)
        labeled = root / "data" / "labeled"
        return cls(
            docs=root / "data" / "raw" / "docs.jsonl",
            cache=labeled / "_teacher_cache.jsonl",
            batches=labeled / "_batches.json",
            stats=root / "results" / "teacher_stats.json",
            train=labeled / "train.jsonl",
            dev=labeled / "dev.jsonl",
            eval_pool=labeled / "eval_pool.jsonl",
        )

    def for_split(self, split: str) -> Path:
        return {"train": self.train, "dev": self.dev, "eval_pool": self.eval_pool}[split]


def _resolve(paths: TeacherPaths | None) -> TeacherPaths:
    return paths if paths is not None else TeacherPaths.default()


def _clock(
    sleep: Callable[[float], None] | None, monotonic: Callable[[], float] | None = None
) -> tuple[Callable[[float], None], Callable[[], float]]:
    """Resolve the injected clock **at call time**, not at import.

    A `sleep=time.sleep` default argument would bind the real function when this
    module is first imported, so a test could not neutralize it and every polling
    test would wait a real 60 seconds.
    """
    return (sleep or time.sleep, monotonic or time.monotonic)


# --- cache and ledger ---------------------------------------------------------


def load_cache(paths: TeacherPaths | None = None) -> dict[str, dict[str, Any]]:
    """Read `_teacher_cache.jsonl` keyed by `doc_id`, **last row wins**.

    Last-wins *is* the per-document retry mechanism: the retry round appends a fresh
    row that supersedes the failed one. The cache also stores raw output rather than
    a bucket, so fixing `extract_json` or a schema bug re-buckets every document from
    disk at zero cost.
    """
    paths = _resolve(paths)
    if not paths.cache.exists():
        return {}
    return {row["doc_id"]: row for row in read_jsonl(paths.cache)}


def cached_doc_ids(
    cache: Mapping[str, Mapping[str, Any]], *, model: str, prompt_sha_: str
) -> set[str]:
    """Documents already paid for at this `(model, prompt_sha)`.

    Filters on `row["ok"]` as well: an `api_error` row is a *record of a failed
    attempt*, not a paid label, so it must not suppress a re-request on a later run.
    Within a run the retry is bounded by `TEACHER_DOC_RETRY_ROUNDS`; across runs, the
    user re-invoking the command is an explicit decision to retry.
    """
    return {
        doc_id
        for doc_id, row in cache.items()
        if row.get("teacher_model") == model
        and row.get("prompt_sha") == prompt_sha_
        and row.get("ok")
    }


def append_cache(row: Mapping[str, Any], paths: TeacherPaths | None = None) -> None:
    """Append one response. Called as each result arrives, so a crash loses nothing."""
    append_jsonl(_resolve(paths).cache, dict(row))


def load_ledger(paths: TeacherPaths | None = None) -> dict[str, Any]:
    paths = _resolve(paths)
    if not paths.batches.exists():
        return {"version": 1, "model": None, "prompt_sha": None, "batches": []}
    with open(paths.batches, encoding="utf-8") as fh:
        return json.load(fh)


def save_ledger(ledger: Mapping[str, Any], paths: TeacherPaths | None = None) -> None:
    write_json(_resolve(paths).batches, dict(ledger))


def _check_ledger_consistency(ledger: Mapping[str, Any], *, model: str, prompt_sha_: str) -> None:
    """Mixing two teachers or two prompts within one run is forbidden (F2 out-of-scope §4)."""
    if not ledger.get("batches"):
        return
    for key, want in (("model", model), ("prompt_sha", prompt_sha_)):
        got = ledger.get(key)
        if got is not None and got != want:
            raise TeacherError(
                f"in-flight batches were submitted with {key}={got!r} but this run uses "
                f"{want!r}. Finish that run with `sxl teacher label --resume`, or delete "
                "data/labeled/_batches.json, before switching."
            )


# --- client and batch lifecycle -----------------------------------------------


def make_client(api_key: str | None = None) -> Any:
    """Build an OpenAI client from `.env`.

    `.env` is loaded here rather than in `config.py`, whose docstring promises
    stdlib-only imports and which would otherwise mutate `os.environ` for every
    unrelated `sxl` command.

    `max_retries=0` is deliberate: the SDK's automatic retry could re-issue a
    `batches.create` whose first attempt actually succeeded, creating and billing a
    duplicate batch. We own retries so every one is preceded by an adoption scan.
    """
    import os

    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise MissingApiKey(
            "OPENAI_API_KEY is unset. Copy .env.example to .env and fill it in "
            "(.env is gitignored — never commit a real key)."
        )

    from openai import OpenAI

    return OpenAI(api_key=key, max_retries=0, timeout=60.0)


def transient_errors() -> tuple[type[BaseException], ...]:
    """Errors worth retrying. Everything else is a contract violation (SPEC §5.5).

    Notably absent: `BadRequestError`, `AuthenticationError`, `PermissionDeniedError`,
    `NotFoundError`. A malformed request body is malformed on every attempt; retrying
    it three times only delays the diagnosis.
    """
    import openai

    return (
        openai.APIConnectionError,
        openai.APITimeoutError,
        openai.RateLimitError,
        openai.InternalServerError,
    )


def _with_retry(
    fn: Callable[[], Any],
    *,
    what: str,
    attempts: int = TEACHER_MAX_RETRIES,
    backoff: Sequence[float] = TEACHER_RETRY_BACKOFF_S,
    sleep: Callable[[float], None] | None = None,
    before_retry: Callable[[], Any] | None = None,
) -> Any:
    """Retry `fn` on transient API errors only, with exponential backoff.

    `before_retry` is the adoption hook: `submit_batch` uses it so a retried
    `batches.create` can never double-bill a batch whose first attempt landed.
    """
    sleep, _ = _clock(sleep)
    transient = transient_errors()
    last: BaseException | None = None
    for attempt in range(attempts):
        try:
            return fn()
        except transient as exc:
            last = exc
            if attempt == attempts - 1:
                break
            wait = backoff[min(attempt, len(backoff) - 1)]
            print(f"[teacher] {what} failed ({type(exc).__name__}); retrying in {wait}s")
            sleep(wait)
            if before_retry is not None:
                adopted = before_retry()
                if adopted is not None:
                    return adopted
    raise TeacherError(f"{what} failed after {attempts} attempts: {last}") from last


def _docs_sha(doc_ids: Sequence[str]) -> str:
    return prompt_sha(",".join(sorted(doc_ids)))


def _adopt_existing_batch(client: Any, *, input_file_id: str, docs_sha: str) -> Any | None:
    """Find a batch that a lost `batches.create` may already have created.

    Without this scan, resuming after a crash between `files.create` and the ledger's
    `batch_id` write either duplicate-bills or abandons a paid batch.
    """
    try:
        page = client.batches.list(limit=100)
    except Exception as exc:  # noqa: BLE001 — adoption is best-effort; the caller resubmits
        print(f"[teacher] could not list batches for adoption ({exc}); assuming none")
        return None
    for batch in getattr(page, "data", None) or []:
        meta = getattr(batch, "metadata", None) or {}
        if (
            getattr(batch, "input_file_id", None) == input_file_id
            or meta.get("docs_sha") == docs_sha
        ):
            print(f"[teacher] adopted orphaned batch {batch.id} (input_file_id={input_file_id})")
            return batch
    return None


def submit_batch(
    client: Any,
    requests: Sequence[Mapping[str, Any]],
    *,
    tag: str,
    model: str,
    prompt_sha_: str,
    ledger: dict[str, Any],
    entry: dict[str, Any] | None = None,
    paths: TeacherPaths | None = None,
    sleep: Callable[[float], None] | None = None,
) -> dict[str, Any]:
    """Upload, record, create. Returns the ledger entry.

    The order matters and is the crash-window narrowing: the ledger entry carrying
    `input_file_id` is persisted **before** `batches.create`, so a crash in between
    leaves enough on disk for `_adopt_existing_batch` to find the batch if it landed.
    """
    paths = _resolve(paths)
    sleep, _ = _clock(sleep)
    doc_ids = [r["custom_id"] for r in requests]
    docs_sha = _docs_sha(doc_ids)

    if entry is None:
        payload = ("\n".join(json.dumps(r, ensure_ascii=False) for r in requests) + "\n").encode()
        file_obj = _with_retry(
            lambda: client.files.create(file=("batch.jsonl", payload), purpose="batch"),
            what=f"files.create({tag})",
            sleep=sleep,
        )
        entry = {
            "tag": tag,
            "attempt": 1,
            "input_file_id": file_obj.id,
            "batch_id": None,
            "output_file_id": None,
            "error_file_id": None,
            "docs_sha": docs_sha,
            "doc_ids": doc_ids,
            "n_requests": len(requests),
            "status": "uploaded",
            "submitted_at": None,
            "completed_at": None,
            "harvested": False,
        }
        ledger["batches"].append(entry)
        save_ledger(ledger, paths)  # <-- persisted BEFORE batches.create

    input_file_id = entry["input_file_id"]

    def _create() -> Any:
        return client.batches.create(
            input_file_id=input_file_id,
            endpoint=BATCH_ENDPOINT,
            completion_window="24h",
            metadata={
                "sxl": "teacher",
                "prompt_sha": prompt_sha_,
                "tag": tag,
                "docs_sha": docs_sha,
            },
        )

    batch = _with_retry(
        _create,
        what=f"batches.create({tag})",
        sleep=sleep,
        before_retry=lambda: _adopt_existing_batch(
            client, input_file_id=input_file_id, docs_sha=docs_sha
        ),
    )

    entry["batch_id"] = batch.id
    entry["status"] = getattr(batch, "status", "validating")
    entry["submitted_at"] = _now()
    ledger["model"], ledger["prompt_sha"] = model, prompt_sha_
    save_ledger(ledger, paths)
    print(f"[teacher] submitted {tag}: {len(requests)} requests as {batch.id}")
    return entry


def poll_batch(
    client: Any,
    batch_id: str,
    *,
    poll_seconds: float = TEACHER_POLL_SECONDS,
    max_wait_s: float = TEACHER_MAX_WAIT_S,
    sleep: Callable[[float], None] | None = None,
    monotonic: Callable[[], float] | None = None,
) -> Any:
    """Poll until terminal and return the **batch object**.

    Deliberately not `-> list[dict]`: the terminal status (`completed` / `expired` /
    `failed` / `cancelled`) drives four different recovery paths, so the caller needs
    the object. Downloading is `harvest_batch`.
    """
    sleep, monotonic = _clock(sleep, monotonic)
    started = monotonic()
    last_status: str | None = None
    while True:
        batch = _with_retry(
            lambda: client.batches.retrieve(batch_id),
            what=f"batches.retrieve({batch_id})",
            sleep=sleep,
        )
        status = getattr(batch, "status", None)
        if status != last_status:  # print on change, not on every poll
            counts = getattr(batch, "request_counts", None)
            done, total = getattr(counts, "completed", None), getattr(counts, "total", None)
            suffix = f" {done}/{total}" if total else ""
            print(f"[teacher] batch {batch_id} {status}{suffix}")
            last_status = status
        if status in TERMINAL_STATUSES:
            return batch
        if monotonic() - started > max_wait_s:
            raise BatchTimeout(
                f"batch {batch_id} was still {status} after {max_wait_s:.0f}s. "
                "Re-run with --resume; the batch is still live and already paid for."
            )
        sleep(poll_seconds)


def _download_lines(
    client: Any, file_id: str | None, *, sleep: Callable[[float], None]
) -> list[dict[str, Any]]:
    if not file_id:
        return []
    content = _with_retry(
        lambda: client.files.content(file_id), what=f"files.content({file_id})", sleep=sleep
    )
    text = getattr(content, "text", None)
    if text is None:
        raw = content.read() if hasattr(content, "read") else content
        text = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def harvest_batch(
    client: Any,
    batch: Any,
    *,
    model: str,
    prompt_sha_: str,
    paths: TeacherPaths | None = None,
    sleep: Callable[[float], None] | None = None,
) -> list[dict[str, Any]]:
    """Download a terminal batch's results, appending each row to the cache as it parses."""
    paths = _resolve(paths)
    sleep, _ = _clock(sleep)
    lines = _download_lines(client, getattr(batch, "output_file_id", None), sleep=sleep)
    lines += _download_lines(client, getattr(batch, "error_file_id", None), sleep=sleep)
    rows = parse_results(lines, model=model, prompt_sha_=prompt_sha_, batch_id=batch.id)
    for row in rows:
        append_cache(row, paths)
    return rows


def _batch_errors_text(batch: Any) -> str:
    errors = getattr(batch, "errors", None)
    data = getattr(errors, "data", None) or []
    if not data:
        return str(errors)
    return "; ".join(f"{getattr(e, 'code', '?')}: {getattr(e, 'message', e)}" for e in data)


def _is_capacity_error(batch: Any) -> bool:
    text = _batch_errors_text(batch).lower()
    return ("token" in text and "limit" in text) or "enqueued" in text


# --- orchestrator -------------------------------------------------------------


def _new_stats(*, model: str, split: str, limit: int) -> dict[str, Any]:
    return {
        "teacher_model": model,
        "prompt_sha": TEACHER_PROMPT_SHA,
        "prompt_version": PROMPT_VERSION,
        "split": split,
        "limit": limit,
        "n_requested": 0,
        "n_cached": 0,
        "n_new_requests": 0,
        "n_ok": 0,
        "n_parse_failed": 0,
        "n_schema_invalid": 0,
        "n_api_error": 0,
        "n_batches": 0,
        "spend_usd": 0.0,
        "spend_usd_cached": 0.0,
        "tokens": {"input": 0, "output": 0},
        "split_counts": {},
        "field_null_rate": {},
        "enum_distribution": {},
        "quality_warnings": [],
    }


def _guard_spend(
    stats: dict[str, Any],
    remaining: Sequence[Mapping[str, Any]],
    *,
    model: str,
    n_priced: int = 0,
) -> float:
    """Project the cost of `remaining` and raise **before** anything is submitted.

    The cap governs *new* spend this run, not lifetime spend: `spend_usd_cached` is
    excluded, or a completed $1.50 run would be one flag away from tripping its own
    cap on a zero-cost re-run.
    """
    if not remaining:
        return 0.0

    projected = cost_of(estimate_usage(remaining), model)
    if n_priced >= TEACHER_MIN_DOCS_FOR_MEAN_COST and stats["spend_usd"] > 0:
        # max() of the observed mean and the char estimate: a spend cap should err
        # high. The 50-document floor stops one anomalous response from
        # extrapolating a fake projection and aborting a healthy run.
        projected = max(stats["spend_usd"] / n_priced * len(remaining), projected)

    if stats["spend_usd"] + projected > MAX_TEACHER_SPEND_USD:
        raise SpendCapExceeded(
            spent=stats["spend_usd"],
            projected=projected,
            cap=MAX_TEACHER_SPEND_USD,
            n_remaining=len(remaining),
        )
    return projected


def _account(stats: dict[str, Any], rows: Iterable[Mapping[str, Any]]) -> None:
    for row in rows:
        usage = row.get("usage") or {"input_tokens": 0, "output_tokens": 0}
        stats["tokens"]["input"] += usage["input_tokens"]
        stats["tokens"]["output"] += usage["output_tokens"]
        stats["spend_usd"] += cost_of(usage, row["teacher_model"])


def _handle_terminal(
    client: Any,
    batch: Any,
    entry: dict[str, Any],
    *,
    model: str,
    prompt_sha_: str,
    paths: TeacherPaths,
    ledger: dict[str, Any],
    stats: dict[str, Any],
    sleep: Callable[[float], None],
) -> list[str]:
    """Harvest a terminal batch. Returns the doc_ids that still need resubmitting."""
    status = getattr(batch, "status", None)
    entry["status"] = status
    entry["output_file_id"] = getattr(batch, "output_file_id", None)
    entry["error_file_id"] = getattr(batch, "error_file_id", None)
    entry["completed_at"] = _now()

    if status == "failed":
        # Validation failure: never processed, never billed.
        entry["harvested"] = True
        save_ledger(ledger, paths)
        if _is_capacity_error(batch):
            print(f"[teacher] batch {batch.id} hit a queue limit; it will be resubmitted")
            return list(entry["doc_ids"])
        raise TeacherError(
            f"batch {batch.id} failed validation and would fail on every attempt: "
            f"{_batch_errors_text(batch)}"
        )

    rows = harvest_batch(
        client, batch, model=model, prompt_sha_=prompt_sha_, paths=paths, sleep=sleep
    )
    _account(stats, rows)
    entry["harvested"] = True
    save_ledger(ledger, paths)

    returned = {r["doc_id"] for r in rows if r.get("ok")}
    missing = [d for d in entry["doc_ids"] if d not in returned]

    if status == "cancelled":
        raise BatchCancelled(
            f"batch {batch.id} was cancelled out of band; {len(returned)} results were "
            f"harvested. Re-run to resubmit the remaining {len(missing)}."
        )
    if status == "expired":
        print(
            f"[teacher] batch {batch.id} expired; {len(returned)}/{entry['n_requests']} "
            f"recovered, {len(missing)} to resubmit (unprocessed requests are not billed)"
        )
        return missing
    return []


def _run_wave(
    client: Any,
    todo: Sequence[Mapping[str, Any]],
    *,
    tag_prefix: str,
    model: str,
    ledger: dict[str, Any],
    paths: TeacherPaths,
    stats: dict[str, Any],
    sleep: Callable[[float], None],
    monotonic: Callable[[], float],
    poll_seconds: float,
    max_wait_s: float,
) -> None:
    """Submit `todo` in chunks with at most `TEACHER_MAX_INFLIGHT_BATCHES` in flight."""
    by_id = {d["doc_id"]: d for d in todo}
    pending: list[list[str]] = [
        [d["doc_id"] for d in todo[i : i + TEACHER_BATCH_SIZE]]
        for i in range(0, len(todo), TEACHER_BATCH_SIZE)
    ]
    inflight: list[dict[str, Any]] = []
    resubmits = 0

    while pending or inflight:
        while pending and len(inflight) < TEACHER_MAX_INFLIGHT_BATCHES:
            chunk = [by_id[d] for d in pending.pop(0)]
            _guard_spend(stats, chunk, model=model, n_priced=stats["n_new_requests"])
            entry = submit_batch(
                client,
                [build_request(d, model) for d in chunk],
                tag=f"{tag_prefix}-{len(ledger['batches']):04d}",
                model=model,
                prompt_sha_=TEACHER_PROMPT_SHA,
                ledger=ledger,
                paths=paths,
                sleep=sleep,
            )
            stats["n_batches"] += 1
            stats["n_new_requests"] += len(chunk)
            inflight.append(entry)

        if not inflight:
            break

        entry = inflight.pop(0)
        batch = poll_batch(
            client,
            entry["batch_id"],
            poll_seconds=poll_seconds,
            max_wait_s=max_wait_s,
            sleep=sleep,
            monotonic=monotonic,
        )
        missing = _handle_terminal(
            client,
            batch,
            entry,
            model=model,
            prompt_sha_=TEACHER_PROMPT_SHA,
            paths=paths,
            ledger=ledger,
            stats=stats,
            sleep=sleep,
        )
        if missing:
            resubmits += 1
            if resubmits >= TEACHER_MAX_RETRIES:
                print(f"[teacher] giving up on {len(missing)} documents after {resubmits} retries")
                continue
            pending.append(missing)


def _resume_inflight(
    client: Any,
    *,
    model: str,
    ledger: dict[str, Any],
    paths: TeacherPaths,
    stats: dict[str, Any],
    sleep: Callable[[float], None],
    monotonic: Callable[[], float],
    poll_seconds: float,
    max_wait_s: float,
) -> None:
    """Harvest every un-harvested ledger entry **before** submitting anything new.

    The work is paid for the moment a batch is created, so this is what makes a crash
    cost wall time and never money.
    """
    for entry in list(ledger["batches"]):
        if entry.get("harvested"):
            continue
        if not entry.get("batch_id"):
            adopted = _adopt_existing_batch(
                client, input_file_id=entry["input_file_id"], docs_sha=entry["docs_sha"]
            )
            if adopted is None:
                # No batch_id and no orphan means `batches.create` never landed:
                # nothing was created and nothing was billed. Close the entry so it
                # stops being rescanned; the documents flow back through `todo`.
                print(f"[teacher] {entry['tag']}: no orphaned batch found; it will be resubmitted")
                entry["status"] = "abandoned"
                entry["harvested"] = True
                save_ledger(ledger, paths)
                continue
            entry["batch_id"] = adopted.id
            save_ledger(ledger, paths)
        print(f"[teacher] resuming {entry['tag']} ({entry['batch_id']})")
        batch = poll_batch(
            client,
            entry["batch_id"],
            poll_seconds=poll_seconds,
            max_wait_s=max_wait_s,
            sleep=sleep,
            monotonic=monotonic,
        )
        _handle_terminal(
            client,
            batch,
            entry,
            model=model,
            prompt_sha_=TEACHER_PROMPT_SHA,
            paths=paths,
            ledger=ledger,
            stats=stats,
            sleep=sleep,
        )


def _retry_round(
    client: Any,
    selected: Sequence[Mapping[str, Any]],
    cache: Mapping[str, Mapping[str, Any]],
    *,
    model: str,
    ledger: dict[str, Any],
    paths: TeacherPaths,
    stats: dict[str, Any],
    sleep: Callable[[float], None],
    monotonic: Callable[[], float],
    poll_seconds: float,
    max_wait_s: float,
) -> None:
    """One follow-up batch for `parse_failed` and `api_error`, with the identical prompt.

    `schema_invalid` is deliberately **not** retried: at `temperature=0` under
    Structured Outputs a shape violation is deterministic and will reproduce, so
    retrying it is pure spend. Mutating the prompt instead would make the training
    data inconsistent, which is why the prompt is reused verbatim.
    """
    if TEACHER_DOC_RETRY_ROUNDS < 1:
        return
    failed = [
        doc
        for doc in selected
        if bucket_for_row(cache.get(doc["doc_id"]))[0] in ("parse_failed", "api_error")
    ]
    if not failed:
        return
    if len(failed) / max(len(selected), 1) > 0.2:
        print(
            f"[teacher] {len(failed)}/{len(selected)} documents failed the first pass — "
            "this is a prompt problem, not transience"
        )
    print(f"[teacher] retrying {len(failed)} documents (one round)")
    _run_wave(
        client,
        failed,
        tag_prefix="retry",
        model=model,
        ledger=ledger,
        paths=paths,
        stats=stats,
        sleep=sleep,
        monotonic=monotonic,
        poll_seconds=poll_seconds,
        max_wait_s=max_wait_s,
    )


def build_rows(
    selected: Sequence[Mapping[str, Any]],
    cache: Mapping[str, Mapping[str, Any]],
    *,
    model: str,
    stats: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """Bucket every selected document and emit the SPEC §3.3 rows for the `ok` ones.

    A document with **no cache row at all** counts as `api_error`, which keeps the
    four buckets a true partition of `selected`.

    `text` is the full, untruncated document: F5/F6 apply `truncate` themselves with
    the same `MAX_INPUT_CHARS` so the arms stay identical (SPEC §3.6), while this
    file stays a faithful copy of the corpus. Truncation is a prompt-time concern.
    """
    out: dict[str, list[dict[str, Any]]] = {s: [] for s in SPLITS}
    for doc in selected:
        bucket, gold = bucket_for_row(cache.get(doc["doc_id"]))
        stats[f"n_{bucket}"] += 1
        if bucket != "ok":
            continue
        out[split_for(doc["doc_id"])].append(
            {
                "doc_id": doc["doc_id"],
                "domain": doc.get("domain", DOMAIN),
                "text": doc["text"],
                "gold": gold,
                "label_source": "teacher",
                "teacher_model": model,
                "verified_by_human": False,
                "verified_at": None,
            }
        )
    for rows in out.values():
        rows.sort(key=lambda r: r["doc_id"])  # byte-identical across runs
    return out


def write_splits(
    rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    split: str,
    paths: TeacherPaths | None = None,
) -> dict[str, int]:
    """Write only the splits in scope, so `--split dev` never truncates `train.jsonl`."""
    paths = _resolve(paths)
    wanted = SPLITS if split == "all" else (split,)
    return {s: write_jsonl(paths.for_split(s), rows_by_split.get(s, [])) for s in wanted}


def report_stats(
    stats: Mapping[str, Any], *, paths: TeacherPaths | None = None, write: bool = True
) -> dict[str, Any]:
    """Validate, print and persist `results/teacher_stats.json`."""
    paths = _resolve(paths)

    total = sum(stats[f"n_{b}"] for b in BUCKETS)
    if total != stats["n_requested"]:
        raise ValueError(
            f"the four failure buckets sum to {total} but {stats['n_requested']} documents "
            "were requested — every document must land in exactly one bucket (F2 §6)"
        )

    out = {**stats, "generated_at": _now(), "git_sha": git_sha()}
    if write:
        write_json(paths.stats, out)
    _print_quality(out)
    return out


def _print_quality(stats: Mapping[str, Any]) -> None:
    """The label-quality smoke test, on stdout (F2 §6)."""
    n_ok = stats["n_ok"]
    if not n_ok:
        print("[teacher] no valid labels — nothing to summarize")
        return
    print(f"[teacher] {n_ok} valid labels; enum distribution (top 3 per field):")
    for field, counts in stats["enum_distribution"].items():
        top = sorted(counts.items(), key=lambda kv: -kv[1])[:3]
        rendered = "  ".join(f"{v}={n} ({100 * n / n_ok:.0f}%)" for v, n in top)
        print(f"[teacher]   {field:18s} {rendered}")
    print("[teacher] highest absence rates:")
    for field, rate in sorted(stats["field_null_rate"].items(), key=lambda kv: -kv[1])[:5]:
        print(f"[teacher]   {field:22s} {100 * rate:.1f}%")


def _selected_counts(selected: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = dict.fromkeys(SPLITS, 0)
    for doc in selected:
        counts[split_for(doc["doc_id"])] += 1
    return counts


def label(
    split: str = "train",
    *,
    limit: int = 0,
    resume: bool = True,
    model: str = TEACHER_MODEL,
    dry_run: bool = False,
    client: Any = None,
    docs: Sequence[Mapping[str, Any]] | None = None,
    paths: TeacherPaths | None = None,
    sleep: Callable[[float], None] | None = None,
    monotonic: Callable[[], float] | None = None,
    poll_seconds: float = TEACHER_POLL_SECONDS,
    max_wait_s: float = TEACHER_MAX_WAIT_S,
) -> dict[str, Any]:
    """Label `split` and materialize the output files. Returns the stats dict.

    `docs`, `client`, `paths`, `sleep` and `monotonic` are the injection seams,
    mirroring `corpus.build(rows=…, out_path=…)`: every test passes all of them, so
    no test touches the network or the real `data/`.
    """
    if split not in SPLIT_CHOICES:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLIT_CHOICES}")

    paths = _resolve(paths)
    sleep, monotonic = _clock(sleep, monotonic)
    stats = _new_stats(model=model, split=split, limit=limit)

    if docs is None:
        ensure_dirs()
        docs = list(read_jsonl(paths.docs))

    selected = select_docs(docs, split=split, limit=limit)
    stats["n_requested"] = len(selected)

    cache = load_cache(paths)
    have = cached_doc_ids(cache, model=model, prompt_sha_=TEACHER_PROMPT_SHA)
    todo = [d for d in selected if d["doc_id"] not in have]
    stats["n_cached"] = len(selected) - len(todo)
    stats["spend_usd_cached"] = round(
        sum(cost_of(cache[d["doc_id"]]["usage"], model) for d in selected if d["doc_id"] in have), 6
    )

    projected = _guard_spend(stats, todo, model=model)

    if dry_run:
        # Zero API calls, zero files written — not even the ledger.
        est = estimate_usage(todo)
        stats["projected_usd"] = round(projected, 4)
        stats["projected_tokens"] = {"input": est["input_tokens"], "output": est["output_tokens"]}
        stats["n_new_requests"] = len(todo)
        stats["n_batches"] = math.ceil(len(todo) / TEACHER_BATCH_SIZE)
        stats["split_counts"] = _selected_counts(selected)
        return stats

    ledger = load_ledger(paths)
    _check_ledger_consistency(ledger, model=model, prompt_sha_=TEACHER_PROMPT_SHA)
    ledger["model"], ledger["prompt_sha"] = model, TEACHER_PROMPT_SHA

    if client is None and (todo or any(not e.get("harvested") for e in ledger["batches"])):
        raise TeacherError("a client is required to request labels; pass client= or use --dry-run")

    if resume and client is not None and ledger["batches"]:
        _resume_inflight(
            client,
            model=model,
            ledger=ledger,
            paths=paths,
            stats=stats,
            sleep=sleep,
            monotonic=monotonic,
            poll_seconds=poll_seconds,
            max_wait_s=max_wait_s,
        )
        cache = load_cache(paths)
        have = cached_doc_ids(cache, model=model, prompt_sha_=TEACHER_PROMPT_SHA)
        todo = [d for d in selected if d["doc_id"] not in have]

    if todo:
        _run_wave(
            client,
            todo,
            tag_prefix="main",
            model=model,
            ledger=ledger,
            paths=paths,
            stats=stats,
            sleep=sleep,
            monotonic=monotonic,
            poll_seconds=poll_seconds,
            max_wait_s=max_wait_s,
        )
        cache = load_cache(paths)
        _retry_round(
            client,
            selected,
            cache,
            model=model,
            ledger=ledger,
            paths=paths,
            stats=stats,
            sleep=sleep,
            monotonic=monotonic,
            poll_seconds=poll_seconds,
            max_wait_s=max_wait_s,
        )
        cache = load_cache(paths)

    rows_by_split = build_rows(selected, cache, model=model, stats=stats)
    stats["split_counts"] = write_splits(rows_by_split, split=split, paths=paths)

    ok_rows = [r for rows in rows_by_split.values() for r in rows]
    stats["field_null_rate"] = field_null_rate(ok_rows)
    stats["enum_distribution"] = enum_distribution(ok_rows)
    stats["quality_warnings"] = quality_warnings(
        n_ok=stats["n_ok"],
        enum_dist=stats["enum_distribution"],
        null_rate=stats["field_null_rate"],
    )
    stats["spend_usd"] = round(stats["spend_usd"], 6)
    return stats
