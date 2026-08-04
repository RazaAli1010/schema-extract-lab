"""Wall-clock latency and standard-rate cost for the `teacher` arm (F7 §Scope 6).

**Laptop, not Kaggle.** This is a network measurement; it needs no GPU and would
learn nothing from having one.

Two things here are measured differently from every other arm, and the output file
says so rather than leaving a reader to assume otherwise:

1. **`measurement: "api_wall_clock"`.** Comparing an HTTP round-trip to a local
   `generate()` is apples-to-oranges. The teacher's number includes queueing,
   TLS, and the width of the Atlantic; the GPU arms' numbers do not. F8 must
   present them as different quantities.
2. **Standard, not Batch, pricing.** F2 labels the corpus through the Batch API at
   a 50% discount, which is correct for a bulk offline job and wrong here: a
   latency-sensitive deployment cannot wait 24 hours. `teacher.cost_of` bakes in
   `TEACHER_BATCH_DISCOUNT`, so this module prices the calls itself at full rate.

Calls are issued **sequentially, one at a time**, matching the single-stream
definition used by the GPU arms. Concurrency would measure the provider's
parallelism, not the per-document latency a user experiences.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from sxl.config import (
    BENCH_TEACHER_N,
    TEACHER_MODEL,
    TEACHER_PRICE_USD,
    git_sha,
    utc_now,
)
from sxl.gpu.bench import BENCH_KEYS, BenchError, summarize_latencies

#: `results/bench/teacher.json`. The SPEC §3.3 shape, so F8 reads one record type,
#: plus the token-accounting fields that make the cost figure reproducible. The GPU
#: fields are present and `null` -- an absent key would look like an oversight,
#: whereas an explicit `null` says "this arm has no GPU".
TEACHER_BENCH_KEYS = BENCH_KEYS + (
    "teacher_model",
    "mean_prompt_tokens",
    "n_calls_ok",
    "n_calls_failed",
    "total_usd",
    "pricing",
    "note",
)

_NOTE = (
    "Wall-clock latency of sequential, non-batch API calls made from the dev laptop: "
    "it includes network round-trip and provider queueing, and is not comparable with "
    "the GPU arms' local generate() timings. Cost is derived from measured token usage "
    "at STANDARD rates, not the 50% Batch API rates F2 used for labeling, because a "
    "latency-sensitive deployment cannot use a 24-hour batch queue."
)


def standard_cost(usage: Mapping[str, int], model: str = TEACHER_MODEL) -> float:
    """USD for one response at **standard** (undiscounted) rates.

    Deliberately not `teacher.cost_of`, which multiplies by
    `TEACHER_BATCH_DISCOUNT`. Reusing it here would understate the cost of the
    thing this file is actually measuring by exactly a factor of two.
    """
    price = TEACHER_PRICE_USD[model]
    return usage["input_tokens"] / 1e6 * price["in"] + usage["output_tokens"] / 1e6 * price["out"]


def call_once(client, body: Mapping[str, Any]) -> dict[str, Any]:
    """One synchronous completion. Returns latency and token usage.

    `body` comes from `teacher.build_request(...)["body"]` unchanged, so this probe
    times the same model, prompt, temperature and strict-JSON response format that
    produced `results/metrics/teacher.json`.
    """
    started = time.perf_counter()
    response = client.chat.completions.create(**dict(body))
    latency_ms = (time.perf_counter() - started) * 1000.0

    usage = getattr(response, "usage", None)
    return {
        "latency_ms": latency_ms,
        "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
    }


def run(
    docs: Sequence[Mapping[str, Any]],
    *,
    n: int = BENCH_TEACHER_N,
    model: str = TEACHER_MODEL,
    client: Any = None,
    echo: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Time `n` sequential teacher calls and build `results/bench/teacher.json`."""
    from sxl.teacher import build_request, make_client

    if model not in TEACHER_PRICE_USD:
        raise BenchError(
            f"no standard pricing for {model!r}. Add it to config.TEACHER_PRICE_USD -- a "
            "cost figure derived from a guessed rate is worse than no cost figure."
        )
    window = list(docs)[:n]
    if len(window) < n:
        raise BenchError(f"need {n} documents but only {len(window)} were supplied")

    client = client if client is not None else make_client()

    latencies: list[float] = []
    completion_tokens: list[int] = []
    prompt_tokens: list[int] = []
    total_usd = 0.0
    n_failed = 0

    for i, doc in enumerate(window, 1):
        body = build_request(doc, model)["body"]
        try:
            result = call_once(client, body)
        except Exception as exc:  # noqa: BLE001 - one flaky call must not lose the run
            n_failed += 1
            echo(f"[bench-teacher] call {i}/{len(window)} failed: {type(exc).__name__}: {exc}")
            continue

        latencies.append(result["latency_ms"])
        completion_tokens.append(result["output_tokens"])
        prompt_tokens.append(result["input_tokens"])
        total_usd += standard_cost(
            {"input_tokens": result["input_tokens"], "output_tokens": result["output_tokens"]},
            model,
        )
        if i % 10 == 0:
            echo(f"[bench-teacher] {i}/{len(window)} calls, {result['latency_ms']:.0f} ms last")

    if not latencies:
        raise BenchError(f"all {len(window)} teacher calls failed; nothing to report")
    if n_failed:
        echo(f"[bench-teacher] WARNING: {n_failed}/{len(window)} calls failed and were skipped")

    stats = summarize_latencies(latencies, completion_tokens)
    n_ok = len(latencies)
    # Cost per document is a token quantity here, not a rented-seconds quantity --
    # which is exactly why it cannot be derived by `cost_per_1k`, and why
    # `gpu_hourly_usd` below is null rather than 0.35.
    cost_per_1k_docs = total_usd / n_ok * 1000.0

    record = {
        "arm": "teacher",
        "gpu_name": None,
        "dtype": None,
        "batch_size": 1,
        "n_docs": n_ok,
        "warmup": 0,
        "repeats": 1,
        "p50_ms": stats["p50_ms"],
        "p95_ms": stats["p95_ms"],
        "mean_ms": stats["mean_ms"],
        "min_ms": stats["min_ms"],
        "max_ms": stats["max_ms"],
        "stdev_ms": stats["stdev_ms"],
        "mean_completion_tokens": stats["mean_completion_tokens"],
        "p50_spread_pct": 0.0,  # one pass; no repeats to spread across
        "throughput_docs_per_s": round(1000.0 / stats["p50_ms"], 4) if stats["p50_ms"] else 0.0,
        "gpu_hourly_usd": None,
        "cost_per_1k_docs_usd": round(cost_per_1k_docs, 6),
        "cache_implementation": None,
        "index_build_s": 0.0,
        "measurement": "api_wall_clock",
        "torch_version": None,
        "transformers_version": None,
        "generated_at": utc_now(),
        "git_sha": git_sha(),
        "teacher_model": model,
        "mean_prompt_tokens": round(sum(prompt_tokens) / n_ok, 1),
        "n_calls_ok": n_ok,
        "n_calls_failed": n_failed,
        "total_usd": round(total_usd, 6),
        "pricing": {"rates_usd_per_1m": dict(TEACHER_PRICE_USD[model]), "batch_discount": False},
        "note": _NOTE,
    }
    if tuple(record) != TEACHER_BENCH_KEYS:  # pragma: no cover - structural guard
        raise BenchError(f"teacher bench key order drifted: {tuple(record)}")
    return record
