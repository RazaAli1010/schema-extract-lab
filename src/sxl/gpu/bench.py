"""Latency, throughput and cost measurement on a single T4 (F7).

**Kaggle only for anything that touches a model**, but -- as in `runner.py` -- the
part that can be wrong in a way that matters is torch-free and unit-tested on the
laptop. That split is not cosmetic here; it is the whole design:

- the *arithmetic* (`cost_per_1k`, `percentile`, `summarize_latencies`) is what
  ends up on a résumé, so it is tested with hand-computed fixtures;
- the *timing loop* (`measure_single_stream`) takes an injected `step_fn` and an
  injected `sync`, so the ordering of synchronize/perf_counter -- the single
  easiest thing to get wrong -- is exercised without a GPU;
- only `measure_sweep` and `run` genuinely need torch, and they import it inside
  the function body.

**What this module is defending against.** The project pitch claims "45 ms on a
T4". SPEC §1.1 disowns that number in advance: a T4 runs a 1.7B model at roughly
40-70 tok/s single-stream, so ~150 tokens of JSON costs *seconds*. The easiest way
to accidentally "confirm" 45 ms is to omit `torch.cuda.synchronize()` and time
CUDA kernel *launches* instead of their execution, which is ~100x too fast. So:
every timed region is bracketed by a synchronize, and `check_plausible` raises on
any result faster than `BENCH_MAX_TOK_PER_S`. A benchmark that cannot fail is not
a benchmark.

**Single-stream and amortized numbers never share a field.** `<arm>.json` is
`batch_size: 1` and its `throughput_docs_per_s` is `1000 / p50_ms` -- what one
user waits for. `<arm>_sweep.json` holds the amortized-at-best-batch figures under
distinct `best_*` names. Conflating the two is the failure this module exists to
prevent (F7 §Context digest).
"""

from __future__ import annotations

import math
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from sxl.config import (
    BENCH_CACHE_IMPL,
    BENCH_MAX_TOK_PER_S,
    BENCH_N_DOCS,
    BENCH_REPEATS,
    BENCH_SPREAD_WARN_PCT,
    BENCH_WARMUP,
    T4_HOURLY_USD,
    git_sha,
    utc_now,
)
from sxl.gpu.runner import Messages
from sxl.schema import validate_prediction
from sxl.teacher import extract_json

#: SPEC §3.3 `results/bench/<arm>.json`, extended by F7's context deltas, in order.
#: A literal, like `runner.RECORD_KEYS`: the key order *is* the code, and
#: `build_bench_record` asserts against it rather than trusting a convention.
BENCH_KEYS = (
    "arm",
    "gpu_name",
    "dtype",
    "batch_size",
    "n_docs",
    "warmup",
    "repeats",
    "p50_ms",
    "p95_ms",
    "mean_ms",
    "min_ms",
    "max_ms",
    "stdev_ms",
    "mean_completion_tokens",
    "p50_spread_pct",
    "throughput_docs_per_s",
    "gpu_hourly_usd",
    "cost_per_1k_docs_usd",
    "cache_implementation",
    "index_build_s",
    "measurement",
    "torch_version",
    "transformers_version",
    "generated_at",
    "git_sha",
)

#: The subset SPEC §3.3 fixes verbatim. F7 may add keys; it may not drop these.
SPEC_BENCH_KEYS = (
    "arm",
    "gpu_name",
    "dtype",
    "batch_size",
    "n_docs",
    "warmup",
    "p50_ms",
    "p95_ms",
    "mean_ms",
    "throughput_docs_per_s",
    "gpu_hourly_usd",
    "cost_per_1k_docs_usd",
    "generated_at",
)

#: `results/bench/<arm>_sweep.json` (F7 context deltas).
SWEEP_KEYS = (
    "arm",
    "gpu_name",
    "dtype",
    "n_docs",
    "warmup",
    "cache_implementation",
    "sweep",
    "best_batch_size",
    "best_throughput_docs_per_s",
    "best_amortized_ms_per_doc",
    "best_cost_per_1k_docs_usd",
    "gpu_hourly_usd",
    "note",
    "generated_at",
    "git_sha",
)

#: One row of the sweep. OOM rows carry the same keys with zeroed measurements, so
#: the table stays rectangular and a reader can see *which* batch sizes failed.
SWEEP_ENTRY_KEYS = (
    "batch_size",
    "throughput_docs_per_s",
    "amortized_ms_per_doc",
    "peak_vram_gb",
    "mean_completion_tokens",
    "oom",
)

#: Times one document end-to-end. Returns at least `raw_output` and
#: `completion_tokens` -- i.e. exactly what `runner.generate_batch` already emits
#: per item, which is why both arms' step functions are three lines.
StepFn = Callable[[Messages], dict[str, Any]]


class BenchError(RuntimeError):
    """A benchmark contract violation. Fatal, never warned (SPEC §5.5)."""


# --- torch-free core: the tested part ----------------------------------------


def percentile(sorted_values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile of an already-sorted sequence.

    Same convention as `train_lora._percentile` and `metrics.py`, deliberately:
    two percentiles in one project computed two ways is a footgun, and the numbers
    here sit beside those in F8's table. No numpy (SPEC §6.1).
    """
    if not sorted_values:
        return 0.0
    idx = min(len(sorted_values) - 1, max(0, math.ceil(q * len(sorted_values)) - 1))
    return float(sorted_values[idx])


def cost_per_1k(throughput_docs_per_s: float, hourly_usd: float) -> float:
    """USD to process 1,000 documents at a given throughput and rental rate.

    `T4_HOURLY_USD = 0.35` is an **assumption** (~GCP on-demand), not a
    measurement -- which is why `hourly_usd` is an argument and is echoed into
    every artifact beside the result. A reader on AWS `g4dn.xlarge` (~$0.53/h)
    rescales with one multiplication.

    Raises rather than returning `inf` on a zero or negative throughput: `inf`
    would serialize to JSON as `Infinity`, which is not valid JSON, and would
    reach F8's table as a silent placeholder instead of a crash.
    """
    if throughput_docs_per_s <= 0 or not math.isfinite(throughput_docs_per_s):
        raise BenchError(
            f"throughput_docs_per_s must be a positive finite number, got "
            f"{throughput_docs_per_s!r}. A zero throughput means the measurement "
            "failed; it does not mean the cost is infinite."
        )
    if hourly_usd < 0 or not math.isfinite(hourly_usd):
        raise BenchError(f"hourly_usd must be a non-negative finite number, got {hourly_usd!r}")
    return (1000.0 / throughput_docs_per_s) / 3600.0 * hourly_usd


def check_plausible(
    p50_ms: float,
    mean_completion_tokens: float,
    max_tok_per_s: float = BENCH_MAX_TOK_PER_S,
) -> None:
    """Raise if the measured latency is too fast to be real on a T4.

    Batch-1 latency is approximately `completion_tokens / tokens_per_second`. At a
    generous 500 tok/s ceiling, 150 tokens cannot come back in under 300 ms on
    Turing. **A measured p50 of 45 ms for a 150-token generation is not a triumph,
    it is a bug** -- almost certainly a missing `torch.cuda.synchronize()`, or a
    run that hit EOS after two tokens (F7 §Implementation notes).

    This guard is the reason the module can be trusted at all, so it raises rather
    than warning, and it runs on the real numbers before anything is written.
    """
    if mean_completion_tokens <= 0:
        raise BenchError(
            "mean_completion_tokens is 0: the model generated nothing. A latency "
            "measured over empty generations describes the tokenizer, not the model."
        )
    floor_ms = mean_completion_tokens * (1000.0 / max_tok_per_s)
    if p50_ms <= floor_ms:
        raise BenchError(
            f"implausible latency: p50={p50_ms:.1f} ms for {mean_completion_tokens:.1f} "
            f"completion tokens implies >{max_tok_per_s:.0f} tok/s, which a T4 (sm_75) "
            f"cannot do. Expected at least {floor_ms:.1f} ms. This is almost always a "
            "missing torch.cuda.synchronize() around the timed region -- you are timing "
            "kernel launches, not kernel execution."
        )


def summarize_latencies(
    latencies_ms: Sequence[float],
    completion_tokens: Sequence[int],
) -> dict[str, Any]:
    """Percentiles and spread for one single-stream pass.

    `mean_completion_tokens` is reported alongside because **latency here is almost
    entirely a function of output length**. An arm that emits 400 tokens is slow
    for a reason the reader deserves to see, and F8 must attribute a fast `lora_ft`
    to tighter JSON rather than to the adapter being intrinsically quicker -- it is
    not; an unmerged LoRA adds compute per token (F7 §Implementation notes).
    """
    if not latencies_ms:
        raise BenchError("no latencies recorded: the measurement loop never ran")
    if len(latencies_ms) != len(completion_tokens):
        raise BenchError(f"{len(latencies_ms)} latencies but {len(completion_tokens)} token counts")
    ordered = sorted(float(x) for x in latencies_ms)
    return {
        "n": len(ordered),
        "p50_ms": round(percentile(ordered, 0.50), 1),
        "p95_ms": round(percentile(ordered, 0.95), 1),
        "mean_ms": round(statistics.fmean(ordered), 1),
        "min_ms": round(ordered[0], 1),
        "max_ms": round(ordered[-1], 1),
        # `stdev` needs two points; one document is a legitimate smoke run.
        "stdev_ms": round(statistics.stdev(ordered), 1) if len(ordered) > 1 else 0.0,
        "mean_completion_tokens": round(statistics.fmean(completion_tokens), 1),
    }


def spread_pct(values: Sequence[float]) -> float:
    """`(max - min) / median * 100` across repeats.

    Kaggle GPUs are shared infrastructure and thermally variable. Reporting the
    spread across `BENCH_REPEATS` is the minimum honest treatment; quoting the best
    of three is not (F7 §Implementation notes).
    """
    if not values:
        return 0.0
    mid = statistics.median(values)
    if mid <= 0:
        return 0.0
    return round((max(values) - min(values)) / mid * 100.0, 2)


def best_batch_size(sweep: Sequence[Mapping[str, Any]]) -> int:
    """The largest batch size that did not OOM.

    OOM entries stay in the file -- knowing *where* a 16 GB card runs out is a
    result, not an error -- but they are excluded from selection.
    """
    ok = [int(e["batch_size"]) for e in sweep if not e.get("oom", False)]
    if not ok:
        raise BenchError("every batch size OOMed; there is no best batch size to report")
    return max(ok)


def build_bench_record(
    *,
    arm: str,
    gpu_name: str | None,
    dtype: str | None,
    stats: Mapping[str, Any],
    p50_spread_pct: float,
    n_docs: int,
    warmup: int,
    repeats: int,
    hourly_usd: float,
    cache_implementation: str,
    index_build_s: float,
    measurement: str,
    torch_version: str | None,
    transformers_version: str | None,
) -> dict[str, Any]:
    """Assemble one `results/bench/<arm>.json`, deriving every number it reports.

    `throughput_docs_per_s` here is the **single-stream** figure, `1000 / p50_ms`:
    one document at a time, which is what a latency-sensitive caller actually
    waits for. The amortized-at-best-batch throughput is a different quantity and
    lives in `<arm>_sweep.json` under `best_*` names (SPEC §1.1).

    `cost_per_1k_docs_usd` is computed by calling `cost_per_1k` with the very
    `gpu_hourly_usd` this record writes down -- never two independent literals, so
    the two cannot drift and a test can assert their consistency.
    """
    p50_ms = float(stats["p50_ms"])
    if p50_ms <= 0:
        raise BenchError(f"p50_ms must be positive, got {p50_ms!r}")
    check_plausible(p50_ms, float(stats["mean_completion_tokens"]))

    # Rounded *before* the cost is derived, not after: the cost must be
    # reproducible from the numbers this file actually prints. Deriving it from a
    # higher-precision throughput than the one written down would leave a reader
    # who recomputes it with a different answer than the file claims.
    throughput = round(1000.0 / p50_ms, 4)
    record = {
        "arm": arm,
        "gpu_name": gpu_name,
        "dtype": dtype,
        "batch_size": 1,
        "n_docs": int(n_docs),
        "warmup": int(warmup),
        "repeats": int(repeats),
        "p50_ms": round(p50_ms, 1),
        "p95_ms": round(float(stats["p95_ms"]), 1),
        "mean_ms": round(float(stats["mean_ms"]), 1),
        "min_ms": round(float(stats["min_ms"]), 1),
        "max_ms": round(float(stats["max_ms"]), 1),
        "stdev_ms": round(float(stats["stdev_ms"]), 1),
        "mean_completion_tokens": round(float(stats["mean_completion_tokens"]), 1),
        "p50_spread_pct": round(float(p50_spread_pct), 2),
        "throughput_docs_per_s": throughput,
        "gpu_hourly_usd": float(hourly_usd),
        "cost_per_1k_docs_usd": round(cost_per_1k(throughput, hourly_usd), 6),
        "cache_implementation": cache_implementation,
        "index_build_s": round(float(index_build_s), 3),
        "measurement": measurement,
        "torch_version": torch_version,
        "transformers_version": transformers_version,
        "generated_at": utc_now(),
        "git_sha": git_sha(),
    }
    if tuple(record) != BENCH_KEYS:  # pragma: no cover - structural guard
        raise BenchError(f"bench key order drifted from the SPEC §3.3 contract: {tuple(record)}")
    return record


def build_sweep_record(
    *,
    arm: str,
    gpu_name: str | None,
    dtype: str | None,
    sweep: Sequence[Mapping[str, Any]],
    n_docs: int,
    warmup: int,
    hourly_usd: float,
    cache_implementation: str,
    note: str = "",
) -> dict[str, Any]:
    """Assemble one `results/bench/<arm>_sweep.json`.

    The `best_*` fields are the amortized figures: what a throughput-oriented
    batch deployment pays per document. They are named apart from the
    single-stream fields on purpose and F8 must not merge the two columns.
    """
    for entry in sweep:
        if tuple(entry) != SWEEP_ENTRY_KEYS:  # pragma: no cover - structural guard
            raise BenchError(f"sweep entry keys drifted: {tuple(entry)}")

    best = best_batch_size(sweep)
    best_entry = next(e for e in sweep if int(e["batch_size"]) == best)
    # As in `build_bench_record`: round first, then derive, so the file's own cost
    # figure is reproducible from the file's own throughput figure.
    best_throughput = round(float(best_entry["throughput_docs_per_s"]), 4)

    record = {
        "arm": arm,
        "gpu_name": gpu_name,
        "dtype": dtype,
        "n_docs": int(n_docs),
        "warmup": int(warmup),
        "cache_implementation": cache_implementation,
        "sweep": [dict(e) for e in sweep],
        "best_batch_size": best,
        "best_throughput_docs_per_s": best_throughput,
        "best_amortized_ms_per_doc": round(1000.0 / best_throughput, 2),
        "best_cost_per_1k_docs_usd": round(cost_per_1k(best_throughput, hourly_usd), 6),
        "gpu_hourly_usd": float(hourly_usd),
        "note": note,
        "generated_at": utc_now(),
        "git_sha": git_sha(),
    }
    if tuple(record) != SWEEP_KEYS:  # pragma: no cover - structural guard
        raise BenchError(f"sweep key order drifted: {tuple(record)}")
    return record


def measure_single_stream(
    step_fn: StepFn,
    prompts: Sequence[Messages],
    *,
    warmup: int = BENCH_WARMUP,
    n_docs: int = BENCH_N_DOCS,
    sync: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Time `n_docs` documents one at a time, end to end. The honest headline number.

    **The synchronize placement is the whole ballgame.** CUDA kernel launches are
    asynchronous, so `sync()` is called immediately before `t0` (to drain work
    queued by the previous iteration, which would otherwise be billed to this one)
    and again *inside* the timed region before `t1` (to wait for this iteration's
    kernels to actually finish). Without the second one you measure how fast Python
    can enqueue work -- roughly 100x too fast, and in exactly the direction that
    would appear to validate the 45 ms hypothesis SPEC §1.1 rejects.

    Parsing is deliberately **inside** the timed region: a user of this model pays
    for `extract_json` and `validate_prediction`, so the benchmark does too (F7
    §Scope 1). Both are cheap, but "cheap" is a measurement, not an assumption.

    `warmup` iterations run untimed first. On the constrained arms this is also
    where Outlines compiles its grammar index -- a real one-time deployment cost,
    reported separately as `index_build_s` rather than smeared across every
    document (F7 §Scope 5).

    `step_fn` and `sync` are injected so this loop is testable on a laptop with no
    torch: the ordering above is the part that can be silently wrong.
    """
    if n_docs <= 0:
        raise BenchError(f"n_docs must be positive, got {n_docs}")
    if len(prompts) < n_docs:
        raise BenchError(
            f"need {n_docs} prompts to measure but only {len(prompts)} were supplied. "
            "Pass --n-docs no larger than the eval set, so every arm measures the same "
            "documents."
        )

    for messages in prompts[: max(warmup, 0)]:
        step_fn(messages)

    latencies_ms: list[float] = []
    completion_tokens: list[int] = []
    for messages in prompts[:n_docs]:
        if sync is not None:
            sync()
        started = time.perf_counter()

        generated = step_fn(messages)
        # Counted, not stored: F7 writes no accuracy number anywhere in
        # results/bench/ (F4 owns schema_valid_rate). This call is here purely
        # because it costs time that a real deployment also pays.
        validate_prediction(extract_json(generated["raw_output"]))

        if sync is not None:
            sync()
        latencies_ms.append((time.perf_counter() - started) * 1000.0)
        completion_tokens.append(int(generated["completion_tokens"]))

    return summarize_latencies(latencies_ms, completion_tokens)


# --- torch-only from here ----------------------------------------------------


def _cuda_sync() -> Callable[[], None]:
    """A no-argument `torch.cuda.synchronize`, or a no-op off-GPU."""
    import torch

    if not torch.cuda.is_available():  # pragma: no cover - GPU-only path
        return lambda: None
    return torch.cuda.synchronize


def assert_t4() -> str:
    """Confirm this session is on a T4 and return the device name.

    Kaggle's `GPU T4 x2` and `GPU P100` are different hardware. A session that
    silently landed on a P100 would produce numbers that are not comparable across
    arms -- and nothing in the output would say so, which is worse than crashing
    (F7 §Kaggle setup).
    """
    import torch

    if not torch.cuda.is_available():
        raise BenchError(
            "no CUDA device. F7 measures a GPU; there is nothing meaningful to "
            "benchmark on the laptop (SPEC §2.1)."
        )
    capability = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0)
    if capability != (7, 5):
        raise BenchError(
            f"expected a Turing T4 (compute capability 7.5) but found {name} with "
            f"capability {capability}. Kaggle's `GPU P100` is different hardware and its "
            "numbers are not comparable with the other arms -- switch the accelerator to "
            "`GPU T4 x2` and re-run, or every arm must be re-measured here."
        )
    return name


def make_step_fn(model, tok, *, max_new_tokens: int) -> StepFn:
    """One-document generation through **F5's exact batched path**.

    A batch of one, via `runner.generate_batch`: left padding is a no-op at size 1,
    and reusing it means F7 benchmarks literally the code that produced
    `artifacts/predictions/*.jsonl`. Writing a second generation path here -- even
    an identical-looking one -- would make the latency table and the accuracy table
    describe different programs (F7 §Reuse from F5).

    `generate_batch` runs its own synchronize after `generate`; that is harmless
    inside our timed region (the second sync is then free) and it is not load-
    bearing for us, because `measure_single_stream` brackets the whole call.
    """
    from sxl.gpu.runner import generate_batch, render_prompt

    def _step(messages: Messages) -> dict[str, Any]:
        prompt = render_prompt(tok, messages)
        return generate_batch(model, tok, [prompt], max_new_tokens=max_new_tokens)[0]

    return _step


def make_constrained_step_fn(generate_fn) -> StepFn:
    """Wrap `constrained.build_constrained_generator`'s `GenerateFn` as a `StepFn`.

    Outlines' logit masking has real per-token overhead, which is precisely what
    this arm is here to measure (F7 §Scope 5).
    """

    def _step(messages: Messages) -> dict[str, Any]:
        return generate_fn([messages])[0]

    return _step


def _oom_errors() -> tuple[type[BaseException], ...]:
    """The exceptions a too-large batch raises. Built defensively across versions."""
    import torch

    errors: list[type[BaseException]] = [RuntimeError]
    oom = getattr(torch.cuda, "OutOfMemoryError", None)
    if isinstance(oom, type) and issubclass(oom, BaseException):
        errors.insert(0, oom)
    return tuple(errors)


def measure_sweep(
    model,
    tok,
    prompts: Sequence[Messages],
    *,
    batch_sizes: Sequence[int],
    n_docs: int,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    """Throughput and peak VRAM across a batch-size sweep.

    Uses `runner.generate_batch` -- the same batched path F5/F6 ran predictions
    with -- so the amortized numbers describe a deployment that actually exists.

    An OOM is **recorded, not raised**: the point of sweeping to 32 on a 16 GB card
    is to find the wall, and crashing there would throw away every smaller batch
    size that already succeeded, along with the GPU minutes they cost (F7 §Scope 2).
    """
    import torch

    from sxl.gpu.runner import generate_batch, render_prompt

    sync = _cuda_sync()
    oom_errors = _oom_errors()
    window = list(prompts[:n_docs])
    rendered = [render_prompt(tok, m) for m in window]

    results: list[dict[str, Any]] = []
    for batch_size in batch_sizes:
        torch.cuda.reset_peak_memory_stats()
        entry: dict[str, Any] = {
            "batch_size": int(batch_size),
            "throughput_docs_per_s": 0.0,
            "amortized_ms_per_doc": 0.0,
            "peak_vram_gb": 0.0,
            "mean_completion_tokens": 0.0,
            "oom": False,
        }
        try:
            # Warm up at **exactly this width**, not at `warmup` documents: a new
            # batch shape re-plans kernels and re-grows the KV cache, so warming at
            # width 10 would leave that cost inside the timed region for width 32.
            # It also means an over-large batch OOMs here rather than mid-measurement.
            generate_batch(
                model,
                tok,
                rendered[: min(batch_size, len(rendered))],
                max_new_tokens=max_new_tokens,
            )

            tokens: list[int] = []
            sync()
            started = time.perf_counter()
            for start in range(0, len(rendered), batch_size):
                chunk = rendered[start : start + batch_size]
                for item in generate_batch(model, tok, chunk, max_new_tokens=max_new_tokens):
                    tokens.append(int(item["completion_tokens"]))
            sync()
            elapsed_s = time.perf_counter() - started

            throughput = len(window) / elapsed_s if elapsed_s > 0 else 0.0
            entry["throughput_docs_per_s"] = round(throughput, 4)
            entry["amortized_ms_per_doc"] = round(1000.0 / throughput, 2) if throughput else 0.0
            entry["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
            entry["mean_completion_tokens"] = round(statistics.fmean(tokens), 1) if tokens else 0.0
        except oom_errors as exc:
            if not _is_oom(exc):
                raise
            entry["oom"] = True
            entry["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
            torch.cuda.empty_cache()

        results.append(entry)
    return results


def _is_oom(exc: BaseException) -> bool:
    """`RuntimeError` is in the catch tuple for old torch; only OOM should be swallowed."""
    import torch

    oom = getattr(torch.cuda, "OutOfMemoryError", None)
    if isinstance(oom, type) and isinstance(exc, oom):
        return True
    return "out of memory" in str(exc).lower()


def run(
    arm: str,
    prompts: Sequence[Messages],
    *,
    model_id: str,
    adapter: str | None = None,
    constrained: bool = False,
    n_docs: int = BENCH_N_DOCS,
    warmup: int = BENCH_WARMUP,
    repeats: int = BENCH_REPEATS,
    batch_sizes: Sequence[int] = (),
    max_new_tokens: int,
    hourly_usd: float = T4_HOURLY_USD,
    cache_implementation: str = BENCH_CACHE_IMPL,
    echo: Callable[[str], None] = print,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Benchmark one arm and return `(bench_record, sweep_record)`.

    `prompts` are already-built chat messages. The caller builds them with the
    same prompt builder that produced this arm's predictions in F5/F6 -- the CLI
    does it with one shared code path rather than a second copy of the arm
    dispatch, because benchmarking a different prompt than the accuracy run used
    would make the two tables incomparable (F7 §Reuse from F5).
    """
    import torch
    import transformers

    from sxl.gpu.runner import load_model

    gpu_name = assert_t4()
    echo(f"[bench] arm={arm} gpu={gpu_name} torch={torch.__version__} n_docs={n_docs}")

    model, tok = load_model(model_id, adapter=adapter)

    index_build_s = 0.0
    if constrained:
        from sxl.gpu.constrained import build_constrained_generator

        build_seconds: list[float] = []
        generate_fn = build_constrained_generator(
            model,
            tok,
            max_new_tokens=max_new_tokens,
            on_index_built=build_seconds.append,
        )
        step_fn = make_constrained_step_fn(generate_fn)
        index_build_s = _measure_index_build(step_fn, prompts, warmup, build_seconds)
        echo(f"[bench] outlines index build: {index_build_s:.2f}s (excluded from per-doc latency)")
    else:
        step_fn = make_step_fn(model, tok, max_new_tokens=max_new_tokens)

    sync = _cuda_sync()
    passes: list[dict[str, Any]] = []
    for i in range(max(repeats, 1)):
        # Only the first pass pays for warmup; the GPU is hot after that, and
        # re-warming would spend Kaggle quota to re-measure a settled number.
        stats = measure_single_stream(
            step_fn,
            prompts,
            warmup=warmup if i == 0 and not constrained else 0,
            n_docs=n_docs,
            sync=sync,
        )
        passes.append(stats)
        echo(
            f"[bench] pass {i + 1}/{repeats}: p50={stats['p50_ms']:.1f}ms "
            f"p95={stats['p95_ms']:.1f}ms tok={stats['mean_completion_tokens']:.1f}"
        )

    # Median of the repeats' p50s, not the best of them (F7 §Scope 4).
    p50s = [float(p["p50_ms"]) for p in passes]
    headline = min(passes, key=lambda p: abs(float(p["p50_ms"]) - statistics.median(p50s)))
    p50_spread = spread_pct(p50s)
    if p50_spread > BENCH_SPREAD_WARN_PCT:
        echo(
            f"[bench] WARNING: p50 spread across repeats is {p50_spread:.1f}% "
            f"(>{BENCH_SPREAD_WARN_PCT:.0f}%). Kaggle GPUs are shared; F8 must report this "
            "rather than quoting the fastest pass."
        )

    record = build_bench_record(
        arm=arm,
        gpu_name=gpu_name,
        dtype="float16",
        stats=headline,
        p50_spread_pct=p50_spread,
        n_docs=n_docs,
        warmup=warmup,
        repeats=max(repeats, 1),
        hourly_usd=hourly_usd,
        cache_implementation=cache_implementation,
        index_build_s=index_build_s,
        measurement="local_gpu",
        torch_version=torch.__version__,
        transformers_version=transformers.__version__,
    )

    if constrained:
        # Outlines generates strictly one document per call
        # (`constrained.build_constrained_generator`), so batch size is inert here
        # by construction. Recording that fact beats manufacturing a flat curve
        # and spending GPU quota to do it.
        sweep = [
            {
                "batch_size": 1,
                "throughput_docs_per_s": record["throughput_docs_per_s"],
                "amortized_ms_per_doc": record["p50_ms"],
                "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
                "mean_completion_tokens": record["mean_completion_tokens"],
                "oom": False,
            }
        ]
        note = (
            "No batch sweep: the Outlines path generates one document per call, so batch "
            "size has no effect. Figures mirror the single-stream measurement."
        )
    else:
        sweep = measure_sweep(
            model,
            tok,
            prompts,
            batch_sizes=batch_sizes,
            n_docs=n_docs,
            max_new_tokens=max_new_tokens,
        )
        note = ""

    sweep_record = build_sweep_record(
        arm=arm,
        gpu_name=gpu_name,
        dtype="float16",
        sweep=sweep,
        n_docs=n_docs,
        warmup=warmup,
        hourly_usd=hourly_usd,
        cache_implementation=cache_implementation,
        note=note,
    )
    return record, sweep_record


def _measure_index_build(
    step_fn: StepFn,
    prompts: Sequence[Messages],
    warmup: int,
    build_seconds: Sequence[float],
) -> float:
    """One-time Outlines cost: generator construction plus grammar compilation.

    Outlines compiles a schema to its index lazily, on the **first** generation
    that uses it, and caches it thereafter. So the compile shows up as the excess
    of call #1 over the steady state, and this runs the warmup calls explicitly to
    measure exactly that: `construction + max(0, first_call - median(rest))`.

    It is reported as `index_build_s` and excluded from per-document latency,
    because it is paid once at process start. Hiding it entirely would be
    dishonest in the opposite direction -- it is a real deployment cost (F7 §Scope 5).
    """
    n_calls = max(warmup, 2)
    if len(prompts) < n_calls:
        n_calls = max(len(prompts), 2)

    sync = _cuda_sync()
    timings: list[float] = []
    for messages in prompts[:n_calls]:
        sync()
        started = time.perf_counter()
        step_fn(messages)
        sync()
        timings.append(time.perf_counter() - started)

    construction_s = float(sum(build_seconds))
    if len(timings) < 2:  # pragma: no cover - defensive
        return round(construction_s, 3)
    excess = timings[0] - statistics.median(timings[1:])
    return round(construction_s + max(0.0, excess), 3)
