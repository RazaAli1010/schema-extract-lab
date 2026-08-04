"""Percentiles, the implausible-speed guard, and the synchronize/clock ordering.

The ordering test is the one that matters most. CUDA kernel launches are
asynchronous, so a timed region that does not `synchronize()` before *both*
`perf_counter()` calls measures how fast Python enqueues work -- roughly 100x too
fast, and in precisely the direction that would appear to confirm the 45 ms/doc
figure SPEC §1.1 disowns. `measure_single_stream` takes an injected `sync` so that
ordering is provable on a laptop with no GPU (F7 §Implementation notes).
"""

from __future__ import annotations

import pytest

from sxl.gpu.bench import (
    BenchError,
    check_plausible,
    measure_single_stream,
    percentile,
    spread_pct,
    summarize_latencies,
)


def messages(i: int) -> list[dict[str, str]]:
    return [{"role": "user", "content": f"doc {i}"}]


def fake_step(tokens: int = 150, raw: str = "{}"):
    """A `StepFn` that records the prompts it saw. No torch, no GPU."""

    def _step(msgs):
        _step.seen.append(msgs)
        return {"raw_output": raw, "completion_tokens": tokens}

    _step.seen = []
    return _step


# --- percentiles --------------------------------------------------------------


def test_percentiles_on_one_to_two_hundred_are_the_hand_computed_values():
    # Nearest rank: p50 -> ceil(0.50 * 200) - 1 = index 99 -> the 100th value.
    #               p95 -> ceil(0.95 * 200) - 1 = index 189 -> the 190th value.
    values = [float(x) for x in range(1, 201)]
    assert percentile(values, 0.50) == 100.0
    assert percentile(values, 0.95) == 190.0


def test_percentile_of_a_single_value_is_that_value():
    assert percentile([42.0], 0.50) == 42.0
    assert percentile([42.0], 0.95) == 42.0


def test_percentile_of_an_empty_sequence_is_zero_rather_than_an_index_error():
    assert percentile([], 0.5) == 0.0


def test_percentile_matches_the_convention_train_lora_already_uses():
    # Two percentiles in one project computed two ways is a footgun: these numbers
    # sit beside `train_stats.json`'s token_p95 in F8's write-up.
    from sxl.gpu.train_lora import _percentile

    values = list(range(1, 201))
    assert percentile([float(v) for v in values], 0.95) == float(_percentile(values, 0.95))


# --- summarize_latencies ------------------------------------------------------


def test_summarize_reports_every_statistic_f7_requires():
    stats = summarize_latencies([float(x) for x in range(1, 201)], [150] * 200)
    assert stats["p50_ms"] == 100.0
    assert stats["p95_ms"] == 190.0
    assert stats["mean_ms"] == 100.5
    assert stats["min_ms"] == 1.0
    assert stats["max_ms"] == 200.0
    assert stats["stdev_ms"] > 0
    assert stats["mean_completion_tokens"] == 150.0
    assert stats["n"] == 200


def test_summarize_sorts_before_taking_percentiles():
    # Measurements arrive in run order, not sorted order.
    forward = summarize_latencies([1.0, 2.0, 3.0, 4.0], [10] * 4)
    shuffled = summarize_latencies([3.0, 1.0, 4.0, 2.0], [10] * 4)
    assert forward == shuffled


def test_a_single_measurement_has_zero_stdev_rather_than_raising():
    # One document is a legitimate smoke run; `statistics.stdev` needs two points.
    assert summarize_latencies([12.0], [10])["stdev_ms"] == 0.0


def test_mismatched_latency_and_token_counts_raise():
    with pytest.raises(BenchError):
        summarize_latencies([1.0, 2.0], [10])


def test_an_empty_measurement_raises_rather_than_reporting_zeros():
    with pytest.raises(BenchError):
        summarize_latencies([], [])


# --- the implausible-speed guard ---------------------------------------------


def test_the_guard_rejects_forty_five_milliseconds_for_a_hundred_and_fifty_tokens():
    # The headline claim F7 exists to test honestly. 150 tokens at 45 ms is
    # 3,333 tok/s, which Turing cannot do -- it is a missing synchronize.
    with pytest.raises(BenchError, match="implausible"):
        check_plausible(45.0, 150.0)


def test_the_guard_accepts_three_thousand_milliseconds_for_a_hundred_and_fifty_tokens():
    # 50 tok/s: the range SPEC §1.1 predicts for a 1.7B fp16 model on a T4.
    check_plausible(3000.0, 150.0)


def test_the_guard_boundary_is_exactly_five_hundred_tokens_per_second():
    check_plausible(300.1, 150.0)  # just under 500 tok/s
    with pytest.raises(BenchError):
        check_plausible(300.0, 150.0)  # exactly 500 tok/s is already implausible


def test_the_guard_rejects_a_run_that_generated_nothing():
    # A generation that hit EOS immediately would otherwise pass any speed check
    # while measuring only the tokenizer.
    with pytest.raises(BenchError, match="generated nothing"):
        check_plausible(3000.0, 0.0)


# --- spread across repeats ----------------------------------------------------


def test_spread_is_the_range_over_the_median_as_a_percentage():
    assert spread_pct([100.0, 110.0, 105.0]) == pytest.approx(9.52, abs=0.01)


def test_identical_repeats_have_zero_spread():
    assert spread_pct([100.0, 100.0, 100.0]) == 0.0


# --- the timed loop: synchronize placement -----------------------------------


def test_every_timed_iteration_is_bracketed_by_two_synchronizes():
    calls: list[str] = []
    step = fake_step()

    def sync():
        calls.append("sync")

    def counting_step(msgs):
        calls.append("step")
        return step(msgs)

    measure_single_stream(
        counting_step, [messages(i) for i in range(5)], warmup=0, n_docs=5, sync=sync
    )
    # One sync before the clock starts (draining the previous iteration's work, so
    # it is not billed to this one) and one inside the timed region before the
    # clock stops (waiting for this iteration's kernels to actually finish).
    # Never two steps between two syncs.
    assert calls == ["sync", "step", "sync"] * 5


def test_warmup_iterations_run_but_are_not_timed():
    step = fake_step()
    stats = measure_single_stream(
        step, [messages(i) for i in range(20)], warmup=10, n_docs=10, sync=None
    )
    # 10 warmup + 10 measured calls, but only the 10 measured ones are reported.
    assert len(step.seen) == 20
    assert stats["n"] == 10


def test_measurement_uses_the_first_n_docs_so_every_arm_sees_the_same_documents():
    step = fake_step()
    prompts = [messages(i) for i in range(50)]
    measure_single_stream(step, prompts, warmup=0, n_docs=5, sync=None)
    assert step.seen == prompts[:5]


def test_asking_for_more_documents_than_exist_raises_before_spending_gpu_time():
    with pytest.raises(BenchError, match="only 3"):
        measure_single_stream(fake_step(), [messages(i) for i in range(3)], warmup=0, n_docs=100)


def test_parsing_happens_inside_the_timed_region_for_invalid_output_too():
    # A user pays for extract_json/validate_prediction whatever the model emits, so
    # an unparseable output must not skip the work or crash the benchmark.
    stats = measure_single_stream(
        fake_step(raw="not json at all"),
        [messages(i) for i in range(3)],
        warmup=0,
        n_docs=3,
    )
    assert stats["n"] == 3
