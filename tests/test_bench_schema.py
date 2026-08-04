"""The shape of `results/bench/*.json` and the consistency of the numbers in it.

Two contracts are defended here:

1. **Key set.** F8 reads these files to build the headline table. A dropped or
   renamed key is a broken table, and SPEC §3.3 fixes the names.
2. **Internal consistency.** `cost_per_1k_docs_usd` must be derivable from the
   `throughput_docs_per_s` and `gpu_hourly_usd` in the *same file*, so that nobody
   can hand-edit one without the other and no reader has to trust that the rate
   quoted is the rate used (F7 §Test plan).
"""

from __future__ import annotations

import pytest

from sxl.gpu.bench import (
    BENCH_KEYS,
    SPEC_BENCH_KEYS,
    SWEEP_ENTRY_KEYS,
    SWEEP_KEYS,
    BenchError,
    best_batch_size,
    build_bench_record,
    build_sweep_record,
    cost_per_1k,
    whole_batches,
)

# A plausible T4 measurement: ~150 tokens at ~50 tok/s.
STATS = {
    "n": 100,
    "p50_ms": 3000.0,
    "p95_ms": 3600.0,
    "mean_ms": 3050.0,
    "min_ms": 2800.0,
    "max_ms": 4100.0,
    "stdev_ms": 210.0,
    "mean_completion_tokens": 150.0,
}


def record(**overrides):
    kwargs = {
        "arm": "lora_ft",
        "gpu_name": "Tesla T4",
        "dtype": "float16",
        "stats": STATS,
        "p50_spread_pct": 4.2,
        "n_docs": 100,
        "warmup": 10,
        "repeats": 3,
        "hourly_usd": 0.35,
        "cache_implementation": "dynamic",
        "index_build_s": 0.0,
        "measurement": "local_gpu",
        "torch_version": "2.13.0",
        "transformers_version": "5.14.1",
    }
    return build_bench_record(**{**kwargs, **overrides})


def sweep_entry(batch_size, throughput=0.0, *, oom=False, vram=1.0):
    return {
        "batch_size": batch_size,
        "throughput_docs_per_s": throughput,
        "amortized_ms_per_doc": round(1000.0 / throughput, 2) if throughput else 0.0,
        "peak_vram_gb": vram,
        "mean_completion_tokens": 150.0 if not oom else 0.0,
        "oom": oom,
    }


# --- key contracts ------------------------------------------------------------


def test_the_bench_record_has_exactly_the_contracted_keys_in_order():
    assert tuple(record()) == BENCH_KEYS


def test_every_key_spec_section_3_3_fixes_survives_f7s_additions():
    # F7's context deltas may *extend* the contract; they may not drop from it.
    missing = [k for k in SPEC_BENCH_KEYS if k not in BENCH_KEYS]
    assert missing == []


def test_the_sweep_record_has_exactly_the_contracted_keys_in_order():
    sweep = build_sweep_record(
        arm="lora_ft",
        gpu_name="Tesla T4",
        dtype="float16",
        sweep=[sweep_entry(1, 0.33), sweep_entry(2, 0.61)],
        n_docs=100,
        warmup=10,
        hourly_usd=0.35,
        cache_implementation="dynamic",
    )
    assert tuple(sweep) == SWEEP_KEYS
    for entry in sweep["sweep"]:
        assert tuple(entry) == SWEEP_ENTRY_KEYS


def test_a_sweep_entry_with_drifted_keys_is_rejected():
    with pytest.raises(BenchError):
        build_sweep_record(
            arm="lora_ft",
            gpu_name="Tesla T4",
            dtype="float16",
            sweep=[{"batch_size": 1, "throughput_docs_per_s": 0.33}],
            n_docs=100,
            warmup=10,
            hourly_usd=0.35,
            cache_implementation="dynamic",
        )


# --- internal consistency of the cost figure ---------------------------------


def test_cost_is_derived_from_the_throughput_and_rate_the_same_file_reports():
    r = record()
    assert r["cost_per_1k_docs_usd"] == pytest.approx(
        cost_per_1k(r["throughput_docs_per_s"], r["gpu_hourly_usd"]), abs=1e-6
    )


def test_the_rate_written_down_is_the_rate_used():
    # If someone changes --hourly-usd, both the echoed assumption and the derived
    # cost must move together.
    cheap, dear = record(hourly_usd=0.35), record(hourly_usd=0.70)
    assert cheap["gpu_hourly_usd"] == 0.35
    assert dear["gpu_hourly_usd"] == 0.70
    # Tolerance of 2e-6 because both are stored rounded to 6 decimal places, which
    # is a tenth of a cent per million documents.
    assert dear["cost_per_1k_docs_usd"] == pytest.approx(
        cheap["cost_per_1k_docs_usd"] * 2, abs=2e-6
    )


def test_single_stream_throughput_is_the_reciprocal_of_p50_not_a_batch_figure():
    # The amortized number belongs in the sweep file under `best_*`. A file that
    # quietly reported batched throughput beside a single-stream p50 is exactly
    # the conflation SPEC §1.1 forbids.
    r = record()
    assert r["batch_size"] == 1
    assert r["throughput_docs_per_s"] == pytest.approx(1000.0 / r["p50_ms"], abs=1e-4)


def test_the_sweep_reports_its_amortized_figures_under_distinctly_named_keys():
    sweep = build_sweep_record(
        arm="lora_ft",
        gpu_name="Tesla T4",
        dtype="float16",
        sweep=[sweep_entry(1, 0.33), sweep_entry(8, 2.5)],
        n_docs=100,
        warmup=10,
        hourly_usd=0.35,
        cache_implementation="dynamic",
    )
    assert sweep["best_batch_size"] == 8
    assert sweep["best_throughput_docs_per_s"] == pytest.approx(2.5)
    assert sweep["best_amortized_ms_per_doc"] == pytest.approx(400.0)
    assert sweep["best_cost_per_1k_docs_usd"] == pytest.approx(cost_per_1k(2.5, 0.35), abs=1e-6)
    # None of the single-stream names leak into the sweep file.
    assert "p50_ms" not in sweep


# --- OOM handling -------------------------------------------------------------


def test_oom_entries_are_preserved_in_the_file_but_excluded_from_selection():
    sweep = [
        sweep_entry(1, 0.33),
        sweep_entry(8, 2.5),
        sweep_entry(16, oom=True, vram=14.8),
        sweep_entry(32, oom=True, vram=14.9),
    ]
    assert best_batch_size(sweep) == 8

    built = build_sweep_record(
        arm="lora_ft",
        gpu_name="Tesla T4",
        dtype="float16",
        sweep=sweep,
        n_docs=100,
        warmup=10,
        hourly_usd=0.35,
        cache_implementation="dynamic",
    )
    # Knowing *where* a 16 GB card runs out is a result, not an error to discard.
    assert [e["batch_size"] for e in built["sweep"]] == [1, 8, 16, 32]
    assert [e["oom"] for e in built["sweep"]] == [False, False, True, True]
    assert built["best_batch_size"] == 8


def test_best_batch_size_is_the_largest_that_survived_not_the_fastest_measured():
    # Throughput saturates before it OOMs; the spec defines best as the largest
    # non-OOM size, and reports it as such.
    sweep = [sweep_entry(1, 0.33), sweep_entry(8, 2.5), sweep_entry(16, 2.4)]
    assert best_batch_size(sweep) == 16


def test_an_all_oom_sweep_raises_rather_than_inventing_a_best():
    with pytest.raises(BenchError):
        best_batch_size([sweep_entry(16, oom=True), sweep_entry(32, oom=True)])


# --- the sweep only measures whole batches -----------------------------------


def test_a_sweep_measures_only_whole_batches():
    # A trailing partial batch (8 then 2) runs at a narrower width than the entry's
    # label claims and drags the measured throughput down.
    assert whole_batches(100, 8) == 96
    assert whole_batches(10, 4) == 8
    assert whole_batches(100, 1) == 100


def test_a_batch_size_larger_than_the_corpus_yields_nothing_to_measure():
    # The bug this exists to prevent: with n_docs=10, a batch_size=32 entry would
    # otherwise measure a batch of 10 and report it as 32.
    assert whole_batches(10, 16) == 0
    assert whole_batches(10, 32) == 0
    assert whole_batches(0, 1) == 0


def test_an_exactly_divisible_corpus_uses_every_document():
    assert whole_batches(32, 32) == 32
    assert whole_batches(96, 8) == 96


def test_a_non_positive_batch_size_raises():
    with pytest.raises(BenchError):
        whole_batches(100, 0)


# --- the guard runs on the way to disk ---------------------------------------


def test_building_a_record_from_an_implausible_measurement_raises():
    # The guard is not advisory: a bad measurement must never reach results/bench/.
    fast = {**STATS, "p50_ms": 45.0}
    with pytest.raises(BenchError, match="implausible"):
        record(stats=fast)


def test_no_accuracy_metric_appears_anywhere_in_a_bench_record():
    # F4 owns macro_f1 and schema_valid_rate. Two features writing the same
    # quantity is how a headline table ends up contradicting itself
    # (F7 §Out of scope).
    forbidden = {"macro_f1", "schema_valid_rate", "per_field", "em", "precision", "recall", "f1"}
    assert forbidden.isdisjoint(BENCH_KEYS)
    assert forbidden.isdisjoint(SWEEP_KEYS)
    assert forbidden.isdisjoint(SWEEP_ENTRY_KEYS)


def test_the_gpu_arms_record_the_hardware_they_ran_on():
    r = record()
    assert "T4" in r["gpu_name"]
    assert r["dtype"] == "float16"
    assert r["measurement"] == "local_gpu"
    assert r["cache_implementation"] == "dynamic"
