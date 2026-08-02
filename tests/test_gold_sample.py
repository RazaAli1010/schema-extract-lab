"""Stratified candidate sampling and the split-leakage guard (SPEC §3.4, F3 Scope 1-2)."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap

import pytest

from _fakes import eval_pool_rows, write_pool
from sxl.io import write_jsonl
from sxl.splits import split_for
from sxl.verify import (
    GoldPaths,
    LeakageDetected,
    PoolTooSmall,
    assert_no_leakage,
    sample_candidates,
    strata_labels,
    stratum_of,
)

SHORT = "s" * 800  # -> 400-1500
LONG = "l" * 5000  # -> 3000-8000


def _paths(tmp_path):
    return GoldPaths.in_dir(tmp_path)


def test_stratum_edges_land_in_the_documented_bins():
    assert strata_labels() == ("400-1500", "1500-3000", "3000-8000", "8000-40000")
    assert stratum_of(400) == "400-1500"
    assert stratum_of(1499) == "400-1500"
    assert stratum_of(1500) == "1500-3000"
    assert stratum_of(7999) == "3000-8000"
    assert stratum_of(8000) == "8000-40000"
    # Out-of-range values clamp rather than raise.
    assert stratum_of(1) == "400-1500"
    assert stratum_of(10**9) == "8000-40000"


def test_same_seed_yields_the_same_ids_twice(tmp_path):
    paths = _paths(tmp_path)
    write_pool(paths, eval_pool_rows(120))

    first = sample_candidates(n=40, seed=1337, paths=paths, write=False)
    second = sample_candidates(n=40, seed=1337, paths=paths, write=False)
    assert [r["doc_id"] for r in first] == [r["doc_id"] for r in second]
    assert first == sorted(first, key=lambda r: r["doc_id"]), "output must be doc_id-sorted"


def test_same_seed_yields_the_same_ids_in_a_fresh_process(tmp_path):
    """Reproducibility must not depend on this interpreter's hash seed."""
    paths = _paths(tmp_path)
    write_pool(paths, eval_pool_rows(120))
    expected = [r["doc_id"] for r in sample_candidates(n=40, seed=1337, paths=paths, write=False)]

    code = textwrap.dedent(f"""
        import json
        from sxl.verify import GoldPaths, sample_candidates
        paths = GoldPaths.in_dir(r{str(tmp_path)!r})
        rows = sample_candidates(n=40, seed=1337, paths=paths, write=False)
        print(json.dumps([r["doc_id"] for r in rows]))
    """)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert json.loads(out.stdout) == expected


def test_a_different_seed_picks_a_different_sample(tmp_path):
    paths = _paths(tmp_path)
    write_pool(paths, eval_pool_rows(120))
    a = {r["doc_id"] for r in sample_candidates(n=40, seed=1337, paths=paths, write=False)}
    b = {r["doc_id"] for r in sample_candidates(n=40, seed=7, paths=paths, write=False)}
    assert a != b


def test_every_sampled_id_is_in_the_eval_pool_bucket(tmp_path):
    paths = _paths(tmp_path)
    write_pool(paths, eval_pool_rows(100))
    for row in sample_candidates(n=50, paths=paths, write=False):
        assert split_for(row["doc_id"]) == "eval_pool"


def test_sampling_is_stratified_by_document_length(tmp_path):
    """A 90/10 short/long pool must come back roughly 90/10, not 100/0."""
    paths = _paths(tmp_path)
    rows = eval_pool_rows(200)
    for i, row in enumerate(rows):  # 10% long, deterministically placed
        row["text"] = LONG if i % 10 == 0 else SHORT
    write_pool(paths, rows)

    picked = sample_candidates(n=100, paths=paths, write=False)
    long_fraction = sum(1 for r in picked if stratum_of(len(r["text"])) == "3000-8000") / len(
        picked
    )
    assert abs(long_fraction - 0.10) <= 0.05, long_fraction


def test_quotas_sum_to_n_exactly(tmp_path):
    """Largest-remainder allocation, not naive rounding — 330 must mean 330."""
    paths = _paths(tmp_path)
    rows = eval_pool_rows(361)
    for i, row in enumerate(rows):
        row["text"] = LONG if i % 3 == 0 else SHORT
    write_pool(paths, rows)
    assert len(sample_candidates(n=330, paths=paths, write=False)) == 330


def test_sample_writes_the_candidates_file(tmp_path):
    paths = _paths(tmp_path)
    write_pool(paths, eval_pool_rows(80))
    picked = sample_candidates(n=30, paths=paths)
    assert paths.candidates.exists()
    assert len(paths.candidates.read_text(encoding="utf-8").strip().splitlines()) == 30
    assert [json.loads(line)["doc_id"] for line in paths.candidates.read_text("utf-8").splitlines()]
    assert not list(paths.candidates.parent.glob("*.tmp")), "atomic write left a tmp file"
    assert len(picked) == 30


def test_a_short_pool_raises_naming_both_numbers(tmp_path):
    paths = _paths(tmp_path)
    write_pool(paths, eval_pool_rows(300))
    with pytest.raises(PoolTooSmall) as exc:
        sample_candidates(n=330, paths=paths, write=False)
    assert "300" in str(exc.value) and "330" in str(exc.value)


def test_a_missing_pool_tells_the_user_to_run_f2(tmp_path):
    with pytest.raises(PoolTooSmall, match="teacher label"):
        sample_candidates(n=10, paths=_paths(tmp_path), write=False)


def test_leakage_raises_and_names_the_offending_id(tmp_path):
    paths = _paths(tmp_path)
    rows = eval_pool_rows(60)
    write_pool(paths, rows)
    planted = rows[0]["doc_id"]
    write_jsonl(paths.train, [{"doc_id": planted}])

    with pytest.raises(LeakageDetected) as exc:
        sample_candidates(n=60, paths=paths, write=False)
    assert planted in str(exc.value)
    assert "train.jsonl" in str(exc.value)


def test_leakage_check_skips_files_that_do_not_exist(tmp_path):
    """F2 can be run one split at a time, leaving train/dev absent."""
    paths = _paths(tmp_path)
    assert_no_leakage({"jp_whatever"}, paths.train, paths.dev)


def test_leakage_check_looks_at_dev_too(tmp_path):
    paths = _paths(tmp_path)
    write_jsonl(paths.dev, [{"doc_id": "jp_planted"}])
    with pytest.raises(LeakageDetected, match="jp_planted"):
        assert_no_leakage({"jp_planted"}, paths.train, paths.dev)
