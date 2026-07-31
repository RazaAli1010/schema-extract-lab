"""F1 x SPEC §3.4 — the corpus must partition cleanly into the three splits.

This is the early warning that a corpus is big enough for F3, checked here on
synthetic ids so it runs without a built corpus.
"""

from __future__ import annotations

from sxl.config import CORPUS_MIN_N, CORPUS_MIN_SPLIT_N
from sxl.corpus import make_doc_id
from sxl.splits import split_for

SOURCE = "test/postings"
IDS = [make_doc_id(SOURCE, f"key-{i}") for i in range(CORPUS_MIN_N)]


def test_split_for_partitions_the_corpus():
    buckets: dict[str, set[str]] = {"train": set(), "dev": set(), "eval_pool": set()}
    for doc_id in IDS:
        buckets[split_for(doc_id)].add(doc_id)

    assert all(buckets.values()), "every split must be non-empty"
    assert buckets["train"].isdisjoint(buckets["dev"])
    assert buckets["train"].isdisjoint(buckets["eval_pool"])
    assert buckets["dev"].isdisjoint(buckets["eval_pool"])
    assert set().union(*buckets.values()) == set(IDS)


def test_a_minimum_sized_corpus_feeds_f3():
    """CORPUS_MIN_N documents must yield >= 340 in both dev and eval_pool."""
    counts = {"train": 0, "dev": 0, "eval_pool": 0}
    for doc_id in IDS:
        counts[split_for(doc_id)] += 1

    assert counts["dev"] >= CORPUS_MIN_SPLIT_N, counts
    assert counts["eval_pool"] >= CORPUS_MIN_SPLIT_N, counts
    assert counts["train"] >= 5000, counts


def test_hashed_doc_ids_do_not_collide():
    assert len(set(IDS)) == len(IDS)
