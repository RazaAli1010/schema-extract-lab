"""Deterministic split assignment by `doc_id` hash (SPEC §3.4).

No randomness, no seed, no dependence on iteration order or corpus size: the same
`doc_id` lands in the same split forever, so re-running any stage can never move a
document between splits.
"""

from __future__ import annotations

import hashlib
from typing import Literal

Split = Literal["train", "dev", "eval_pool"]


def bucket_for(doc_id: str) -> int:
    """0..99. Exposed so tests can assert the distribution directly."""
    return int(hashlib.sha256(doc_id.encode()).hexdigest()[:8], 16) % 100


def split_for(doc_id: str) -> Split:
    """0-4 -> eval_pool (5%) | 5-9 -> dev (5%) | 10-99 -> train (90%)."""
    bucket = bucket_for(doc_id)
    if bucket < 5:
        return "eval_pool"
    if bucket < 10:
        return "dev"
    return "train"
