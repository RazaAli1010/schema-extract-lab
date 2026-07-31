"""The no-double-billing guarantee (SPEC §5.3, F2 acceptance criteria).

"A crashed run must never re-bill." Everything here is a variation on that: what is
already paid for is subtracted before anything is submitted, a changed prompt or
model invalidates the cache wholesale, and a mid-run crash keeps every result that
had already arrived.
"""

from __future__ import annotations

import pytest

from _fakes import FakeOpenAI, docs_spanning_splits, gold_json
from sxl.io import append_jsonl
from sxl.prompts import TEACHER_PROMPT_SHA
from sxl.teacher import (
    TeacherPaths,
    cached_doc_ids,
    label,
    load_cache,
)

MODEL = "gpt-4o-mini"


def cache_row(doc_id: str, *, model: str = MODEL, sha: str = TEACHER_PROMPT_SHA, ok: bool = True):
    return {
        "doc_id": doc_id,
        "teacher_model": model,
        "prompt_sha": sha,
        "raw_output": gold_json(title="cached") if ok else None,
        "usage": {"input_tokens": 2000, "output_tokens": 300},
        "ok": ok,
        "error": None if ok else "request_error: server_error: boom",
        "finish_reason": "stop" if ok else None,
        "batch_id": "batch-old",
        "at": "2026-07-31T00:00:00Z",
    }


def seed_cache(paths: TeacherPaths, rows) -> None:
    for row in rows:
        append_jsonl(paths.cache, row)


def run(paths, docs, client, **kw):
    return label("all", client=client, docs=docs, paths=paths, sleep=lambda _: None, **kw)


def test_only_uncached_documents_are_requested(tmp_path):
    docs = docs_spanning_splits(2)  # 6 documents
    paths = TeacherPaths.in_dir(tmp_path)
    already = [d["doc_id"] for d in docs[:4]]
    seed_cache(paths, [cache_row(d) for d in already])

    client = FakeOpenAI()
    stats = run(paths, docs, client)

    assert sorted(client.custom_ids_requested()) == sorted(d["doc_id"] for d in docs[4:])
    assert stats["n_cached"] == 4
    assert stats["n_new_requests"] == 2


def test_changing_the_prompt_invalidates_the_whole_cache(tmp_path, monkeypatch):
    """The cache key is derived from the system prompt, so a new prompt re-labels."""
    docs = docs_spanning_splits(2)
    paths = TeacherPaths.in_dir(tmp_path)
    seed_cache(paths, [cache_row(d["doc_id"]) for d in docs])

    monkeypatch.setattr("sxl.teacher.TEACHER_PROMPT_SHA", "0000deadbeef0000")
    client = FakeOpenAI()
    stats = run(paths, docs, client)

    assert len(client.custom_ids_requested()) == len(docs)
    assert stats["n_cached"] == 0


def test_a_different_teacher_model_invalidates_the_cache(tmp_path):
    docs = docs_spanning_splits(2)
    paths = TeacherPaths.in_dir(tmp_path)
    seed_cache(paths, [cache_row(d["doc_id"], model="gpt-4o") for d in docs])

    client = FakeOpenAI()
    stats = run(paths, docs, client)

    assert len(client.custom_ids_requested()) == len(docs)
    assert stats["n_cached"] == 0


def test_an_api_error_row_does_not_suppress_a_re_request(tmp_path):
    """A failed attempt is a record, not a paid label."""
    docs = docs_spanning_splits(2)
    paths = TeacherPaths.in_dir(tmp_path)
    seed_cache(paths, [cache_row(d["doc_id"], ok=False) for d in docs])

    client = FakeOpenAI()
    run(paths, docs, client)

    assert sorted(set(client.custom_ids_requested())) == sorted(d["doc_id"] for d in docs)


def test_load_cache_keeps_the_last_row_per_doc_id(tmp_path):
    """Last-wins *is* the per-document retry mechanism."""
    paths = TeacherPaths.in_dir(tmp_path)
    first = cache_row("jp_x", ok=False)
    second = cache_row("jp_x", ok=True)
    seed_cache(paths, [first, second])

    cache = load_cache(paths)
    assert len(cache) == 1
    assert cache["jp_x"]["ok"] is True


def test_cached_doc_ids_filters_on_model_sha_and_ok():
    cache = {
        "a": cache_row("a"),
        "b": cache_row("b", model="gpt-4o"),
        "c": cache_row("c", sha="other"),
        "d": cache_row("d", ok=False),
    }
    assert cached_doc_ids(cache, model=MODEL, prompt_sha_=TEACHER_PROMPT_SHA) == {"a"}


def test_a_crash_mid_run_keeps_every_result_that_had_arrived(tmp_path, monkeypatch):
    """The crashed run's first chunk stays paid-for; the re-run requests only the rest."""
    monkeypatch.setattr("sxl.teacher.TEACHER_BATCH_SIZE", 2)
    monkeypatch.setattr("sxl.teacher.TEACHER_MAX_INFLIGHT_BATCHES", 1)

    docs = docs_spanning_splits(2)  # 6 documents -> 3 chunks of 2
    paths = TeacherPaths.in_dir(tmp_path)

    crashing = FakeOpenAI(raise_on={"batches.create": [None, RuntimeError("network died")]})
    with pytest.raises(RuntimeError, match="network died"):
        run(paths, docs, crashing)

    harvested = set(load_cache(paths))
    assert len(harvested) == 2, "the first chunk must survive the crash"

    resumed = FakeOpenAI()
    stats = run(paths, docs, resumed)

    assert harvested.isdisjoint(resumed.custom_ids_requested()), (
        "re-billed an already-paid document"
    )
    assert stats["n_cached"] == 2
    assert stats["n_ok"] == 6


def test_the_cache_file_is_append_only(tmp_path):
    docs = docs_spanning_splits(1)
    paths = TeacherPaths.in_dir(tmp_path)
    run(paths, docs, FakeOpenAI())
    before = paths.cache.read_bytes()

    more = docs + docs_spanning_splits(2)
    run(paths, {d["doc_id"]: d for d in more}.values(), FakeOpenAI())
    after = paths.cache.read_bytes()

    assert after.startswith(before), "earlier rows were rewritten, not appended to"
    assert len(after) > len(before)


def test_re_running_a_completed_run_makes_zero_api_calls(tmp_path):
    """F2 acceptance: the second run costs $0 and produces byte-identical files."""
    docs = docs_spanning_splits(3)
    paths = TeacherPaths.in_dir(tmp_path)

    first = run(paths, docs, FakeOpenAI())
    snapshots = {s: paths.for_split(s).read_bytes() for s in ("train", "dev", "eval_pool")}

    client = FakeOpenAI()
    second = run(paths, docs, client)

    assert client.calls == []
    assert second["spend_usd"] == 0.0
    assert second["spend_usd_cached"] == pytest.approx(first["spend_usd"])
    assert second["n_new_requests"] == 0
    assert second["n_cached"] == second["n_requested"]
    for split, blob in snapshots.items():
        assert paths.for_split(split).read_bytes() == blob, split


def test_a_second_run_leaves_no_tmp_files_behind(tmp_path):
    paths = TeacherPaths.in_dir(tmp_path)
    run(paths, docs_spanning_splits(2), FakeOpenAI())
    assert not list(tmp_path.rglob("*.tmp"))
