"""The output contract: row shape, split routing, and the four-bucket partition.

Every literal in `ROW_KEYS` below is hand-copied from SPEC §3.3. If a change to
`teacher.py` makes one of these fail, the *contract* changed — that is the point of
the test, not a reason to edit the expected values.

The bucket partition (`ok + parse_failed + schema_invalid + api_error == n_requested`)
is the difference between an honest failure count and a self-congratulatory one: a
document that never came back must be counted, not quietly dropped from the total.
"""

from __future__ import annotations

import json

from _fakes import FakeOpenAI, doc, docs_spanning_splits, gold_json, ok_line, refusal_line
from sxl.schema import FIELD_NAMES, validate_prediction
from sxl.splits import split_for
from sxl.teacher import (
    TeacherPaths,
    build_rows,
    label,
    select_docs,
)

SPEC_ROW_KEYS = [
    "doc_id",
    "domain",
    "text",
    "gold",
    "label_source",
    "teacher_model",
    "verified_by_human",
    "verified_at",
]
SPLITS = ("train", "dev", "eval_pool")


def run(paths, docs, client=None, **kw):
    return label(
        "all", client=client or FakeOpenAI(), docs=docs, paths=paths, sleep=lambda _: None, **kw
    )


def read(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def all_rows(paths):
    return {s: read(paths.for_split(s)) for s in SPLITS}


# --- row shape ----------------------------------------------------------------


def test_output_rows_have_the_eight_spec_keys_in_order(tmp_path):
    paths = TeacherPaths.in_dir(tmp_path)
    run(paths, docs_spanning_splits(2))
    for rows in all_rows(paths).values():
        assert rows
        for row in rows:
            assert list(row) == SPEC_ROW_KEYS


def test_output_rows_carry_the_teacher_provenance(tmp_path):
    """F3 is the only feature permitted to write `"human"` (F2 out-of-scope §1)."""
    paths = TeacherPaths.in_dir(tmp_path)
    run(paths, docs_spanning_splits(2))
    models = set()
    for rows in all_rows(paths).values():
        for row in rows:
            assert row["label_source"] == "teacher"
            assert row["verified_by_human"] is False
            assert row["verified_at"] is None
            assert row["domain"] == "job_posting"
            models.add(row["teacher_model"])
    assert models == {"gpt-4o-mini"}


def test_gold_has_the_sixteen_fields_in_spec_order_and_validates(tmp_path):
    paths = TeacherPaths.in_dir(tmp_path)
    run(paths, docs_spanning_splits(2))
    for rows in all_rows(paths).values():
        for row in rows:
            assert list(row["gold"]) == list(FIELD_NAMES)
            assert validate_prediction(row["gold"]) is not None


def test_gold_is_json_serializable(tmp_path):
    """Pins `model_dump(mode="json")`: raw Enum members would raise at write time."""
    paths = TeacherPaths.in_dir(tmp_path)
    client = FakeOpenAI(
        lambda r: ok_line(
            r["custom_id"], gold_json(seniority="senior", employment_type="full_time")
        )
    )
    run(paths, docs_spanning_splits(1), client)
    for rows in all_rows(paths).values():
        for row in rows:
            json.dumps(row["gold"])  # must not raise
            assert row["gold"]["seniority"] == "senior"


def test_text_is_the_full_untruncated_document(tmp_path):
    """F5/F6 truncate at prompt time; this file stays a faithful copy of the corpus."""
    long_text = "Job Description\n\n" + ("x" * 20_000)
    docs = [doc(i, text=long_text) for i in range(6)]
    paths = TeacherPaths.in_dir(tmp_path)
    run(paths, docs)
    for rows in all_rows(paths).values():
        for row in rows:
            assert len(row["text"]) == len(long_text)


# --- routing ------------------------------------------------------------------


def test_every_row_lands_in_the_file_matching_split_for(tmp_path):
    paths = TeacherPaths.in_dir(tmp_path)
    run(paths, docs_spanning_splits(3))
    for split, rows in all_rows(paths).items():
        assert rows, split
        assert all(split_for(row["doc_id"]) == split for row in rows)


def test_the_three_output_files_are_disjoint_by_doc_id(tmp_path):
    paths = TeacherPaths.in_dir(tmp_path)
    run(paths, docs_spanning_splits(3))
    rows = all_rows(paths)
    ids = {s: {r["doc_id"] for r in rows[s]} for s in SPLITS}
    assert not ids["train"] & ids["dev"]
    assert not ids["train"] & ids["eval_pool"]
    assert not ids["dev"] & ids["eval_pool"]


def test_labeling_dev_does_not_touch_the_train_file(tmp_path):
    """`--split dev` must never truncate a train file a previous run paid for."""
    docs = docs_spanning_splits(2)
    paths = TeacherPaths.in_dir(tmp_path)
    run(paths, docs)
    before = paths.train.read_bytes()

    label("dev", client=FakeOpenAI(), docs=docs, paths=paths, sleep=lambda _: None)

    assert paths.train.read_bytes() == before


def test_output_is_byte_identical_across_runs(tmp_path):
    docs = docs_spanning_splits(3)
    a, b = TeacherPaths.in_dir(tmp_path / "a"), TeacherPaths.in_dir(tmp_path / "b")
    run(a, docs)
    run(b, docs)
    for split in SPLITS:
        assert a.for_split(split).read_bytes() == b.for_split(split).read_bytes(), split


def test_no_tmp_file_is_left_behind(tmp_path):
    run(TeacherPaths.in_dir(tmp_path), docs_spanning_splits(2))
    assert not list(tmp_path.rglob("*.tmp"))


# --- selection ----------------------------------------------------------------


def test_limit_ten_twice_selects_the_identical_documents():
    """Sorted by doc_id, never sampled — otherwise a re-run pays for different documents."""
    docs = docs_spanning_splits(10)
    first = select_docs(docs, split="all", limit=10)
    second = select_docs(list(reversed(docs)), split="all", limit=10)
    assert [d["doc_id"] for d in first] == [d["doc_id"] for d in second]
    assert len(first) == 10


def test_train_and_dev_are_capped_but_eval_pool_is_not(monkeypatch):
    """F3 samples 330 candidates from eval_pool; a cap there would starve the gold set."""
    monkeypatch.setattr("sxl.teacher.N_TRAIN_TARGET", 5)
    monkeypatch.setattr("sxl.teacher.N_DEV_TARGET", 2)
    docs = docs_spanning_splits(8)

    selected = select_docs(docs, split="all")
    counts = {s: sum(1 for d in selected if split_for(d["doc_id"]) == s) for s in SPLITS}

    assert counts["train"] == 5
    assert counts["dev"] == 2
    assert counts["eval_pool"] == 8


def test_selecting_one_split_ignores_the_others():
    docs = docs_spanning_splits(3)
    selected = select_docs(docs, split="dev")
    assert selected
    assert all(split_for(d["doc_id"]) == "dev" for d in selected)


# --- the four buckets ---------------------------------------------------------


def test_a_schema_invalid_gold_is_absent_from_every_file_and_counted(tmp_path):
    docs = docs_spanning_splits(2)
    doomed = docs[0]["doc_id"]
    client = FakeOpenAI(
        lambda r: ok_line(
            r["custom_id"], '{"title": "x"}' if r["custom_id"] == doomed else gold_json()
        )
    )
    paths = TeacherPaths.in_dir(tmp_path)
    stats = run(paths, docs, client)

    written = {r["doc_id"] for rows in all_rows(paths).values() for r in rows}
    assert doomed not in written
    assert stats["n_schema_invalid"] == 1
    assert stats["n_ok"] == len(docs) - 1


def test_an_unparseable_output_is_absent_and_counted(tmp_path):
    docs = docs_spanning_splits(2)
    doomed = docs[0]["doc_id"]
    client = FakeOpenAI(
        lambda r: ok_line(
            r["custom_id"], "sorry, I cannot" if r["custom_id"] == doomed else gold_json()
        )
    )
    paths = TeacherPaths.in_dir(tmp_path)
    stats = run(paths, docs, client)

    written = {r["doc_id"] for rows in all_rows(paths).values() for r in rows}
    assert doomed not in written
    assert stats["n_parse_failed"] == 1


def test_an_api_error_is_absent_and_counted(tmp_path):
    docs = docs_spanning_splits(2)
    doomed = docs[0]["doc_id"]
    client = FakeOpenAI(
        lambda r: (
            refusal_line(r["custom_id"])
            if r["custom_id"] == doomed
            else ok_line(r["custom_id"], gold_json())
        )
    )
    paths = TeacherPaths.in_dir(tmp_path)
    stats = run(paths, docs, client)

    written = {r["doc_id"] for rows in all_rows(paths).values() for r in rows}
    assert doomed not in written
    assert stats["n_api_error"] == 1


def test_a_document_with_no_cache_row_counts_as_an_api_error():
    """Keeps the four buckets a true partition even when a batch never returned."""
    docs = docs_spanning_splits(2)
    stats = {f"n_{b}": 0 for b in ("ok", "parse_failed", "schema_invalid", "api_error")}

    rows = build_rows(docs, {}, model="gpt-4o-mini", stats=stats)

    assert stats["n_api_error"] == len(docs)
    assert all(not v for v in rows.values())


def test_the_four_buckets_partition_n_requested(tmp_path):
    docs = docs_spanning_splits(3)
    bad = {docs[0]["doc_id"], docs[1]["doc_id"], docs[2]["doc_id"]}

    def responder(request):
        cid = request["custom_id"]
        if cid == docs[0]["doc_id"]:
            return ok_line(cid, "nope")
        if cid == docs[1]["doc_id"]:
            return ok_line(cid, '{"title": "x"}')
        if cid == docs[2]["doc_id"]:
            return refusal_line(cid)
        return ok_line(cid, gold_json())

    stats = run(TeacherPaths.in_dir(tmp_path), docs, FakeOpenAI(responder))

    total = sum(stats[f"n_{b}"] for b in ("ok", "parse_failed", "schema_invalid", "api_error"))
    assert total == stats["n_requested"] == len(docs)
    assert stats["n_ok"] == len(docs) - len(bad)
