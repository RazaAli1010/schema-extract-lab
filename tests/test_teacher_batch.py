"""The Batch API lifecycle: request shape, polling, adoption, terminal statuses.

The expensive failure this file guards against is a retried `batches.create` that
duplicates a batch whose first attempt actually landed — you pay twice and only
find out on the invoice. `submit_batch` persists `input_file_id` *before* creating,
and every create-retry is gated on an adoption scan; both are asserted here.

`openai` is imported inside the two tests that need real exception instances. The
module itself never imports it (see `tests/test_no_torch_import.py`).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from _fakes import (
    FakeOpenAI,
    doc,
    docs_spanning_splits,
    error_line,
    gold_json,
    http_line,
    ok_line,
    refusal_line,
)
from sxl.config import TEACHER_MAX_TOKENS
from sxl.io import write_json
from sxl.schema import JSON_SCHEMA
from sxl.teacher import (
    BATCH_ENDPOINT,
    BatchCancelled,
    BatchTimeout,
    TeacherError,
    TeacherPaths,
    build_request,
    label,
    load_cache,
    load_ledger,
    parse_result_line,
    poll_batch,
)

MODEL = "gpt-4o-mini"
SHA = "0123456789abcdef"


def line_args(**kw):
    return {"model": MODEL, "prompt_sha_": SHA, "batch_id": "batch-0", **kw}


def transient(message: str = "connection reset"):
    import httpx
    import openai

    return openai.APIConnectionError(
        message=message, request=httpx.Request("POST", "https://api.openai.com/v1/batches")
    )


# --- request shape ------------------------------------------------------------


def test_build_request_has_the_batch_line_shape():
    request = build_request(doc(0), MODEL)
    assert request["custom_id"] == doc(0)["doc_id"]
    assert request["method"] == "POST"
    assert request["url"] == BATCH_ENDPOINT
    body = request["body"]
    assert body["model"] == MODEL
    assert body["temperature"] == 0.0  # greedy everywhere (SPEC §3.6)
    assert body["max_tokens"] == TEACHER_MAX_TOKENS
    assert [m["role"] for m in body["messages"]] == ["system", "user"]


def test_build_request_uses_strict_structured_outputs():
    """Strict Structured Outputs is what drives parse/schema failures to ~0."""
    fmt = build_request(doc(0))["body"]["response_format"]
    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"]["name"] == "job_posting"
    assert fmt["json_schema"]["strict"] is True
    assert fmt["json_schema"]["schema"] is JSON_SCHEMA


def test_custom_ids_are_unique_within_a_batch():
    docs = docs_spanning_splits(60)
    ids = [build_request(d)["custom_id"] for d in docs]
    assert len(set(ids)) == len(ids)


# --- result-line mapping ------------------------------------------------------


def test_parse_result_line_maps_openai_usage_names():
    """OpenAI says prompt/completion; the cache and `cost_of` say input/output."""
    row = parse_result_line(ok_line("jp_a", gold_json(), in_tok=1234, out_tok=321), **line_args())
    assert row["ok"] is True
    assert row["usage"] == {"input_tokens": 1234, "output_tokens": 321}
    assert row["doc_id"] == "jp_a"
    assert row["batch_id"] == "batch-0"
    assert row["teacher_model"] == MODEL and row["prompt_sha"] == SHA


def test_parse_result_line_records_length_truncation_as_ok():
    """The truncated text is real; it becomes `parse_failed` downstream, not `api_error`."""
    row = parse_result_line(ok_line("jp_a", '{"title": "x"', finish="length"), **line_args())
    assert row["ok"] is True
    assert row["finish_reason"] == "length"


def test_parse_result_line_buckets_a_refusal_as_an_error():
    row = parse_result_line(refusal_line("jp_a"), **line_args())
    assert row["ok"] is False
    assert "refusal" in row["error"]
    assert row["raw_output"] is None


def test_parse_result_line_handles_a_non_200_response():
    row = parse_result_line(http_line("jp_a", 429, "slow down"), **line_args())
    assert row["ok"] is False
    assert "http_429" in row["error"] and "slow down" in row["error"]


def test_parse_result_line_handles_a_request_level_error():
    row = parse_result_line(error_line("jp_a", "token_limit", "too many"), **line_args())
    assert row["ok"] is False
    assert "request_error" in row["error"] and "too many" in row["error"]
    assert row["usage"] == {"input_tokens": 0, "output_tokens": 0}, "errors are not billed"


def test_parse_result_line_handles_empty_content():
    row = parse_result_line(ok_line("jp_a", ""), **line_args())
    assert row["ok"] is False
    assert row["error"] == "empty_content"


# --- polling ------------------------------------------------------------------


def test_poll_batch_returns_the_terminal_batch_object():
    client = FakeOpenAI(statuses=("validating", "in_progress", "completed"))
    client.files.create(file=b'{"custom_id": "a"}\n', purpose="batch")
    created = client.batches.create(
        input_file_id="file-0", endpoint=BATCH_ENDPOINT, completion_window="24h"
    )
    slept: list[float] = []

    batch = poll_batch(client, created.id, poll_seconds=60, sleep=slept.append)

    assert batch.status == "completed"
    assert len(slept) == 2, "one sleep per non-terminal poll"


def test_poll_batch_raises_after_max_wait_and_points_at_resume():
    client = FakeOpenAI(statuses=("in_progress",))
    client.files.create(file=b'{"custom_id": "a"}\n', purpose="batch")
    created = client.batches.create(
        input_file_id="file-0", endpoint=BATCH_ENDPOINT, completion_window="24h"
    )
    clock = iter([0.0, 100_000.0, 200_000.0])

    with pytest.raises(BatchTimeout, match="--resume"):
        poll_batch(
            client,
            created.id,
            poll_seconds=1,
            max_wait_s=60,
            sleep=lambda _: None,
            monotonic=lambda: next(clock),
        )


# --- crash windows and adoption ----------------------------------------------


def test_the_ledger_records_the_input_file_id_before_creating(tmp_path, monkeypatch):
    """The crash window between upload and create must leave a trace on disk."""
    monkeypatch.setattr("sxl.teacher.TEACHER_BATCH_SIZE", 2)
    monkeypatch.setattr("sxl.teacher.TEACHER_MAX_INFLIGHT_BATCHES", 1)
    paths = TeacherPaths.in_dir(tmp_path)
    client = FakeOpenAI(raise_on={"batches.create": [RuntimeError("died mid-create")]})

    with pytest.raises(RuntimeError):
        label("all", client=client, docs=docs_spanning_splits(2), paths=paths, sleep=lambda _: None)

    entry = load_ledger(paths)["batches"][0]
    assert entry["input_file_id"] == "file-0"
    assert entry["batch_id"] is None
    assert entry["status"] == "uploaded"
    assert entry["harvested"] is False


def test_resume_adopts_an_orphaned_batch_instead_of_creating_a_second_one(tmp_path):
    """The batch was already created and billed; resume must harvest it, not repeat it."""
    docs = docs_spanning_splits(1)
    paths = TeacherPaths.in_dir(tmp_path)
    client = FakeOpenAI()

    # Simulate a create that landed but whose batch_id never reached the ledger.
    payload = "\n".join(__import__("json").dumps(build_request(d, MODEL)) for d in docs).encode()
    file_obj = client.files.create(file=("batch.jsonl", payload), purpose="batch")
    orphan = client.batches.create(
        input_file_id=file_obj.id, endpoint=BATCH_ENDPOINT, completion_window="24h"
    )
    write_json(
        paths.batches,
        {
            "version": 1,
            "model": MODEL,
            "prompt_sha": None,
            "batches": [
                {
                    "tag": "main-0000",
                    "attempt": 1,
                    "input_file_id": file_obj.id,
                    "batch_id": None,
                    "output_file_id": None,
                    "error_file_id": None,
                    "docs_sha": "irrelevant",
                    "doc_ids": [d["doc_id"] for d in docs],
                    "n_requests": len(docs),
                    "status": "uploaded",
                    "submitted_at": None,
                    "completed_at": None,
                    "harvested": False,
                }
            ],
        },
    )

    stats = label("all", client=client, docs=docs, paths=paths, sleep=lambda _: None)

    assert client.n_calls("batches.create") == 1, "resume created a duplicate batch"
    assert client.n_calls("files.create") == 1
    assert stats["n_ok"] == len(docs)
    assert load_ledger(paths)["batches"][0]["batch_id"] == orphan.id


def test_resume_closes_the_entry_when_no_orphan_exists(tmp_path):
    """No batch_id and no orphan means `batches.create` never landed — nothing was billed."""
    docs = docs_spanning_splits(1)
    paths = TeacherPaths.in_dir(tmp_path)
    client = FakeOpenAI()
    write_json(
        paths.batches,
        {
            "version": 1,
            "model": MODEL,
            "prompt_sha": None,
            "batches": [
                {
                    "tag": "main-0000",
                    "attempt": 1,
                    "input_file_id": "file-never-created",
                    "batch_id": None,
                    "output_file_id": None,
                    "error_file_id": None,
                    "docs_sha": "nope",
                    "doc_ids": [d["doc_id"] for d in docs],
                    "n_requests": len(docs),
                    "status": "uploaded",
                    "submitted_at": None,
                    "completed_at": None,
                    "harvested": False,
                }
            ],
        },
    )

    stats = label("all", client=client, docs=docs, paths=paths, sleep=lambda _: None)

    assert stats["n_ok"] == len(docs)
    abandoned = load_ledger(paths)["batches"][0]
    assert abandoned["status"] == "abandoned" and abandoned["harvested"] is True


def test_a_run_refuses_to_mix_two_teacher_models(tmp_path):
    """`teacher_model` must be constant within a run (F2 out-of-scope §4)."""
    docs = docs_spanning_splits(1)
    paths = TeacherPaths.in_dir(tmp_path)
    label("all", client=FakeOpenAI(), docs=docs, paths=paths, sleep=lambda _: None)

    with pytest.raises(TeacherError, match="model="):
        label(
            "all",
            client=FakeOpenAI(),
            docs=docs,
            paths=paths,
            model="gpt-4o",
            sleep=lambda _: None,
        )


# --- terminal statuses --------------------------------------------------------


def test_an_expired_batch_harvests_partials_and_resubmits_the_rest(tmp_path, monkeypatch):
    """Unprocessed requests are not billed, so only the missing ids go back out."""
    monkeypatch.setattr("sxl.teacher.TEACHER_BATCH_SIZE", 100)
    docs = docs_spanning_splits(2)  # 6 documents
    paths = TeacherPaths.in_dir(tmp_path)
    unprocessed = {d["doc_id"] for d in docs[4:]}

    def responder(request):
        cid = request["custom_id"]
        if cid in unprocessed:
            return error_line(cid, "batch_expired", "not processed")
        return ok_line(cid, gold_json())

    client = FakeOpenAI(responder, statuses=("in_progress", "expired"))
    # The retry wave succeeds for everyone.
    label("all", client=client, docs=docs, paths=paths, sleep=lambda _: None)

    requested = client.custom_ids_requested()
    resubmitted = [cid for cid in requested[6:]]
    assert set(resubmitted) == unprocessed, "resubmitted more than the unprocessed requests"


def test_a_failed_batch_with_a_non_transient_error_raises_without_resubmitting(tmp_path):
    errors = SimpleNamespace(
        data=[SimpleNamespace(code="invalid_request", message="model does not exist")]
    )
    client = FakeOpenAI(statuses=("failed",), errors=errors)

    with pytest.raises(TeacherError, match="model does not exist"):
        label(
            "all",
            client=client,
            docs=docs_spanning_splits(1),
            paths=TeacherPaths.in_dir(tmp_path),
            sleep=lambda _: None,
        )

    assert client.n_calls("batches.create") == 1, "retried a request that is malformed every time"


def test_a_cancelled_batch_harvests_partials_then_raises(tmp_path):
    docs = docs_spanning_splits(1)
    paths = TeacherPaths.in_dir(tmp_path)
    client = FakeOpenAI(statuses=("in_progress", "cancelled"))

    with pytest.raises(BatchCancelled):
        label("all", client=client, docs=docs, paths=paths, sleep=lambda _: None)

    assert len(load_cache(paths)) == len(docs), "partial results were thrown away"


# --- retries ------------------------------------------------------------------


def test_transient_errors_are_retried_with_exponential_backoff(tmp_path):
    docs = docs_spanning_splits(1)
    client = FakeOpenAI(raise_on={"batches.create": [transient(), transient(), None]})
    slept: list[float] = []

    stats = label(
        "all",
        client=client,
        docs=docs,
        paths=TeacherPaths.in_dir(tmp_path),
        sleep=slept.append,
    )

    assert stats["n_ok"] == len(docs)
    assert slept[:2] == [4, 16], "backoff is not 4s then 16s"


def test_a_bad_request_is_never_retried(tmp_path):
    """A malformed body is malformed on every attempt; retrying only delays the diagnosis."""
    import httpx
    import openai

    request = httpx.Request("POST", "https://api.openai.com/v1/batches")
    bad = openai.BadRequestError(
        "unknown parameter", response=httpx.Response(400, request=request), body=None
    )
    client = FakeOpenAI(raise_on={"batches.create": [bad]})

    with pytest.raises(openai.BadRequestError):
        label(
            "all",
            client=client,
            docs=docs_spanning_splits(1),
            paths=TeacherPaths.in_dir(tmp_path),
            sleep=lambda _: None,
        )

    assert client.attempts["batches.create"] == 1, "retried a permanently bad request"
    assert client.n_calls("batches.create") == 0, "a batch was created despite the error"


def test_at_most_four_batches_are_in_flight_at_once(tmp_path, monkeypatch):
    """OpenAI caps enqueued tokens across in-flight batches; exceeding it fails them all."""
    monkeypatch.setattr("sxl.teacher.TEACHER_BATCH_SIZE", 1)
    monkeypatch.setattr("sxl.teacher.TEACHER_MAX_INFLIGHT_BATCHES", 2)
    docs = docs_spanning_splits(3)  # 9 documents -> 9 batches

    created_when_polled: list[int] = []
    client = FakeOpenAI()
    real_retrieve = client.batches.retrieve

    def counting_retrieve(batch_id):
        unharvested = sum(1 for b in client.created if b.status != "completed")
        created_when_polled.append(unharvested)
        return real_retrieve(batch_id)

    monkeypatch.setattr(client.batches, "retrieve", counting_retrieve)
    label(
        "all", client=client, docs=docs, paths=TeacherPaths.in_dir(tmp_path), sleep=lambda _: None
    )

    assert max(created_when_polled) <= 2, created_when_polled


def test_a_document_is_retried_at_most_once(tmp_path):
    """One follow-up round, then the document is dropped (F2 §7)."""
    docs = docs_spanning_splits(1)
    doomed = docs[0]["doc_id"]

    def responder(request):
        cid = request["custom_id"]
        if cid == doomed:
            return ok_line(cid, "sorry, I cannot extract anything")
        return ok_line(cid, gold_json())

    client = FakeOpenAI(responder)
    stats = label(
        "all",
        client=client,
        docs=docs,
        paths=TeacherPaths.in_dir(tmp_path),
        sleep=lambda _: None,
    )

    assert client.custom_ids_requested().count(doomed) == 2, "retried more or less than once"
    assert stats["n_parse_failed"] == 1


def test_a_schema_invalid_document_is_not_retried(tmp_path):
    """At temperature=0 a shape violation reproduces; retrying it is pure spend."""
    docs = docs_spanning_splits(1)
    doomed = docs[0]["doc_id"]

    def responder(request):
        cid = request["custom_id"]
        if cid == doomed:
            return ok_line(cid, '{"title": "x"}')  # valid JSON, 15 keys missing
        return ok_line(cid, gold_json())

    client = FakeOpenAI(responder)
    stats = label(
        "all",
        client=client,
        docs=docs,
        paths=TeacherPaths.in_dir(tmp_path),
        sleep=lambda _: None,
    )

    assert client.custom_ids_requested().count(doomed) == 1
    assert stats["n_schema_invalid"] == 1
