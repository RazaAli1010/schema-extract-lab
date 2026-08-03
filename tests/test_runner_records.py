"""`predict_arm`'s record contract, driven by a fake generator.

Laptop-only: `sxl.gpu.runner` keeps every torch import inside a function body, so
this file exercises the whole prediction path — parse, validate, checkpoint,
resume, `<think>` — without a GPU, a model, or torch installed (SPEC §2.1).
"""

from __future__ import annotations

import json

import pytest

from _fakes import doc, gold, gold_json, gold_row
from sxl.gpu.runner import RECORD_KEYS, RunnerError, partial_path, predict_arm
from sxl.io import read_jsonl, write_jsonl

ARM = "base_fewshot"


def rows(n: int) -> list[dict]:
    """`n` eval_gold-shaped rows."""
    return [gold_row(doc(i)) for i in range(n)]


def fake_generate(outputs: list[str]):
    """A `GenerateFn` replaying `outputs` in order, recording what it was asked for.

    `.seen` is the audit log the resume test reads: the point of checkpointing is
    that a resumed run does not re-generate, and the only way to prove that is to
    count what reached the generator.
    """
    queue = list(outputs)
    seen: list[list[dict]] = []

    def _generate(batch):
        seen.append(list(batch))
        if len(queue) < len(batch):
            raise AssertionError(f"generator asked for {len(batch)} more than scripted")
        return [
            {
                "raw_output": queue.pop(0),
                "prompt_tokens": 1180,
                "completion_tokens": 143,
                "latency_ms": 2841.3,
            }
            for _ in batch
        ]

    _generate.seen = seen
    return _generate


def run(tmp_path, gold_rows, outputs, *, arm=ARM, batch_size=8):
    out = tmp_path / f"{arm}.jsonl"
    gen = fake_generate(outputs)
    summary = predict_arm(
        arm,
        gold_rows,
        gen,
        out,
        prompt_fn=lambda text: [{"role": "user", "content": text}],
        batch_size=batch_size,
    )
    return out, summary, gen


# --- the record shape ---------------------------------------------------------


def test_valid_json_completion_produces_a_valid_record(tmp_path):
    g = rows(1)
    out, summary, _ = run(tmp_path, g, [gold_json(title="Senior Python Engineer")])

    written = list(read_jsonl(out))
    assert len(written) == 1
    record = written[0]
    assert tuple(record) == RECORD_KEYS  # SPEC §3.3 order, not just membership
    assert record["doc_id"] == g[0]["doc_id"]
    assert record["arm"] == ARM
    assert record["schema_valid"] is True
    assert record["parsed"]["title"] == "Senior Python Engineer"
    assert summary["n"] == 1 and summary["n_valid"] == 1


def test_fenced_json_is_parsed_by_the_tolerant_teacher_parser(tmp_path):
    fenced = f"```json\n{gold_json(company='Acme Robotics')}\n```"
    out, summary, _ = run(tmp_path, rows(1), [fenced])

    record = next(iter(read_jsonl(out)))
    assert record["schema_valid"] is True
    assert record["parsed"]["company"] == "Acme Robotics"
    assert record["raw_output"] == fenced  # raw is stored verbatim, never cleaned
    assert summary["n_valid"] == 1


def test_a_refusal_is_written_as_an_invalid_record_not_skipped(tmp_path):
    # SPEC §3.5b: invalid predictions are wrong, not excluded. F4 needs one row
    # per gold doc_id or its denominator silently shrinks.
    out, summary, _ = run(tmp_path, rows(1), ["I'm sorry, I can't help with that."])

    written = list(read_jsonl(out))
    assert len(written) == 1
    assert written[0]["parsed"] is None
    assert written[0]["schema_valid"] is False
    assert summary["n"] == 1 and summary["n_valid"] == 0 and summary["n_invalid"] == 1


def test_schema_invalid_json_parses_but_does_not_validate(tmp_path):
    # Parseable JSON, but `remote_mode` is not an enum member and the 16 keys are
    # incomplete — `parsed` must be null, per SPEC §3.3.
    out, _, _ = run(tmp_path, rows(1), ['{"title": "Dev", "remote_mode": "sometimes"}'])

    record = next(iter(read_jsonl(out)))
    assert record["parsed"] is None
    assert record["schema_valid"] is False


def test_schema_valid_agrees_with_validate_prediction_for_every_row(tmp_path):
    # The invariant `metrics._check_prediction_rows` re-derives and raises on.
    from sxl.schema import validate_prediction

    out, _, _ = run(
        tmp_path,
        rows(3),
        [gold_json(), "not json at all", '{"title": "Dev"}'],
        batch_size=3,
    )
    for record in read_jsonl(out):
        assert record["schema_valid"] == (validate_prediction(record["parsed"]) is not None)


# --- thinking mode ------------------------------------------------------------


def test_a_think_block_raises(tmp_path):
    # SPEC §3.7 / F5: do not "handle" it by stripping — that would hide a
    # train/inference template mismatch F6 would inherit.
    with pytest.raises(RunnerError, match="<think>"):
        run(tmp_path, rows(1), [f"<think>let me see</think>{gold_json()}"])


def test_a_think_block_raises_even_when_the_json_is_otherwise_perfect(tmp_path):
    with pytest.raises(RunnerError, match="enable_thinking"):
        run(tmp_path, rows(2), [gold_json(), f"<think>hmm</think>{gold_json()}"], batch_size=2)


# --- checkpoint and resume ----------------------------------------------------


def test_resume_skips_documents_already_in_the_partial_file(tmp_path):
    g = rows(8)
    partial = partial_path(tmp_path / f"{ARM}.jsonl")
    write_jsonl(
        partial,
        [
            {
                "doc_id": row["doc_id"],
                "arm": ARM,
                "raw_output": gold_json(title="from the partial"),
                "parsed": gold(title="from the partial"),
                "schema_valid": True,
                "latency_ms": 1.0,
                "prompt_tokens": 1,
                "completion_tokens": 1,
            }
            for row in g[:5]
        ],
    )

    out, summary, gen = run(tmp_path, g, [gold_json()] * 3, batch_size=8)

    # Only the 3 outstanding documents reached the generator.
    assert sum(len(batch) for batch in gen.seen) == 3
    assert summary["n"] == 8 and summary["n_resumed"] == 5

    written = list(read_jsonl(out))
    assert [r["doc_id"] for r in written] == [r["doc_id"] for r in g]  # gold order
    assert written[0]["parsed"]["title"] == "from the partial"  # not regenerated


def test_the_partial_file_is_removed_once_the_run_completes(tmp_path):
    out, _, _ = run(tmp_path, rows(2), [gold_json()] * 2, batch_size=2)
    assert out.exists()
    assert not partial_path(out).exists()


def test_a_partial_from_a_different_arm_is_fatal(tmp_path):
    partial = partial_path(tmp_path / f"{ARM}.jsonl")
    write_jsonl(partial, [{"doc_id": "jp_x", "arm": "lora_ft", "raw_output": "", "parsed": None}])

    with pytest.raises(RunnerError, match="lora_ft"):
        run(tmp_path, rows(1), [gold_json()])


def test_checkpoints_survive_a_crash_midway(tmp_path):
    # Batch 1 succeeds, batch 2 raises. The partial must hold batch 1 so the
    # rerun does not pay for it again.
    g = rows(4)
    out = tmp_path / f"{ARM}.jsonl"

    def exploding(batch):
        if getattr(exploding, "called", False):
            raise RuntimeError("session died")
        exploding.called = True
        return [
            {
                "raw_output": gold_json(),
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "latency_ms": 1.0,
            }
            for _ in batch
        ]

    with pytest.raises(RuntimeError, match="session died"):
        predict_arm(
            ARM,
            g,
            exploding,
            out,
            prompt_fn=lambda text: [{"role": "user", "content": text}],
            batch_size=2,
        )

    assert not out.exists()
    assert len(list(read_jsonl(partial_path(out)))) == 2


# --- batching and summary -----------------------------------------------------


def test_documents_are_sent_in_batches_of_the_requested_size(tmp_path):
    _, _, gen = run(tmp_path, rows(5), [gold_json()] * 5, batch_size=2)
    assert [len(batch) for batch in gen.seen] == [2, 2, 1]


def test_the_prompt_fn_receives_the_document_text(tmp_path):
    g = rows(1)
    _, _, gen = run(tmp_path, g, [gold_json()])
    assert gen.seen[0][0] == [{"role": "user", "content": g[0]["text"]}]


def test_truncation_is_counted(tmp_path):
    # A completion at the cap is a truncated one. For the constrained arm this is
    # a reportable cost, not a footnote (F5 §Implementation notes).
    g = rows(2)
    out = tmp_path / f"{ARM}.jsonl"

    def at_the_cap(batch):
        return [
            {
                "raw_output": gold_json(),
                "prompt_tokens": 10,
                "completion_tokens": 512,
                "latency_ms": 1.0,
            }
            for _ in batch
        ]

    summary = predict_arm(
        ARM,
        g,
        at_the_cap,
        out,
        prompt_fn=lambda text: [{"role": "user", "content": text}],
        batch_size=2,
        max_new_tokens=512,
    )
    assert summary["n_truncated"] == 2
    assert summary["mean_completion_tokens"] == 512.0


def test_a_generator_returning_the_wrong_count_is_fatal(tmp_path):
    def short(batch):
        return [
            {"raw_output": gold_json(), "prompt_tokens": 1, "completion_tokens": 1, "latency_ms": 1}
        ]

    with pytest.raises(RunnerError, match="returned 1 outputs for 2"):
        predict_arm(
            ARM,
            rows(2),
            short,
            tmp_path / f"{ARM}.jsonl",
            prompt_fn=lambda text: [{"role": "user", "content": text}],
            batch_size=2,
        )


def test_output_is_valid_jsonl_one_object_per_line(tmp_path):
    out, _, _ = run(tmp_path, rows(3), [gold_json()] * 3, batch_size=3)
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    for line in lines:
        assert tuple(json.loads(line)) == RECORD_KEYS
