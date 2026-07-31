"""Money. Every assertion here exists because the alternative is an unexpected bill.

The spend guard's contract is "do not submit and then apologize": the projection is
computed and checked **before** any client method is touched, so the tests assert on
`client.calls == []` rather than on a message.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from _fakes import FakeOpenAI, docs_spanning_splits, patch_cli, write_docs
from sxl.cli import app
from sxl.config import TEACHER_BATCH_DISCOUNT, TEACHER_PRICE_USD
from sxl.teacher import (
    SpendCapExceeded,
    TeacherPaths,
    cost_of,
    estimate_usage,
    label,
)

runner = CliRunner()

ONE_MILLION_EACH = {"input_tokens": 1_000_000, "output_tokens": 1_000_000}


def test_price_table_holds_standard_not_discounted_rates():
    """`cost_of` applies the discount. Pre-discounting the table double-applies it."""
    assert TEACHER_PRICE_USD["gpt-4o-mini"] == {"in": 0.15, "out": 0.60}
    assert TEACHER_BATCH_DISCOUNT == 0.5


def test_cost_of_applies_the_batch_discount():
    """0.5 * ($0.15 in + $0.60 out) per 1M+1M tokens."""
    assert cost_of(ONE_MILLION_EACH, "gpt-4o-mini") == pytest.approx(0.375)


def test_cost_of_is_zero_for_zero_usage():
    assert cost_of({"input_tokens": 0, "output_tokens": 0}) == 0.0


def test_cost_of_raises_loudly_for_an_unpriced_model():
    with pytest.raises(KeyError):
        cost_of(ONE_MILLION_EACH, "some-model-nobody-priced")


def test_estimate_usage_scales_with_document_count():
    one = estimate_usage(docs_spanning_splits(1)[:1])
    ten = estimate_usage(docs_spanning_splits(4)[:10])
    assert ten["output_tokens"] == 350 * 10
    assert one["output_tokens"] == 350
    assert ten["input_tokens"] > one["input_tokens"]


def test_estimate_usage_is_empty_for_no_documents():
    assert estimate_usage([]) == {"input_tokens": 0, "output_tokens": 0}


def test_dry_run_makes_zero_api_calls_and_writes_nothing(tmp_path):
    client = FakeOpenAI()
    paths = TeacherPaths.in_dir(tmp_path)
    stats = label("all", client=client, docs=docs_spanning_splits(2), paths=paths, dry_run=True)

    assert client.calls == []
    assert stats["projected_usd"] > 0
    assert stats["n_new_requests"] == stats["n_requested"]
    for path in (paths.cache, paths.batches, paths.train, paths.dev, paths.eval_pool, paths.stats):
        assert not path.exists(), path


def test_orchestrator_refuses_to_submit_over_the_cap(tmp_path, monkeypatch):
    monkeypatch.setattr("sxl.teacher.MAX_TEACHER_SPEND_USD", 0.0001)
    client = FakeOpenAI()

    with pytest.raises(SpendCapExceeded) as excinfo:
        label(
            "all",
            client=client,
            docs=docs_spanning_splits(2),
            paths=TeacherPaths.in_dir(tmp_path),
        )

    assert client.calls == []  # nothing uploaded, nothing created
    assert excinfo.value.cap == 0.0001
    assert excinfo.value.n_remaining == 6


def test_the_cap_ignores_spend_that_was_already_paid(tmp_path, monkeypatch):
    """A completed run must not be one flag away from tripping its own cap on a re-run."""
    docs = docs_spanning_splits(2)
    paths = TeacherPaths.in_dir(tmp_path)
    first = label("all", client=FakeOpenAI(), docs=docs, paths=paths, sleep=lambda _: None)
    assert first["spend_usd"] > 0

    monkeypatch.setattr("sxl.teacher.MAX_TEACHER_SPEND_USD", first["spend_usd"] / 2)
    client = FakeOpenAI()
    second = label("all", client=client, docs=docs, paths=paths, sleep=lambda _: None)

    assert client.calls == []
    assert second["spend_usd"] == 0.0
    assert second["spend_usd_cached"] == pytest.approx(first["spend_usd"])


def test_cli_exits_1_when_the_cap_would_be_crossed(tmp_path, monkeypatch):
    client = FakeOpenAI()
    paths = patch_cli(monkeypatch, tmp_path, client)
    write_docs(paths, docs_spanning_splits(2))
    monkeypatch.setattr("sxl.teacher.MAX_TEACHER_SPEND_USD", 0.0001)

    result = runner.invoke(app, ["teacher", "label", "--split", "all"])

    assert result.exit_code == 1
    assert "spend cap" in result.output
    assert client.calls == []


def test_cli_dry_run_prints_a_projection_and_calls_nothing(tmp_path, monkeypatch):
    client = FakeOpenAI()
    paths = patch_cli(monkeypatch, tmp_path, client)
    write_docs(paths, docs_spanning_splits(2))

    result = runner.invoke(app, ["teacher", "label", "--split", "all", "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "projected $" in result.output
    assert "no API calls made" in result.output
    assert client.calls == []


def test_cli_rejects_an_unpriced_model_before_doing_anything(tmp_path, monkeypatch):
    client = FakeOpenAI()
    patch_cli(monkeypatch, tmp_path, client)

    result = runner.invoke(app, ["teacher", "label", "--model", "gpt-9-imaginary"])

    assert result.exit_code == 1
    assert "no price for" in result.output
    assert client.calls == []


def test_cli_rejects_an_unknown_split_with_exit_1_not_2(tmp_path, monkeypatch):
    """Exit 2 means "not implemented" in this repo and must stay unambiguous."""
    patch_cli(monkeypatch, tmp_path, FakeOpenAI())

    result = runner.invoke(app, ["teacher", "label", "--split", "eval_gold"])

    assert result.exit_code == 1
    assert "unknown --split" in result.output
