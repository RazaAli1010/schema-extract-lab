"""F6 §Scope-3/5: the guards that must fire before a GPU is ever touched.

Discovering split leakage or a malformed target *after* a three-hour Kaggle run is
the worst outcome available in this project, so every one of these assertions is
tested for the thing that actually matters: that it raises **early**.

Laptop only. Nothing here imports torch — which is itself the point of
`test_train_raises_before_a_trainer_is_constructed`.
"""

from __future__ import annotations

import pytest

from _fakes import FakeQwenTokenizer, train_row
from sxl.gpu import train_lora
from sxl.gpu.train_lora import (
    TrainError,
    assert_no_leakage,
    newest_checkpoint,
    resolve_resume,
    token_length_stats,
)
from sxl.io import write_jsonl

TOK = FakeQwenTokenizer()

TRAIN = [train_row(i) for i in range(6)]
DEV = [train_row(i) for i in range(10, 13)]
GOLD = [train_row(i) for i in range(20, 23)]


def write_splits(tmp_path, train=None, dev=None, gold=None):
    paths = {name: tmp_path / f"{name}.jsonl" for name in ("train", "dev", "gold")}
    write_jsonl(paths["train"], train if train is not None else TRAIN)
    write_jsonl(paths["dev"], dev if dev is not None else DEV)
    write_jsonl(paths["gold"], gold if gold is not None else GOLD)
    return paths


# --- assert_no_leakage --------------------------------------------------------


def test_a_clean_corpus_passes():
    assert_no_leakage(TRAIN, DEV, GOLD)  # the negative control: must not raise


def test_leakage_of_a_gold_doc_id_into_train_raises_and_names_it():
    planted = GOLD[0]["doc_id"]

    with pytest.raises(TrainError, match=planted):
        assert_no_leakage([*TRAIN, GOLD[0]], DEV, GOLD)


def test_leakage_into_dev_raises_and_names_the_split():
    planted = GOLD[1]["doc_id"]

    with pytest.raises(TrainError, match=f"in dev.*{planted}"):
        assert_no_leakage(TRAIN, [*DEV, GOLD[1]], GOLD)


def test_a_duplicate_train_doc_id_raises():
    with pytest.raises(TrainError, match="duplicate"):
        assert_no_leakage([*TRAIN, TRAIN[0]], DEV, GOLD)


def test_a_train_row_whose_gold_fails_the_schema_raises():
    """Training on malformed targets teaches the model to emit malformed JSON."""
    broken = train_row(7)
    broken["gold"]["employment_type"] = "freelance"  # not an EmploymentType member

    with pytest.raises(TrainError, match=broken["doc_id"]):
        assert_no_leakage([*TRAIN, broken], DEV, GOLD)


def test_a_dev_row_missing_a_field_raises():
    broken = train_row(8)
    del broken["gold"]["salary_period"]

    with pytest.raises(TrainError, match="dev"):
        assert_no_leakage(TRAIN, [*DEV, broken], GOLD)


# --- train() ordering ---------------------------------------------------------


def test_train_raises_before_a_trainer_is_constructed(tmp_path, monkeypatch):
    """The ordering guarantee: no GPU work happens on a contaminated corpus.

    Monkeypatching `_build_trainer` is what makes this observable — the call
    never happens, so the assertion is on an empty list rather than on an
    exception type that a later failure could also produce.
    """
    reached: list[object] = []

    def _boom(*a, **kw):
        reached.append(a)
        raise AssertionError("a trainer was constructed on a leaking corpus")

    monkeypatch.setattr(train_lora, "_build_trainer", _boom)
    paths = write_splits(tmp_path, train=[*TRAIN, GOLD[0]])

    with pytest.raises(TrainError, match=GOLD[0]["doc_id"]):
        train_lora.train(train_path=paths["train"], dev_path=paths["dev"], gold_path=paths["gold"])
    assert reached == []


def test_train_raises_before_a_trainer_is_constructed_on_an_invalid_target(tmp_path, monkeypatch):
    reached: list[object] = []
    monkeypatch.setattr(train_lora, "_build_trainer", lambda *a, **kw: reached.append(a) or None)

    broken = train_row(9)
    broken["gold"]["seniority"] = "wizard"
    paths = write_splits(tmp_path, train=[*TRAIN, broken])

    with pytest.raises(TrainError, match="fails the schema"):
        train_lora.train(train_path=paths["train"], dev_path=paths["dev"], gold_path=paths["gold"])
    assert reached == []


# --- checkpoint resolution ----------------------------------------------------


def test_newest_checkpoint_sorts_by_step_not_lexically(tmp_path):
    """Lexically, `checkpoint-900` beats `checkpoint-1000`. It is not newer."""
    for step in (90, 100, 1000, 900):
        (tmp_path / f"checkpoint-{step}").mkdir()
    (tmp_path / "not-a-checkpoint").mkdir()

    assert newest_checkpoint(tmp_path).name == "checkpoint-1000"


def test_newest_checkpoint_returns_none_for_an_empty_or_missing_directory(tmp_path):
    assert newest_checkpoint(tmp_path) is None
    assert newest_checkpoint(tmp_path / "nope") is None


def test_resolve_resume_prefers_a_live_checkpoint(tmp_path):
    live, mirror = tmp_path / "ckpt", tmp_path / "mirror"
    (live / "checkpoint-200").mkdir(parents=True)
    (mirror / "checkpoint-100").mkdir(parents=True)

    assert resolve_resume("auto", live, mirror).endswith("checkpoint-200")


def test_resolve_resume_auto_falls_back_to_the_mirror_and_stages_it(tmp_path):
    """The whole reason the mirror exists: /kaggle/tmp does not survive a session."""
    live, mirror = tmp_path / "ckpt", tmp_path / "mirror"
    (mirror / "checkpoint-300").mkdir(parents=True)
    (mirror / "checkpoint-300" / "adapter_model.safetensors").write_text("weights")

    resolved = resolve_resume("auto", live, mirror)

    # Staged into output_dir, because Trainer resumes relative to its own dir.
    assert resolved == str(live / "checkpoint-300")
    assert (live / "checkpoint-300" / "adapter_model.safetensors").read_text() == "weights"


def test_resolve_resume_returns_none_when_nothing_exists(tmp_path):
    assert resolve_resume("auto", tmp_path / "ckpt", tmp_path / "mirror") is None
    assert resolve_resume("", tmp_path) is None
    assert resolve_resume("none", tmp_path) is None


def test_resolve_resume_rejects_a_missing_explicit_path(tmp_path):
    with pytest.raises(TrainError, match="not a directory"):
        resolve_resume(str(tmp_path / "gone"), tmp_path)


# --- token_length_stats -------------------------------------------------------


def test_token_length_stats_is_deterministic_and_respects_the_sample_cap():
    rows = [train_row(i) for i in range(50)]

    first = token_length_stats(rows, TOK, sample=10)
    assert first == token_length_stats(rows, TOK, sample=10)
    assert first["n_sampled"] == 10
    assert 0 < first["p50"] <= first["p95"] <= first["max"]


def test_token_length_stats_handles_an_empty_corpus():
    assert token_length_stats([], TOK) == {"n_sampled": 0, "p50": 0, "p95": 0, "max": 0}


def test_a_longer_document_moves_the_maximum():
    short = [train_row(i, text="a b c") for i in range(4)]
    long = [*short, train_row(9, text=" ".join(["word"] * 500))]

    assert token_length_stats(long, TOK)["max"] > token_length_stats(short, TOK)["max"]
