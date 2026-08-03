"""F5's student prompt: coverage, contamination, determinism.

Laptop-only. Nothing here imports torch or loads a model (SPEC §2.1).
"""

from __future__ import annotations

import json

import pytest

from _fakes import DOC_TEXT, doc, gold, labeled_row
from sxl.config import DEV_PATH, EVAL_GOLD_PATH, MAX_INPUT_CHARS, N_FEWSHOT, TRAIN_PATH
from sxl.io import read_jsonl
from sxl.prompts import (
    FEWSHOT_IDS,
    NULL_UNKNOWN_RULE,
    SYSTEM_STUDENT,
    build_student_prompt,
    build_teacher_prompt,
    pick_fewshot,
)
from sxl.schema import ENUM_TYPES, FIELD_NAMES

# A stand-in "train split" for the tests that must not depend on the real corpus
# being present. Shaped like `train.jsonl` rows, which is all `pick_fewshot` and
# `build_student_prompt` ever look at.
FAKE_TRAIN = [
    labeled_row(doc(0), title="Senior Python Engineer", salary_min=150000.0, remote_mode="hybrid"),
    labeled_row(doc(1), required_skills=["python", "sql"], education_level="bachelor"),
    labeled_row(doc(2), years_experience_min=6),
    labeled_row(doc(3)),
]


def _shots() -> list[dict]:
    """The real exemplars when the corpus is present, else the fake stand-ins."""
    if TRAIN_PATH.exists():
        return pick_fewshot(read_jsonl(TRAIN_PATH))
    return FAKE_TRAIN[:N_FEWSHOT]


# --- the rendered prompt ------------------------------------------------------


def test_system_prompt_names_every_field():
    for field in FIELD_NAMES:
        assert field in SYSTEM_STUDENT, field


def test_system_prompt_names_every_enum_member():
    for field, enum_type in ENUM_TYPES.items():
        for member in enum_type:
            assert f'"{member.value}"' in SYSTEM_STUDENT, f"{field}.{member.value}"


def test_system_prompt_states_the_none_vs_unknown_distinction():
    # The single most common labeling error (SPEC §3.2). Asserted on the sentence,
    # not just the words, so deleting the explanation fails the test.
    assert NULL_UNKNOWN_RULE in SYSTEM_STUDENT
    assert "no formal education is required" in SYSTEM_STUDENT
    assert "SILENT about education" in SYSTEM_STUDENT


def test_student_prompt_is_not_the_teacher_prompt():
    # SPEC/F5 §Scope-2: the teacher prompt is labeling machinery and carries
    # production fixes; the student's is part of a measured arm.
    teacher_system, _ = build_teacher_prompt(DOC_TEXT)
    assert SYSTEM_STUDENT != teacher_system


def test_message_roles_alternate_and_end_on_the_document():
    messages = build_student_prompt(DOC_TEXT, _shots())
    assert [m["role"] for m in messages] == (
        ["system"] + ["user", "assistant"] * N_FEWSHOT + ["user"]
    )
    assert messages[-1]["content"] == DOC_TEXT


def test_document_is_truncated_at_exactly_max_input_chars():
    messages = build_student_prompt("x" * (MAX_INPUT_CHARS + 500), _shots())
    assert len(messages[-1]["content"]) == MAX_INPUT_CHARS


def test_exemplar_text_is_truncated_too():
    shot = labeled_row(doc(0)) | {"text": "y" * (MAX_INPUT_CHARS + 500)}
    messages = build_student_prompt(DOC_TEXT, [shot])
    assert len(messages[1]["content"]) == MAX_INPUT_CHARS


def test_assistant_turns_are_compact_parseable_json_of_the_gold_object():
    shot = labeled_row(doc(0), title="Senior Python Engineer")
    messages = build_student_prompt(DOC_TEXT, [shot])
    answer = messages[2]["content"]
    assert ", " not in answer and '": ' not in answer  # compact separators
    assert json.loads(answer) == shot["gold"]


# --- exemplar selection -------------------------------------------------------


def test_pick_fewshot_is_deterministic():
    # Same rows in, same rows out — twice, and with the frozen-id path bypassed so
    # the seeded bootstrap branch is what is under test.
    from sxl import prompts

    original = prompts.FEWSHOT_IDS
    prompts.FEWSHOT_IDS = ()
    try:
        first = [r["doc_id"] for r in pick_fewshot(list(FAKE_TRAIN))]
        second = [r["doc_id"] for r in pick_fewshot(list(FAKE_TRAIN))]
    finally:
        prompts.FEWSHOT_IDS = original
    assert first == second
    assert len(first) == N_FEWSHOT


def test_pick_fewshot_raises_when_a_frozen_id_is_missing():
    # A corpus rebuild that drops an exemplar must fail loudly rather than
    # silently redrawing and moving the baseline.
    with pytest.raises(ValueError, match="FEWSHOT_IDS not found"):
        pick_fewshot(FAKE_TRAIN)


def test_fewshot_ids_are_frozen_and_the_right_count():
    assert len(FEWSHOT_IDS) == N_FEWSHOT
    assert len(set(FEWSHOT_IDS)) == N_FEWSHOT


@pytest.mark.skipif(not TRAIN_PATH.exists(), reason="corpus not built")
def test_fewshot_ids_all_come_from_train():
    train_ids = {row["doc_id"] for row in read_jsonl(TRAIN_PATH)}
    assert set(FEWSHOT_IDS) <= train_ids


@pytest.mark.parametrize("path", [DEV_PATH, EVAL_GOLD_PATH])
def test_fewshot_ids_never_touch_dev_or_eval_gold(path):
    # Drawing a shot from dev or eval_gold is test-set contamination and would
    # invalidate every number the baseline arm produces.
    if not path.exists():
        pytest.skip(f"{path.name} not built")
    held_out = {row["doc_id"] for row in read_jsonl(path)}
    assert set(FEWSHOT_IDS).isdisjoint(held_out)


@pytest.mark.skipif(not TRAIN_PATH.exists(), reason="corpus not built")
def test_the_three_exemplars_cover_salary_remote_mode_and_skills():
    # F5 §Scope-1: exemplars that are all `null` teach the model to answer `null`.
    shots = pick_fewshot(read_jsonl(TRAIN_PATH))
    golds = [s["gold"] for s in shots]
    assert any(g["salary_min"] is not None or g["salary_max"] is not None for g in golds)
    assert any(g["remote_mode"] != "unknown" for g in golds)
    assert any(g["required_skills"] for g in golds)


def test_build_student_prompt_accepts_a_minimal_gold_object():
    # The all-absent object is valid input; rendering must not depend on any
    # particular field being populated.
    shot = {"doc_id": "jp_x", "text": DOC_TEXT, "gold": gold()}
    messages = build_student_prompt(DOC_TEXT, [shot])
    assert json.loads(messages[2]["content"]) == gold()
