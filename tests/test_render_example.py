"""F6 §Scope-1: the training text, and its parity with the inference prompt.

Laptop only. The tokenizer is `_fakes.FakeQwenTokenizer`, which reproduces the one
property that makes this feature dangerous: `add_generation_prompt=True,
enable_thinking=False` emits an empty `<think></think>` block that an assistant
message inside the message list does not get. A stub without that asymmetry would
make every assertion here pass for the wrong reason.
"""

from __future__ import annotations

import json
import re

import pytest

from _fakes import DOC_TEXT, FakeQwenTokenizer, train_row
from sxl.config import MAX_INPUT_CHARS
from sxl.gpu.runner import render_prompt
from sxl.gpu.train_lora import TrainError, render_example
from sxl.prompts import SCHEMA_BLOCK, build_ft_prompt
from sxl.schema import FIELD_NAMES

TOK = FakeQwenTokenizer()
ROW = train_row(0, title="Senior Python Engineer", required_skills=["python", "sql"])


def _target(example: str) -> str:
    """The JSON the model is trained to emit: everything after the prefix, less EOS."""
    prefix = render_prompt(TOK, build_ft_prompt(ROW["text"]))
    return example[len(prefix) :].removesuffix(TOK.eos_token)


def test_the_training_prefix_is_byte_identical_to_the_inference_prompt():
    """The F6 acceptance criterion, and the reason `render_example` concatenates.

    If this ever fails, the model is being trained to answer at a position it
    never occupies at inference — a mismatch whose only symptom is a bad arm.
    """
    example = render_example(ROW, TOK)
    assert example.startswith(render_prompt(TOK, build_ft_prompt(ROW["text"])))


def test_the_example_ends_with_the_json_target_and_exactly_one_extra_eos():
    example = render_example(ROW, TOK)
    prefix = render_prompt(TOK, build_ft_prompt(ROW["text"]))

    assert example.endswith(TOK.eos_token)
    assert json.loads(_target(example))["title"] == "Senior Python Engineer"
    # One EOS more than the prompt: the stop signal the model must learn to emit.
    assert example.count(TOK.eos_token) == prefix.count(TOK.eos_token) + 1


def test_no_thinking_content_appears_anywhere():
    """SPEC §3.7. An *empty* think block is legitimate — thinking content is not.

    The spec's literal wording is "contains no `<think>`", which the real Qwen3
    template makes unsatisfiable: `enable_thinking=False` still emits an empty
    block. What actually matters is that no reasoning text is being trained on,
    and that the target itself is pure JSON.
    """
    example = render_example(ROW, TOK)

    assert all(
        span.strip() == "" for span in re.findall(r"<think>(.*?)</think>", example, re.DOTALL)
    )
    assert "<think>" not in _target(example)


def test_target_keys_are_in_field_names_order_not_alphabetical():
    """The model learns one fixed key order, which is free accuracy (SPEC §3.2)."""
    keys = list(json.loads(_target(render_example(ROW, TOK))))

    assert keys == list(FIELD_NAMES)
    # Guards the guard: if FIELD_NAMES were already sorted, the check above would
    # pass under `sort_keys=True` and prove nothing.
    assert keys != sorted(FIELD_NAMES)


def test_the_target_is_compact_json():
    target = _target(render_example(ROW, TOK))

    assert ", " not in target
    assert '": ' not in target


def test_the_prompt_carries_no_schema_and_no_exemplars():
    """The mechanism behind the cost claim: the schema lives in the weights now."""
    example = render_example(ROW, TOK)

    assert SCHEMA_BLOCK not in example
    assert '"$defs"' not in example
    assert example.count("<|im_start|>user") == 1
    assert example.count("<|im_start|>assistant") == 1


def test_document_text_is_truncated_at_max_input_chars():
    """SPEC §3.6 requires the identical truncation in every arm."""
    long_row = train_row(1, text="x" * (MAX_INPUT_CHARS + 500))
    example = render_example(long_row, TOK)

    assert "x" * MAX_INPUT_CHARS in example
    assert "x" * (MAX_INPUT_CHARS + 1) not in example


def test_render_example_never_enables_thinking():
    """`FakeQwenTokenizer` raises when `enable_thinking` is left at its default."""

    class ForgetfulTokenizer(FakeQwenTokenizer):
        def apply_chat_template(self, messages, **kw):
            kw.pop("enable_thinking", None)  # simulate the call site dropping it
            return super().apply_chat_template(messages, **kw)

    with pytest.raises(AssertionError, match="enable_thinking"):
        render_example(ROW, ForgetfulTokenizer())


def test_a_gold_object_with_drifted_keys_raises():
    drifted = train_row(2)
    drifted["gold"] = {"title": "x"}

    with pytest.raises(TrainError, match="FIELD_NAMES"):
        render_example(drifted, TOK)


def test_two_documents_render_to_different_examples():
    """A sanity check that the fixture is not accidentally constant."""
    other = train_row(3, text=DOC_TEXT + "\n\nRemote friendly.")

    assert render_example(ROW, TOK) != render_example(other, TOK)
