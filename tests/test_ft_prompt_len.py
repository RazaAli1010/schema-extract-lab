"""F6 acceptance: the `lora_ft` prompt is >=5x shorter than `base_fewshot`'s.

This is not a style check. The project's headline claim is that a fine-tune is
cheaper and faster than a prompted baseline, and the entire mechanism behind that
claim is prompt length: `base_fewshot` carries `SCHEMA_BLOCK` plus three exemplars
on every single call, and `lora_ft` carries one sentence. A future session that
"helpfully" adds the schema back to the fine-tuned prompt would erase the result
while every other test still passed.

Token-free by design — there is no tokenizer on the laptop (SPEC §2.1) — but
characters and tokens do not disagree by a factor of five.
"""

from __future__ import annotations

from _fakes import DOC_TEXT, FakeQwenTokenizer, train_row
from sxl.prompts import (
    FT_PROMPT_SHA,
    NULL_UNKNOWN_RULE,
    SCHEMA_BLOCK,
    STUDENT_PROMPT_SHA,
    SYSTEM_FT,
    build_ft_prompt,
    build_student_prompt,
)

#: The floor the acceptance criterion names. The real ratio is far larger; this
#: fails only if someone puts structure back into the fine-tuned prompt.
RATIO = 5

TOK = FakeQwenTokenizer()
SHOTS = [train_row(i) for i in range(3)]


def _joined(messages) -> str:
    return "".join(m["content"] for m in messages)


def test_the_ft_prompt_is_at_least_five_times_shorter():
    ft = _joined(build_ft_prompt(DOC_TEXT))
    fewshot = _joined(build_student_prompt(DOC_TEXT, SHOTS))

    assert len(fewshot) >= RATIO * len(ft), (
        f"ratio is only {len(fewshot) / len(ft):.1f}x — the fine-tuned arm's cost advantage "
        "comes from prompt length and nothing else"
    )


def test_the_same_holds_after_the_chat_template():
    """Template framing must not be what rescues a bloated system prompt."""
    ft = TOK.apply_chat_template(
        build_ft_prompt(DOC_TEXT),
        add_generation_prompt=True,
        enable_thinking=False,
    )
    fewshot = TOK.apply_chat_template(
        build_student_prompt(DOC_TEXT, SHOTS),
        add_generation_prompt=True,
        enable_thinking=False,
    )

    assert len(fewshot) >= RATIO * len(ft)


def test_the_ft_prompt_contains_no_schema_and_no_exemplar():
    messages = build_ft_prompt(DOC_TEXT)
    joined = _joined(messages)

    # Two turns, exactly: an exemplar would show up as an extra user/assistant
    # pair. Comparing against the shots' *text* would prove nothing here — they
    # share `DOC_TEXT` with the document under extraction.
    assert [m["role"] for m in messages] == ["system", "user"]
    assert SCHEMA_BLOCK not in joined
    assert NULL_UNKNOWN_RULE not in joined
    assert len(build_student_prompt(DOC_TEXT, SHOTS)) == len(messages) + 2 * len(SHOTS)


def test_the_ft_system_prompt_stays_short():
    """~30 tokens, per F6 §Scope-1. This is the constant that would drift."""
    assert len(SYSTEM_FT) < 400


def test_ft_prompt_sha_is_stable_and_distinct_from_the_student_sha():
    """Two arms, two prompts, two traceable hashes in the run log."""
    assert FT_PROMPT_SHA != STUDENT_PROMPT_SHA
    assert len(FT_PROMPT_SHA) == 16
    assert all(c in "0123456789abcdef" for c in FT_PROMPT_SHA)
