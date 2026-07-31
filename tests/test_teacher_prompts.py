"""The teacher prompt carries the whole labeling contract (SPEC §3.2, F2 scope §1).

`TEACHER_PROMPT_SHA` is the cache key for ~5,700 paid API calls. If it stops
changing when the schema changes, a stale prompt silently keeps producing labels
against a schema that has moved. If it starts varying per document, every re-run
re-bills. Both failure modes are asserted here.
"""

from __future__ import annotations

import string

from sxl.config import MAX_INPUT_CHARS
from sxl.prompts import (
    NULL_UNKNOWN_RULE,
    PROMPT_VERSION,
    SCHEMA_BLOCK,
    SYSTEM_TEACHER,
    TEACHER_PROMPT_SHA,
    build_teacher_prompt,
    prompt_sha,
    truncate,
)
from sxl.schema import ENUM_TYPES, FIELD_NAMES

SHORT = "Senior Python Engineer at Acme. Remote, US. $120,000-$150,000 per year."
LONG = "Job Description\n\n" + ("lorem ipsum dolor sit amet " * 2000)

assert len(LONG) > MAX_INPUT_CHARS, "the fixture must exceed the truncation window to exercise it"


def test_build_teacher_prompt_returns_system_and_user():
    system, user = build_teacher_prompt(SHORT)
    assert system == SYSTEM_TEACHER
    assert user == SHORT


def test_system_prompt_embeds_the_verbatim_json_schema():
    assert SCHEMA_BLOCK in SYSTEM_TEACHER
    for field in FIELD_NAMES:
        assert field in SYSTEM_TEACHER, field


def test_system_prompt_names_every_enum_member():
    """A member the teacher never sees is a member it can never emit."""
    for field, enum_type in ENUM_TYPES.items():
        for member in enum_type:
            assert f'"{member.value}"' in SYSTEM_TEACHER, f"{field}.{member.value}"


def test_system_prompt_states_the_none_vs_unknown_rule():
    """SPEC §3.2: `none` means "no degree required", `unknown` means "the posting is silent"."""
    assert NULL_UNKNOWN_RULE in SYSTEM_TEACHER
    assert "SILENT" in NULL_UNKNOWN_RULE
    assert '"none"' in NULL_UNKNOWN_RULE and '"unknown"' in NULL_UNKNOWN_RULE


def test_system_prompt_forbids_inference_from_world_knowledge():
    lowered = SYSTEM_TEACHER.lower()
    assert "never infer" in lowered
    assert "market estimate" in lowered


def test_system_prompt_forbids_prose_and_code_fences():
    assert "no markdown code fences" in SYSTEM_TEACHER.lower()


def test_user_prompt_is_truncated_at_max_input_chars():
    _, user = build_teacher_prompt(LONG)
    assert len(user) == MAX_INPUT_CHARS
    assert user == LONG[:MAX_INPUT_CHARS]


def test_short_text_is_not_truncated():
    assert truncate(SHORT) == SHORT


def test_prompt_sha_is_sixteen_hex_and_stable():
    sha = prompt_sha("a", "b")
    assert len(sha) == 16
    assert set(sha) <= set(string.hexdigits.lower())
    assert prompt_sha("a", "b") == sha


def test_prompt_sha_separates_its_parts():
    """The NUL join is what stops ("ab","c") colliding with ("a","bc")."""
    assert prompt_sha("ab", "c") != prompt_sha("a", "bc")
    assert prompt_sha("a", "b") != prompt_sha("b", "a")


def test_teacher_prompt_sha_does_not_vary_by_document():
    """The cache key must be document-independent, or every re-run re-bills."""
    assert isinstance(TEACHER_PROMPT_SHA, str)
    assert TEACHER_PROMPT_SHA == prompt_sha(PROMPT_VERSION, SYSTEM_TEACHER, str(MAX_INPUT_CHARS))


def test_changing_the_system_prompt_changes_the_sha():
    """The cache-invalidation contract: a new prompt must not reuse old labels."""
    assert prompt_sha(PROMPT_VERSION, SYSTEM_TEACHER + " extra", str(MAX_INPUT_CHARS)) != (
        TEACHER_PROMPT_SHA
    )


def test_changing_max_input_chars_changes_the_sha():
    """It is the only thing that alters the *user* half of the prompt."""
    assert (
        prompt_sha(PROMPT_VERSION, SYSTEM_TEACHER, str(MAX_INPUT_CHARS + 1)) != TEACHER_PROMPT_SHA
    )
