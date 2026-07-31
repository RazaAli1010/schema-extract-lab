"""The tolerant response parser (F2 scope §2).

The rule this file pins down is "drop, never guess": a document the teacher could
not label cleanly is excluded and counted, never repaired by regex into something
that looks like a label.

The non-dict cases matter more than they look. `json.loads` succeeds on
`"[1,2,3]"`, `"null"` and `"5"`, and without an `isinstance` guard each would sail
into `validate_prediction` and be counted as `schema_invalid` when the honest
bucket is `parse_failed` — quietly misattributing why the teacher failed.
"""

from __future__ import annotations

import json

import pytest

from sxl.teacher import extract_json

OBJ = {"title": "Senior Python Engineer", "company": "Acme", "salary_min": 150000.0}
BARE = json.dumps(OBJ)


def test_parses_a_bare_object():
    assert extract_json(BARE) == OBJ


def test_parses_an_object_with_surrounding_whitespace():
    assert extract_json(f"\n  {BARE}\n\n") == OBJ


def test_strips_json_code_fences():
    assert extract_json(f"```json\n{BARE}\n```") == OBJ


def test_strips_bare_code_fences():
    assert extract_json(f"```\n{BARE}\n```") == OBJ


def test_recovers_from_a_leading_sentence():
    assert extract_json(f"Here is the extraction:\n{BARE}") == OBJ


def test_recovers_from_a_trailing_sentence():
    assert extract_json(f"{BARE}\n\nLet me know if you need anything else.") == OBJ


def test_recovers_from_prose_on_both_sides():
    assert extract_json(f"Sure! {BARE} Hope that helps.") == OBJ


@pytest.mark.parametrize(
    "raw",
    [
        "sorry, I cannot",
        "I don't have enough information to extract anything.",
        '{"title": "x"',  # truncated — the finish_reason == "length" case
        '{"title": "x",}',  # trailing comma: malformed, and NOT repaired
        "{",
        "}",
    ],
)
def test_returns_none_for_unparseable_output(raw):
    assert extract_json(raw) is None


@pytest.mark.parametrize("raw", ["[1,2,3]", "null", "5", '"text"', "true"])
def test_returns_none_for_valid_json_that_is_not_an_object(raw):
    """A list or scalar is `parse_failed`, not `schema_invalid` (see the module docstring)."""
    assert extract_json(raw) is None


@pytest.mark.parametrize("raw", ["", "   \n\t ", None])
def test_returns_none_for_empty_and_none(raw):
    assert extract_json(raw) is None


def test_prefers_the_whole_string_over_the_brace_substring():
    """A nested object must not be shredded by the first-brace/last-brace fallback."""
    nested = json.dumps({"title": "x", "meta": {"a": 1}})
    assert extract_json(nested) == {"title": "x", "meta": {"a": 1}}


def test_an_object_containing_braces_in_a_string_survives():
    payload = json.dumps({"title": "Engineer {contract}"})
    assert extract_json(f"Here you go: {payload}") == {"title": "Engineer {contract}"}
