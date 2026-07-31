"""Normalization rules from SPEC §3.5."""

from __future__ import annotations

import pytest

from sxl.normalize import (
    is_absent,
    norm_country,
    norm_currency,
    norm_field,
    norm_number,
    norm_skills,
    norm_str,
)
from sxl.schema import ENUM_FIELDS, FIELD_NAMES


def test_norm_str_strips_collapses_and_lowercases():
    assert norm_str("  Senior   DATA Engineer ") == "senior data engineer"


def test_norm_str_passes_through_none_and_blanks():
    assert norm_str(None) is None
    assert norm_str("   ") is None


def test_norm_country_and_currency_uppercase():
    assert norm_country(" us ") == "US"
    assert norm_currency("usd") == "USD"
    assert norm_country(None) is None


def test_norm_skills_dedupes_as_a_set():
    assert norm_skills(["Python ", "python", "SQL"]) == frozenset({"python", "sql"})
    assert norm_skills(None) == frozenset()
    assert norm_skills([]) == frozenset()


def test_norm_number_casts_to_float():
    assert norm_number(3) == 3.0
    assert norm_number("120000") == 120000.0  # normalization is lenient; validation is not
    assert norm_number(None) is None
    assert norm_number("n/a") is None


def test_norm_field_dispatches_every_one_of_the_16():
    for field in FIELD_NAMES:
        norm_field(field, None)  # must not raise for any real field


def test_norm_field_rejects_an_unknown_name():
    with pytest.raises(KeyError):
        norm_field("nope", 1)


def test_norm_field_date_is_a_literal_string_compare():
    assert norm_field("posting_date", " 2026-07-30 ") == "2026-07-30"


def test_is_absent_for_none_on_every_field():
    for field in FIELD_NAMES:
        assert is_absent(field, None) is True


def test_is_absent_for_unknown_on_every_enum_field():
    for field in ENUM_FIELDS:
        assert is_absent(field, "unknown") is True


def test_is_absent_for_empty_skill_list():
    assert is_absent("required_skills", []) is True
    assert is_absent("required_skills", ["python"]) is False


def test_education_none_is_present_not_absent():
    """SPEC §3.2: "none" means the posting says no degree is required."""
    assert is_absent("education_level", "none") is False
    assert is_absent("education_level", "unknown") is True


def test_is_absent_for_populated_values():
    assert is_absent("title", "Data Engineer") is False
    assert is_absent("salary_min", 0.0) is False  # a real zero is not absence
