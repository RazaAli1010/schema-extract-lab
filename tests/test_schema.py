"""The schema contract. Every literal here is hand-copied from SPEC §3.2.

If a change to `sxl.schema` makes one of these fail, the *contract* changed —
that is the point of the test, not a reason to edit the expected values.
"""

from __future__ import annotations

import pytest

from sxl.schema import (
    ENUM_FIELDS,
    FIELD_NAMES,
    JSON_SCHEMA,
    NUMERIC_FIELDS,
    SET_FIELDS,
    EducationLevel,
    EmploymentType,
    JobPosting,
    RemoteMode,
    SalaryPeriod,
    Seniority,
    empty_posting,
    validate_prediction,
)

SPEC_FIELD_ORDER = (
    "title",
    "company",
    "employment_type",
    "seniority",
    "remote_mode",
    "location_city",
    "location_region",
    "location_country",
    "salary_min",
    "salary_max",
    "salary_currency",
    "salary_period",
    "years_experience_min",
    "education_level",
    "required_skills",
    "posting_date",
)


def test_field_names_match_spec_table_order():
    assert FIELD_NAMES == SPEC_FIELD_ORDER
    assert len(FIELD_NAMES) == 16
    assert tuple(JobPosting.model_fields) == SPEC_FIELD_ORDER


def test_json_schema_requires_all_16_and_forbids_extras():
    assert JSON_SCHEMA["additionalProperties"] is False
    assert JSON_SCHEMA["required"] == list(SPEC_FIELD_ORDER)
    assert len(JSON_SCHEMA["required"]) == 16
    assert set(JSON_SCHEMA["properties"]) == set(SPEC_FIELD_ORDER)


def test_enum_defs_survive_for_outlines():
    """F5 feeds JSON_SCHEMA to Outlines, which consumes the $defs enum entries."""
    defs = JSON_SCHEMA["$defs"]
    assert set(defs) == {
        "EmploymentType",
        "Seniority",
        "RemoteMode",
        "SalaryPeriod",
        "EducationLevel",
    }
    assert set(defs["RemoteMode"]["enum"]) == {"onsite", "hybrid", "remote", "unknown"}


@pytest.mark.parametrize(
    ("enum_cls", "expected"),
    [
        (
            EmploymentType,
            {"full_time", "part_time", "contract", "internship", "temporary", "unknown"},
        ),
        (Seniority, {"intern", "junior", "mid", "senior", "lead", "principal", "unknown"}),
        (RemoteMode, {"onsite", "hybrid", "remote", "unknown"}),
        (SalaryPeriod, {"hourly", "daily", "weekly", "monthly", "yearly", "unknown"}),
        (
            EducationLevel,
            {
                "none",
                "high_school",
                "associate",
                "bachelor",
                "master",
                "doctorate",
                "unknown",
            },
        ),
    ],
)
def test_enum_members_match_spec(enum_cls, expected):
    assert {m.value for m in enum_cls} == expected


def test_field_taxonomy():
    assert ENUM_FIELDS == frozenset(
        {"employment_type", "seniority", "remote_mode", "salary_period", "education_level"}
    )
    assert SET_FIELDS == frozenset({"required_skills"})
    assert NUMERIC_FIELDS == frozenset({"salary_min", "salary_max", "years_experience_min"})
    assert ENUM_FIELDS | SET_FIELDS | NUMERIC_FIELDS <= set(FIELD_NAMES)


def test_validate_prediction_accepts_empty_posting():
    payload = empty_posting().model_dump(mode="json")
    assert validate_prediction(payload) is not None


def _valid_payload() -> dict:
    payload = empty_posting().model_dump(mode="json")
    payload["salary_min"] = 120000.0
    return payload


def test_validate_prediction_accepts_a_populated_payload():
    assert validate_prediction(_valid_payload()) is not None


def test_validate_prediction_rejects_missing_key():
    payload = _valid_payload()
    del payload["posting_date"]
    assert validate_prediction(payload) is None


def test_validate_prediction_rejects_extra_key():
    payload = _valid_payload()
    payload["salary_equity"] = "0.1%"
    assert validate_prediction(payload) is None


def test_validate_prediction_rejects_unknown_enum_member():
    payload = _valid_payload()
    payload["seniority"] = "staff"  # not a Seniority member
    assert validate_prediction(payload) is None


def test_validate_prediction_rejects_null_enum():
    """SPEC §3.2: enum fields are never null — absence is the member "unknown"."""
    payload = _valid_payload()
    payload["employment_type"] = None
    assert validate_prediction(payload) is None


def test_validate_prediction_rejects_quoted_number():
    """Lax pydantic would coerce "120000" to 120000.0; a quoted number is invalid."""
    payload = _valid_payload()
    payload["salary_min"] = "120000"
    assert validate_prediction(payload) is None


def test_validate_prediction_accepts_json_int_for_a_float_field():
    """A JSON number without a decimal point is still a number."""
    payload = _valid_payload()
    payload["salary_min"] = 120000
    assert validate_prediction(payload) is not None


def test_validate_prediction_rejects_null_required_skills():
    """SPEC §3.2: required_skills uses [] for absence, never null."""
    payload = _valid_payload()
    payload["required_skills"] = None
    assert validate_prediction(payload) is None


def test_validate_prediction_never_raises_on_garbage():
    for garbage in (None, [], "not a dict", 42, {"nope": 1}):
        assert validate_prediction(garbage) is None
