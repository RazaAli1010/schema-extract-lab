"""The contract module: `JobPosting` is the single source of truth for all 16 fields.

SPEC §3.2. Flat, 16 fields, no nesting — flatness is deliberate, the metric
contract (SPEC §3.5) depends on it.

**The null/unknown rule.** Enum fields are never `null`; absence is the member
`"unknown"`. `EducationLevel.none` ("the posting states no formal education is
required") is *not* `EducationLevel.unknown` ("the posting does not say") — these
are different and must not be conflated. Non-enum fields use `null` for absence.
`required_skills` uses `[]`.

**No field has a default.** pydantic drops a key from JSON Schema `required` if it
has *any* default, including `= None`. A prediction missing a key must be invalid,
not partially credited, so nullable fields are declared `X | None` with no default
and the import-time check at the bottom of this module enforces the result.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class EmploymentType(str, Enum):
    full_time = "full_time"
    part_time = "part_time"
    contract = "contract"
    internship = "internship"
    temporary = "temporary"
    unknown = "unknown"


class Seniority(str, Enum):
    intern = "intern"
    junior = "junior"
    mid = "mid"
    senior = "senior"
    lead = "lead"
    principal = "principal"
    unknown = "unknown"


class RemoteMode(str, Enum):
    onsite = "onsite"
    hybrid = "hybrid"
    remote = "remote"
    unknown = "unknown"


class SalaryPeriod(str, Enum):
    hourly = "hourly"
    daily = "daily"
    weekly = "weekly"
    monthly = "monthly"
    yearly = "yearly"
    unknown = "unknown"


class EducationLevel(str, Enum):
    none = "none"  # posting states no formal education is required
    high_school = "high_school"
    associate = "associate"
    bachelor = "bachelor"
    master = "master"
    doctorate = "doctorate"
    unknown = "unknown"  # posting is silent


class JobPosting(BaseModel):
    """The 16 fields of SPEC §3.2, in table order."""

    model_config = ConfigDict(extra="forbid")

    title: str | None
    company: str | None
    employment_type: EmploymentType
    seniority: Seniority
    remote_mode: RemoteMode
    location_city: str | None
    location_region: str | None
    location_country: str | None  # ISO-3166 alpha-2, uppercase
    # strict=True on the numerics: pydantic's default lax mode would silently
    # coerce "120000" into 120000.0, and a model that emits a quoted number has
    # not produced a schema-valid object.
    salary_min: float | None = Field(strict=True)
    salary_max: float | None = Field(strict=True)
    salary_currency: str | None  # ISO-4217, uppercase
    salary_period: SalaryPeriod
    years_experience_min: int | None = Field(strict=True)
    education_level: EducationLevel
    required_skills: list[str]  # may be [], never null
    posting_date: str | None  # YYYY-MM-DD


#: The 16 field names in SPEC §3.2 table order. Everything downstream iterates
#: this, never `JobPosting.model_fields`, so ordering is stable across pydantic
#: versions and refactors.
FIELD_NAMES: tuple[str, ...] = (
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

ENUM_FIELDS: frozenset[str] = frozenset(
    {"employment_type", "seniority", "remote_mode", "salary_period", "education_level"}
)
SET_FIELDS: frozenset[str] = frozenset({"required_skills"})
NUMERIC_FIELDS: frozenset[str] = frozenset({"salary_min", "salary_max", "years_experience_min"})

#: Enum class per enum field, so callers do not re-derive the mapping.
ENUM_TYPES: dict[str, type[Enum]] = {
    "employment_type": EmploymentType,
    "seniority": Seniority,
    "remote_mode": RemoteMode,
    "salary_period": SalaryPeriod,
    "education_level": EducationLevel,
}


def _build_json_schema() -> dict[str, Any]:
    """Draft 2020-12 schema with all 16 keys required and no extra properties.

    The `$defs` entries pydantic emits for the `str, Enum` subclasses are left
    intact — Outlines (F5) consumes them directly.
    """
    schema = JobPosting.model_json_schema()

    # The `required` trap: any default at all (including `= None`) would drop the
    # key here. Catch it at import time rather than at scoring time.
    produced = set(schema.get("required", ()))
    missing = [f for f in FIELD_NAMES if f not in produced]
    if missing:
        raise AssertionError(
            f"fields missing from JSON Schema `required`: {missing}. "
            "A field was given a default — remove it (see SPEC §3.2)."
        )

    schema["required"] = list(FIELD_NAMES)  # deterministic order
    schema["additionalProperties"] = False
    return schema


JSON_SCHEMA: dict[str, Any] = _build_json_schema()


def validate_prediction(obj: Any) -> JobPosting | None:
    """Return the validated model, or `None` for anything invalid. Never raises.

    This is the single function that defines `schema_valid` for the whole project
    (SPEC §3.5a).
    """
    try:
        return JobPosting.model_validate(obj)
    except (ValidationError, TypeError, ValueError):
        return None


def empty_posting() -> JobPosting:
    """All nullables `None`, all enums `"unknown"`, `required_skills` empty.

    Used as a test fixture and as the degenerate-baseline reference.
    """
    return JobPosting(
        title=None,
        company=None,
        employment_type=EmploymentType.unknown,
        seniority=Seniority.unknown,
        remote_mode=RemoteMode.unknown,
        location_city=None,
        location_region=None,
        location_country=None,
        salary_min=None,
        salary_max=None,
        salary_currency=None,
        salary_period=SalaryPeriod.unknown,
        years_experience_min=None,
        education_level=EducationLevel.unknown,
        required_skills=[],
        posting_date=None,
    )


# --- import-time contract checks (SPEC §3.2; F0 acceptance criteria) ----------
# Plain `raise`, not `assert`: these must survive `python -O`.
if len(FIELD_NAMES) != 16:
    raise AssertionError(f"FIELD_NAMES must hold 16 fields, got {len(FIELD_NAMES)}")
if tuple(JobPosting.model_fields) != FIELD_NAMES:
    raise AssertionError(
        "JobPosting field order drifted from FIELD_NAMES: "
        f"{tuple(JobPosting.model_fields)!r} != {FIELD_NAMES!r}"
    )
if len(JSON_SCHEMA["required"]) != 16:
    raise AssertionError(f"JSON_SCHEMA.required must hold 16 keys, got {JSON_SCHEMA['required']}")
if JSON_SCHEMA["additionalProperties"] is not False:
    raise AssertionError("JSON_SCHEMA.additionalProperties must be False")
