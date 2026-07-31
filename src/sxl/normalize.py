"""Deterministic normalization applied before any comparison (SPEC §3.5).

Pure functions, no I/O. Both the gold value and the predicted value go through
the same call, so scoring never compares a raw string to a normalized one.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any

from sxl.schema import FIELD_NAMES, NUMERIC_FIELDS, SET_FIELDS

_WS = re.compile(r"\s+")

#: Case-insensitive marker for "the document does not say" (SPEC §3.5).
ABSENT_MARKER = "unknown"


def _unwrap(v: Any) -> Any:
    """Enum members compare as their `.value`."""
    return v.value if isinstance(v, Enum) else v


def norm_str(v: str | None) -> str | None:
    """Strip, collapse internal whitespace, lowercase.

    An empty or whitespace-only string normalizes to `None`: "present but empty"
    is not a real extraction, and treating it as absent keeps it from scoring as
    a false positive against an absent gold value.
    """
    v = _unwrap(v)
    if v is None:
        return None
    s = _WS.sub(" ", str(v).strip()).lower()
    return s or None


def norm_country(v: str | None) -> str | None:
    """`norm_str` then uppercase — ISO-3166 alpha-2 (SPEC §3.2)."""
    s = norm_str(v)
    return s.upper() if s is not None else None


def norm_currency(v: str | None) -> str | None:
    """`norm_str` then uppercase — ISO-4217 (SPEC §3.2)."""
    s = norm_str(v)
    return s.upper() if s is not None else None


def norm_skills(v: list[str] | None) -> frozenset[str]:
    """Normalize each element and compare as a set (SPEC §3.5). Never `None`."""
    if v is None:
        return frozenset()
    if isinstance(v, (str, bytes)):  # a bare string is not a list of skills
        v = [v]
    return frozenset(s for s in (norm_str(x) for x in v) if s is not None)


def norm_number(v: Any) -> float | None:
    """Cast to `float`; compared with `==` (money and years, not measurements)."""
    v = _unwrap(v)
    if v is None or isinstance(v, bool):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def norm_date(v: str | None) -> str | None:
    """`posting_date` is compared as the literal `YYYY-MM-DD` string."""
    v = _unwrap(v)
    if v is None:
        return None
    return str(v).strip() or None


_SPECIAL: dict[str, Any] = {
    "location_country": norm_country,
    "salary_currency": norm_currency,
    "posting_date": norm_date,
}


def norm_field(field: str, value: Any) -> Any:
    """Normalize `value` according to `field`'s rule.

    Raises `KeyError` for a name that is not one of the 16 — a silent
    pass-through would let a typo'd field score as perfect.
    """
    if field not in FIELD_NAMES:
        raise KeyError(f"unknown field: {field!r}")
    if field in SET_FIELDS:
        return norm_skills(value)
    if field in NUMERIC_FIELDS:
        return norm_number(value)
    fn = _SPECIAL.get(field)
    return fn(value) if fn is not None else norm_str(value)


def is_absent(field: str, value: Any) -> bool:
    """Is this value the absent marker for `field`?

    `None`, `"unknown"` (case-insensitive), and the empty skill set all mean
    absent (SPEC §3.5). Note `education_level == "none"` is *present* — it means
    the posting states no formal education is required.
    """
    normalized = norm_field(field, value)
    if normalized is None:
        return True
    if isinstance(normalized, frozenset):
        return not normalized
    if isinstance(normalized, str):
        return normalized.lower() == ABSENT_MARKER
    return False
