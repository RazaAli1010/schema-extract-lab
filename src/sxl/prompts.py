"""Prompt construction and the constants of SPEC §3.6.

Created by F2 for the teacher. **F5 appends its student prompts below the marker
at the bottom of this file and edits nothing above it** -- the teacher prompt is
production labeling machinery, the student prompt is part of a measured arm, and
conflating them would make the baseline arm untraceable.

`truncate` is shared infrastructure, not a teacher detail: SPEC §3.6 requires the
`MAX_INPUT_CHARS` truncation to be *identical* in every arm, and one function is
how that is enforced.
"""

from __future__ import annotations

import hashlib
import json

# Re-exported so callers can say "the prompt constants live in prompts.py" (SPEC
# §3.6) without a second copy of the values drifting from config.
from sxl.config import (
    MAX_INPUT_CHARS,
    MAX_NEW_TOKENS,
    N_FEWSHOT,
    TEMPERATURE,
)
from sxl.schema import JSON_SCHEMA

__all__ = [
    "MAX_INPUT_CHARS",
    "MAX_NEW_TOKENS",
    "NULL_UNKNOWN_RULE",
    "N_FEWSHOT",
    "PROMPT_VERSION",
    "SCHEMA_BLOCK",
    "SYSTEM_TEACHER",
    "TEACHER_PROMPT_SHA",
    "TEMPERATURE",
    "build_teacher_prompt",
    "prompt_sha",
    "truncate",
]

#: Bump to force a re-label after a change the hash cannot see (e.g. a decision
#: to change `max_tokens`). Changes to the schema or the system text below are
#: picked up automatically -- see `TEACHER_PROMPT_SHA`.
PROMPT_VERSION = "v1"


# --- shared helpers (used by every arm) --------------------------------------


def truncate(text: str) -> str:
    """Cut `text` to `MAX_INPUT_CHARS`. Applied identically in all arms (SPEC §3.6)."""
    return text[:MAX_INPUT_CHARS]


def prompt_sha(*parts: str) -> str:
    """sha256 of `parts` joined by NUL, first 16 hex characters.

    Variadic on purpose: the same function produces a per-request digest
    (`prompt_sha(system, user)`) and the stable, document-independent cache key
    (`TEACHER_PROMPT_SHA`). The NUL join means `("ab", "c")` and `("a", "bc")`
    hash differently.
    """
    h = hashlib.sha256()
    for i, part in enumerate(parts):
        if i:
            h.update(b"\x00")
        h.update(part.encode("utf-8"))
    return h.hexdigest()[:16]


# --- F2: the teacher prompt ---------------------------------------------------

#: The verbatim contract, `sort_keys=True` because key ordering is pydantic-version
#: noise while content is not. Matches `io.write_json` and `sxl schema dump`.
SCHEMA_BLOCK = json.dumps(JSON_SCHEMA, indent=2, sort_keys=True)

#: SPEC §3.2 calls this the single most common labeling error, so it is stated
#: explicitly rather than left implicit in the schema's enum lists.
NULL_UNKNOWN_RULE = """\
Absence is encoded differently for different field types, and getting this wrong
is the most common mistake:

- The five enum fields (employment_type, seniority, remote_mode, salary_period,
  education_level) are NEVER null. When the posting does not say, use the member
  "unknown".
- education_level has two distinct absence-like values. Use "none" when the
  posting states that no formal education is required. Use "unknown" when the
  posting is SILENT about education. These are different facts; do not conflate
  them.
- Every other field uses null when the posting does not say.
- required_skills is never null. Use [] when the posting lists no skills."""

_COPY_DONT_INFER_RULE = """\
Every value must be copied or normalized from the posting text. Never infer from
world knowledge:

- A posting that does not mention pay gets salary_min: null, salary_max: null and
  salary_currency: null -- not a market estimate for the role.
- A posting that does not name a city gets location_city: null -- not the
  company's headquarters.
- A posting that does not state a degree requirement gets
  education_level: "unknown" -- not the degree typical for the role.
- required_skills contains only skills the posting actually names."""

_FORMAT_RULE = """\
Normalize these formats:

- location_country: ISO-3166 alpha-2, uppercase (e.g. "US", "GB", "DE").
- salary_currency: ISO-4217, uppercase (e.g. "USD", "EUR").
- posting_date: YYYY-MM-DD.
- salary_min / salary_max: plain numbers, no currency symbols, no thousands
  separators. Keep the period the posting uses and record it in salary_period.
- required_skills: short lowercase skill names, deduplicated (e.g.
  ["python", "sql", "aws"]), not full sentences from the posting."""

SYSTEM_TEACHER = f"""\
You extract structured data from job postings. You return one JSON object and \
nothing else.

The object must conform to this JSON Schema:

{SCHEMA_BLOCK}

{NULL_UNKNOWN_RULE}

{_COPY_DONT_INFER_RULE}

{_FORMAT_RULE}

Emit only the JSON object. No prose, no explanation, no markdown code fences."""

#: The cache key (F2). Document-independent by construction: `doc_id` is already a
#: content hash (F1), so a document's text is immutable and only these three
#: inputs can change its prompt. Because `SYSTEM_TEACHER` embeds `SCHEMA_BLOCK`
#: verbatim, adding a 17th field or renaming an enum member invalidates every
#: cached label automatically -- that property is the whole point.
#:
#: The model is deliberately NOT folded in: cache rows record `teacher_model`
#: separately and the orchestrator filters on both fields independently.
TEACHER_PROMPT_SHA = prompt_sha(PROMPT_VERSION, SYSTEM_TEACHER, str(MAX_INPUT_CHARS))


def build_teacher_prompt(text: str) -> tuple[str, str]:
    """Return `(system, user)` for one document."""
    return SYSTEM_TEACHER, truncate(text)


# --- F5 appends its student prompts below this line. Nothing above is edited. --
