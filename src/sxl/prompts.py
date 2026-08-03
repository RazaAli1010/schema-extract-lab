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
import random
from collections.abc import Iterable, Sequence

# Re-exported so callers can say "the prompt constants live in prompts.py" (SPEC
# §3.6) without a second copy of the values drifting from config.
from sxl.config import (
    MAX_INPUT_CHARS,
    MAX_NEW_TOKENS,
    N_FEWSHOT,
    SEED,
    TEMPERATURE,
)
from sxl.schema import JSON_SCHEMA

__all__ = [
    "FEWSHOT_IDS",
    "MAX_INPUT_CHARS",
    "MAX_NEW_TOKENS",
    "NULL_UNKNOWN_RULE",
    "N_FEWSHOT",
    "PROMPT_VERSION",
    "SCHEMA_BLOCK",
    "STUDENT_PROMPT_SHA",
    "SYSTEM_STUDENT",
    "SYSTEM_TEACHER",
    "TEACHER_PROMPT_SHA",
    "TEMPERATURE",
    "build_student_prompt",
    "build_teacher_prompt",
    "pick_fewshot",
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
- A posting that does not carry a date gets posting_date: null -- never
  today's date, never a guess from the writing style, never a date carried
  over from an earlier posting.
- required_skills contains only skills the posting actually names."""

#: F4 measured the teacher at posting_date precision 0.039 / recall 1.000 against
#: the gold set: it emitted the *same* invented date (2023-10-01) on 257 of 330
#: documents while missing none of the 10 real ones. That is a pure
#: false-positive failure, so the rule below is stated as its own block rather
#: than as one more bullet in `_COPY_DONT_INFER_RULE` -- the field needs a
#: default-to-null instruction, not just a don't-infer instruction.
_POSTING_DATE_RULE = """\
posting_date needs one more rule, because it is the field most often invented:

- Emit a date ONLY if the posting text literally contains the date it was
  posted. Point to the characters you are copying. If you cannot, emit null.
- Most postings carry no date at all. null is the expected answer, not the
  exceptional one. Do not fill this field to make the object look complete.
- A relative phrase ("posted 3 days ago", "reposted last week") is NOT a date:
  you have no reference point to resolve it against. Emit null.
- Deadlines, start dates, interview dates and copyright years are not the
  posting date. Emit null unless the text dates the posting itself.
- posting_date is not an enum. The string "unknown" is not a valid value --
  absence is null."""

_FORMAT_RULE = """\
Normalize these formats:

- location_country: ISO-3166 alpha-2, uppercase (e.g. "US", "GB", "DE").
- salary_currency: ISO-4217, uppercase (e.g. "USD", "EUR").
- posting_date: YYYY-MM-DD when the posting carries a date, null otherwise. A
  posting dated only "October 2023" is not a YYYY-MM-DD date; emit null rather
  than inventing a day.
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

{_POSTING_DATE_RULE}

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

#: The three exemplars of the `base_fewshot` arm, frozen at implementation time.
#:
#: Committed as a constant rather than re-selected per run because the few-shot
#: prompt is *part of the measured arm*: if a corpus rebuild reshuffled these, the
#: baseline number would change without a single line of code changing with it.
#: All three are drawn from `train.jsonl` -- a shot from `dev` or `eval_gold`
#: would be test-set contamination, and `tests/test_prompts.py` proves it isn't.
#:
#: Chosen by the `pick_fewshot` bootstrap branch below (seeded, deterministic) and
#: pasted back here. Between them they cover a populated salary, a non-`unknown`
#: `remote_mode` and a non-empty `required_skills`, per F5 §Scope-1: three
#: all-`null` exemplars would teach the model that `null` is the expected answer.
#: What the three demonstrate, in order: a fully-populated posting (salary,
#: onsite, bachelor, 7 years, 10 skills); a second populated one with a weekly
#: salary period and `education_level: "unknown"`; and one whose title, salary and
#: remote_mode are genuinely absent -- the third is not filler, it is the exemplar
#: that shows `null` and `"unknown"` are correct answers rather than failures.
FEWSHOT_IDS: tuple[str, ...] = ("jp_704dcd7afd", "jp_5b53acc701", "jp_0f665c2948")

#: Upper bound on exemplar length for the bootstrap selection. Not a quality
#: judgement -- three median-length postings already put the prompt around 5k
#: tokens, and at `GEN_BATCH_SIZE = 8` the KV cache is what decides whether the
#: arm runs on a 16 GB T4. The document under extraction is still truncated at
#: `MAX_INPUT_CHARS` exactly like every other arm (SPEC §3.6).
FEWSHOT_MAX_CHARS = 3500


#: Properties the three exemplars must cover *between them*. F5 §Scope-1 names the
#: first three; the rest are here for the same reason. A field the exemplars only
#: ever show as `null`/`"unknown"` is a field the model is being taught to leave
#: empty, and a baseline weakened that way is the most common way this genre of
#: project lies (F5 §Out-of-scope). This is exemplar coverage, not prompt tuning:
#: nothing here is chosen by looking at a score.
_COVERAGE = (
    ("salary", lambda g: g.get("salary_min") is not None or g.get("salary_max") is not None),
    ("remote_mode", lambda g: g.get("remote_mode") not in (None, "unknown")),
    ("required_skills", lambda g: bool(g.get("required_skills"))),
    ("education_level", lambda g: g.get("education_level") not in (None, "unknown")),
    ("title", lambda g: g.get("title") is not None),
    ("years_experience_min", lambda g: g.get("years_experience_min") is not None),
)


def _covers(row: dict) -> tuple[bool, ...]:
    """Which `_COVERAGE` properties one labeled row demonstrates."""
    g = row["gold"]
    return tuple(fn(g) for _, fn in _COVERAGE)


def pick_fewshot(train_rows: Iterable[dict], n: int = N_FEWSHOT) -> list[dict]:
    """Return `n` few-shot exemplars, in a fixed order.

    Two modes, deliberately:

    - **`FEWSHOT_IDS` populated (the normal path).** Look the frozen ids up in
      `train_rows` and return them in `FEWSHOT_IDS` order. A missing id raises:
      silently falling back to a fresh draw would change the baseline arm behind
      the measurement's back, which is the exact failure this constant exists to
      prevent.
    - **`FEWSHOT_IDS` empty (the bootstrap path).** Seeded, deterministic greedy
      selection used *once* to choose the ids that then get frozen above.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")

    if FEWSHOT_IDS:
        wanted = FEWSHOT_IDS[:n]
        by_id = {row["doc_id"]: row for row in train_rows if row["doc_id"] in wanted}
        missing = [doc_id for doc_id in wanted if doc_id not in by_id]
        if missing:
            raise ValueError(
                f"FEWSHOT_IDS not found in the given rows: {missing}. The few-shot arm is "
                "reproducible only because these ids are frozen -- re-select them and update "
                "prompts.FEWSHOT_IDS deliberately rather than letting the prompt drift."
            )
        return [by_id[doc_id] for doc_id in wanted]

    # Bootstrap. Seeded shuffle, then `n` greedy rounds each taking the row that
    # adds the most still-uncovered `_COVERAGE` properties. Ties break on shuffle
    # position, so the whole thing is a pure function of SEED and the input rows.
    pool = [row for row in train_rows if len(row["text"]) <= FEWSHOT_MAX_CHARS]
    random.Random(SEED).shuffle(pool)
    if len(pool) < n:
        raise ValueError(f"only {len(pool)} exemplars available, need {n}")

    picked: list[dict] = []
    taken: set[int] = set()
    covered = [False] * len(_COVERAGE)
    for _ in range(n):
        best_i, best_gain = None, -1
        for i, row in enumerate(pool):
            if i in taken:
                continue
            gain = sum(g and not c for g, c in zip(_covers(row), covered, strict=True))
            if gain > best_gain:
                best_i, best_gain = i, gain
        assert best_i is not None  # guarded by the len(pool) < n check above
        taken.add(best_i)
        picked.append(pool[best_i])
        covered = [c or g for c, g in zip(covered, _covers(pool[best_i]), strict=True)]
    return picked


_STUDENT_TASK_RULE = """\
Read the job posting and emit one JSON object holding exactly the 16 fields of \
the schema. Copy values from the posting; do not infer them from world knowledge."""

#: Deliberately shorter than `SYSTEM_TEACHER`. The teacher's system prompt carries
#: production labeling fixes (`_COPY_DONT_INFER_RULE`, `_POSTING_DATE_RULE`) that
#: were added *after* F4 measured the teacher's failures. Folding those into the
#: student's prompt would be tuning the baseline against the eval set -- F5's
#: "one honest schema-in-prompt few-shot setup, documented, and that is the
#: number". The schema and the null/unknown rule are shared because they are the
#: contract, not a tuning knob.
SYSTEM_STUDENT = f"""\
You extract structured data from job postings. You return one JSON object and \
nothing else.

{_STUDENT_TASK_RULE}

The object must conform to this JSON Schema:

{SCHEMA_BLOCK}

{NULL_UNKNOWN_RULE}

Emit only the JSON object. No prose, no explanation, no markdown code fences."""

#: Recorded in the `sxl gpu predict` run log so a prediction file can be traced
#: to the prompt that produced it. `FEWSHOT_IDS` is folded in because changing an
#: exemplar changes the arm just as surely as changing the system text does.
STUDENT_PROMPT_SHA = prompt_sha(PROMPT_VERSION, SYSTEM_STUDENT, str(MAX_INPUT_CHARS), *FEWSHOT_IDS)


def build_student_prompt(text: str, shots: Sequence[dict]) -> list[dict[str, str]]:
    """Return chat messages for one document: system, `shots` pairs, then the doc.

    Separate from `build_teacher_prompt` on purpose (F5 §Scope-2): the teacher
    prompt is labeling machinery, this one is part of a measured arm, and sharing
    a builder would make a change to either silently move the other's numbers.

    Assistant turns are compact JSON (`separators=(",", ":")`) -- the model is
    being shown the answer format, and pretty-printing would spend context on
    whitespace and invite pretty-printed, longer completions.
    """
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_STUDENT}]
    for shot in shots:
        messages.append({"role": "user", "content": truncate(shot["text"])})
        messages.append(
            {
                "role": "assistant",
                "content": json.dumps(shot["gold"], ensure_ascii=False, separators=(",", ":")),
            }
        )
    messages.append({"role": "user", "content": truncate(text)})
    return messages
