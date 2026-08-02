"""Eval-set sampling and human verification (F3). Laptop-only, torch-free.

`data/gold/eval_gold.jsonl` is **the only file in this project that is not
machine-generated**, and its integrity is the project's integrity. Every arm --
including the `teacher` arm -- is scored against it, so without it "the student
is within N macro-F1 of the teacher" compares the student to its own supervisor
and is circular.

Three properties matter here:

1. **Resumability.** 330 documents is several hours of human work. Every state
   change appends one line to `_progress.jsonl` *before* the next document is
   printed, so a closed terminal or a `Ctrl-C` costs at most the document in
   progress. `KeyboardInterrupt` is a save, not a crash.
2. **Structural correctness.** An enum can only be set from a numbered menu of
   its exact members, so `"Senior"` vs `"senior"` and `none` vs `unknown` are
   impossible to get wrong by accident (SPEC §3.2). Every edit is re-validated
   and reverted if it would make the record invalid.
3. **Independence.** Split leakage raises rather than warns, and is checked twice
   -- once at sample time and once at finalize (SPEC §5.5).

No new dependencies: `input()` and bare ANSI escapes only. A review loop does not
need a TUI framework, and the 5 GB laptop disk budget is real (SPEC §2.1).
"""

from __future__ import annotations

import copy
import random
import re
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

from sxl.config import (
    DEV_PATH,
    EVAL_GOLD_PATH,
    EVAL_POOL_PATH,
    GOLD_CANDIDATES_PATH,
    GOLD_MAX_YEARS,
    GOLD_PROGRESS_EVERY,
    GOLD_PROGRESS_PATH,
    GOLD_STATS_PATH,
    GOLD_STRATA_BINS,
    N_EVAL_GOLD,
    N_GOLD_CANDIDATES,
    SEED,
    TRAIN_PATH,
    git_sha,
)
from sxl.config import utc_now as _now  # the one `generated_at` clock (SPEC §5.2)
from sxl.io import append_jsonl, read_jsonl, write_json, write_jsonl
from sxl.normalize import is_absent, norm_field, norm_skills, norm_str
from sxl.prompts import truncate
from sxl.schema import (
    ENUM_TYPES,
    FIELD_NAMES,
    NUMERIC_FIELDS,
    SET_FIELDS,
    validate_prediction,
)
from sxl.teacher import ROW_KEYS

#: Progress-log actions. `edit` carries field/old/new; the other two carry nulls.
ACTIONS = ("accept", "edit", "reject_doc")

#: Who performed the review, recorded verbatim into `eval_gold.label_source`.
#:
#: SPEC §3.3 assumes `"human"`, because the `teacher` arm exists to measure
#: teacher-vs-**human** disagreement (SPEC §3.6). `"model_verified"` records a
#: pass done by a language model instead. It is not an equivalent substitute and
#: must never be relabeled as one: against a model-verified eval set the teacher
#: arm measures model-vs-model agreement, and every other arm's macro-F1 is
#: measured against labels no person ever checked. F8 must say so in the README.
REVIEWERS = ("human", "model_verified")

#: Fields whose value should literally appear in the source document. A value the
#: teacher produced that is nowhere in the text is a likely hallucination, so it is
#: surfaced as a `low` hint for the reviewer -- never auto-corrected.
GROUNDED_FIELDS = frozenset(
    {
        "title",
        "company",
        "location_city",
        "location_region",
        "location_country",
        "salary_currency",
        "required_skills",
    }
)

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

#: Every prompt loop is capped (SPEC §5.4). After this many invalid answers the
#: edit is cancelled rather than looping forever -- which also keeps a scripted
#: test from hanging when its input script runs out.
MAX_PROMPT_ATTEMPTS = 5

#: Cap on how many offending ids a leakage error lists before summarizing.
_LEAK_SAMPLE = 10


# --- exceptions ---------------------------------------------------------------
# The library raises with the numbers attached; `cli.py` owns the exit codes,
# exactly as corpus.py and teacher.py do.


class GoldError(RuntimeError):
    """Any contract violation in the gold-set pipeline."""


class PoolTooSmall(GoldError):
    """`eval_pool` cannot supply the requested number of candidates."""


class LeakageDetected(GoldError):
    """A gold `doc_id` appears in `train.jsonl` or `dev.jsonl` (SPEC §3.4)."""


class NotEnoughVerified(GoldError):
    """Fewer than `N_EVAL_GOLD` documents survived review."""


# --- paths --------------------------------------------------------------------


@dataclass(frozen=True)
class GoldPaths:
    """Every file F3 touches, in one injectable object.

    Mirrors `teacher.TeacherPaths`: passing `GoldPaths.in_dir(tmp_path)` makes it
    structurally impossible for a test to write into the real `data/`.
    """

    eval_pool: Path
    train: Path
    dev: Path
    candidates: Path
    progress: Path
    gold: Path
    stats: Path

    @classmethod
    def default(cls) -> GoldPaths:
        return cls(
            eval_pool=EVAL_POOL_PATH,
            train=TRAIN_PATH,
            dev=DEV_PATH,
            candidates=GOLD_CANDIDATES_PATH,
            progress=GOLD_PROGRESS_PATH,
            gold=EVAL_GOLD_PATH,
            stats=GOLD_STATS_PATH,
        )

    @classmethod
    def in_dir(cls, root: Path) -> GoldPaths:
        root = Path(root)
        labeled, gold = root / "data" / "labeled", root / "data" / "gold"
        return cls(
            eval_pool=labeled / "eval_pool.jsonl",
            train=labeled / "train.jsonl",
            dev=labeled / "dev.jsonl",
            candidates=gold / "_candidates.jsonl",
            progress=gold / "_progress.jsonl",
            gold=gold / "eval_gold.jsonl",
            stats=root / "results" / "gold_stats.json",
        )


def _resolve(paths: GoldPaths | None) -> GoldPaths:
    return paths if paths is not None else GoldPaths.default()


# --- stratified sampling ------------------------------------------------------


def strata_labels() -> tuple[str, ...]:
    """The bin labels in fixed ascending order, e.g. `("400-1500", ...)`."""
    return tuple(f"{lo}-{hi}" for lo, hi in pairwise(GOLD_STRATA_BINS))


def stratum_of(char_len: int) -> str:
    """Bin `char_len` into one of `strata_labels()`.

    Values outside the outermost edges clamp into the first/last bin rather than
    raising: F1 already enforces `CORPUS_MIN_CHARS`/`CORPUS_MAX_CHARS`, and losing
    a candidate to an off-by-one edge would be worse than a slightly wide bin.
    """
    labels = strata_labels()
    for i, hi in enumerate(GOLD_STRATA_BINS[1:]):
        if char_len < hi:
            return labels[i]
    return labels[-1]


def _quotas(counts: Mapping[str, int], n: int) -> dict[str, int]:
    """Largest-remainder allocation of `n` across strata, proportional to `counts`.

    Proportional so the sample mirrors the pool's own length distribution;
    largest-remainder so the parts sum to exactly `n` instead of `n - 3` after
    rounding. A stratum can never be allocated more documents than it holds --
    the surplus is redistributed to the strata that still have room.
    """
    total = sum(counts.values())
    if total == 0:
        return dict.fromkeys(counts, 0)

    exact = {k: counts[k] * n / total for k in counts}
    quota = {k: min(int(v), counts[k]) for k, v in exact.items()}

    # Hand out the remaining seats by descending fractional part, then by label so
    # ties resolve deterministically. Loop is capped by `n`.
    for _ in range(n - sum(quota.values())):
        room = [k for k in counts if quota[k] < counts[k]]
        if not room:
            break
        pick = max(room, key=lambda k: (exact[k] - quota[k], k))
        quota[pick] += 1
    return quota


def sample_candidates(
    n: int = N_GOLD_CANDIDATES,
    seed: int = SEED,
    *,
    paths: GoldPaths | None = None,
    write: bool = True,
) -> list[dict[str, Any]]:
    """Draw `n` candidates from `eval_pool`, stratified by document length.

    `n` is 330 rather than 300: 10% slack for documents a reviewer rejects as
    unusable (not a job posting, wrong language, truncated scrape).

    An unstratified sample under-represents long postings, which is exactly where
    extraction fails -- a gold set of short easy documents would inflate every arm
    equally and hide the variance this project exists to measure.

    Deterministic: `random.Random(seed)` over `doc_id`-sorted strata, result sorted
    by `doc_id`. The same seed yields the same 330 ids in any process.
    """
    paths = _resolve(paths)
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")

    try:
        pool = list(read_jsonl(paths.eval_pool))
    except FileNotFoundError as exc:
        raise PoolTooSmall(
            f"{paths.eval_pool} does not exist — run `sxl teacher label --split eval_pool` first"
        ) from exc

    if len(pool) < n:
        raise PoolTooSmall(
            f"{paths.eval_pool} has {len(pool)} labeled rows, need >= {n} to sample "
            f"{n} candidates. Re-run `sxl teacher label --split eval_pool` (a short pool "
            "usually means documents landed in parse_failed / schema_invalid / api_error "
            "— see results/teacher_stats.json)."
        )

    by_stratum: dict[str, list[dict[str, Any]]] = {label: [] for label in strata_labels()}
    for row in pool:
        by_stratum[stratum_of(len(row["text"]))].append(row)
    for rows in by_stratum.values():
        rows.sort(key=lambda r: r["doc_id"])

    quota = _quotas({k: len(v) for k, v in by_stratum.items()}, n)
    rng = random.Random(seed)
    picked: list[dict[str, Any]] = []
    for label in strata_labels():  # fixed order, so the rng stream is reproducible
        picked.extend(rng.sample(by_stratum[label], quota[label]))
    picked.sort(key=lambda r: r["doc_id"])

    assert_no_leakage({r["doc_id"] for r in picked}, paths.train, paths.dev)

    if write:
        write_jsonl(paths.candidates, picked)
    return picked


def assert_no_leakage(gold_ids: Iterable[str], train_path: Path, dev_path: Path) -> None:
    """Raise `LeakageDetected` if any gold id appears in `train` or `dev` (SPEC §3.4).

    A *missing* train/dev file is skipped rather than treated as an error: F2 can
    be run one split at a time, and `splits.split_for` already makes the three
    buckets disjoint by hash. This check exists to catch the case where that
    invariant has been broken by hand -- the single mistake that would invalidate
    every number in the project -- so it is cheap and it runs twice.
    """
    gold = set(gold_ids)
    for path in (train_path, dev_path):
        if not Path(path).exists():
            continue
        overlap = sorted(gold & {row["doc_id"] for row in read_jsonl(path)})
        if overlap:
            shown = ", ".join(overlap[:_LEAK_SAMPLE])
            more = f" (+{len(overlap) - _LEAK_SAMPLE} more)" if len(overlap) > _LEAK_SAMPLE else ""
            raise LeakageDetected(
                f"{len(overlap)} gold doc_id(s) also appear in {path}: {shown}{more}. "
                "eval_gold must never overlap train/dev (SPEC §3.4)."
            )


# --- review hints -------------------------------------------------------------


def field_review_order(row: Mapping[str, Any]) -> list[str]:
    """The fixed SPEC §3.2 order. Reviewers read the same layout on every document.

    Kept as a function rather than inlining `FIELD_NAMES` so the review loop has a
    single place to change if a later domain reorders its fields.
    """
    del row  # order is domain-level, not row-level
    return list(FIELD_NAMES)


#: A posting writes `$150,000`, not `USD 150,000`, so an ISO-4217 code is normally
#: *inferred* from a symbol rather than copied. Without this the currency hint fires
#: on roughly a third of all documents, and a hint that cries wolf is worse than no
#: hint -- reviewer fatigue is the main quality risk on a 330-item task.
CURRENCY_SYMBOLS = {
    "USD": ("$", "us$", "usd"),
    "EUR": ("€", "eur"),
    "GBP": ("£", "gbp"),
    "JPY": ("¥", "jpy"),
    "INR": ("₹", "inr", "rs."),
    "CAD": ("c$", "cad", "$"),
    "AUD": ("a$", "aud", "$"),
}


def _grounded(name: str, value: str, text: str) -> bool:
    """Does `value` actually appear in the (normalized, lowercased) document text?"""
    if name == "salary_currency":
        code = value.upper()
        return (
            any(token in text for token in CURRENCY_SYMBOLS.get(code, ())) or code.lower() in text
        )
    return value.lower() in text


def confidence_hints(row: Mapping[str, Any]) -> dict[str, str]:
    """Per-field `"low"` / `"normal"`, to direct human attention (F3 Scope 3).

    A *hint for the reviewer*, never an automatic edit. F3 corrects nothing on its
    own; a `low` mark only means "look at this one first".
    """
    gold = row["gold"]
    text = norm_str(row.get("text")) or ""
    hints = dict.fromkeys(FIELD_NAMES, "normal")

    for name in GROUNDED_FIELDS:
        value = gold.get(name)
        if is_absent(name, value):
            continue
        normalized = norm_field(name, value)
        # Lowercase both sides: norm_country/norm_currency uppercase their output,
        # which would never substring-match the lowercased document text.
        wanted = normalized if isinstance(normalized, frozenset) else {str(normalized)}
        if any(not _grounded(name, w, text) for w in wanted):
            hints[name] = "low"

    lo, hi = (
        norm_field("salary_min", gold.get("salary_min")),
        norm_field("salary_max", gold.get("salary_max")),
    )
    if lo is not None and hi is not None and lo > hi:
        hints["salary_min"] = hints["salary_max"] = "low"

    years = norm_field("years_experience_min", gold.get("years_experience_min"))
    if years is not None and years > GOLD_MAX_YEARS:
        hints["years_experience_min"] = "low"

    date = gold.get("posting_date")
    if date is not None and not _DATE_RE.match(str(date)):
        hints["posting_date"] = "low"

    return hints


# --- progress log -------------------------------------------------------------


@dataclass
class DocState:
    """What the progress log says has happened to one candidate."""

    status: str = "unreviewed"  # unreviewed | accepted | rejected
    verified_at: str | None = None
    edited_fields: set[str] = field(default_factory=set)
    reviewed_at: str | None = None  # accept *or* reject, for the pace estimate

    @property
    def reviewed(self) -> bool:
        return self.status != "unreviewed"


def progress_line(
    doc_id: str,
    action: str,
    *,
    at: str,
    field_: str | None = None,
    old: Any = None,
    new: Any = None,
) -> dict[str, Any]:
    """One append-only log line (F3 Context deltas). Key order is the contract."""
    if action not in ACTIONS:
        raise ValueError(f"unknown action {action!r}; expected one of {' | '.join(ACTIONS)}")
    return {"doc_id": doc_id, "field": field_, "old": old, "new": new, "at": at, "action": action}


def load_progress(*, paths: GoldPaths | None = None) -> list[dict[str, Any]]:
    """Every logged line, in order. Missing file means "nothing reviewed yet"."""
    paths = _resolve(paths)
    if not paths.progress.exists():
        return []
    return list(read_jsonl(paths.progress))


def replay(
    candidates: Sequence[Mapping[str, Any]], progress: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, DocState]]:
    """Apply the log to the candidates. Pure -- the source of truth is the log.

    Lines are applied in file order, so re-reviewing a document (via `b`) simply
    overwrites the earlier decision and refreshes its `verified_at`.
    """
    rows = [copy.deepcopy(dict(c)) for c in candidates]
    by_id = {r["doc_id"]: r for r in rows}
    state = {r["doc_id"]: DocState() for r in rows}

    for line in progress:
        doc_id = line["doc_id"]
        row, st = by_id.get(doc_id), state.get(doc_id)
        if row is None or st is None:  # a line for a document no longer sampled
            continue
        action = line["action"]
        if action == "edit":
            row["gold"][line["field"]] = line["new"]
            st.edited_fields.add(line["field"])
        elif action == "accept":
            st.status, st.verified_at, st.reviewed_at = "accepted", line["at"], line["at"]
        elif action == "reject_doc":
            st.status, st.verified_at, st.reviewed_at = "rejected", None, line["at"]

    return rows, state


def _counts(state: Mapping[str, DocState]) -> dict[str, int]:
    accepted = [s for s in state.values() if s.status == "accepted"]
    return {
        "n_candidates": len(state),
        "n_reviewed": sum(1 for s in state.values() if s.reviewed),
        "n_accepted": len(accepted),
        "n_accepted_as_is": sum(1 for s in accepted if not s.edited_fields),
        "n_edited": sum(1 for s in accepted if s.edited_fields),
        "n_rejected": sum(1 for s in state.values() if s.status == "rejected"),
    }


def _parse_at(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    except ValueError:
        return None


def median_seconds_per_doc(progress: Sequence[Mapping[str, Any]]) -> float:
    """Median gap between consecutive reviewed documents.

    The median, not the mean: a reviewer who stops overnight leaves one 40,000-second
    gap that would wreck a mean and leaves the median untouched.
    """
    stamps = [
        t
        for t in (_parse_at(line.get("at")) for line in progress if line["action"] != "edit")
        if t is not None
    ]
    deltas = sorted((b - a).total_seconds() for a, b in pairwise(stamps))
    if not deltas:
        return 0.0
    mid = len(deltas) // 2
    value = deltas[mid] if len(deltas) % 2 else (deltas[mid - 1] + deltas[mid]) / 2
    return round(value, 1)


# --- terminal rendering -------------------------------------------------------

_ANSI = {"dim": "\x1b[2m", "bold": "\x1b[1m", "warn": "\x1b[33m", "off": "\x1b[0m"}


def _paint(enabled: bool) -> Callable[[str, str], str]:
    if not enabled:
        return lambda _style, text: text
    return lambda style, text: f"{_ANSI[style]}{text}{_ANSI['off']}"


def render_value(name: str, value: Any) -> str:
    """`null` and `unknown` must read as visibly different tokens (SPEC §3.2)."""
    if value is None:
        return "null"
    if name in SET_FIELDS:
        return ", ".join(value) if value else "[]"
    return str(value)


def render_document(
    row: Mapping[str, Any], *, index: int, total: int, full_text: bool, color: bool
) -> str:
    paint = _paint(color)
    text = row["text"] if full_text else truncate(row["text"])
    hidden = len(row["text"]) - len(text)
    more = "" if not hidden else paint("dim", f"\n[+more] {hidden} more chars - press `m`")

    hints = confidence_hints(row)
    lines = []
    for i, name in enumerate(field_review_order(row), 1):
        mark = paint("warn", " ! ") if hints[name] == "low" else "   "
        shown = render_value(name, row["gold"][name])
        if row["gold"][name] is None or shown == "unknown":
            shown = paint("dim", shown)
        lines.append(f"{i:>3}{mark}{name:<22}{shown}")

    header = paint(
        "bold",
        f"[{index + 1}/{total}] {row['doc_id']} | {len(row['text'])} chars "
        f"| stratum {stratum_of(len(row['text']))}",
    )
    # ASCII chrome only: a Windows console defaults to cp1252, where a box-drawing
    # rule raises UnicodeEncodeError and takes the whole review session with it.
    rule = "-" * 78
    return "\n".join(
        [
            "",
            header,
            rule,
            text,
            more,
            rule,
            *lines,
            rule,
            paint(
                "dim",
                "Enter accept | e <field> edit | x reject | b back | m more | s save+quit | ?",
            ),
        ]
    )


HELP = """\
  Enter        accept every field as shown and move to the next document
  e <field>    edit one field, by name (e seniority) or number (e 4)
  x            reject this document as unusable (not a posting, wrong language,
               truncated scrape). Rejected documents never reach eval_gold.
  b            go back one document to re-review it
  m            toggle the full document text
  s            save and quit - every decision so far is already on disk
  ?            this help

Enum fields are edited from a numbered menu of their exact members, so `none`
(the posting says no degree is needed) can never be confused with `unknown`
(the posting is silent). Nullable fields take an empty line to mean `null`.
`required_skills` takes a comma-separated list."""


# --- editing ------------------------------------------------------------------


def _prompt_enum(name: str, read_line: Callable[[str], str], echo: Callable[[str], None]) -> Any:
    """Numbered menu of the field's exact members. Free text is refused.

    Returns the chosen member string, or `None` to cancel. There is deliberately
    **no empty option**: an enum field is never `null` (SPEC §3.2).
    """
    members = [m.value for m in ENUM_TYPES[name]]
    for i, member in enumerate(members, 1):
        echo(f"    {i:>2}) {member}")
    for _ in range(MAX_PROMPT_ATTEMPTS):
        answer = read_line(f"  {name} [1-{len(members)}, or `c` to cancel]: ").strip()
        if answer.lower() == "c":
            return None
        if answer.isdigit() and 1 <= int(answer) <= len(members):
            return members[int(answer) - 1]
        echo(f"    not a menu number: {answer!r} — enums are chosen from the list, not typed")
    echo("    too many invalid answers; edit cancelled")
    return None


def _prompt_value(name: str, read_line: Callable[[str], str], echo: Callable[[str], None]) -> Any:
    """Read a non-enum field. Returns `(value)` or the sentinel `_CANCEL`."""
    if name in SET_FIELDS:
        raw = read_line(f"  {name} [comma-separated, empty = none]: ").strip()
        return sorted(norm_skills([s for s in raw.split(",")]))

    if name in NUMERIC_FIELDS:
        cast: Callable[[str], Any] = int if name == "years_experience_min" else float
        for _ in range(MAX_PROMPT_ATTEMPTS):
            raw = read_line(f"  {name} [number, empty = null]: ").strip()
            if not raw:
                return None
            try:
                return cast(raw)
            except ValueError:
                echo(f"    not a {cast.__name__}: {raw!r}")
        echo("    too many invalid answers; edit cancelled")
        return _CANCEL

    raw = read_line(f"  {name} [text, empty = null]: ").strip()
    return raw or None


class _Cancel:
    """Sentinel: the reviewer backed out of an edit. `None` means `null`."""


_CANCEL = _Cancel()


def _resolve_field(token: str) -> str | None:
    """Accept a field name or a 1-based index from the rendered list."""
    token = token.strip()
    if token.isdigit() and 1 <= int(token) <= len(FIELD_NAMES):
        return FIELD_NAMES[int(token) - 1]
    return token if token in FIELD_NAMES else None


# --- the review loop ----------------------------------------------------------


def _console_echo() -> Callable[[str], None]:
    """`print`, but a character the console cannot encode never kills the session.

    Real postings carry curly quotes, en dashes and emoji, and a Windows console
    defaults to cp1252 -- so a plain `print` raises `UnicodeEncodeError` partway
    through a document and takes hours of review with it. Unencodable characters
    are replaced; the reviewer sees `?` for a glyph instead of a stack trace.
    """
    try:
        sys.stdout.reconfigure(errors="replace")  # keeps the console's own encoding
    except (AttributeError, OSError, ValueError):  # pragma: no cover - exotic stdout
        pass

    def echo(text: str) -> None:
        try:
            print(text)
        except UnicodeEncodeError:  # pragma: no cover - reconfigure normally prevents this
            encoding = getattr(sys.stdout, "encoding", None) or "ascii"
            print(text.encode(encoding, "replace").decode(encoding))

    return echo


def review(
    *,
    paths: GoldPaths | None = None,
    read_line: Callable[[str], str] | None = None,
    echo: Callable[[str], None] | None = None,
    now: Callable[[], str] | None = None,
    color: bool | None = None,
) -> dict[str, Any]:
    """The terminal review loop. Resumes at the first unreviewed document.

    `read_line` / `echo` / `now` are injected so tests can script a whole session
    without monkeypatching builtins; the defaults are `input`, `print` and `_now`.
    """
    paths = _resolve(paths)
    read_line = read_line or input
    echo = echo or _console_echo()
    now = now or _now
    if color is None:
        color = sys.stdout.isatty()

    candidates = list(read_jsonl(paths.candidates))
    if not candidates:
        raise GoldError(f"{paths.candidates} is empty — run `sxl gold sample` first")
    candidates.sort(key=lambda r: r["doc_id"])

    rows, state = replay(candidates, load_progress(paths=paths))
    total = len(rows)

    def log(line: dict[str, Any]) -> None:
        """Append *before* the next document is drawn. `append_jsonl` opens and
        closes the file per line, so every decision is durable by construction."""
        append_jsonl(paths.progress, line)

    def report() -> None:
        c = _counts(state)
        median = median_seconds_per_doc(load_progress(paths=paths))
        left = total - c["n_reviewed"]
        eta = "unknown"
        if median > 0:
            secs = int(median * left)
            eta = f"{secs // 3600:02d}:{secs % 3600 // 60:02d}"
        echo(
            f"reviewed {c['n_reviewed']}/{total} · accepted-as-is {c['n_accepted_as_is']} "
            f"· edited {c['n_edited']} · rejected {c['n_rejected']} · est. remaining {eta}"
        )

    index = next((i for i, r in enumerate(rows) if not state[r["doc_id"]].reviewed), total)
    if index >= total:
        echo(f"all {total} candidates reviewed — run `sxl gold finalize`")
        report()
        return {**_counts(state), "quit_early": False}

    report()
    full_text = False
    quit_early = False
    since_report = 0

    try:
        while index < total:
            row = rows[index]
            doc_id = row["doc_id"]
            echo(render_document(row, index=index, total=total, full_text=full_text, color=color))

            command = read_line("> ").strip()

            if command == "":
                at = now()
                log(progress_line(doc_id, "accept", at=at))
                st = state[doc_id]
                st.status, st.verified_at, st.reviewed_at = "accepted", at, at
                index, full_text, since_report = index + 1, False, since_report + 1
            elif command in ("x", "X"):
                at = now()
                log(progress_line(doc_id, "reject_doc", at=at))
                st = state[doc_id]
                st.status, st.verified_at, st.reviewed_at = "rejected", None, at
                index, full_text, since_report = index + 1, False, since_report + 1
            elif command in ("s", "q"):
                quit_early = True
                break
            elif command in ("b", "B"):
                index, full_text = max(0, index - 1), False
            elif command in ("m", "M"):
                full_text = not full_text
            elif command in ("?", "h", "help"):
                echo(HELP)
            elif command == "e" or command.startswith("e "):
                _edit(row, command[1:], state[doc_id], read_line, echo, now, log)
            else:
                echo(f"unknown command {command!r} — `?` for help")

            if since_report >= GOLD_PROGRESS_EVERY:
                report()
                since_report = 0
    except KeyboardInterrupt:
        # A save, not a crash: everything up to the document in progress is on disk.
        echo("\ninterrupted — progress saved; re-run `sxl gold verify` to resume")
        quit_early = True

    report()
    return {**_counts(state), "quit_early": quit_early}


def _last_at(paths: GoldPaths) -> str | None:
    """The `at` of the line just written, so in-memory state matches the log."""
    lines = load_progress(paths=paths)
    return lines[-1]["at"] if lines else None


def _edit(
    row: dict[str, Any],
    token: str,
    st: DocState,
    read_line: Callable[[str], str],
    echo: Callable[[str], None],
    now: Callable[[], str],
    log: Callable[[dict[str, Any]], None],
) -> None:
    """Edit one field in place, or explain why nothing changed.

    The record is re-validated after every edit and reverted if it became invalid,
    so the loop can never advance past a broken record.
    """
    if not token.strip():
        token = read_line("  field [name or 1-16]: ")
    name = _resolve_field(token)
    if name is None:
        echo(f"    unknown field {token.strip()!r} — use a name or a number from the list")
        return

    old = row["gold"][name]
    new = (
        _prompt_enum(name, read_line, echo)
        if name in ENUM_TYPES
        else _prompt_value(name, read_line, echo)
    )
    if isinstance(new, _Cancel) or (name in ENUM_TYPES and new is None):
        echo("    unchanged")
        return
    if new == old:
        echo("    unchanged")
        return

    candidate = {**row["gold"], name: new}
    if validate_prediction(candidate) is None:
        echo(f"    rejected: {name}={new!r} makes the record fail the JobPosting schema")
        return

    row["gold"] = candidate
    st.edited_fields.add(name)
    log(progress_line(row["doc_id"], "edit", at=now(), field_=name, old=old, new=new))
    echo(f"    {name}: {render_value(name, old)} -> {render_value(name, new)}")


# --- finalize -----------------------------------------------------------------


def teacher_field_agreement(
    candidates: Sequence[Mapping[str, Any]], final: Sequence[Mapping[str, Any]]
) -> dict[str, float]:
    """Per-field rate at which the teacher's original label survived human review.

    This is **the teacher's own accuracy ceiling**, not a diagnostic (F3 Scope 7).
    A student within 3 macro-F1 of a teacher that is itself only 93% right on
    `seniority` is a very different claim from one made against a perfect oracle,
    and F8 quotes this number directly.

    Compared with `normalize.norm_field`, so it uses exactly the comparison logic
    F4's metrics use and the two numbers are commensurable.
    """
    original = {c["doc_id"]: c["gold"] for c in candidates}
    n = len(final)
    if not n:
        return dict.fromkeys(FIELD_NAMES, 0.0)
    return {
        name: round(
            sum(
                norm_field(name, original[r["doc_id"]][name]) == norm_field(name, r["gold"][name])
                for r in final
            )
            / n,
            4,
        )
        for name in FIELD_NAMES
    }


def finalize(
    *, paths: GoldPaths | None = None, n: int = N_EVAL_GOLD, reviewer: str = "human"
) -> dict[str, Any]:
    """Replay the log, drop rejects, and write exactly `n` verified rows.

    The first `n` by `doc_id` -- not the first `n` reviewed -- so the output is a
    pure function of (candidates, progress) and does not depend on review order.

    `reviewer` is written straight through to `label_source`, and
    `verified_by_human` is derived from it rather than hardcoded: an eval set that
    no person checked must not claim on disk that one did (see `REVIEWERS`).
    """
    paths = _resolve(paths)
    if reviewer not in REVIEWERS:
        raise GoldError(f"unknown reviewer {reviewer!r}; expected one of {' | '.join(REVIEWERS)}")
    candidates = sorted(read_jsonl(paths.candidates), key=lambda r: r["doc_id"])
    if not candidates:
        raise GoldError(f"{paths.candidates} is empty — run `sxl gold sample` first")

    progress = load_progress(paths=paths)
    rows, state = replay(candidates, progress)

    verified = [r for r in rows if state[r["doc_id"]].status == "accepted"]
    if len(verified) < n:
        raise NotEnoughVerified(
            f"only {len(verified)} of {len(candidates)} candidates are verified, need {n}. "
            f"{_counts(state)['n_rejected']} were rejected and "
            f"{len(candidates) - _counts(state)['n_reviewed']} are still unreviewed — "
            "run `sxl gold verify` to continue."
        )

    final = []
    for row in verified[:n]:
        st = state[row["doc_id"]]
        out = {
            "doc_id": row["doc_id"],
            "domain": row["domain"],
            "text": row["text"],
            "gold": {name: row["gold"][name] for name in FIELD_NAMES},
            "label_source": reviewer,
            # Retained so the audit trail records *what the reviewer was correcting*.
            "teacher_model": row["teacher_model"],
            "verified_by_human": reviewer == "human",
            "verified_at": st.verified_at,
        }
        if tuple(out) != ROW_KEYS:
            raise GoldError(f"row key order drifted from SPEC §3.3: {tuple(out)!r}")
        if validate_prediction(out["gold"]) is None:
            raise GoldError(f"{row['doc_id']}: verified gold does not validate against JobPosting")
        final.append(out)

    assert_no_leakage({r["doc_id"] for r in final}, paths.train, paths.dev)
    write_jsonl(paths.gold, final)

    kept = {r["doc_id"] for r in final}
    stats = {
        "reviewer": reviewer,
        "n_candidates": len(candidates),
        "n_rejected": _counts(state)["n_rejected"],
        "n_final": len(final),
        "n_docs_edited": sum(1 for r in final if state[r["doc_id"]].edited_fields),
        # Process metric: did a human touch this field? `teacher_field_agreement` is
        # the outcome metric -- they differ when an edit is reverted to its original
        # value, and both are worth having.
        "edit_rate_by_field": {
            name: round(sum(1 for i in kept if name in state[i].edited_fields) / len(final), 4)
            for name in FIELD_NAMES
        },
        "teacher_field_agreement": teacher_field_agreement(candidates, final),
        "median_seconds_per_doc": median_seconds_per_doc(progress),
        "char_len_bins": _char_len_bins(final),
        "generated_at": _now(),
        "git_sha": git_sha(),
    }
    write_json(paths.stats, stats)
    return stats


def _char_len_bins(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    bins = dict.fromkeys(strata_labels(), 0)
    for row in rows:
        bins[stratum_of(len(row["text"]))] += 1
    return bins


#: Below this, a field's teacher prompt is ambiguous enough that the 4,500
#: training rows are being poisoned by it (F3 Implementation notes). Cheaper to
#: catch here than in F8.
AGREEMENT_FLOOR = 0.75


def agreement_warnings(agreement: Mapping[str, float]) -> list[str]:
    return [
        f"teacher_field_agreement[{name}] = {value:.2f} < {AGREEMENT_FLOOR} — that field's "
        "teacher prompt is ambiguous; fix prompts.py and re-run F2 before training on it"
        for name, value in sorted(agreement.items())
        if value < AGREEMENT_FLOOR
    ]
