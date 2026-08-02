"""The metric contract: one formula, sixteen fields (SPEC §3.5).

Every arm and every report in this project uses these functions and only these.
The arithmetic is deliberately plain — integer counters summed across documents,
divided exactly once at the end — so this module needs **no numpy, scipy, sklearn
or pandas**. That is what keeps the laptop install under 100 MB (SPEC §2.1), and
`tests/test_no_torch_import.py` enforces it.

Two rules carry the whole feature, and both exist to stop a number from
flattering itself:

- **Invalid predictions are wrong, not excluded** (SPEC §3.5b). A prediction that
  did not parse or did not validate contributes zero true positives and counts as
  a miss on every field with a present gold value. It is never dropped from the
  denominator.
- **Missing predictions are counted, not skipped.** A predictions file that omits
  documents is a silent way to inflate every score, so omissions are synthesized
  as invalid and reported in `n_missing_predictions`.

EM and F1 deliberately disagree about one case: a document where gold and
prediction are *both* absent is a success for EM and invisible to F1. Reporting
both is what stops an arm that answers `null` to everything from looking good —
and `macro_f1_null_baseline` puts that arm's exact score in the output for
comparison.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sxl.config import (
    ARMS,
    DEV_PATH,
    EVAL_GOLD_PATH,
    EVAL_POOL_PATH,
    METRICS_DIR,
    PREDICTIONS_DIR,
    TRAIN_PATH,
    git_sha,
    utc_now,
)
from sxl.io import read_jsonl, write_json
from sxl.normalize import is_absent, norm_field
from sxl.schema import FIELD_NAMES, SET_FIELDS, empty_posting, validate_prediction

__all__ = [
    "Counts",
    "MetricsError",
    "MetricsPaths",
    "assert_no_leakage",
    "load_metrics",
    "score_arm",
    "score_arm_files",
    "score_field",
    "score_teacher_arm",
    "write_metrics",
]

#: Ratios are rounded here and **only** here — never in intermediate math.
_ROUND = 4

#: Offending ids listed in an integrity error before it truncates. Matches
#: `verify._LEAK_SAMPLE`, so the two error messages read the same way.
_ERROR_SAMPLE = 10

#: The key order of a `per_field` entry, part of the SPEC §3.3 output contract.
_FIELD_KEYS = ("em", "precision", "recall", "f1", "support")


class MetricsError(RuntimeError):
    """A contract violation that must stop scoring (SPEC §5.5).

    Subclasses `RuntimeError`, which the F4 spec names as the type raised for an
    extra prediction doc_id, an arm mismatch, and a `schema_valid`/`parsed`
    disagreement.
    """


# --- paths --------------------------------------------------------------------


@dataclass(frozen=True)
class MetricsPaths:
    """Every file F4 touches, in one injectable object.

    Mirrors `teacher.TeacherPaths` and `verify.GoldPaths`: passing
    `MetricsPaths.in_dir(tmp_path)` makes it structurally impossible for a test to
    read the real `data/` or write the real `results/`.
    """

    gold: Path
    eval_pool: Path
    train: Path
    dev: Path
    predictions: Path
    metrics: Path

    @classmethod
    def default(cls) -> MetricsPaths:
        return cls(
            gold=EVAL_GOLD_PATH,
            eval_pool=EVAL_POOL_PATH,
            train=TRAIN_PATH,
            dev=DEV_PATH,
            predictions=PREDICTIONS_DIR,
            metrics=METRICS_DIR,
        )

    @classmethod
    def in_dir(cls, root: Path) -> MetricsPaths:
        root = Path(root)
        labeled = root / "data" / "labeled"
        return cls(
            gold=root / "data" / "gold" / "eval_gold.jsonl",
            eval_pool=labeled / "eval_pool.jsonl",
            train=labeled / "train.jsonl",
            dev=labeled / "dev.jsonl",
            predictions=root / "artifacts" / "predictions",
            metrics=root / "results" / "metrics",
        )

    def pred_for(self, arm: str) -> Path:
        return self.predictions / f"{arm}.jsonl"

    def out_for(self, arm: str) -> Path:
        return self.metrics / f"{arm}.json"


def _resolve(paths: MetricsPaths | None) -> MetricsPaths:
    return paths if paths is not None else MetricsPaths.default()


# --- counters -----------------------------------------------------------------


@dataclass(frozen=True)
class Counts:
    """Per-field integer tallies, summed across the documents of a split.

    `support` is the number of documents whose **gold** value is present — the
    denominator that makes an F1 interpretable, since a field with support 4 out
    of 300 has an F1 that means almost nothing.

    The exception is `required_skills`: being a set field, its `support` counts
    gold skill *elements* across the whole split, not documents. A reader
    comparing `support: 1840` for skills against `support: 210` for `title` is
    looking at two different units, and F8 says so.
    """

    tp: int = 0
    fp: int = 0
    fn: int = 0
    em_hits: int = 0
    support: int = 0

    @classmethod
    def zero(cls) -> Counts:
        return cls()

    def __add__(self, other: Counts) -> Counts:
        if not isinstance(other, Counts):  # pragma: no cover - defensive
            return NotImplemented
        return Counts(
            self.tp + other.tp,
            self.fp + other.fp,
            self.fn + other.fn,
            self.em_hits + other.em_hits,
            self.support + other.support,
        )


def _ratio(num: int, den: int) -> float:
    """`num/den`, or `0.0` when the denominator is 0 — never `nan`, never raising."""
    return num / den if den else 0.0


def _prf(counts: Counts) -> tuple[float, float, float]:
    """Precision, recall, F1 — each `0.0` when its own denominator is 0 (SPEC §3.5c)."""
    precision = _ratio(counts.tp, counts.tp + counts.fp)
    recall = _ratio(counts.tp, counts.tp + counts.fn)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def _mean(values: Sequence[float]) -> float:
    """Arithmetic mean of the 16 per-field F1s — every field weighs the same (SPEC §3.5e)."""
    return sum(values) / len(values) if values else 0.0


# --- the single formula -------------------------------------------------------


def score_field(field: str, gold_val: Any, pred_val: Any, pred_valid: bool) -> Counts:
    """Score one field of one document (SPEC §3.5c/d). The whole metric lives here.

    |                        | gold absent          | gold present    |
    |------------------------|----------------------|-----------------|
    | pred absent            | — (ignored, EM hit)  | FN              |
    | pred present, matches  | n/a                  | TP              |
    | pred present, differs  | FP                   | FP **and** FN   |

    For `required_skills` the same table is applied per element:
    `TP += |P ∩ G|`, `FP += |P \\ G|`, `FN += |G \\ P|`.

    When `pred_valid` is False the prediction is treated as absent **and**
    `em_hits` is forced to 0 even if gold is also absent: an unparseable output is
    not credited with correctly omitting a field.
    """
    g = norm_field(field, gold_val)
    is_set = field in SET_FIELDS

    if not pred_valid:
        if is_set:
            return Counts(tp=0, fp=0, fn=len(g), em_hits=0, support=len(g))
        missing = 0 if is_absent(field, gold_val) else 1
        return Counts(tp=0, fp=0, fn=missing, em_hits=0, support=missing)

    p = norm_field(field, pred_val)

    if is_set:
        return Counts(
            tp=len(p & g),
            fp=len(p - g),
            fn=len(g - p),
            em_hits=int(p == g),  # both-empty is an exact match
            support=len(g),
        )

    g_abs, p_abs = is_absent(field, gold_val), is_absent(field, pred_val)
    if g_abs and p_abs:
        return Counts(em_hits=1)  # true negative: an EM hit, invisible to F1
    if g_abs:  # pred present, gold absent
        return Counts(fp=1)
    if p_abs:  # gold present, pred absent
        return Counts(fn=1, support=1)
    if p == g:
        return Counts(tp=1, em_hits=1, support=1)
    return Counts(fp=1, fn=1, support=1)


# --- integrity gates ----------------------------------------------------------


def assert_no_leakage(
    gold_rows: Iterable[Mapping[str, Any]], *, paths: MetricsPaths | None = None
) -> None:
    """Re-run F3's split-isolation check at scoring time (SPEC §3.4).

    F3 already guarantees this; F4 asserts it because this is the last gate before
    a number becomes a claim, and leakage is the single mistake that would
    invalidate every result in the project. Delegates to `verify.assert_no_leakage`
    rather than reimplementing it — one check, one error message.
    """
    from sxl.verify import assert_no_leakage as _verify_no_leakage

    p = _resolve(paths)
    _verify_no_leakage({row["doc_id"] for row in gold_rows}, p.train, p.dev)


def _sample(ids: Sequence[str]) -> str:
    shown = ", ".join(ids[:_ERROR_SAMPLE])
    more = f" (+{len(ids) - _ERROR_SAMPLE} more)" if len(ids) > _ERROR_SAMPLE else ""
    return f"{shown}{more}"


def _index(rows: Iterable[Mapping[str, Any]], what: str) -> dict[str, Mapping[str, Any]]:
    """`{doc_id: row}`, raising on a duplicate id rather than silently keeping one."""
    out: dict[str, Mapping[str, Any]] = {}
    dupes: list[str] = []
    for row in rows:
        doc_id = row["doc_id"]
        if doc_id in out:
            dupes.append(doc_id)
        out[doc_id] = row
    if dupes:
        raise MetricsError(f"{len(dupes)} duplicate doc_id(s) in {what}: {_sample(sorted(dupes))}")
    return out


def _check_prediction_rows(preds: Mapping[str, Mapping[str, Any]], arm: str) -> None:
    """Arm labels and `schema_valid`/`parsed` agreement, before any counting.

    The arm check catches the copy-paste error of scoring `lora_ft` predictions as
    `base_fewshot`. The revalidation catches an upstream feature writing
    `schema_valid` by hand instead of taking it from `validate_prediction`.
    """
    wrong_arm = sorted(d for d, r in preds.items() if r.get("arm") != arm)
    if wrong_arm:
        got = sorted({str(preds[d].get("arm")) for d in wrong_arm})
        raise MetricsError(
            f"{len(wrong_arm)} prediction row(s) are labeled {' / '.join(got)}, not {arm!r}: "
            f"{_sample(wrong_arm)}. Scoring one arm's predictions as another's is a "
            "contract violation, not a warning."
        )

    disagree = sorted(
        doc_id
        for doc_id, row in preds.items()
        if bool(row.get("schema_valid")) != (validate_prediction(row.get("parsed")) is not None)
    )
    if disagree:
        raise MetricsError(
            f"{len(disagree)} prediction row(s) disagree with `validate_prediction`: "
            f"{_sample(disagree)}. `schema_valid` must come from "
            "`schema.validate_prediction`, never be written by hand."
        )


# --- scoring an arm -----------------------------------------------------------


def score_arm(
    gold_rows: Sequence[Mapping[str, Any]],
    pred_rows: Sequence[Mapping[str, Any]],
    arm: str,
    *,
    split: str = "eval_gold",
    paths: MetricsPaths | None = None,
    check_leakage: bool = True,
) -> dict[str, Any]:
    """Score one arm against the gold split, emitting the SPEC §3.3 metrics dict.

    N is always the full gold split size: predictions are never dropped for being
    invalid (SPEC §3.5b) or for being absent from the predictions file.
    """
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {' | '.join(ARMS)}")

    gold = _index(gold_rows, "the gold file")
    preds = _index(pred_rows, f"{arm} predictions")

    extra = sorted(set(preds) - set(gold))
    if extra:
        raise MetricsError(
            f"{len(extra)} prediction doc_id(s) are not in the gold split: {_sample(extra)}. "
            "Scoring predictions for documents with no gold label is meaningless."
        )
    _check_prediction_rows(preds, arm)

    if check_leakage:
        assert_no_leakage(gold.values(), paths=paths)

    n = len(gold)
    n_missing = len(set(gold) - set(preds))
    n_valid = 0

    totals = {f: Counts.zero() for f in FIELD_NAMES}
    null_totals = {f: Counts.zero() for f in FIELD_NAMES}
    null_pred = empty_posting().model_dump(mode="json")

    for doc_id, gold_row in gold.items():
        gold_obj = gold_row["gold"]
        pred_row = preds.get(doc_id)
        # A missing prediction is an invalid one, not an excluded one.
        valid = bool(pred_row["schema_valid"]) if pred_row is not None else False
        parsed = pred_row.get("parsed") if pred_row is not None else None
        n_valid += int(valid)

        for f in FIELD_NAMES:
            gold_val = gold_obj.get(f)
            pred_val = parsed.get(f) if valid and parsed is not None else None
            totals[f] = totals[f] + score_field(f, gold_val, pred_val, valid)
            # The degenerate floor, scored through the identical formula.
            null_totals[f] = null_totals[f] + score_field(f, gold_val, null_pred[f], True)

    per_field: dict[str, dict[str, Any]] = {}
    f1s: list[float] = []
    for f in FIELD_NAMES:
        counts = totals[f]
        precision, recall, f1 = _prf(counts)
        f1s.append(f1)
        per_field[f] = {
            "em": round(_ratio(counts.em_hits, n), _ROUND),
            "precision": round(precision, _ROUND),
            "recall": round(recall, _ROUND),
            "f1": round(f1, _ROUND),
            "support": counts.support,
        }
        if tuple(per_field[f]) != _FIELD_KEYS:  # pragma: no cover - defensive
            raise MetricsError(f"per_field[{f!r}] key order drifted from {_FIELD_KEYS}")

    null_f1s = [_prf(null_totals[f])[2] for f in FIELD_NAMES]

    out = {
        "arm": arm,
        "split": split,
        "n": n,
        "schema_valid_rate": round(_ratio(n_valid, n), _ROUND),
        "macro_f1": round(_mean(f1s), _ROUND),
        "macro_f1_null_baseline": round(_mean(null_f1s), _ROUND),
        "n_missing_predictions": n_missing,
        "per_field": per_field,
        "generated_at": utc_now(),
        "git_sha": git_sha(),
    }
    if tuple(out["per_field"]) != FIELD_NAMES:  # pragma: no cover - defensive
        raise MetricsError("per_field drifted from FIELD_NAMES order (SPEC §3.2)")
    return out


# --- the teacher arm ----------------------------------------------------------


def score_teacher_arm(
    *, paths: MetricsPaths | None = None, check_leakage: bool = True
) -> dict[str, Any]:
    """Score the teacher against the hand-verified gold (SPEC §3.6).

    The teacher arm has no predictions file: its "prediction" is the teacher label
    already sitting in `eval_pool.jsonl`, and the gold is the human-corrected
    version of the same document in `eval_gold.jsonl`. So this synthesizes the
    prediction rows and hands them to the identical `score_arm` — the metric code
    is never forked for this arm. The result measures teacher-vs-human
    disagreement, without which "within N F1 of the teacher" is meaningless.
    """
    p = _resolve(paths)
    gold_rows = list(read_jsonl(p.gold))
    gold_ids = {row["doc_id"] for row in gold_rows}

    # eval_pool is a superset of the gold set (F3 samples from it), so filter
    # before scoring or every unsampled document trips the extra-doc_id guard.
    pred_rows = [
        {
            "doc_id": row["doc_id"],
            "arm": "teacher",
            "raw_output": "",
            "parsed": row["gold"],
            # F2 only ever wrote rows whose label validated, so this is True by
            # construction -- and `_check_prediction_rows` re-verifies it.
            "schema_valid": True,
            "latency_ms": None,
            "prompt_tokens": None,
            "completion_tokens": None,
        }
        for row in read_jsonl(p.eval_pool)
        if row["doc_id"] in gold_ids
    ]
    return score_arm(gold_rows, pred_rows, "teacher", paths=paths, check_leakage=check_leakage)


# --- file-level entrypoints ---------------------------------------------------


def score_arm_files(
    arm: str,
    *,
    pred_path: Path | None = None,
    gold_path: Path | None = None,
    paths: MetricsPaths | None = None,
    check_leakage: bool = True,
) -> dict[str, Any]:
    """Read the JSONL inputs for `arm` and score them. The CLI's one call."""
    p = _resolve(paths)
    if arm == "teacher":  # no predictions file exists for this arm
        return score_teacher_arm(paths=paths, check_leakage=check_leakage)

    gold_rows = list(read_jsonl(gold_path if gold_path is not None else p.gold))
    pred_rows = list(read_jsonl(pred_path if pred_path is not None else p.pred_for(arm)))
    return score_arm(gold_rows, pred_rows, arm, paths=paths, check_leakage=check_leakage)


def write_metrics(result: Mapping[str, Any], out_path: Path) -> None:
    """Persist `results/metrics/<arm>.json`, preserving `per_field` ordering."""
    write_json(out_path, dict(result), sort_keys=False)


def load_metrics(metrics_dir: Path) -> list[dict[str, Any]]:
    """Every `results/metrics/*.json` on disk — the input to `sxl metrics compare`."""
    out: list[dict[str, Any]] = []
    for path in sorted(Path(metrics_dir).glob("*.json")):
        with open(path, encoding="utf-8") as fh:
            out.append(json.load(fh))
    return out


# --- human-readable rendering -------------------------------------------------


def report_lines(result: Mapping[str, Any]) -> list[str]:
    """The `sxl metrics score` table, per-field rows sorted by **ascending f1**.

    Weakest fields first: the interesting part of a result is where it fails, and
    a reader who stops after three lines should have seen that, not `title`.
    """
    per_field: Mapping[str, Mapping[str, Any]] = result["per_field"]
    headline = (
        f"[metrics] schema_valid_rate {result['schema_valid_rate']:.4f}   "
        f"macro_f1 {result['macro_f1']:.4f}   "
        f"null-baseline macro_f1 {result['macro_f1_null_baseline']:.4f}"
    )
    missing = (
        f"[metrics] missing predictions: {result['n_missing_predictions']} "
        f"(scored as invalid, not dropped)"
    )
    header = (
        f"[metrics]   {'field':22s} {'f1':>7s} {'prec':>7s} {'rec':>7s} {'em':>7s} {'support':>8s}"
    )
    lines = [
        f"[metrics] arm={result['arm']}  split={result['split']}  n={result['n']}",
        headline,
        missing,
        header,
    ]
    for name, m in sorted(per_field.items(), key=lambda kv: (kv[1]["f1"], kv[0])):
        lines.append(
            f"[metrics]   {name:22s} {m['f1']:7.4f} {m['precision']:7.4f} "
            f"{m['recall']:7.4f} {m['em']:7.4f} {m['support']:8d}"
        )
    zero_support = [f for f, m in per_field.items() if m["support"] == 0]
    if zero_support:
        # Named, never silently tolerated: a field absent for every gold document
        # scores f1 0.0 by definition and drags macro_f1 down for a reason that has
        # nothing to do with the model (F4 Verify).
        lines.append(
            f"[metrics] {len(zero_support)} field(s) have zero gold support and so score "
            f"f1 0.0 by definition: {', '.join(sorted(zero_support))}"
        )
    return lines


def compare_lines(results: Sequence[Mapping[str, Any]]) -> list[str]:
    """One row per arm, sorted by `macro_f1` descending — `sxl metrics compare`."""
    lines = [f"{'arm':26s} {'n':>5s} {'schema_valid':>13s} {'macro_f1':>9s}"]
    for r in sorted(results, key=lambda r: r["macro_f1"], reverse=True):
        lines.append(
            f"{r['arm']:26s} {r['n']:5d} {r['schema_valid_rate']:13.4f} {r['macro_f1']:9.4f}"
        )
    return lines
