"""Finalizing eval_gold and the teacher's own accuracy ceiling (F3 Scope 6-7).

`teacher_field_agreement` is a headline result, not a diagnostic: F8 quotes it
directly, because "within 3 macro-F1 of the teacher" means something very
different when the teacher is itself only 93% right on `seniority`.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from _fakes import eval_pool_rows, write_candidates
from sxl.config import N_EVAL_GOLD
from sxl.io import read_jsonl, write_jsonl
from sxl.schema import FIELD_NAMES, validate_prediction
from sxl.splits import split_for
from sxl.teacher import ROW_KEYS
from sxl.verify import (
    AGREEMENT_FLOOR,
    GoldPaths,
    LeakageDetected,
    NotEnoughVerified,
    agreement_warnings,
    finalize,
    median_seconds_per_doc,
    progress_line,
    teacher_field_agreement,
)


def _at(i: int) -> str:
    """One accepted document per minute, so the pace estimate has real gaps."""
    return f"2026-08-02T{14 + i // 60:02d}:{i % 60:02d}:00Z"


def _setup(tmp_path, *, n_accept, n_reject=0, edits=None):
    """Write `n_accept + n_reject` candidates and a progress log covering them."""
    paths = GoldPaths.in_dir(tmp_path)
    rows = eval_pool_rows(n_accept + n_reject)
    write_candidates(paths, rows)

    log = []
    for i, row in enumerate(rows):
        for name, new in (edits or {}).get(row["doc_id"], {}).items():
            log.append(
                progress_line(
                    row["doc_id"],
                    "edit",
                    at=_at(i),
                    field_=name,
                    old=row["gold"][name],
                    new=new,
                )
            )
        action = "reject_doc" if i >= n_accept else "accept"
        log.append(progress_line(row["doc_id"], action, at=_at(i)))
    write_jsonl(paths.progress, log)
    return paths, rows


# --- the output file ----------------------------------------------------------


def test_finalize_takes_exactly_300_from_305_accepted_plus_5_rejected(tmp_path):
    paths, _ = _setup(tmp_path, n_accept=305, n_reject=5)
    stats = finalize(paths=paths)

    out = list(read_jsonl(paths.gold))
    assert len(out) == N_EVAL_GOLD == 300
    assert stats["n_candidates"] == 310
    assert stats["n_rejected"] == 5
    assert stats["n_final"] == 300


def test_every_output_row_matches_the_spec_record_shape(tmp_path):
    paths, _ = _setup(tmp_path, n_accept=302)
    finalize(paths=paths)

    for row in read_jsonl(paths.gold):
        assert tuple(row) == ROW_KEYS, "keys must be in SPEC §3.3 order"
        assert row["label_source"] == "human"
        assert row["verified_by_human"] is True
        assert row["teacher_model"] == "gpt-4o-mini", "retained: what the human corrected"
        assert validate_prediction(row["gold"]) is not None
        assert tuple(row["gold"]) == FIELD_NAMES
        assert split_for(row["doc_id"]) == "eval_pool"
        # parseable ISO-8601 UTC
        assert (
            datetime.strptime(row["verified_at"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC).tzinfo
            is UTC
        )


def test_rejected_documents_never_reach_the_output(tmp_path):
    paths, rows = _setup(tmp_path, n_accept=300, n_reject=5)
    finalize(paths=paths)
    kept = {r["doc_id"] for r in read_jsonl(paths.gold)}
    assert not (kept & {r["doc_id"] for r in rows[300:]})


def test_output_is_the_first_300_by_doc_id_not_by_review_order(tmp_path):
    """A pure function of (candidates, progress) — re-running must not reshuffle."""
    paths, rows = _setup(tmp_path, n_accept=310)
    finalize(paths=paths)
    ids = [r["doc_id"] for r in read_jsonl(paths.gold)]
    assert ids == sorted(ids)
    assert ids == [r["doc_id"] for r in rows][:300]


def test_finalize_is_idempotent(tmp_path):
    paths, _ = _setup(tmp_path, n_accept=302)
    finalize(paths=paths)
    first = paths.gold.read_bytes()
    finalize(paths=paths)
    assert paths.gold.read_bytes() == first
    assert not list(paths.gold.parent.glob("*.tmp")), "atomic write left a tmp file"


def test_finalize_exits_with_a_count_when_too_few_survive(tmp_path):
    paths, _ = _setup(tmp_path, n_accept=290, n_reject=5)
    with pytest.raises(NotEnoughVerified) as exc:
        finalize(paths=paths)
    assert "290" in str(exc.value) and "300" in str(exc.value)
    assert not paths.gold.exists(), "a short run must not leave a partial gold file"


def test_unreviewed_candidates_do_not_count_as_verified(tmp_path):
    paths = GoldPaths.in_dir(tmp_path)
    write_candidates(paths, eval_pool_rows(310))
    with pytest.raises(NotEnoughVerified, match="unreviewed"):
        finalize(paths=paths)


def test_finalize_re_asserts_no_leakage(tmp_path):
    """Second of the two chances to catch the one mistake that voids the project."""
    paths, rows = _setup(tmp_path, n_accept=302)
    write_jsonl(paths.train, [{"doc_id": rows[0]["doc_id"]}])
    with pytest.raises(LeakageDetected, match=rows[0]["doc_id"]):
        finalize(paths=paths)


# --- teacher_field_agreement --------------------------------------------------


def test_agreement_is_hand_computed_on_two_of_ten_edited():
    """2 of 10 `seniority` labels edited -> the teacher was right 8 times."""
    rows = eval_pool_rows(10)
    final = [{**r, "gold": dict(r["gold"])} for r in rows]
    for row in final[:2]:
        row["gold"]["seniority"] = "senior"

    agreement = teacher_field_agreement(rows, final)
    assert agreement["seniority"] == 0.8
    assert agreement["title"] == 1.0
    assert set(agreement) == set(FIELD_NAMES), "all 16 fields reported"


def test_agreement_uses_the_same_normalization_as_the_metrics():
    """A case/whitespace-only correction is not a disagreement (SPEC §3.5)."""
    rows = eval_pool_rows(2)
    for row in rows:
        row["gold"]["title"] = "Senior Python Engineer"
    final = [{**r, "gold": {**r["gold"], "title": "  senior   python engineer "}} for r in rows]
    assert teacher_field_agreement(rows, final)["title"] == 1.0


def test_agreement_treats_skills_as_a_set():
    rows = eval_pool_rows(2)
    for row in rows:
        row["gold"]["required_skills"] = ["python", "sql"]
    final = [{**r, "gold": {**r["gold"], "required_skills": ["sql", "python"]}} for r in rows]
    assert teacher_field_agreement(rows, final)["required_skills"] == 1.0


def test_finalize_reports_agreement_and_edit_rate_for_all_16_fields(tmp_path):
    rows = eval_pool_rows(305)
    edited = {r["doc_id"]: {"seniority": "senior"} for r in rows[:30]}
    paths = GoldPaths.in_dir(tmp_path)
    write_candidates(paths, rows)
    _setup_log(paths, rows, edited)

    stats = finalize(paths=paths)
    assert set(stats["teacher_field_agreement"]) == set(FIELD_NAMES)
    assert set(stats["edit_rate_by_field"]) == set(FIELD_NAMES)
    assert stats["n_docs_edited"] == 30
    assert stats["edit_rate_by_field"]["seniority"] == 0.1
    assert stats["teacher_field_agreement"]["seniority"] == 0.9
    assert stats["teacher_field_agreement"]["title"] == 1.0


def _setup_log(paths, rows, edits):
    log = []
    for i, row in enumerate(rows):
        for name, new in edits.get(row["doc_id"], {}).items():
            log.append(
                progress_line(
                    row["doc_id"], "edit", at=_at(i), field_=name, old=row["gold"][name], new=new
                )
            )
        log.append(progress_line(row["doc_id"], "accept", at=_at(i)))
    write_jsonl(paths.progress, log)


# --- gold_stats.json ----------------------------------------------------------


def test_stats_file_has_every_documented_key(tmp_path):
    paths, _ = _setup(tmp_path, n_accept=302, n_reject=3)
    stats = finalize(paths=paths)

    on_disk = json.loads(paths.stats.read_text(encoding="utf-8"))
    assert on_disk == stats
    for key in (
        "n_candidates",
        "n_rejected",
        "n_final",
        "n_docs_edited",
        "edit_rate_by_field",
        "teacher_field_agreement",
        "median_seconds_per_doc",
        "char_len_bins",
        "generated_at",
        "git_sha",
    ):
        assert key in on_disk, key
    assert sum(on_disk["char_len_bins"].values()) == 300


def test_median_seconds_ignores_an_overnight_break():
    """The median, not the mean: one 12-hour gap must not distort the estimate."""
    log = [progress_line(f"jp_{i}", "accept", at=_at(i)) for i in range(5)]
    log.append(progress_line("jp_9", "accept", at="2026-08-03T09:00:00Z"))
    assert median_seconds_per_doc(log) == 60.0


def test_median_seconds_is_zero_before_anything_is_reviewed():
    assert median_seconds_per_doc([]) == 0.0


# --- the agreement floor ------------------------------------------------------


def test_a_field_below_the_floor_is_flagged_by_name():
    warnings = agreement_warnings({"seniority": 0.61, "title": 0.99})
    assert len(warnings) == 1
    assert "seniority" in warnings[0] and "prompts.py" in warnings[0]


def test_a_healthy_run_produces_no_warnings():
    assert agreement_warnings(dict.fromkeys(FIELD_NAMES, AGREEMENT_FLOOR)) == []
