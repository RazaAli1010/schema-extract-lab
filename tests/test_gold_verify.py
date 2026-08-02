"""The resumable review loop (F3 Scope 4). Every session is a scripted stdin.

The loop's contract is that the append-only progress log is the source of truth:
anything the reviewer did survives a closed terminal, and nothing else does.
"""

from __future__ import annotations

import io
import json
import sys

from _fakes import DOC_TEXT, eval_pool_rows, scripted, write_candidates
from sxl.io import read_jsonl
from sxl.schema import ENUM_TYPES, FIELD_NAMES
from sxl.verify import (
    GoldPaths,
    confidence_hints,
    field_review_order,
    load_progress,
    render_document,
    render_value,
    review,
)


def _clock():
    """A monotonic fake clock: one document per minute, so the pace estimate is real."""
    state = {"n": 0}

    def now() -> str:
        state["n"] += 1
        return f"2026-08-02T14:{state['n']:02d}:00Z"

    return now


def _session(tmp_path, answers, n=3, rows=None):
    paths = GoldPaths.in_dir(tmp_path)
    write_candidates(paths, rows if rows is not None else eval_pool_rows(n))
    out: list[str] = []
    summary = review(
        paths=paths,
        read_line=scripted(answers),
        echo=out.append,
        now=_clock(),
        color=False,
    )
    return paths, summary, "\n".join(out)


def _log(paths):
    return list(read_jsonl(paths.progress))


# --- accepting ----------------------------------------------------------------


def test_accept_all_logs_one_line_per_document_and_changes_nothing(tmp_path):
    rows = eval_pool_rows(3)
    paths, summary, _ = _session(tmp_path, ["", "", ""], rows=rows)

    log = _log(paths)
    assert [line["action"] for line in log] == ["accept"] * 3
    assert [line["doc_id"] for line in log] == [r["doc_id"] for r in rows]
    assert all(line["field"] is None and line["old"] is None for line in log)
    assert summary["n_accepted"] == 3 and summary["n_edited"] == 0

    # The candidate file is never rewritten; gold survives byte-identical.
    assert list(read_jsonl(paths.candidates)) == rows


def test_the_log_records_a_timestamp_per_document_not_a_batch_one(tmp_path):
    paths, _, _ = _session(tmp_path, ["", "", ""])
    stamps = [line["at"] for line in _log(paths)]
    assert len(set(stamps)) == 3, "verified_at must be per-document (it is an audit trail)"


# --- editing ------------------------------------------------------------------


def test_editing_an_enum_from_the_menu_logs_old_and_new(tmp_path):
    members = [m.value for m in ENUM_TYPES["seniority"]]
    choice = members.index("senior") + 1

    paths, summary, _ = _session(tmp_path, ["e seniority", str(choice), "", "", ""])

    edits = [line for line in _log(paths) if line["action"] == "edit"]
    assert len(edits) == 1
    assert edits[0]["field"] == "seniority"
    assert edits[0]["old"] == "unknown"
    assert edits[0]["new"] == "senior"
    assert summary["n_edited"] == 1 and summary["n_accepted_as_is"] == 2


def test_a_field_can_be_named_by_number(tmp_path):
    index = FIELD_NAMES.index("seniority") + 1
    members = [m.value for m in ENUM_TYPES["seniority"]]
    paths, _, _ = _session(tmp_path, [f"e {index}", str(members.index("lead") + 1), "", "", ""])
    edits = [line for line in _log(paths) if line["action"] == "edit"]
    assert edits and edits[0]["field"] == "seniority" and edits[0]["new"] == "lead"


def test_free_text_on_an_enum_is_rejected_and_reprompts(tmp_path):
    """`"Senior"` vs `"senior"` must be structurally impossible, not merely discouraged."""
    members = [m.value for m in ENUM_TYPES["seniority"]]
    paths, _, out = _session(
        tmp_path,
        ["e seniority", "Senior", "senior", str(members.index("senior") + 1), "", "", ""],
    )
    assert "not a menu number" in out
    edits = [line for line in _log(paths) if line["action"] == "edit"]
    assert len(edits) == 1, "only the menu choice may take effect"
    assert edits[0]["new"] == "senior"


def test_an_enum_menu_offers_no_empty_option(tmp_path):
    """Enum fields are never null — absence is the member `unknown` (SPEC §3.2)."""
    _, _, out = _session(tmp_path, ["e seniority", "c", "", "", ""])
    menu = [line for line in out.splitlines() if ") " in line and "seniority" not in line]
    assert len(menu) == len(ENUM_TYPES["seniority"]), "every member, and only members"
    assert all(line.split(") ", 1)[1].strip() for line in menu), "no blank option to pick"
    # `education_level` is the field where this actually bites: `none` ("the posting
    # says no degree is needed") and `unknown` ("the posting is silent") are both
    # real members and must both be offered (SPEC §3.2).
    levels = [m.value for m in ENUM_TYPES["education_level"]]
    assert "none" in levels and "unknown" in levels


def test_cancelling_an_enum_edit_logs_nothing(tmp_path):
    paths, _, out = _session(tmp_path, ["e seniority", "c", "", "", ""])
    assert not [line for line in _log(paths) if line["action"] == "edit"]
    assert "unchanged" in out


def test_empty_input_on_a_nullable_number_yields_null(tmp_path):
    rows = eval_pool_rows(3)
    rows[0]["gold"]["salary_min"] = 150000.0
    paths, _, _ = _session(tmp_path, ["e salary_min", "", "", "", ""], rows=rows)

    edits = [line for line in _log(paths) if line["action"] == "edit"]
    assert len(edits) == 1
    assert edits[0]["old"] == 150000.0
    assert edits[0]["new"] is None


def test_a_non_number_on_a_numeric_field_reprompts(tmp_path):
    paths, _, out = _session(tmp_path, ["e salary_min", "lots", "150000", "", "", ""])
    assert "not a float" in out
    edits = [line for line in _log(paths) if line["action"] == "edit"]
    assert edits[0]["new"] == 150000.0


def test_years_experience_is_edited_as_an_int(tmp_path):
    """`years_experience_min` is a strict int — a float would fail validation."""
    paths, _, _ = _session(tmp_path, ["e years_experience_min", "6", "", "", ""])
    edits = [line for line in _log(paths) if line["action"] == "edit"]
    assert edits[0]["new"] == 6
    assert isinstance(edits[0]["new"], int)


def test_required_skills_is_deduplicated_and_sorted(tmp_path):
    paths, _, _ = _session(tmp_path, ["e required_skills", "Python, sql , python,AWS", "", "", ""])
    edits = [line for line in _log(paths) if line["action"] == "edit"]
    assert edits[0]["new"] == ["aws", "python", "sql"]


def test_editing_a_text_field_to_empty_yields_null(tmp_path):
    rows = eval_pool_rows(3)
    rows[0]["gold"]["title"] = "Senior Python Engineer"
    paths, _, _ = _session(tmp_path, ["e title", "", "", "", ""], rows=rows)
    edits = [line for line in _log(paths) if line["action"] == "edit"]
    assert edits[0]["new"] is None


def test_an_unknown_field_name_is_refused(tmp_path):
    paths, _, out = _session(tmp_path, ["e salery", "", "", ""])
    assert "unknown field" in out
    assert not [line for line in _log(paths) if line["action"] == "edit"]


def test_setting_a_field_to_its_current_value_logs_nothing(tmp_path):
    members = [m.value for m in ENUM_TYPES["seniority"]]
    paths, _, out = _session(
        tmp_path, ["e seniority", str(members.index("unknown") + 1), "", "", ""]
    )
    assert "unchanged" in out
    assert not [line for line in _log(paths) if line["action"] == "edit"]


# --- rejecting, going back, quitting -----------------------------------------


def test_x_marks_the_document_rejected(tmp_path):
    rows = eval_pool_rows(3)
    paths, summary, _ = _session(tmp_path, ["x", "", ""], rows=rows)
    log = _log(paths)
    assert log[0]["action"] == "reject_doc" and log[0]["doc_id"] == rows[0]["doc_id"]
    assert summary["n_rejected"] == 1 and summary["n_accepted"] == 2


def test_b_goes_back_and_the_later_decision_wins(tmp_path):
    rows = eval_pool_rows(3)
    members = [m.value for m in ENUM_TYPES["seniority"]]
    paths, _, _ = _session(
        tmp_path,
        ["", "b", "e seniority", str(members.index("lead") + 1), "", "", ""],
        rows=rows,
    )
    first_id = rows[0]["doc_id"]
    accepts = [ln for ln in _log(paths) if ln["action"] == "accept" and ln["doc_id"] == first_id]
    assert len(accepts) == 2, "re-reviewing re-accepts and refreshes verified_at"
    assert accepts[0]["at"] != accepts[1]["at"]


def test_s_saves_and_quits_without_reviewing_the_rest(tmp_path):
    paths, summary, _ = _session(tmp_path, ["", "s"], n=5)
    assert summary["n_reviewed"] == 1 and summary["quit_early"] is True
    assert len(_log(paths)) == 1


def test_help_is_available_and_does_not_advance(tmp_path):
    paths, _, out = _session(tmp_path, ["?", "", "", ""])
    assert "Enter" in out and "reject this document" in out
    assert [line["action"] for line in _log(paths)] == ["accept"] * 3


def test_an_unknown_command_says_so_and_does_not_advance(tmp_path):
    paths, _, out = _session(tmp_path, ["zzz", "", "", ""])
    assert "unknown command" in out
    assert len(_log(paths)) == 3


# --- resumability -------------------------------------------------------------


def test_interrupting_after_two_of_five_resumes_at_the_third(tmp_path):
    """The acceptance criterion: no lost edits across a killed session."""
    paths = GoldPaths.in_dir(tmp_path)
    rows = eval_pool_rows(5)
    write_candidates(paths, rows)
    members = [m.value for m in ENUM_TYPES["seniority"]]

    # Session one: accept #1, edit and accept #2, then the terminal dies.
    review(
        paths=paths,
        read_line=scripted(["", "e seniority", str(members.index("senior") + 1), ""]),
        echo=lambda _s: None,
        now=_clock(),
        color=False,
    )
    assert sum(1 for line in _log(paths) if line["action"] == "accept") == 2

    # Session two: a fresh process would see exactly this state.
    seen: list[str] = []
    summary = review(
        paths=paths,
        read_line=scripted(["", "", ""]),
        echo=seen.append,
        now=_clock(),
        color=False,
    )
    assert summary["n_reviewed"] == 5
    accepted_ids = [line["doc_id"] for line in _log(paths) if line["action"] == "accept"]
    assert accepted_ids == [r["doc_id"] for r in rows], "resumed at #3, in doc_id order"

    # The edit from session one survived and is still applied.
    edits = [line for line in _log(paths) if line["action"] == "edit"]
    assert len(edits) == 1 and edits[0]["new"] == "senior"
    assert rows[1]["doc_id"] not in "".join(seen[:2]), "reviewed documents are not redrawn"


def test_a_finished_candidate_set_says_so_instead_of_relooping(tmp_path):
    paths = GoldPaths.in_dir(tmp_path)
    write_candidates(paths, eval_pool_rows(2))
    review(paths=paths, read_line=scripted(["", ""]), echo=lambda _s: None, now=_clock())

    out: list[str] = []
    summary = review(paths=paths, read_line=scripted([]), echo=out.append, now=_clock())
    assert summary["n_reviewed"] == 2 and summary["quit_early"] is False
    assert "gold finalize" in "\n".join(out)


def test_keyboard_interrupt_is_a_save_not_a_crash(tmp_path):
    """`scripted([])` raises KeyboardInterrupt the moment input runs out."""
    paths, summary, out = _session(tmp_path, [""], n=3)
    assert summary["quit_early"] is True
    assert "progress saved" in out
    assert len(_log(paths)) == 1


# --- progress reporting -------------------------------------------------------


def test_progress_reports_counts_and_an_eta(tmp_path):
    _, _, out = _session(tmp_path, ["", "", ""])
    assert "reviewed 3/3" in out
    assert "accepted-as-is 3" in out
    assert "est. remaining" in out


# --- rendering and hints ------------------------------------------------------


def test_null_and_unknown_render_as_different_tokens(tmp_path):
    assert render_value("title", None) == "null"
    assert render_value("seniority", "unknown") == "unknown"
    assert render_value("required_skills", []) == "[]"
    assert render_value("required_skills", ["python", "sql"]) == "python, sql"


def test_the_frame_is_ascii_so_a_cp1252_console_can_print_it():
    """A Windows console is cp1252: a box-drawing rule raises mid-session."""
    row = eval_pool_rows(1)[0]
    frame = render_document(row, index=0, total=330, full_text=False, color=False)
    chrome = frame.replace(row["text"], "")  # the posting itself is not ours to control
    assert chrome.isascii(), [c for c in set(chrome) if not c.isascii()]


def test_a_posting_the_console_cannot_encode_does_not_kill_the_session(tmp_path, monkeypatch):
    """Curly quotes and emoji are normal in real postings; a crash here costs hours."""
    sink = io.TextIOWrapper(io.BytesIO(), encoding="cp1252", newline="")
    monkeypatch.setattr(sys, "stdout", sink)

    rows = eval_pool_rows(2, text="Café — “Senior” Engineer \U0001f680 " + "x" * 500)
    paths = GoldPaths.in_dir(tmp_path)
    write_candidates(paths, rows)
    summary = review(  # echo=None -> the real console writer, not a list
        paths=paths, read_line=scripted(["", ""]), now=_clock(), color=False
    )
    assert summary["n_accepted"] == 2
    assert len(_log(paths)) == 2


def test_field_review_order_is_the_spec_order():
    assert field_review_order({"gold": {}}) == list(FIELD_NAMES)


def test_a_value_absent_from_the_document_is_flagged_low():
    row = {"text": DOC_TEXT, "gold": json.loads(json.dumps({**_blank(), "company": "Globex"}))}
    assert confidence_hints(row)["company"] == "low"

    row["gold"]["company"] = "Acme Robotics"
    assert confidence_hints(row)["company"] == "normal"


def test_country_is_compared_case_insensitively():
    """`norm_country` uppercases, so a naive substring test would flag every country."""
    row = {
        "text": "Remote role based in us offices.",
        "gold": {**_blank(), "location_country": "US"},
    }
    assert confidence_hints(row)["location_country"] == "normal"


def test_a_currency_inferred_from_its_symbol_is_not_flagged():
    """Postings write `$150,000`, never `USD 150,000` — flagging that is crying wolf."""
    row = {"text": "Pay: $150,000 per year.", "gold": {**_blank(), "salary_currency": "USD"}}
    assert confidence_hints(row)["salary_currency"] == "normal"

    row = {"text": "Salary: £80,000.", "gold": {**_blank(), "salary_currency": "GBP"}}
    assert confidence_hints(row)["salary_currency"] == "normal"

    # A currency with neither its code nor its symbol present is still suspicious.
    row = {"text": "Pay: $150,000 per year.", "gold": {**_blank(), "salary_currency": "JPY"}}
    assert confidence_hints(row)["salary_currency"] == "low"


def test_impossible_numbers_are_flagged_low():
    row = {"text": DOC_TEXT, "gold": {**_blank(), "salary_min": 200000.0, "salary_max": 100000.0}}
    hints = confidence_hints(row)
    assert hints["salary_min"] == "low" and hints["salary_max"] == "low"

    row = {"text": DOC_TEXT, "gold": {**_blank(), "years_experience_min": 99}}
    assert confidence_hints(row)["years_experience_min"] == "low"

    row = {"text": DOC_TEXT, "gold": {**_blank(), "posting_date": "June 2026"}}
    assert confidence_hints(row)["posting_date"] == "low"

    row = {"text": DOC_TEXT, "gold": {**_blank(), "posting_date": "2026-06-14"}}
    assert confidence_hints(row)["posting_date"] == "normal"


def test_absent_fields_are_never_flagged():
    hints = confidence_hints({"text": DOC_TEXT, "gold": _blank()})
    assert set(hints.values()) == {"normal"}


def test_a_skill_missing_from_the_text_is_flagged():
    row = {"text": DOC_TEXT, "gold": {**_blank(), "required_skills": ["python", "cobol"]}}
    assert confidence_hints(row)["required_skills"] == "low"

    row["gold"]["required_skills"] = ["python", "sql"]
    assert confidence_hints(row)["required_skills"] == "normal"


def _blank():
    from sxl.schema import empty_posting

    return empty_posting().model_dump(mode="json")


def test_load_progress_on_a_missing_file_is_empty(tmp_path):
    assert load_progress(paths=GoldPaths.in_dir(tmp_path)) == []
