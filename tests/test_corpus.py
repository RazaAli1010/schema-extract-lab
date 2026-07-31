"""F1 — corpus acquisition. Every test is offline; no network, no `datasets`.

`build(rows=...)` is the injection seam: fixtures stand in for the HF stream.
"""

from __future__ import annotations

import json

import pytest

from sxl.config import HF_CACHE_DIR
from sxl.corpus import (
    _set_hf_cache,
    build,
    clean_text,
    dedupe,
    make_doc_id,
    pick_source_key_column,
    pick_text_column,
)

SOURCE = "test/postings"

# Long enough to clear CORPUS_MIN_CHARS = 400 and to exercise the 600-char
# near-duplicate window.
BODY = (
    "We are hiring a Senior Platform Engineer to join our Infrastructure team in Berlin. "
    "You will own our Kubernetes estate, our Terraform modules, and the CI pipeline that "
    "ships roughly forty services a day. The role is hybrid, three days on site. "
    "Requirements: five years of production experience with Go or Python, deep knowledge "
    "of Linux networking, and a track record of running systems that page people at night. "
    "We offer a salary of EUR 95,000 to 120,000 per year, a learning budget, and equity. "
    "Applications close at the end of the quarter and we interview on a rolling basis. "
    "Our stack is Go, Python, PostgreSQL, Kafka and a great deal of YAML nobody enjoys. "
    "You will report to the Head of Platform and mentor two junior engineers. "
)

assert len(BODY) > 600, "the fixture must exceed the near-duplicate window to exercise it"


def raw(source_key: str, text: str) -> dict[str, str]:
    return {"source_key": source_key, "text": text}


def fixture_rows() -> list[dict[str, str]]:
    """Five distinct postings, no exact or near-duplicates between them."""
    return [
        raw(f"k{i}", f"Posting number {i}. Role {i} at Company {i}.\n\n{BODY}") for i in range(5)
    ]


# --- clean_text --------------------------------------------------------------


def test_clean_text_is_idempotent():
    messy = (
        "Senior Engineer &amp; Architect\r\n\r\n"
        "Salary &gt; 100k &amp;lt;negotiable&amp;gt;   \r\n"
        "\n\n\n\n\n"
        "Apply now.   "
    )
    once = clean_text(messy)
    assert clean_text(once) == once, "clean_text(clean_text(x)) must equal clean_text(x)"


def test_clean_text_collapses_blank_runs_and_normalizes_newlines():
    cleaned = clean_text("A\r\nB\r\rC\n\n\n\n\nD")
    assert "\r" not in cleaned
    assert "\n\n\n" not in cleaned
    assert cleaned == "A\nB\n\nC\n\nD"


def test_clean_text_preserves_case_and_punctuation():
    cleaned = clean_text("Senior SRE (Kubernetes) — apply by 2026-08-01!")
    assert "SRE" in cleaned, "uppercase tokens must survive"
    assert "(Kubernetes)" in cleaned
    assert "!" in cleaned


def test_clean_text_leaves_bare_angle_brackets_alone():
    """Real postings say "<5 years"; that is not markup and must not be eaten."""
    cleaned = clean_text("Requires <5 years experience and a salary <100k.")
    assert "<5 years experience" in cleaned
    assert "<100k" in cleaned


def test_clean_text_strips_real_html_tags():
    cleaned = clean_text("<p>Senior <strong>Engineer</strong></p><br/><li>Remote</li>")
    assert "<p>" not in cleaned and "<li>" not in cleaned
    assert "Senior" in cleaned and "Engineer" in cleaned and "Remote" in cleaned


def test_clean_text_handles_empty_input():
    assert clean_text("") == ""


# --- make_doc_id -------------------------------------------------------------


def test_make_doc_id_is_stable_and_prefixed():
    first = make_doc_id(SOURCE, "abc")
    assert first == make_doc_id(SOURCE, "abc")
    assert first.startswith("jp_")
    assert len(first) == len("jp_") + 10


def test_make_doc_id_differs_per_source_key():
    assert make_doc_id(SOURCE, "abc") != make_doc_id(SOURCE, "abd")


def test_make_doc_id_differs_across_sources():
    """Same upstream key under two sources must never collide."""
    assert make_doc_id("source/one", "abc") != make_doc_id("source/two", "abc")


def test_make_doc_id_is_not_positional():
    """A sequence number would move documents between splits on any reorder."""
    assert make_doc_id(SOURCE, "k0") != "jp_000001"


# --- dedupe ------------------------------------------------------------------


def test_dedupe_drops_an_exact_repeat():
    rows = [{"doc_id": "jp_a", "text": BODY}, {"doc_id": "jp_a", "text": BODY}]
    stats: dict[str, int] = {}
    assert len(list(dedupe(rows, stats))) == 1
    assert stats["n_dropped_dup_id"] == 1


def test_dedupe_drops_a_shared_600_char_prefix():
    """A repost differing only in a trailing block is still the same posting."""
    rows = [
        {"doc_id": "jp_a", "text": BODY},
        {"doc_id": "jp_b", "text": BODY + "\n\nApply here: https://example.com/jobs/2"},
    ]
    stats: dict[str, int] = {}
    assert len(list(dedupe(rows, stats))) == 1
    assert stats["n_dropped_dup_prefix"] == 1


def test_dedupe_keeps_a_row_differing_within_the_first_600_chars():
    rows = [
        {"doc_id": "jp_a", "text": BODY},
        {"doc_id": "jp_b", "text": "A completely different opening paragraph. " + BODY},
    ]
    stats: dict[str, int] = {}
    assert len(list(dedupe(rows, stats))) == 2
    assert stats["n_dropped_dup_prefix"] == 0


# --- build -------------------------------------------------------------------


def read_records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_build_writes_the_five_spec_keys_in_order(tmp_path):
    out = tmp_path / "docs.jsonl"
    stats = build(target_n=10, source=SOURCE, rows=fixture_rows(), out_path=out)

    assert stats["n_kept"] == 5
    for record in read_records(out):
        assert list(record) == ["doc_id", "domain", "text", "source", "char_len"]
        assert record["domain"] == "job_posting"
        assert record["source"] == SOURCE
        assert record["char_len"] == len(record["text"])


def test_build_is_byte_identical_across_runs(tmp_path):
    first, second = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    build(target_n=10, source=SOURCE, rows=fixture_rows(), out_path=first)
    build(target_n=10, source=SOURCE, rows=fixture_rows(), out_path=second)
    assert first.read_bytes() == second.read_bytes()


def test_build_output_is_sorted_by_doc_id(tmp_path):
    out = tmp_path / "docs.jsonl"
    build(target_n=10, source=SOURCE, rows=fixture_rows(), out_path=out)
    ids = [r["doc_id"] for r in read_records(out)]
    assert ids == sorted(ids)


def test_build_is_order_independent(tmp_path):
    """Reversing the input yields the same id set — proves ids are content-derived.

    The fixture is deliberately free of near-duplicates: `dedupe` is first-seen-
    wins, so a collision would legitimately resolve differently under reversal.
    """
    forward, backward = tmp_path / "f.jsonl", tmp_path / "b.jsonl"
    build(target_n=10, source=SOURCE, rows=fixture_rows(), out_path=forward)
    build(target_n=10, source=SOURCE, rows=list(reversed(fixture_rows())), out_path=backward)
    assert forward.read_bytes() == backward.read_bytes()


def test_build_drops_short_and_overlong_rows(tmp_path):
    out = tmp_path / "docs.jsonl"
    rows = [
        *fixture_rows(),
        raw("short", "x" * 100),
        raw("long", "y" * 50_000),
    ]
    stats = build(target_n=10, source=SOURCE, rows=rows, out_path=out)

    assert stats["n_dropped_short"] == 1
    assert stats["n_dropped_long"] == 1
    assert stats["n_kept"] == 5
    assert stats["n_seen"] == 7


def test_build_stops_at_target_n(tmp_path):
    out = tmp_path / "docs.jsonl"
    stats = build(target_n=2, source=SOURCE, rows=fixture_rows(), out_path=out)
    assert stats["n_kept"] == 2


def test_build_reports_split_counts_and_percentiles(tmp_path):
    out = tmp_path / "docs.jsonl"
    stats = build(target_n=10, source=SOURCE, rows=fixture_rows(), out_path=out)

    assert set(stats["split_counts"]) == {"train", "dev", "eval_pool"}
    assert sum(stats["split_counts"].values()) == stats["n_kept"]
    assert set(stats["char_len"]) == {"p5", "p50", "p95", "max"}
    assert stats["char_len"]["max"] >= stats["char_len"]["p50"]


def test_build_leaves_no_tmp_file_behind(tmp_path):
    out = tmp_path / "docs.jsonl"
    build(target_n=10, source=SOURCE, rows=fixture_rows(), out_path=out)
    assert not list(tmp_path.glob("*.tmp")), "atomic write left its tmp file behind"


# --- HF cache redirection ----------------------------------------------------


def test_set_hf_cache_redirects_every_hf_variable(monkeypatch):
    """The laptop has 5 GB free; nothing may land in ~/.cache/huggingface."""
    for var in ("HF_HOME", "HF_HUB_CACHE", "HF_DATASETS_CACHE"):
        monkeypatch.setenv(var, "sentinel")  # registered so pytest restores it

    _set_hf_cache()

    import os

    assert os.environ["HF_HOME"] == str(HF_CACHE_DIR)
    assert os.environ["HF_HUB_CACHE"].startswith(str(HF_CACHE_DIR))
    assert os.environ["HF_DATASETS_CACHE"].startswith(str(HF_CACHE_DIR))


def test_set_hf_cache_does_not_create_the_directory(monkeypatch, tmp_path):
    """Otherwise a metadata call after cleanup resurrects an empty data/.hfcache."""
    cache = tmp_path / "hfcache"
    monkeypatch.setattr("sxl.corpus.HF_CACHE_DIR", cache)
    for var in ("HF_HOME", "HF_HUB_CACHE", "HF_DATASETS_CACHE"):
        monkeypatch.setenv(var, "sentinel")

    _set_hf_cache()

    assert not cache.exists()


# --- column detection --------------------------------------------------------


def test_pick_text_column_finds_the_long_field():
    rows = [{"title": "Senior Engineer", "description": BODY, "url": "https://x/y"}] * 5
    assert pick_text_column(rows, SOURCE) == "description"


def test_pick_text_column_uses_the_median_not_the_first_row():
    """A null description in row 1 must not hand the crown to a URL column."""
    rows = [{"description": None, "application_url": "https://example.com/" + "a" * 80}]
    rows += [{"description": BODY, "application_url": "https://example.com/" + "a" * 80}] * 9
    assert pick_text_column(rows, SOURCE) == "description"


def test_pick_text_column_rejects_a_metadata_only_dataset():
    """Regression: `lukebarousse/data_jobs` has no description column at all.

    Its longest string is an ~85-char job title, so every row would fail the
    400-char minimum and the build would have emitted an empty corpus. That must
    be a loud crash, not a silent zero.
    """
    rows = [
        {
            "job_title": "Senior Data Analyst, Enterprise Reporting and Business Intelligence Team",
            "job_location": "Berlin, Germany",
            "salary_year_avg": 95000.0,
            "job_skills": "['sql', 'python']",
        }
    ] * 5
    with pytest.raises(ValueError, match="no free-text column"):
        pick_text_column(rows, "lukebarousse/data_jobs")


def test_pick_text_column_raises_on_no_rows():
    with pytest.raises(ValueError, match="no rows"):
        pick_text_column([], SOURCE)


def test_pick_source_key_column_prefers_an_upstream_id():
    rows = [{"job_id": 3757759040, "description": BODY}] * 3
    assert pick_source_key_column(rows, "description") == "job_id"


def test_pick_source_key_column_returns_none_when_no_id_exists():
    rows = [{"title": "Engineer", "description": BODY}] * 3
    assert pick_source_key_column(rows, "description") is None


def test_pick_source_key_column_rejects_a_partially_null_id():
    rows = [{"job_id": 1, "description": BODY}, {"job_id": None, "description": BODY}]
    assert pick_source_key_column(rows, "description") is None
