"""JSONL helpers — every later feature depends on these being atomic and lossless."""

from __future__ import annotations

import json

import pytest

from sxl.config import git_sha
from sxl.io import append_jsonl, read_jsonl, write_json, write_jsonl

ROWS = [
    {"doc_id": "jp_000001", "domain": "job_posting", "text": "héllo", "char_len": 5},
    {"doc_id": "jp_000002", "domain": "job_posting", "text": "b", "char_len": 1},
]


def test_write_then_read_roundtrips_and_preserves_key_order(tmp_path):
    path = tmp_path / "docs.jsonl"
    assert write_jsonl(path, ROWS) == 2

    back = list(read_jsonl(path))
    assert back == ROWS
    assert list(back[0]) == list(ROWS[0])  # SPEC §3.3: keys in the given order


def test_write_jsonl_leaves_no_tmp_file(tmp_path):
    path = tmp_path / "docs.jsonl"
    write_jsonl(path, ROWS)
    assert not list(tmp_path.glob("*.tmp"))


def test_write_jsonl_creates_parent_dirs(tmp_path):
    path = tmp_path / "deep" / "nested" / "docs.jsonl"
    write_jsonl(path, ROWS)
    assert path.exists()


def test_append_jsonl_is_resumable(tmp_path):
    path = tmp_path / "cache.jsonl"
    for row in ROWS:
        append_jsonl(path, row)
    assert list(read_jsonl(path)) == ROWS


def test_read_jsonl_skips_blank_lines(tmp_path):
    path = tmp_path / "docs.jsonl"
    path.write_text('{"a": 1}\n\n{"a": 2}\n', encoding="utf-8")
    assert list(read_jsonl(path)) == [{"a": 1}, {"a": 2}]


def test_read_jsonl_fails_loudly_on_malformed_json(tmp_path):
    path = tmp_path / "docs.jsonl"
    path.write_text('{"a": 1}\nnot json\n', encoding="utf-8")
    with pytest.raises(ValueError, match=":2:"):
        list(read_jsonl(path))


def test_write_json_is_sorted_and_newline_terminated(tmp_path):
    path = tmp_path / "metrics.json"
    write_json(path, {"b": 2, "a": 1})

    text = path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert list(json.loads(text)) == ["a", "b"]
    assert not list(tmp_path.glob("*.tmp"))


def test_git_sha_never_raises():
    sha = git_sha()
    assert isinstance(sha, str) and sha
