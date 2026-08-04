"""JSONL/JSON helpers used by every later feature, so nobody hand-rolls them.

All files are UTF-8, one JSON object per line, keys written in insertion order
(SPEC §3.3). This module does not shadow the stdlib `io` — Python 3 uses absolute
imports inside packages.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield one dict per non-empty line. Streams — never loads the whole file."""
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:  # fail loudly (SPEC §5.5)
                raise ValueError(f"{path}:{lineno}: malformed JSON: {exc}") from exc


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    """Write rows atomically and return the count.

    Writes to a sibling `.tmp` then `os.replace`, so a killed run never leaves a
    half-written split behind for the next stage to read.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    n = 0
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    os.replace(tmp, path)
    return n


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    """Append a single row. Used by the resumable, cached teacher pipeline (F2)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_json(path: Path) -> Any:
    """Parse a whole JSON file, naming the path on malformed input.

    The `results/*.json` artifacts are small and read whole, so unlike
    `read_jsonl` this does not stream. Fails loudly (SPEC §5.5) — a truncated
    artifact must not reach a report as a silently empty dict.
    """
    path = Path(path)
    with open(path, encoding="utf-8") as fh:
        try:
            return json.load(fh)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: malformed JSON: {exc}") from exc


def write_text(path: Path, text: str) -> None:
    """Write UTF-8 text atomically with `\\n` line endings.

    The markdown counterpart of `write_json`, and atomic for the same reason: a
    killed run must never leave half a results table where the next reader — or a
    `git diff` — will treat it as the finished artifact.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    os.replace(tmp, path)


def write_json(path: Path, obj: Any, *, sort_keys: bool = True) -> None:
    """Write pretty JSON atomically: sorted keys, indent 2, trailing newline.

    `sort_keys=False` preserves insertion order, for the one artifact where key
    order is part of the contract: F4's `per_field` must read in `FIELD_NAMES`
    order (SPEC §3.2), which alphabetizing would destroy.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2, sort_keys=sort_keys)
        fh.write("\n")
    os.replace(tmp, path)
