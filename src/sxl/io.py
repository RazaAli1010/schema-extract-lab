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


def write_json(path: Path, obj: Any) -> None:
    """Write pretty JSON atomically: sorted keys, indent 2, trailing newline."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")
    os.replace(tmp, path)
