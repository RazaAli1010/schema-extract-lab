"""Batch review harness for `sxl gold verify` (F3), for a non-interactive reviewer.

`verify.review()` is a terminal TUI driven by `input()`. A reviewer that cannot
sit at a TTY needs the same append-only progress log written some other way, so
this script exposes the two halves separately:

    python scripts/review_batch.py dump  --start 0 --count 15
    python scripts/review_batch.py apply --decisions <file.json>

`dump` renders candidates (teacher label + source text) for reading. `apply`
appends `edit`/`accept`/`reject_doc` lines through `verify.progress_line`, so the
log this writes is byte-identical in shape to one the TUI would have produced and
`verify.replay`/`finalize` consume it unchanged.

This does not make a model-driven pass equivalent to a human one — see
`verify.REVIEWERS`. It only keeps the *log format* honest so `sxl gold finalize
--reviewer model_verified` can record what actually happened.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from sxl.config import utc_now
from sxl.io import append_jsonl, read_jsonl
from sxl.schema import FIELD_NAMES, validate_prediction
from sxl.verify import GoldPaths, load_progress, progress_line, replay, confidence_hints

_BLANKS = re.compile(r"\n{3,}")
_TRAILING_WS = re.compile(r"[ \t]+$", re.MULTILINE)


def _candidates(paths: GoldPaths) -> list[dict]:
    return sorted(read_jsonl(paths.candidates), key=lambda r: r["doc_id"])


def cmd_dump(args: argparse.Namespace) -> None:
    paths = GoldPaths.default()
    rows, state = replay(_candidates(paths), load_progress(paths=paths))
    if args.queue:
        # Work a named queue (see the targeted-review plan) instead of raw order,
        # skipping anything already decided so a resumed run never re-reads.
        wanted = json.loads(Path(args.queue).read_text(encoding="utf-8"))[args.bucket]
        order = {d: i for i, d in enumerate(wanted)}
        rows = sorted(
            (r for r in rows if r["doc_id"] in order and not state[r["doc_id"]].reviewed),
            key=lambda r: order[r["doc_id"]],
        )
    batch = rows[args.start : args.start + args.count]

    out = []
    for i, row in enumerate(batch, start=args.start):
        done = state[row["doc_id"]].status
        text = _TRAILING_WS.sub("", row["text"])
        text = _BLANKS.sub("\n\n", text).strip()
        hints = {k: v for k, v in confidence_hints(row).items() if v == "low"}
        out.append(f"{'=' * 78}\n[{i}] {row['doc_id']}  chars={len(row['text'])}  status={done}")
        out.append("--- teacher label " + "-" * 60)
        for name in FIELD_NAMES:
            flag = "   <-- CHECK (not grounded in text)" if name in hints else ""
            out.append(f"  {name:22s} {json.dumps(row['gold'][name], ensure_ascii=False)}{flag}")
        out.append("--- text " + "-" * 69)
        out.append(text)
    Path(args.out).write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"wrote {args.out}: documents {args.start}..{args.start + len(batch) - 1}")


def cmd_apply(args: argparse.Namespace) -> None:
    """Apply a decisions file: {"<doc_id>": {"reject": true} | {"<field>": <new>}}."""
    paths = GoldPaths.default()
    rows = {r["doc_id"]: r for r in _candidates(paths)}
    decisions = json.loads(Path(args.decisions).read_text(encoding="utf-8"))

    n_edits = n_accept = n_reject = 0
    for doc_id, changes in decisions.items():
        if doc_id not in rows:
            raise SystemExit(f"{doc_id} is not a sampled candidate")
        at = utc_now()
        if changes.get("reject"):
            append_jsonl(paths.progress, progress_line(doc_id, "reject_doc", at=at))
            n_reject += 1
            continue

        gold = dict(rows[doc_id]["gold"])
        for field_, new in changes.items():
            if field_ not in FIELD_NAMES:
                raise SystemExit(f"{doc_id}: unknown field {field_!r}")
            old = gold[field_]
            if old == new:
                continue  # a no-op edit would pollute `edit_rate_by_field`
            append_jsonl(
                paths.progress,
                progress_line(doc_id, "edit", at=at, field_=field_, old=old, new=new),
            )
            gold[field_] = new
            n_edits += 1
        # Validate the corrected object before accepting it, so a bad edit fails
        # here rather than inside `finalize` after 300 documents of work.
        if validate_prediction(gold) is None:
            raise SystemExit(f"{doc_id}: corrected gold does not validate against JobPosting")
        # `--no-accept` lets a mechanical pass (one field, applied by rule across
        # the split) run ahead of the per-document read that actually accepts.
        if not args.no_accept:
            append_jsonl(paths.progress, progress_line(doc_id, "accept", at=at))
            n_accept += 1

    print(f"applied: {n_accept} accepted, {n_edits} field edits, {n_reject} rejected")


def cmd_status(_args: argparse.Namespace) -> None:
    paths = GoldPaths.default()
    rows, state = replay(_candidates(paths), load_progress(paths=paths))
    done = sum(1 for r in rows if state[r["doc_id"]].reviewed)
    accepted = sum(1 for r in rows if state[r["doc_id"]].status == "accepted")
    edited = sum(1 for r in rows if state[r["doc_id"]].edited_fields)
    print(f"{done}/{len(rows)} reviewed · {accepted} accepted · {edited} edited")
    nxt = next((i for i, r in enumerate(rows) if not state[r["doc_id"]].reviewed), None)
    print(f"next unreviewed index: {nxt}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(required=True)

    d = sub.add_parser("dump")
    d.add_argument("--start", type=int, default=0)
    d.add_argument("--count", type=int, default=15)
    d.add_argument("--out", required=True)
    d.add_argument("--queue", help="queue.json produced by the review plan")
    d.add_argument("--bucket", default="targeted", help="targeted | sample | unread")
    d.set_defaults(func=cmd_dump)

    a = sub.add_parser("apply")
    a.add_argument("--decisions", required=True)
    a.add_argument("--no-accept", action="store_true", help="log edits without accepting")
    a.set_defaults(func=cmd_apply)

    s = sub.add_parser("status")
    s.set_defaults(func=cmd_status)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
