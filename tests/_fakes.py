"""Offline fixtures, a stub OpenAI client, and a fake tokenizer for the tests.

A plain importable module rather than a `conftest.py`, keeping the repo's "no
conftest, no pytest fixtures — module-level constants and plain helpers"
convention while avoiding a 70-line fake duplicated across five test files. The
leading underscore keeps pytest from collecting it.

Every returned object is a `SimpleNamespace`, so attribute access matches the SDK
(`.id`, `.status`, `.output_file_id`, `.request_counts.completed`) **without
importing `openai`** — the suite must run against the stub with no SDK present.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from sxl.corpus import make_doc_id
from sxl.schema import empty_posting
from sxl.splits import split_for

SOURCE = "test/postings"

DOC_TEXT = (
    "Senior Python Engineer\n\n"
    "Acme Robotics is hiring a Senior Python Engineer in Austin, Texas. "
    "This is a full-time, hybrid role: three days per week in our downtown office.\n\n"
    "Compensation: $150,000 - $185,000 per year, plus equity.\n\n"
    "Requirements:\n"
    "- 6+ years of professional Python experience\n"
    "- Strong SQL and PostgreSQL\n"
    "- Experience with AWS and Docker\n"
    "- Bachelor's degree in Computer Science or equivalent experience\n\n"
    "Posted 2026-06-14."
)


def doc(i: int, *, text: str | None = None) -> dict[str, Any]:
    """One `docs.jsonl` row, keyed by the real F1 `make_doc_id`.

    Using the real id function matters: `split_for` is a hash of `doc_id`, so these
    documents distribute across train/dev/eval_pool exactly as production data does.
    """
    body = DOC_TEXT if text is None else text
    return {
        "doc_id": make_doc_id(SOURCE, f"k{i}"),
        "domain": "job_posting",
        "text": body,
        "source": SOURCE,
        "char_len": len(body),
    }


def docs_spanning_splits(per_split: int = 2) -> list[dict[str, Any]]:
    """Documents guaranteed to populate all three splits."""
    found: dict[str, list[dict[str, Any]]] = {"train": [], "dev": [], "eval_pool": []}
    i = 0
    while any(len(v) < per_split for v in found.values()):
        d = doc(i)
        bucket = found[split_for(d["doc_id"])]
        if len(bucket) < per_split:
            bucket.append(d)
        i += 1
        if i > 100_000:  # every loop has a cap (SPEC §5.4)
            raise RuntimeError("could not populate all three splits")
    return sorted((d for v in found.values() for d in v), key=lambda d: d["doc_id"])


def gold(**over: Any) -> dict[str, Any]:
    """A valid 16-field object: all nullables None, all enums "unknown"."""
    return empty_posting().model_dump(mode="json") | over


def gold_json(**over: Any) -> str:
    return json.dumps(gold(**over))


# --- F3 gold-set fixtures -----------------------------------------------------


def labeled_row(doc: dict[str, Any], **over: Any) -> dict[str, Any]:
    """One `eval_pool.jsonl` row in `teacher.ROW_KEYS` shape.

    `over` patches the 16-field `gold` object, not the envelope, because that is
    what every F3 test actually varies.
    """
    return {
        "doc_id": doc["doc_id"],
        "domain": doc["domain"],
        "text": doc["text"],
        "gold": gold(**over),
        "label_source": "teacher",
        "teacher_model": "gpt-4o-mini",
        "verified_by_human": False,
        "verified_at": None,
    }


def eval_pool_rows(n: int, *, text: str | None = None) -> list[dict[str, Any]]:
    """`n` labeled rows whose `doc_id`s really hash into the `eval_pool` bucket.

    Built through the real `make_doc_id`/`split_for` pair rather than by faking
    ids, so `sample_candidates` sees production-shaped data and the "every sampled
    id is in eval_pool" assertion means something.
    """
    rows: list[dict[str, Any]] = []
    i = 0
    while len(rows) < n:
        d = doc(i, text=text)
        if split_for(d["doc_id"]) == "eval_pool":
            rows.append(labeled_row(d))
        i += 1
        if i > 500_000:  # every loop has a cap (SPEC §5.4)
            raise RuntimeError(f"could not find {n} eval_pool documents")
    return sorted(rows, key=lambda r: r["doc_id"])


def write_pool(paths: Any, rows: list[dict[str, Any]]) -> None:
    from sxl.io import write_jsonl

    write_jsonl(paths.eval_pool, rows)


def write_candidates(paths: Any, rows: list[dict[str, Any]]) -> None:
    from sxl.io import write_jsonl

    write_jsonl(paths.candidates, sorted(rows, key=lambda r: r["doc_id"]))


# --- F4 metrics fixtures ------------------------------------------------------


def gold_row(doc: dict[str, Any], **over: Any) -> dict[str, Any]:
    """One `eval_gold.jsonl` row: `labeled_row` after human verification."""
    return labeled_row(doc, **over) | {
        "label_source": "human",
        "verified_by_human": True,
        "verified_at": "2026-08-01T00:00:00Z",
    }


def prediction_row(
    doc_id: str,
    parsed: dict[str, Any] | None,
    *,
    arm: str = "base_fewshot",
    schema_valid: bool | None = None,
    **over: Any,
) -> dict[str, Any]:
    """One `artifacts/predictions/<arm>.jsonl` row, keys in SPEC §3.3 order.

    `schema_valid` is derived from the real `validate_prediction` unless a test
    passes it explicitly — the integrity tests need to write the two out of sync
    on purpose, but nothing else should have to think about it.
    """
    from sxl.schema import validate_prediction

    valid = (validate_prediction(parsed) is not None) if schema_valid is None else schema_valid
    return {
        "doc_id": doc_id,
        "arm": arm,
        "raw_output": "" if parsed is None else json.dumps(parsed),
        "parsed": parsed,
        "schema_valid": valid,
        "latency_ms": 1234.5,
        "prompt_tokens": 1180,
        "completion_tokens": 143,
    } | over


def scripted(answers: list[str]) -> Any:
    """A `read_line` that replays `answers`, then behaves like a closed terminal.

    Running out of input raises `KeyboardInterrupt`, which `review` treats as
    save-and-quit — so a test script that ends mid-session exercises exactly the
    crash path resumability is supposed to survive.
    """
    queue = list(answers)

    def read_line(_prompt: str = "") -> str:
        if not queue:
            raise KeyboardInterrupt
        return queue.pop(0)

    return read_line


# --- batch result lines -------------------------------------------------------


def ok_line(
    custom_id: str,
    content: str,
    *,
    in_tok: int = 2000,
    out_tok: int = 300,
    finish: str = "stop",
) -> dict[str, Any]:
    return {
        "id": f"resp-{custom_id}",
        "custom_id": custom_id,
        "response": {
            "status_code": 200,
            "request_id": f"req-{custom_id}",
            "body": {
                "choices": [{"index": 0, "finish_reason": finish, "message": {"content": content}}],
                "usage": {"prompt_tokens": in_tok, "completion_tokens": out_tok},
            },
        },
        "error": None,
    }


def refusal_line(custom_id: str, text: str = "I cannot help with that") -> dict[str, Any]:
    line = ok_line(custom_id, "")
    line["response"]["body"]["choices"][0]["message"] = {"content": None, "refusal": text}
    return line


def http_line(custom_id: str, status: int = 429, message: str = "rate limited") -> dict[str, Any]:
    return {
        "id": f"resp-{custom_id}",
        "custom_id": custom_id,
        "response": {"status_code": status, "body": {"error": {"message": message}}},
        "error": None,
    }


def error_line(
    custom_id: str, code: str = "invalid_request", message: str = "bad thing"
) -> dict[str, Any]:
    """The shape of an `error_file_id` line — e.g. an expired batch's unprocessed requests."""
    return {
        "id": f"resp-{custom_id}",
        "custom_id": custom_id,
        "response": None,
        "error": {"code": code, "message": message},
    }


def always_ok(request: dict[str, Any]) -> dict[str, Any]:
    """The default responder: every document gets a valid, all-absent label."""
    return ok_line(request["custom_id"], gold_json())


# --- the stub client ----------------------------------------------------------


class _Files:
    def __init__(self, owner: FakeOpenAI) -> None:
        self._owner = owner

    def create(self, *, file: Any, purpose: str) -> SimpleNamespace:
        self._owner._maybe_raise("files.create")
        payload = file[1] if isinstance(file, tuple) else file
        text = payload.decode("utf-8") if isinstance(payload, bytes) else str(payload)
        requests = [json.loads(line) for line in text.splitlines() if line.strip()]
        file_id = f"file-{len(self._owner.files_store)}"
        self._owner.files_store[file_id] = text
        self._owner.requests_by_file[file_id] = requests
        self._owner.calls.append(("files.create", file_id))
        return SimpleNamespace(id=file_id, purpose=purpose, bytes=len(text))

    def content(self, file_id: str) -> SimpleNamespace:
        self._owner._maybe_raise("files.content")
        self._owner.calls.append(("files.content", file_id))
        text = self._owner.files_store[file_id]
        return SimpleNamespace(text=text, content=text.encode("utf-8"))


class _Batches:
    def __init__(self, owner: FakeOpenAI) -> None:
        self._owner = owner

    def create(
        self, *, input_file_id: str, endpoint: str, completion_window: str, metadata: Any = None
    ) -> SimpleNamespace:
        owner = self._owner
        owner._maybe_raise("batches.create")

        batch_id = f"batch-{len(owner.batches_store)}"
        requests = owner.requests_by_file[input_file_id]

        # The responder runs now, so results are ready the moment polling reaches a
        # terminal status. Tests script the *statuses*, not the timing of work.
        out_lines, err_lines = [], []
        for request in requests:
            line = owner.responder(request)
            (err_lines if line.get("error") else out_lines).append(line)

        out_id = err_id = None
        if out_lines:
            out_id = f"file-out-{batch_id}"
            owner.files_store[out_id] = "\n".join(json.dumps(x) for x in out_lines) + "\n"
        if err_lines:
            err_id = f"file-err-{batch_id}"
            owner.files_store[err_id] = "\n".join(json.dumps(x) for x in err_lines) + "\n"

        batch = SimpleNamespace(
            id=batch_id,
            status="validating",
            input_file_id=input_file_id,
            output_file_id=out_id,
            error_file_id=err_id,
            metadata=metadata or {},
            errors=owner.errors,
            request_counts=SimpleNamespace(
                total=len(requests), completed=len(out_lines), failed=len(err_lines)
            ),
        )
        owner.batches_store[batch_id] = batch
        owner.status_script[batch_id] = list(owner.statuses)
        owner.created.append(batch)
        owner.calls.append(("batches.create", batch_id))
        return batch

    def retrieve(self, batch_id: str) -> SimpleNamespace:
        owner = self._owner
        owner._maybe_raise("batches.retrieve")
        owner.calls.append(("batches.retrieve", batch_id))
        batch = owner.batches_store[batch_id]
        script = owner.status_script[batch_id]
        batch.status = script.pop(0) if len(script) > 1 else script[0]
        return batch

    def list(self, *, limit: int = 20) -> SimpleNamespace:
        owner = self._owner
        owner._maybe_raise("batches.list")
        owner.calls.append(("batches.list", limit))
        return SimpleNamespace(data=list(owner.batches_store.values())[-limit:])


class FakeOpenAI:
    """A stub with the four surfaces F2 touches.

    - `responder(request) -> result_line` decides what each document comes back as.
    - `statuses` is the script `batches.retrieve` walks; the last value sticks, so
      `("in_progress", "expired")` exercises the expiry path.
    - `raise_on={"batches.create": [exc, None]}` scripts per-method failures for the
      retry and crash tests; `None` means "succeed this time".
    - `.calls` is the audit log that every "made zero API calls" assertion reads.
    """

    def __init__(
        self,
        responder: Any = always_ok,
        *,
        statuses: tuple[str, ...] = ("in_progress", "completed"),
        raise_on: dict[str, list[BaseException | None]] | None = None,
        errors: Any = None,
    ) -> None:
        self.responder = responder
        self.statuses = statuses
        self.raise_on = {k: list(v) for k, v in (raise_on or {}).items()}
        self.errors = errors
        self.calls: list[tuple[str, Any]] = []  # successful calls only
        self.attempts: dict[str, int] = {}  # including ones that raised
        self.files_store: dict[str, str] = {}
        self.requests_by_file: dict[str, list[dict[str, Any]]] = {}
        self.batches_store: dict[str, SimpleNamespace] = {}
        self.status_script: dict[str, list[str]] = {}
        self.created: list[SimpleNamespace] = []
        self.files = _Files(self)
        self.batches = _Batches(self)

    def _maybe_raise(self, method: str) -> None:
        # Counted before the raise: an attempt that errors is still an attempt, and
        # the retry tests need to distinguish "tried once" from "tried three times".
        self.attempts[method] = self.attempts.get(method, 0) + 1
        script = self.raise_on.get(method)
        if not script:
            return
        exc = script.pop(0)
        if exc is not None:
            raise exc

    # --- assertion helpers -----------------------------------------------------

    def custom_ids_requested(self) -> list[str]:
        """Every `custom_id` this client was actually asked to process."""
        return [r["custom_id"] for reqs in self.requests_by_file.values() for r in reqs]

    def n_calls(self, method: str) -> int:
        return sum(1 for name, _ in self.calls if name == method)


# --- CLI plumbing -------------------------------------------------------------


def patch_cli(monkeypatch: Any, tmp_path: Any, client: FakeOpenAI | None = None) -> Any:
    """Point the CLI at a tmp tree and (optionally) a stub client.

    The CLI calls `label()` with no `paths=`/`client=`, so redirecting
    `TeacherPaths.default` is how a CLI test avoids touching the real `data/` and
    the network. Also neutralizes `sleep` so polling is instant.
    """
    from sxl.teacher import TeacherPaths

    paths = TeacherPaths.in_dir(tmp_path)
    monkeypatch.setattr(TeacherPaths, "default", classmethod(lambda cls: paths))
    monkeypatch.setattr("sxl.teacher.time.sleep", lambda _: None)
    if client is not None:
        monkeypatch.setattr("sxl.teacher.make_client", lambda *a, **k: client)
    return paths


def write_docs(paths: Any, docs: list[dict[str, Any]]) -> None:
    from sxl.io import write_jsonl

    write_jsonl(paths.docs, docs)


# --- F6 fine-tune fixtures ----------------------------------------------------


class FakeQwenTokenizer:
    """A chat template close enough to Qwen3's to test train/inference parity.

    The behaviour that matters, and that a naive stub would paper over: with
    `add_generation_prompt=True, enable_thinking=False` the real template opens the
    assistant turn *and* emits an **empty** `<think></think>` block, while an
    assistant message passed inside the message list is rendered without one. That
    asymmetry is the entire reason `train_lora.render_example` concatenates the
    inference prefix instead of handing the assistant turn to
    `apply_chat_template` — so the fake has to reproduce it, or the parity test
    proves nothing.

    `enable_thinking` defaults to `True` and raises: a call site that forgets to
    pass it fails loudly here on the laptop rather than silently on a T4 (SPEC
    §3.7), where the only symptom is a ruined `schema_valid_rate`.
    """

    eos_token = "<|im_end|>"

    def apply_chat_template(
        self,
        messages: Any,
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        enable_thinking: bool = True,
        **_kw: Any,
    ) -> str:
        if tokenize:
            raise NotImplementedError("the laptop never tokenizes for real (SPEC §2.1)")
        if enable_thinking:
            raise AssertionError(
                "SPEC §3.7: every apply_chat_template call passes enable_thinking=False"
            )
        out = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        if add_generation_prompt:
            out += "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        return out

    def __call__(self, text: str) -> dict[str, Any]:
        """A crude whitespace tokenizer — `token_length_stats` only needs lengths."""
        return {"input_ids": text.split()}


def train_row(i: int = 0, *, text: str | None = None, **over: Any) -> dict[str, Any]:
    """One `train.jsonl` row: `labeled_row` is already exactly that shape."""
    return labeled_row(doc(i, text=text), **over)
