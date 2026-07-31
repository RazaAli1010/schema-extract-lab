## F2 — Teacher labeling pipeline (batched, cached, spend-capped)

**Goal:** `sxl teacher label` turns `docs.jsonl` into `train.jsonl` (5,000),
`dev.jsonl` (300), and `eval_pool.jsonl` (361) of schema-valid `gold` labels
produced by `gpt-4o-mini`, for well under $25, resumable after a crash, and
without ever paying twice for the same document.

**Depends on:** F0 (`schema.py`, `config.py`, `io.py`, `splits.py`), F1 (`docs.jsonl`)

> **Provider note.** An earlier draft of this spec specified `claude-sonnet-5` via
> the Anthropic Message Batches API. **SPEC.md §6.5 is the source of truth and says
> `gpt-4o-mini` via the OpenAI Batch API**; `config.py`, `pyproject.toml` and
> `.env.example` have said so since commit `7bc3aad`. This file was retargeted
> accordingly. The orphaned `anthropic` SDK was uninstalled and
> `tests/test_no_torch_import.py` keeps it uninstalled.

---

### Context digest

**Hardware (SPEC §2.1):** laptop, 8 GB RAM, no GPU. This feature is
network-bound, not compute-bound — it is exactly the kind of work the laptop
*should* do. Nothing here touches torch.

**Record shape (SPEC §3.3)** — output rows, keys in this order:
```json
{"doc_id": "...", "domain": "job_posting", "text": "...",
 "gold": { <the 16 JobPosting fields> },
 "label_source": "teacher",
 "teacher_model": "gpt-4o-mini",
 "verified_by_human": false,
 "verified_at": null}
```
`text` is the **full, untruncated** document. Truncation to `MAX_INPUT_CHARS` is a
prompt-time concern; F5/F6 apply the same `prompts.truncate` themselves so the arms
stay identical (SPEC §3.6) while this file stays a faithful copy of the corpus.

**Schema (SPEC §3.2):** the 16 fields, the 5 enums, and the **null/unknown rule**
— enum fields are never `null`, absence is `"unknown"`; `EducationLevel.none`
("no degree required") ≠ `"unknown"` ("posting is silent"); non-enum fields use
`null`; `required_skills` uses `[]`. The teacher prompt states this rule
explicitly, because it is the single most common labeling error.

**Validation (F0):** `sxl.schema.validate_prediction(obj) -> JobPosting | None`
is the *only* definition of schema-validity in this project. F2 calls it, never
re-implements it, and dumps with `mode="json"` so `Enum` members do not leak into
the written row.

**Splits (SPEC §3.4):** `splits.split_for(doc_id)` → `train` | `dev` | `eval_pool`.
F2 materializes the three files by routing each labeled row through this function.
Never re-derive or override the bucket.

**Teacher config (SPEC §6.5, from F0 `config.py`):**
```python
TEACHER_MODEL = "gpt-4o-mini"            # alt: "gpt-4o"
MAX_TEACHER_SPEND_USD = 25.0             # hard stop
TEACHER_MAX_RETRIES = 3
MAX_INPUT_CHARS = 6000
N_TRAIN_TARGET, N_DEV_TARGET = 5000, 300
```

**Budget, measured.** The real selection is **5,661 documents** (train 5,000 of
6,752 available + dev 300 of 387 + eval_pool 361 uncapped). `sxl teacher label
--split all --dry-run` projects **13.25M input + 1.98M output tokens ≈ $1.59** at
`gpt-4o-mini` Batch rates ($0.075/$0.30 per M — half the $0.15/$0.60 standard
rates). Input runs ~2.3k tokens/doc rather than the ~1.2k first estimated, because
the verbatim `JSON_SCHEMA` in the system prompt is ~800 tokens on its own. The cap
therefore sits ~15x above the expected spend and acts as a runaway-loop guard, not
a live budget ceiling.

**Principles (SPEC §5):** idempotent and resumable — "a crashed run must never
re-bill". Every loop capped, caps in config. Fail loudly on contract violations.

### Context deltas

Additions to `config.py` (applied to SPEC §6 before implementing):

```python
TEACHER_CACHE_PATH   = DATA/"labeled"/"_teacher_cache.jsonl"  # append-only, gitignored
TEACHER_BATCHES_PATH = DATA/"labeled"/"_batches.json"         # in-flight ledger, gitignored
TEACHER_STATS_PATH   = RESULTS/"teacher_stats.json"           # committed

TEACHER_BATCH_SIZE           = 500      # requests per batch: 5,661 docs -> 12 batches
TEACHER_MAX_INFLIGHT_BATCHES = 4        # OpenAI caps *enqueued* tokens per model
TEACHER_POLL_SECONDS         = 60
TEACHER_MAX_WAIT_S           = 86400    # 24h Batch SLA; exit non-zero past this

TEACHER_MAX_TOKENS       = 1024
TEACHER_DOC_RETRY_ROUNDS = 1
TEACHER_RETRY_BACKOFF_S  = (4, 16, 64)  # len() == TEACHER_MAX_RETRIES

# STANDARD rates per 1M tokens. Do NOT pre-discount -- `cost_of` multiplies by
# TEACHER_BATCH_DISCOUNT, so halving the table double-applies the discount.
TEACHER_PRICE_USD      = {"gpt-4o-mini": {"in": 0.15, "out": 0.60},
                          "gpt-4o":      {"in": 2.50, "out": 10.00}}
TEACHER_BATCH_DISCOUNT = 0.5

# Pre-flight projection only; no tokenizer on the laptop, no tiktoken dependency.
TEACHER_CHARS_PER_TOKEN         = 4.0
TEACHER_REQUEST_OVERHEAD_TOKENS = 32
TEACHER_EST_OUTPUT_TOKENS       = 350
TEACHER_MIN_DOCS_FOR_MEAN_COST  = 50

TEACHER_MAX_ENUM_SHARE         = 0.95   # no enum may be >95% a single value
TEACHER_QUALITY_MIN_N          = 200    # ...but only at this sample size
TEACHER_MAX_SKILLS_EMPTY_SHARE = 0.5
```

`TEACHER_MAX_INFLIGHT_BATCHES` and `TEACHER_QUALITY_MIN_N` are load-bearing and were
not in the original draft. Submitting all 12 batches at once is ~13M enqueued input
tokens, which a low usage tier returns as status `failed` for every batch. And
without the sample-size gate, this spec's own `--split dev --limit 20` smoke run
trips the >95% enum guard and exits 1 on a perfectly healthy run — 20 postings
really are all `education_level: "unknown"`.

Addition to SPEC §3.3: the cache file `_teacher_cache.jsonl` is an internal
artifact with shape
```json
{"doc_id": "...", "teacher_model": "gpt-4o-mini", "prompt_sha": "...",
 "raw_output": "...", "usage": {"input_tokens": 0, "output_tokens": 0},
 "ok": true, "error": null, "finish_reason": "stop",
 "batch_id": "batch_abc123", "at": "..."}
```
`input_tokens`/`output_tokens` are provider-neutral names mapped once, in
`parse_result_line`, from OpenAI's `prompt_tokens`/`completion_tokens`.

The ledger `_batches.json` is the second internal artifact:
```json
{"version": 1, "model": "gpt-4o-mini", "prompt_sha": "...",
 "batches": [{"tag": "main-0000", "attempt": 1, "input_file_id": "file-xyz",
              "batch_id": "batch_abc123", "output_file_id": "...", "error_file_id": null,
              "docs_sha": "...", "doc_ids": ["..."], "n_requests": 500,
              "status": "completed", "submitted_at": "...", "completed_at": "...",
              "harvested": true}]}
```
`harvested` means "the results are in the cache" — it is the resume flag. Both files
are **not** contract files and are gitignored by `data/**`.

---

### Scope

1. **`src/sxl/prompts.py`** — created here, **extended** (not rewritten) by F5,
   which appends below the marker at the bottom and edits nothing above it.
   - `SYSTEM_TEACHER`: the extraction role, the **verbatim `JSON_SCHEMA`** from
     `sxl.schema` serialized with `json.dumps(..., indent=2, sort_keys=True)`, the
     null/unknown rule with the `none` vs `unknown` distinction spelled out, the
     copy-don't-infer rule (a posting that does not mention salary gets `null`, not
     a market estimate), format normalization (ISO-3166 alpha-2, ISO-4217,
     `YYYY-MM-DD`), and an instruction to emit only a JSON object.
   - `build_teacher_prompt(text) -> tuple[str, str]` → `(system, user)`, the user
     half truncated via the shared `truncate`.
   - `prompt_sha(*parts) -> str` — sha256 of the NUL-joined parts, first 16 hex.
     **Variadic**, replacing the draft's `prompt_sha(system, user)`: a per-document
     digest cannot serve as a cache key (`teacher_stats.json` has one scalar
     `prompt_sha`, and cache subtraction would have to re-render every prompt to
     compute it). It also buys nothing — `doc_id` is already a content hash, so a
     document's text is immutable.
   - `TEACHER_PROMPT_SHA = prompt_sha(PROMPT_VERSION, SYSTEM_TEACHER, str(MAX_INPUT_CHARS))`
     is the stable cache key. Because `SYSTEM_TEACHER` embeds the schema verbatim,
     adding a 17th field or renaming an enum member invalidates every cached label
     automatically. `PROMPT_VERSION` is the manual escape hatch. The model is
     deliberately *not* folded in — cache rows record `teacher_model` separately.

2. **`teacher.py::extract_json(raw) -> dict | None`**
   Tolerant parser, tried in order: whole string → strip ```` ```json ```` fences →
   substring from the first `{` to the last `}`. Every branch checks
   `isinstance(obj, dict)`, because `json.loads` succeeds on `"[1,2,3]"`, `"null"`
   and `"5"` — without the guard those land in `schema_invalid` when the honest
   bucket is `parse_failed`. Never repair malformed JSON by regex: a document the
   teacher could not label cleanly is dropped, not guessed at.

3. **The Batch API surface.** `openai==2.51.0`, verified against the installed SDK:
   `client.files.create(file=..., purpose="batch")`,
   `client.batches.create(input_file_id=..., endpoint="/v1/chat/completions",
   completion_window="24h", metadata=...)`, `client.batches.retrieve(id)` →
   `status` in `validating|in_progress|finalizing|completed|failed|expired|cancelled`,
   `client.batches.list(limit=...)`, `client.files.content(file_id).text`.
   One input line per document:
   ```json
   {"custom_id": "<doc_id>", "method": "POST", "url": "/v1/chat/completions",
    "body": {"model": "gpt-4o-mini", "messages": [...], "temperature": 0.0,
             "max_tokens": 1024,
             "response_format": {"type": "json_schema",
                                 "json_schema": {"name": "job_posting", "strict": true,
                                                 "schema": <JSON_SCHEMA>}}}}
   ```
   `custom_id` **is** the `doc_id`, which is how results are rejoined.

   **Structured Outputs** (`strict: true`) is what drives `n_parse_failed` and
   `n_schema_invalid` to ~0. `JSON_SCHEMA` was verified strict-compatible: only
   `$defs`/`$ref`/`anyOf`/`enum`/array-of-string, `additionalProperties: false`, all
   16 keys required, no unsupported keywords.

   `submit_batch` persists the ledger entry carrying `input_file_id` **before**
   calling `batches.create`, and every create-retry is preceded by
   `_adopt_existing_batch`, which scans `batches.list` for a batch matching that
   `input_file_id` or `metadata["docs_sha"]`. Without that scan, a crash in the
   window between upload and ledger-write either duplicate-bills or abandons a paid
   batch.

   `poll_batch` returns the **batch object**, not a list of results — the terminal
   status drives four different recovery paths. `harvest_batch` does the downloading
   and appends each row to the cache as it parses; `parse_results` stays pure and
   client-free so the whole mapping is unit-testable with plain dicts.

   At most `TEACHER_MAX_INFLIGHT_BATCHES` batches are in flight at once.

4. **`teacher.py::cost_of(usage, model) -> float`**
   ```python
   p = TEACHER_PRICE_USD[model]
   return TEACHER_BATCH_DISCOUNT * (usage["input_tokens"]/1e6*p["in"]
                                  + usage["output_tokens"]/1e6*p["out"])
   ```
   The spend guard runs in two places, both before money moves: once in `label()`
   right after `todo` is computed and before any client is touched (this is the
   number `--dry-run` prints), and again immediately before each chunk's
   `files.create`. Before any cost data exists the projection is the char-based
   `estimate_usage`; once ≥ `TEACHER_MIN_DOCS_FOR_MEAN_COST` documents are priced it
   is `max(observed mean × remaining, char estimate)` — a spend cap should err high,
   and the 50-document floor stops one anomalous response from extrapolating a fake
   projection and aborting a healthy run. (The draft's "project from the mean
   cost-per-doc so far" was written for a sequential loop, where a mean exists from
   the first response.)

   The cap governs **new spend this run**, not lifetime spend: `spend_usd_cached` is
   excluded, or a completed $1.59 run would be one flag away from tripping its own
   cap on a zero-cost re-run. On breach, raise `SpendCapExceeded` with the figures —
   the CLI turns it into exit 1. Do not submit and then apologize.

5. **`teacher.py::label(...) -> dict`** — orchestrator:
   - load `docs.jsonl`; `select_docs` filters by `split_for(doc_id)` and caps `train`
     at `N_TRAIN_TARGET` and `dev` at `N_DEV_TARGET`; **`eval_pool` is never capped**
     (F3 needs ≥ 330). Caps and `--limit` are taken by **sorted `doc_id`**, never by
     random sample, so re-runs select the identical documents
   - **subtract everything already in the cache with a matching `prompt_sha` and
     `teacher_model` *and* `ok: true`** — this is the no-double-billing guarantee.
     The `ok` filter matters: an `api_error` row is a record of a failed attempt, not
     a paid label, so it must not suppress a re-request on a later run
   - with `--resume`, harvest every un-harvested ledger entry **before** submitting
     anything new; the work is paid for the moment a batch is created
   - chunk into `TEACHER_BATCH_SIZE`, submit, poll, append every result to the cache
     **as it arrives** via `io.append_jsonl` (crash-safe)
   - one retry round, then parse → `validate_prediction` → build rows → `write_jsonl`
   - **output files are a pure function of (docs, cache, caps, limit)**: never
     appended to, never merged with a previous version on disk, and only the splits
     in scope are written, so `--split dev` never truncates `train.jsonl`

6. **Failure accounting.** Every document ends in exactly one bucket:
   `ok` (valid), `parse_failed`, `schema_invalid`, `api_error`. A document with no
   cache row at all counts as `api_error`, which keeps the four buckets a true
   partition — `report_stats` raises `ValueError` if they do not sum to
   `n_requested`. Rows that are not `ok` are excluded from the output files and
   listed in `results/teacher_stats.json`:
   ```json
   {"teacher_model": "gpt-4o-mini", "prompt_sha": "...", "prompt_version": "v1",
    "split": "all", "limit": 0,
    "n_requested": 0, "n_cached": 0, "n_new_requests": 0,
    "n_ok": 0, "n_parse_failed": 0, "n_schema_invalid": 0, "n_api_error": 0,
    "n_batches": 0, "spend_usd": 0.0, "spend_usd_cached": 0.0,
    "tokens": {"input": 0, "output": 0},
    "split_counts": {"train": 0, "dev": 0, "eval_pool": 0},
    "field_null_rate": {"salary_min": 0.0},
    "enum_distribution": {"seniority": {"senior": 0}},
    "quality_warnings": [],
    "generated_at": "...", "git_sha": "..."}
   ```
   `field_null_rate` and `enum_distribution` are the label-quality smoke test:
   if `remote_mode` is 99% `"unknown"` or `required_skills` is empty for half the
   corpus, the prompt is broken and F3's human verification will waste hours
   discovering it. **They are printed to stdout at the end of the run.**
   `enum_distribution` seeds every member of every enum at zero, because
   `"principal": 0` across 5,000 postings is real signal that a plain `Counter` would
   omit. `field_null_rate` keeps its name but is really an *absence* rate — for enum
   fields absence is `"unknown"`, not `null`; the predicate `is_absent` is defined
   once and F4 reuses it.

   The >95% guard fires only at `n_ok >= TEACHER_QUALITY_MIN_N`. When it fires,
   **every file is written first and the CLI then exits 1**: the money is already
   spent and the data is worth inspecting, but SPEC §5.5 says fail loudly and an exit
   0 with a red line scrolled off the top of a long run is not loud.

7. **Retry policy.** `TEACHER_MAX_RETRIES = 3` applies to transient API failures
   (`APIConnectionError`, `APITimeoutError`, `RateLimitError`, `InternalServerError`)
   with exponential backoff (4s, 16s, 64s). `BadRequestError`, `AuthenticationError`,
   `PermissionDeniedError` and `NotFoundError` are **never** retried — a malformed
   request body is malformed on every attempt, and retrying only delays the
   diagnosis. Terminal batch statuses:

   | status | handling |
   |---|---|
   | `completed` | harvest and move on |
   | `expired` | partials are in `output_file_id`, unprocessed requests in `error_file_id` and **not billed**; harvest both, resubmit only the missing ids |
   | `failed` | never processed, never billed. A queue/token-limit error is resubmitted; anything else raises `TeacherError` with `batch.errors` verbatim |
   | `cancelled` | harvest partials, then raise `BatchCancelled` → exit 1. Never silently resubmit work a human cancelled |

   Individual documents that come back unparseable are retried **at most once**, in a
   single follow-up batch, with the same prompt (no prompt mutation — that would make
   the training data inconsistent). Their new cache rows supersede by last-wins.
   `schema_invalid` is deliberately **not** retried: at `temperature=0` under
   Structured Outputs a shape violation is deterministic and will reproduce, so
   retrying it is pure spend. If the retry set exceeds 20% of the wave, print a loud
   "this is a prompt problem, not transience" line and still run the single round.

8. **`sxl teacher label` CLI** (replaces the F0 stub):
   `--split [train|dev|eval_pool|all]`, `--limit INT` (0 = no cap),
   `--resume/--no-resume`, `--model TEXT` (default `TEACHER_MODEL`), `--dry-run`.
   `--dry-run` renders prompts, counts documents, prints the projected cost, and
   **makes zero API calls and writes zero files** — run it before any real
   invocation. It never constructs a client, so it works with no `.env` present.
   An unknown `--split` or an unpriced `--model` exits **1, not 2** — exit 2 means
   "not implemented" in this repo and must stay unambiguous. A spend cap that cannot
   price the model is not a spend cap.

---

### Out of scope

- Human verification of any label — F3. F2 writes `verified_by_human: false` and
  `label_source: "teacher"` on **every** row, including `eval_pool`. F3 is the
  only feature permitted to write `"human"`.
- Scoring the teacher as an arm — F4 computes `results/metrics/teacher.json` from
  `eval_pool` predictions against F3's `eval_gold`.
- Prompt engineering for the *student* few-shot arms — F5 extends `prompts.py`
  with `build_student_prompt`; F2's teacher prompt is separate and must not be
  reused for the student (the student's prompt is part of the measured arm).
- Any second teacher / ensemble labeling — not in scope, and mixing two teachers
  within one run is explicitly forbidden. `label()` raises `TeacherError` when the
  ledger's `model`/`prompt_sha` disagree with the current run.

---

### Implementation notes

- **`openai==2.51.0`.** The SDK surface above was verified against the installed
  version (`dir(OpenAI().batches)` → `cancel, create, list, retrieve`;
  `dir(OpenAI().files)` → `content, create, delete, list, retrieve`). Re-verify
  before changing the loop — this is the one API most likely to move.
- **`max_retries=0` on the client.** The SDK's automatic retry could re-issue a
  `batches.create` whose first attempt actually succeeded, creating and billing a
  duplicate batch. F2 owns retries so that every one is preceded by an adoption scan.
- **`max_tokens` vs `max_completion_tokens`.** `max_tokens` is deprecated in favour
  of `max_completion_tokens` but still accepted on `/v1/chat/completions` for
  non-reasoning models, which `gpt-4o-mini` is. If a smoke run returns
  `invalid_request_error` naming the parameter, it is a one-line swap — which is what
  the 20-document smoke run is for.
- **No prompt-caching directive.** There is no OpenAI equivalent of Anthropic's
  `cache_control` on batch requests. The system message goes first so any automatic
  prefix caching can apply, but the projection assumes none: if it applies, the run
  comes in under budget, which is the safe direction.
- **`finish_reason == "length"` is `ok: true`.** The truncated text is real,
  `extract_json` fails on it, and it lands in `parse_failed`. That is the honest
  bucket and it makes a too-low `TEACHER_MAX_TOKENS` visible in the stats rather than
  hiding as a transport error.
- **The cache stores raw output, not the bucket.** Improving `extract_json` or
  fixing a schema bug then re-buckets every document from disk at **zero cost**.
  Storing the bucket would force a re-bill to recover from a parser bug.
- **`temperature=0`** on teacher calls. `max_tokens = 1024` (the 16-field object is
  ~350 tokens; 1024 leaves room without inviting rambling).
- **Do not put the API key in code.** `make_client` reads `OPENAI_API_KEY` via
  `python-dotenv` from `.env`, which is gitignored (F0), and raises `MissingApiKey`
  with a clear message if unset. `.env` is loaded **there**, not in `config.py`,
  whose docstring promises stdlib-only imports and which would otherwise mutate
  `os.environ` for every unrelated `sxl` command.
- **Batch sizing.** 500-request chunks keep each poll cycle's result download small
  on a laptop and mean a crash loses at most one chunk of *wall time* — never money;
  the cache has everything that already returned.
- **`eval_pool` gets labeled too, in full.** F3 needs teacher labels as the
  *starting point* for human correction — verifying is far faster than labeling
  from scratch. **Do not cap `eval_pool`**: F3 samples 330 candidates from it and
  a cap below that would starve the gold set. F1 delivered 361 documents in this
  bucket, so the total failure budget is **31 documents (8.6%)**.

---

### Test plan

All offline against `tests/_fakes.py::FakeOpenAI`; **no test makes a network call
or writes outside `tmp_path`**. `TeacherPaths.in_dir(tmp_path)` is passed to every
call, which makes it structurally impossible for a test to touch the real `data/`.

- `tests/test_teacher_parse.py` — `extract_json`: bare / fenced / leading prose /
  trailing prose all parse; prose, truncated JSON, trailing commas, **JSON arrays and
  scalars**, empty and `None` all return `None`.
- `tests/test_teacher_prompts.py` — the system prompt embeds every field name and
  every enum member and states the `none` vs `unknown` rule; truncation at
  `MAX_INPUT_CHARS`; `prompt_sha` is 16 hex, stable and order-sensitive;
  `TEACHER_PROMPT_SHA` does not vary by document and *does* change when the system
  prompt or `MAX_INPUT_CHARS` changes.
- `tests/test_teacher_cost.py` — `cost_of({"input_tokens": 1e6, "output_tokens": 1e6},
  "gpt-4o-mini") == 0.375`; the price table holds standard rates; `--dry-run` and a
  breached cap both leave `client.calls == []`; cached spend is excluded from the cap;
  the CLI exits 1 for a breached cap, an unpriced model, and an unknown split.
- `tests/test_teacher_cache.py` — 3-of-5 cached → exactly 2 requested; a changed
  `prompt_sha` or `teacher_model` invalidates everything; an `ok: false` row does not
  suppress a re-request; last-row-wins; the file is append-only; a crash mid-run keeps
  every arrived result and the re-run requests only the remainder; **re-running a
  completed run makes zero API calls, costs $0, and produces byte-identical files**.
- `tests/test_teacher_batch.py` — request shape and strict Structured Outputs; the
  ledger records `input_file_id` before creating; **resume adopts an orphaned batch
  and never creates a second one**; `poll_batch` returns the terminal object and
  raises `BatchTimeout` naming `--resume`; usage-name mapping; refusal / non-200 /
  `length` bucketing; expired harvests partials and resubmits only the rest;
  non-transient `failed` raises without resubmitting; `cancelled` harvests then
  raises; transient errors retry with `[4, 16]` backoff; `BadRequestError` is tried
  once; the in-flight cap holds; a document is retried at most once and a
  `schema_invalid` one not at all.
- `tests/test_teacher_routing.py` — the 8 SPEC §3.3 keys in order; `label_source`,
  `verified_by_human`, `verified_at` and a constant `teacher_model`; `gold` has the 16
  fields in `FIELD_NAMES` order, validates, and is JSON-serializable; `text` is
  untruncated; every row matches `split_for`; the three files are disjoint;
  `--split dev` leaves `train.jsonl` byte-unchanged; output is byte-identical across
  runs; `--limit 10` twice selects the identical documents; train and dev are capped
  and `eval_pool` is not; the four buckets partition `n_requested`.
- `tests/test_teacher_stats.py` — the exact key set; `field_null_rate` covers all 16
  fields and treats `"unknown"`/`[]` as absent while distinguishing `none`;
  `enum_distribution` includes zero-count members and sums to `n_ok`; the >95% guard
  fires at 300 rows and is suppressed at 20; the `required_skills` guard; the CLI
  exits 1 on a warning **but still writes all four files**; `report_stats` raises when
  the buckets do not partition.
- `tests/test_no_torch_import.py` — importing `sxl.teacher` must not import `openai`,
  and `anthropic` must not be installed.
- `tests/test_cli.py` — the `("teacher", "label")` entry was removed from
  `UNIMPLEMENTED` and replaced with a `--help`-only test (never invoked; it spends
  money).

---

### Verify

```bash
sxl teacher label --split all --dry-run          # prints doc counts + projected USD, no calls
sxl teacher label --split dev --limit 20         # small real run first — check the money works
python - <<'PY'
import json
rows=[json.loads(l) for l in open("data/labeled/dev.jsonl", encoding="utf-8")]
from sxl.schema import validate_prediction, FIELD_NAMES
KEYS=["doc_id","domain","text","gold","label_source","teacher_model",
      "verified_by_human","verified_at"]
assert all(list(r)==KEYS for r in rows)
assert all(validate_prediction(r["gold"]) for r in rows)
assert all(list(r["gold"])==list(FIELD_NAMES) for r in rows)
print("valid:", len(rows))
PY

# idempotence: must cost $0 and make zero calls
sxl teacher label --split dev --limit 20
python -c "import json;s=json.load(open('results/teacher_stats.json'));\
assert s['spend_usd']==0.0 and s['n_new_requests']==0 and s['spend_usd_cached']>0;print('cached OK')"

# resume: kill during polling, then re-run
sxl teacher label --split eval_pool --limit 40    # Ctrl-C once "in_progress" appears
cat data/labeled/_batches.json                    # one entry, harvested: false, batch_id present
sxl teacher label --split eval_pool --limit 40 --resume
python -c "import json;s=json.load(open('results/teacher_stats.json'));\
assert s['n_new_requests']==0;print('resume re-billed nothing:', s['spend_usd'])"

sxl teacher label --split all                     # the full ~$1.6 run
python - <<'PY'
import json
from sxl.splits import split_for
L=lambda p:[json.loads(l) for l in open(p, encoding="utf-8")]
tr,dv,ep=L("data/labeled/train.jsonl"),L("data/labeled/dev.jsonl"),L("data/labeled/eval_pool.jsonl")
ids=lambda R:{r["doc_id"] for r in R}
assert not (ids(tr)&ids(dv)) and not (ids(tr)&ids(ep)) and not (ids(dv)&ids(ep))
assert len(tr)>=4500 and len(dv)>=280 and len(ep)>=330, (len(tr),len(dv),len(ep))
for rows,name in ((tr,"train"),(dv,"dev"),(ep,"eval_pool")):
    assert all(split_for(r["doc_id"])==name for r in rows), name
print("OK", len(tr), len(dv), len(ep))
PY
cat results/teacher_stats.json                    # spend_usd < 25, inspect enum_distribution
```

Expected: the dry run projects ≈$1.59 on 5,661 documents; the 20-doc run costs under
a cent and produces 20 valid labels; the full run prints `OK` with ≥4,500 / ≥280 /
≥330 and a `spend_usd` far under the cap. Re-running `sxl teacher label --split all`
immediately after must make **zero** API calls and cost **$0**.

`eval_pool >= 330` out of 361 available is the tightest constraint in the feature. If
it comes in short, the fix is re-running the failed documents, **not** lowering the
threshold.

---

### Acceptance criteria

- [ ] `train.jsonl` ≥ 4,500 rows, `dev.jsonl` ≥ 280, `eval_pool.jsonl` ≥ 330
      (F3 samples 330 candidates from it); all three mutually disjoint by `doc_id`.
- [ ] Every row in every output file passes `validate_prediction(row["gold"])`
      and has exactly the 8 SPEC §3.3 keys in order.
- [ ] Every row has `label_source: "teacher"`, `verified_by_human: false`,
      `verified_at: null`, and a `teacher_model` identical across the file.
- [ ] Total `spend_usd` in `results/teacher_stats.json` is **< 25.0** (expected
      ≈ $1.59), and the run aborts before submitting if the projection would exceed it.
- [ ] Re-running the same command after a completed run makes zero API calls:
      `n_new_requests == 0 and spend_usd == 0.0 and spend_usd_cached > 0`, and the
      output files are byte-identical to the first run's.
- [ ] Killing the process mid-run and re-running with `--resume` re-requests only
      the documents that had not returned — demonstrated once against the real API
      on a 40-document slice.
- [ ] `--dry-run` makes zero API calls, writes zero files, and prints a projected cost.
- [ ] `results/teacher_stats.json` reports `enum_distribution` and
      `field_null_rate`; no enum field is >95% a single value at `n_ok >= 200` (if one
      is, the run exits 1 and the prompt must be fixed before running F3).
