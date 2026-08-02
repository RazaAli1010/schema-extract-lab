## F1 — Corpus acquisition and normalization

**Goal:** `sxl corpus build` produces `data/raw/docs.jsonl` containing ≥ 7,000
(target 7,500) deduplicated, length-filtered job-posting texts with stable
`doc_id`s, on a laptop with 5 GB of free disk and no GPU.

**Depends on:** F0 (`config.py`, `io.py`, `splits.py`)

**Status:** implemented 2026-07-31.

---

### Context digest

**Hardware (SPEC §2.1):** 8 GB RAM, **~5 GB free disk**, no GPU. This feature is
the one most likely to fill the disk, because dataset hosts love shipping 2 GB
parquet files. Streaming and early column-projection are mandatory, not optional.

**Record shape (SPEC §3.3)** — `data/raw/docs.jsonl`, keys in this order:
```json
{"doc_id": "jp_a1b2c3d4e5", "domain": "job_posting", "text": "...",
 "source": "xanderios/linkedin-job-postings", "char_len": 3412}
```

**Domain (SPEC §3.1):** `job_posting` only. `DOMAIN = "job_posting"` in
`config.py`. Invoices and clinical notes are a future F9; MIMIC-IV is rejected
(credentialed access).

**Sizing (SPEC §3.4):** train ≈ 5000, dev = 300, eval_gold = 300 — the last
sampled by F3 as **330 candidates** drawn from the 5% `eval_pool`. (F2 delivered
4,500 labeled `train` rows, not 5,000; see F2 "Delivered". The corpus-capacity
floors below are unaffected — they count available documents, not labeled ones.)
Split assignment is by `sha256(doc_id) % 100` → 5/5/90. The binding constraint is the
5% `eval_pool` bucket, which must yield **≥ 340** documents: that needs a corpus
of **≥ 7,000**. Target **7,500** for headroom and treat anything below 7,000 as a
build failure.

**Truncation (SPEC §3.6):** `MAX_INPUT_CHARS = 6000`, applied **identically in
every arm**. F1 stores the *untruncated* text and records `char_len`;
truncation happens at prompt-build time so the constant lives in one place.

**Constants available from F0 `config.py`:** `DOCS_PATH`, `DOMAIN`, `SEED`,
`ensure_dirs()`, `git_sha()`. **I/O helpers from F0 `io.py`:** `read_jsonl`,
`write_jsonl`, `write_json`.

**Principles (SPEC §5):** idempotent — re-running `sxl corpus build` over an
existing `docs.jsonl` produces a byte-identical file. Deterministic — `doc_id`
assignment must not depend on download order, because split membership derives
from `doc_id` and a reshuffle would silently leak eval documents into train.

### Context deltas

Applied to SPEC §3.3/§3.4/§4/§6.1 on 2026-07-31. The constants now live in
`config.py`; see SPEC §6.1 for the authoritative block.

```python
CORPUS_SOURCES = ("xanderios/linkedin-job-postings",)  # HF ids, priority order
CORPUS_TARGET_N   = 7500
CORPUS_MIN_N      = 7000     # below this, `sxl corpus build` exits non-zero
# 7000 docs x 5% = ~350 eval_pool, which must exceed F3's 330 candidates.
# 6500 would leave only ~325 and starve F3 -- do not lower these.
CORPUS_MIN_CHARS  = 400      # drop stubs
CORPUS_MAX_CHARS  = 40000    # drop scrape artifacts / concatenated pages
CORPUS_DEDUPE_PREFIX_CHARS = 600
CORPUS_MAX_SCAN   = 200_000  # hard cap on rows read upstream (SPEC §5.4)
CORPUS_PEEK_ROWS  = 20       # rows sampled to auto-detect the text column
CORPUS_MIN_FREE_BYTES = 2 * 1024**3
CORPUS_MIN_SPLIT_N = 340     # required in both `dev` and `eval_pool`
HF_CACHE_DIR = DATA / ".hfcache"
CORPUS_STATS_PATH = RESULTS / "corpus_stats.json"
```

#### Why the source changed — do not re-litigate

`lukebarousse/data_jobs`, the dataset this spec originally named, **has no job
description column.** Its 17 columns are all short metadata (`job_title`,
`job_location`, `salary_year_avg`, `job_skills`, …) and its longest string is an
~85-char job title. The "pick the longest string field" heuristic below would
have selected `job_title`, every row would have failed `CORPUS_MIN_CHARS = 400`,
and the build would have emitted an empty corpus. Verified against the HF
datasets-server first-rows API on 2026-07-31.

Replaced with **`xanderios/linkedin-job-postings`**: 33,246 real LinkedIn
postings, MIT, ungated, 63 MB parquet, `description` of 3,000–5,500 chars of
clean plain text (no HTML), and `job_id` as a stable upstream `source_key`. That
is 4.4× headroom over the 7,500 target, and typical lengths sit just under
`MAX_INPUT_CHARS = 6000` so few documents get truncated at prompt time.

Fallback if it ever disappears or becomes gated:
`lang-uk/recruitment-dataset-job-descriptions-english` (141,897 rows, MIT,
ungated) — but its descriptions average only ~1,200 chars and it is narrow to
Ukrainian IT recruitment, which weakens the claim. As the licensing note below
says: **do not substitute synthetic text.**

#### The `datasets` placement question — resolved

`datasets==5.0.1` is declared in the **base** (not `gpu`) group in SPEC §6.1
precisely for this feature, and F1 had to prove the placement was safe. It is:
verified 2026-07-31 that `importlib.util.find_spec("torch") is None` and no
`nvidia-*` distributions are present in the project venv. `datasets` pulls
`pyarrow`, `pandas`, `numpy`, `fsspec`, `huggingface_hub`, `hf-xet`, `aiohttp`,
`dill`, `multiprocess`, `xxhash` and `tqdm` — all torch-free.

**The httpx / `datasets-server` "plan B" is therefore not needed and must not be
built.** `tests/test_no_torch_import.py` remains the arbiter and still passes.

---

### Scope

1. **`src/sxl/corpus.py::fetch_raw(source: str, limit: int) -> Iterator[dict]`**
   Stream from the HF hub without materializing the dataset:
   ```python
   from datasets import load_dataset
   ds = load_dataset(source, split="train", streaming=True)
   ```
   `streaming=True` is **required** — a non-streaming load writes the full arrow
   cache to `~/.cache/huggingface` and will exhaust the 5 GB budget. Yield only
   the text-bearing column plus whatever is needed for `source_key` (step 3).
   Stop after `limit` *accepted* records, not `limit` *seen* records.

2. **`src/sxl/corpus.py::clean_text(raw: str) -> str`** — deterministic, pure:
   - unescape HTML entities (`html.unescape`), strip HTML tags if present
   - normalize unicode to NFKC; replace `\r\n` and `\r` with `\n`
   - collapse runs of 3+ blank lines to 2; strip trailing whitespace per line
   - strip leading/trailing whitespace overall
   Do **not** lowercase, do **not** strip punctuation — the model must see
   realistic text. Normalization for *scoring* is a separate concern owned by
   `normalize.py`.

3. **`src/sxl/corpus.py::make_doc_id(source: str, source_key: str) -> str`**
   `doc_id` must be a **stable function of content**, never of position:
   ```python
   h = hashlib.sha256(f"{source}\x00{source_key}".encode()).hexdigest()[:10]
   return f"jp_{h}"
   ```
   where `source_key` is the upstream row's own stable identifier if it has one,
   else the sha256 of the cleaned text. Sequential `jp_000001` ids are
   **forbidden** — they would change under any upstream reordering and silently
   move documents between splits (SPEC §3.4).

4. **`src/sxl/corpus.py::dedupe(rows) -> Iterator[dict]`** — two passes, both
   streaming, both `O(1)` memory per row:
   - exact: skip any `doc_id` already seen (a `set` of 6.5k 13-char strings is
     trivial)
   - near-duplicate: skip if the sha1 of the first 600 cleaned characters has
     been seen. Reposted listings differing only in a trailing "apply here" block
     are common in job-board data and would otherwise leak between splits.
   Count and report both drop reasons.

5. **`src/sxl/corpus.py::build(target_n: int) -> dict`** — the orchestrator:
   fetch → clean → filter on `CORPUS_MIN_CHARS`/`CORPUS_MAX_CHARS` → dedupe →
   assign `doc_id` → emit records. **Sort the final list by `doc_id` before
   writing** so the output is byte-identical across runs regardless of stream
   order. Write via `write_jsonl` (atomic). Return a stats dict.

6. **`src/sxl/corpus.py::report(stats) -> dict`** and
   `results/corpus_stats.json` (committed, small):
   ```json
   {"source": "...", "n_seen": 0, "n_dropped_short": 0, "n_dropped_long": 0,
    "n_dropped_dup_id": 0, "n_dropped_dup_prefix": 0, "n_kept": 0,
    "char_len": {"p5": 0, "p50": 0, "p95": 0, "max": 0},
    "split_counts": {"train": 0, "dev": 0, "eval_pool": 0},
    "generated_at": "...", "git_sha": "..."}
   ```
   `split_counts` is computed by calling F0's `splits.split_for` over the kept
   ids — this is the **early warning** that the corpus is big enough. Percentiles
   from `statistics.quantiles`, not numpy.

7. **`sxl corpus build` CLI** (replaces the F0 stub) with
   `--target-n INT`, `--source TEXT`, `--force/--no-force`. Default behavior is
   **idempotent**: if `DOCS_PATH` exists and has ≥ `CORPUS_MIN_N` rows, print the
   existing stats and exit 0 without re-downloading. `--force` re-fetches.
   Exit code 1 with a clear message if `n_kept < CORPUS_MIN_N` or if
   `split_counts["dev"] < 340` or `split_counts["eval_pool"] < 340`.

8. **Disk guard.** At the top of `build()`, check free space on the filesystem
   holding `DATA` with `shutil.disk_usage`. If under **2 GB**, exit 1 with a
   message telling the user to free space — do not begin a download that will
   die 80% through. Also set `HF_HOME` to a path under `data/.hfcache` and
   delete that cache directory at the end of a successful build.

---

### Out of scope

- Any labeling or schema population — F2 owns calling the teacher; `docs.jsonl`
  has **no `gold` key**.
- Writing `train.jsonl` / `dev.jsonl` / `eval_pool.jsonl` — F2 materializes those
  by applying `split_for` during labeling.
- A second domain (invoices, clinical notes) — future F9. Do not add a
  `domains/` registry now; `DOMAIN` is a constant.
- Any prompt construction or truncation to `MAX_INPUT_CHARS` — F5 owns that.

---

### Implementation notes

- **`datasets==5.0.1`, `streaming=True`.** Verify the actual text column name at
  runtime rather than hardcoding it. Job-posting datasets rename their
  description column constantly and a hardcoded `row["description"]` is the most
  likely way this feature breaks six months from now. Sample
  `CORPUS_PEEK_ROWS = 20` rows and pick the column with the greatest **median**
  string length — median, not the first row's, because a single null description
  in row 1 would otherwise hand the crown to `application_url`. Buffer the peeked
  rows and chain them back so none are lost. Log the chosen column.
  Then **raise** if the winner's median is still under `CORPUS_MIN_CHARS`: that
  is exactly the metadata-only failure mode described in the Context deltas, and
  it must be a loud crash (SPEC §5.5), never a silently empty corpus.
- **Import order is load-bearing.** `huggingface_hub` freezes `HF_HOME` into
  module constants at import time, so redirecting the cache *after* importing
  `datasets` is a silent no-op that fills `~/.cache/huggingface`. Set `HF_HOME`
  and `HF_HUB_CACHE` immediately before the local `from datasets import
  load_dataset`, and keep that import inside `fetch_raw`.
- **Licensing.** Record the upstream dataset's license string into
  `results/corpus_stats.json` (`"mit"` for the current source). If a future card
  carries a non-permissive or research-only license, note it in the README rather
  than pretending it does not exist. If the dataset disappears or becomes gated,
  fall back to another public postings dataset and update `CORPUS_SOURCES` —
  **do not** substitute synthetic text silently, because "fine-tuned on synthetic
  postings" is a materially different claim.
- **Memory.** Never hold more than the kept records in RAM: 6,500 × ~3 KB ≈ 20 MB.
  Fine on 8 GB. Do **not** call `list(ds)` on the streaming dataset.
- **Determinism.** The only sort key is `doc_id`. No `random` calls anywhere in
  this feature — `SEED` is unused here and that is correct.
- **`char_len`** is the length of the **cleaned, untruncated** text.

---

### Test plan

`tests/test_corpus.py` — all offline, using a hand-written fixture list of raw
rows; no network in tests.

- `clean_text` is idempotent: `clean_text(clean_text(x)) == clean_text(x)` over a
  fixture with HTML entities, `\r\n`, and 5 consecutive blank lines.
- `clean_text` preserves case and punctuation (assert an uppercase token survives).
- `make_doc_id` is stable across calls and differs for different `source_key`s;
  same text under two different sources yields different ids.
- `dedupe` drops an exact repeat and drops a row sharing the first 600 chars but
  differing in a trailing paragraph; keeps a row differing within the first 600.
- `build` on the fixture writes records with **exactly** the five keys in SPEC
  §3.3 order, and `char_len == len(text)` for every row.
- Running `build` twice over the same fixture produces byte-identical files
  (compare file bytes, not parsed objects).
- Length filter drops a 100-char row and a 50,000-char row.
- `clean_text` does not eat a literal `<` ("<5 years experience") but does strip
  real `<p>`/`<br>` markup.
- `pick_text_column` raises on a metadata-only fixture whose longest column is an
  85-char title — the permanent `lukebarousse/data_jobs` regression test — and
  uses the median, so a null description in row 1 does not mislead it.

> **Fixture-design note.** `dedupe` is first-seen-wins, so which member of a
> near-duplicate pair survives *is* order-dependent. The "reverse the input, get
> the same id set" test must therefore run on a **collision-free** fixture, kept
> separate from the dedupe fixture. Otherwise the test is legitimately flaky and
> the next session will "fix" the hashing to make it pass.

`tests/test_corpus_splits.py`
- Over the kept fixture ids, `split_for` produces three non-empty disjoint sets
  whose union is the whole corpus.
- A `CORPUS_MIN_N`-sized corpus of hashed ids yields ≥ 340 `dev`, ≥ 340
  `eval_pool` and ≥ 5,000 `train`.

`tests/test_no_torch_import.py` gains `sxl.corpus` plus an assertion that
importing it does **not** import `datasets` — the mechanical guard for the
HF-cache import-ordering rule above.

---

### Verify

```bash
sxl corpus build --target-n 7500
wc -l data/raw/docs.jsonl                       # >= 7000
python - <<'PY'
import json, collections
from sxl.splits import split_for
rows=[json.loads(l) for l in open("data/raw/docs.jsonl")]
ids=[r["doc_id"] for r in rows]
assert len(ids)==len(set(ids)), "duplicate doc_id"
assert all(list(r)==["doc_id","domain","text","source","char_len"] for r in rows), "key order"
assert all(r["char_len"]==len(r["text"]) for r in rows)
c=collections.Counter(split_for(i) for i in ids); print(c)
assert c["dev"]>=340 and c["eval_pool"]>=340, c
print("OK", len(rows))
PY
cp data/raw/docs.jsonl /tmp/a && sxl corpus build --force && cmp /tmp/a data/raw/docs.jsonl && echo "deterministic"
du -sh ~/.cache/huggingface data/.hfcache 2>/dev/null   # cache cleaned up
```

Expected: `OK <n>` with n ≥ 7000, the split counter showing ≥340 in both `dev`
and `eval_pool`, `deterministic` printed, and no multi-GB cache left behind.

---

### Acceptance criteria

- [ ] `data/raw/docs.jsonl` has ≥ 7,000 rows with unique `doc_id`s and exactly
      the five SPEC §3.3 keys in order.
- [ ] Applying `split_for` to the corpus yields ≥ 340 `dev` and ≥ 340 `eval_pool`
      documents (F3 needs 330 candidates), and ≥ 5,000 `train`.
- [ ] Two consecutive `sxl corpus build --force` runs produce byte-identical
      `docs.jsonl`.
- [ ] `doc_id` is a hash of content/source key, not a sequence number — verified
      by a test that reverses the input order and gets the same id set.
- [ ] Peak additional disk use during a full build stays under 2 GB, and the HF
      cache directory is removed on success.
- [ ] `results/corpus_stats.json` exists, records the upstream license string, and
      its `n_kept` matches `wc -l` of `docs.jsonl`.
- [ ] `tests/test_no_torch_import.py` still passes after the `datasets` dependency
      is added.
