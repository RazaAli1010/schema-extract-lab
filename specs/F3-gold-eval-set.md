## F3 — Eval-set sampling and human verification (300 gold examples)

**Goal:** `sxl gold sample` + `sxl gold verify` produce
`data/gold/eval_gold.jsonl`: exactly 300 documents whose 16 fields a human has
inspected and corrected, with an audit trail of what was changed — the only
ground truth in the project.

**Depends on:** F0 (`schema.py`, `normalize.py`, `io.py`), F2 (`eval_pool.jsonl`)

---

### Context digest

**Hardware (SPEC §2.1):** laptop, 8 GB RAM, no GPU. This is a terminal
application over small JSONL files. Nothing here touches torch or the network.

**Record shape (SPEC §3.3)** — `eval_gold.jsonl` has the same 8 keys as the
labeled files, but with:
```json
{"doc_id": "...", "domain": "job_posting", "text": "...",
 "gold": { <16 fields, human-corrected> },
 "label_source": "human",
 "teacher_model": "claude-sonnet-5",
 "verified_by_human": true,
 "verified_at": "2026-08-02T14:03:11Z"}
```
`teacher_model` is retained so the audit trail records *what the human was
correcting*.

**Why this file exists (SPEC §3.6):** the `teacher` arm is scored against
`eval_gold`, which measures teacher-vs-human disagreement. Without a
human-verified set, "within N F1 of the teacher" compares the model to its own
supervisor and is circular. **`eval_gold` is the only file in the project that is
not machine-generated**, and its integrity is the project's integrity.

**Split isolation (SPEC §3.4):** `eval_gold` documents come from the `eval_pool`
bucket (`sha256(doc_id) % 100 ∈ [0,4]`). SPEC §3.4: *"`eval_gold` doc_ids never
appear in `train.jsonl` or `dev.jsonl`"* — **F3 owns enforcing this**, F4 owns
re-asserting it at scoring time.

**Schema and the null/unknown rule (SPEC §3.2):** 16 fields; enums never `null`,
absence is `"unknown"`; `EducationLevel.none` ("posting states no degree needed")
≠ `"unknown"` ("posting is silent"); non-enum fields use `null`;
`required_skills` uses `[]`. The verification UI must make this distinction
impossible to get wrong by accident (Scope 4).

**Validation (F0):** `sxl.schema.validate_prediction`, `sxl.schema.FIELD_NAMES`,
`ENUM_FIELDS`, `SET_FIELDS`, `NUMERIC_FIELDS`. **Reuse them; do not restate the
field list in this feature's code.**

**Constants (F0 `config.py`):** `N_EVAL_GOLD = 300`, `SEED = 1337`,
`EVAL_POOL_PATH`, `EVAL_GOLD_PATH`.

**Principle (SPEC §5.5):** fail loudly on contract violations — split leakage
raises, it does not warn.

### Context deltas

Additions to `config.py`:

```python
GOLD_CANDIDATES_PATH = DATA/"gold"/"_candidates.jsonl"   # sampled, pre-verification
GOLD_PROGRESS_PATH   = DATA/"gold"/"_progress.jsonl"     # append-only edit log
GOLD_STRATA_BINS     = (400, 1500, 3000, 8000, 40000)    # char_len bin edges
```

Addition to SPEC §3.3 — the progress log (internal, gitignored):
```json
{"doc_id": "...", "field": "seniority", "old": "mid", "new": "senior",
 "at": "...", "action": "edit|accept|reject_doc"}
```

---

### Scope

1. **`src/sxl/verify.py::sample_candidates(n: int, seed: int) -> list[dict]`**
   Draw `n = 330` candidates (300 target + 10% slack for documents rejected as
   unusable) from `eval_pool.jsonl`, **stratified by `char_len`** across
   `GOLD_STRATA_BINS` in proportion to the pool's own distribution. Rationale: an
   unstratified sample under-represents long postings, which are exactly where
   extraction fails; a gold set of short easy documents would inflate every arm
   equally and hide the interesting variance.
   Sampling uses `random.Random(SEED)` and the candidate list is sorted by
   `doc_id` before writing, so the selection is reproducible. Write to
   `GOLD_CANDIDATES_PATH`.

2. **`src/sxl/verify.py::assert_no_leakage(gold_ids, train_path, dev_path) -> None`**
   Raise `RuntimeError` naming the offending ids if any gold `doc_id` appears in
   `train.jsonl` or `dev.jsonl`. Called at the end of `sample_candidates` **and**
   at the end of `finalize` (Scope 6). Two chances to catch the single mistake
   that would invalidate every number in the project.

3. **`src/sxl/verify.py::field_review_order(row) -> list[str]`**
   Present fields in the fixed `FIELD_NAMES` order, but surface a per-field
   **confidence hint** to direct human attention:
   - `low` — field is non-absent and its value does not appear as a substring of
     the (normalized) document text. For `title`, `company`, `location_*`,
     `salary_currency` and `required_skills` elements, a value the teacher
     produced that is nowhere in the source text is a likely hallucination.
   - `low` — `salary_min > salary_max`, or `years_experience_min > 40`, or
     `posting_date` not matching `^\d{4}-\d{2}-\d{2}$`.
   - `normal` — everything else.
   This is a *hint for the reviewer*, never an automatic edit. F3 must not
   auto-correct anything.

4. **`sxl gold verify` — the terminal review loop.** For each candidate, in
   `doc_id` order:
   - print the document text (paged, `MAX_INPUT_CHARS`-truncated view with a
     `[+more]` toggle) and the 16 proposed values, with `low`-confidence fields
     marked
   - keystrokes: `Enter` accept-all-and-next · `e <field>` edit one field ·
     `x` reject the document (unusable — not a job posting, wrong language,
     truncated scrape) · `b` back one document · `s` save-and-quit · `?` help
   - editing an **enum** field presents a **numbered menu of its exact members**
     — free text is not accepted. This makes `"Senior"` vs `"senior"` and
     `none` vs `unknown` errors structurally impossible.
   - editing a **nullable** field accepts a literal empty input as `null`, and
     shows `null` and `unknown` as visibly different tokens
   - editing `required_skills` accepts a comma-separated list, normalized to a
     deduplicated, sorted list via `normalize.norm_skills`
   - every keystroke that changes state appends one line to `GOLD_PROGRESS_PATH`
     immediately (`io.append_jsonl`) — **the session is resumable after a crash or
     a closed terminal, and re-running resumes at the first unreviewed doc_id.**
     300 documents is several hours of human work; losing it to a dropped SSH
     session must be impossible.
   - after each edit, re-run `validate_prediction` and refuse to advance if the
     record has become invalid

5. **Progress reporting.** On entry and every 25 documents, print
   `reviewed n/330 · accepted-as-is X · edited Y · rejected Z · est. remaining <hh:mm>`
   using the observed median seconds-per-document. Reviewer fatigue is the main
   quality risk on a 300-item task; a visible finish line matters.

6. **`src/sxl/verify.py::finalize() -> dict`** and `sxl gold finalize`
   Replay `GOLD_PROGRESS_PATH` over `GOLD_CANDIDATES_PATH`, drop rejected
   documents, take the **first 300 by `doc_id`** of what remains, set
   `label_source="human"`, `verified_by_human=true`, `verified_at=<now, UTC, Z>`,
   validate every row, run `assert_no_leakage`, and `write_jsonl` to
   `EVAL_GOLD_PATH`. Exit 1 if fewer than 300 verified documents survive.
   Emit `results/gold_stats.json`:
   ```json
   {"n_candidates": 330, "n_rejected": 0, "n_final": 300,
    "n_docs_edited": 0, "edit_rate_by_field": {"seniority": 0.07},
    "teacher_field_agreement": {"seniority": 0.93},
    "median_seconds_per_doc": 0.0, "char_len_bins": {"400-1500": 0},
    "generated_at": "...", "git_sha": "..."}
   ```

7. **`teacher_field_agreement` is a headline result, not a diagnostic.** It is
   the per-field rate at which the teacher's original label survived human review
   — i.e. **the teacher's own accuracy ceiling**. F8 quotes it directly: a
   student within 3 macro-F1 of a teacher that is itself only 93% right on
   `seniority` is a very different claim from one made against a perfect oracle.
   Compute it with `normalize.norm_field` so it uses the same comparison logic as
   F4's metrics.

8. **`sxl gold sample` CLI** with `--n INT` (default 330), `--seed INT`
   (default `SEED`), `--force`. Idempotent: refuses to overwrite an existing
   `GOLD_CANDIDATES_PATH` without `--force`, because resampling after review has
   started would discard human work.

---

### Out of scope

- Computing any model metric — F4. F3 produces the reference file only. The one
  exception is `teacher_field_agreement`, which is a property of the labeling
  process, not of a model arm.
- Writing `results/metrics/teacher.json` — F4 does that from `eval_pool`
  predictions vs `eval_gold`.
- Re-labeling with a second teacher, or adjudicating between two teachers — not
  in scope for v1.
- Any GUI or web interface — terminal only. A browser app is not worth the
  dependency weight on a 5 GB laptop.
- Inter-annotator agreement across two human reviewers — noted in F8 as a
  limitation of a single-reviewer gold set; not implemented.

---

### Implementation notes

- **No new dependencies.** Use `input()` and ANSI escapes from the standard
  library. Do not add `rich`, `textual`, or `prompt_toolkit` — the 5 GB disk
  budget is real and a review loop does not need a TUI framework.
- **Resumability is the load-bearing feature here.** Write the progress line
  *before* printing the next document, and `flush()` the file handle every time.
  Treat `KeyboardInterrupt` as `s` (save-and-quit), not as a crash.
- **`verified_at` must be a single timestamp per document**, taken when that
  document is accepted, not a batch timestamp written at finalize. It is an audit
  trail.
- **Do not let the reviewer see model predictions from any arm.** F3 runs before
  F5/F6 produce any output, and must stay that way — a reviewer who has seen the
  student's answer cannot un-see it, and the gold set stops being independent.
  Enforce by ordering: F3 completes before F5 begins (SPEC §8 dependency order).
- **Expect roughly 3–6 seconds per accepted document and 30–60 for an edited
  one.** At a ~20% edit rate, 330 documents is about 3–5 hours. Plan to run
  `sxl gold verify` across several sessions — which is precisely why Scope 4
  requires resumability.
- **If `teacher_field_agreement` for any field comes back below ~0.75**, stop.
  That field's teacher prompt is ambiguous, and the 4,500 training rows are being
  poisoned by it. Fix `prompts.py`, re-run F2 for that field's sake, and re-verify.
  This is cheaper than discovering it in F8.

---

### Test plan

All offline against synthetic candidate rows; the interactive loop is tested by
feeding a scripted stdin.

`tests/test_gold_sample.py`
- `sample_candidates` with a fixed seed returns identical `doc_id` sets across
  two calls and two processes.
- Every sampled id satisfies `split_for(doc_id) == "eval_pool"`.
- Stratification: with a pool skewed 90% short / 10% long, the sample's long
  fraction is within ±5 points of 10%.
- `assert_no_leakage` raises when a gold id is planted in a fake `train.jsonl`,
  and names that id in the message.

`tests/test_gold_verify.py` (scripted stdin via `monkeypatch`)
- Accept-all on 3 documents writes 3 `accept` lines to the progress log and
  leaves `gold` byte-identical to the candidate.
- `e seniority` + selecting menu index for `senior` writes an `edit` line with
  correct `old`/`new` and updates the record.
- Free-text input on an enum field is rejected and re-prompts.
- Empty input on `salary_min` yields `null`; on `seniority` the menu has no
  "empty" option at all.
- `x` marks the document rejected and it is absent from `finalize()`'s output.
- Killing the loop after 2 of 5 documents and restarting resumes at document 3.

`tests/test_gold_finalize.py`
- `finalize` on 305 accepted + 5 rejected candidates emits exactly 300 rows.
- All output rows have `label_source == "human"`, `verified_by_human is True`,
  a parseable ISO-8601 `verified_at`, and pass `validate_prediction`.
- `finalize` exits 1 when only 290 verified documents exist.
- `teacher_field_agreement["seniority"]` equals a hand-computed value on a
  fixture where 2 of 10 seniority labels were edited.

---

### Verify

```bash
sxl gold sample --n 330
wc -l data/gold/_candidates.jsonl                 # 330
sxl gold verify                                   # interactive; run to completion (several sessions)
sxl gold finalize
python - <<'PY'
import json
from sxl.schema import validate_prediction, FIELD_NAMES
from sxl.splits import split_for
G=[json.loads(l) for l in open("data/gold/eval_gold.jsonl")]
assert len(G)==300, len(G)
assert all(r["verified_by_human"] and r["label_source"]=="human" for r in G)
assert all(validate_prediction(r["gold"]) for r in G)
assert all(list(r["gold"])==list(FIELD_NAMES) for r in G)
assert all(split_for(r["doc_id"])=="eval_pool" for r in G)
gid={r["doc_id"] for r in G}
for p in ("data/labeled/train.jsonl","data/labeled/dev.jsonl"):
    other={json.loads(l)["doc_id"] for l in open(p)}
    assert not (gid & other), (p, sorted(gid & other)[:5])
print("OK 300 gold, no leakage")
PY
python -c "import json;d=json.load(open('results/gold_stats.json'));print(sorted(d['teacher_field_agreement'].items(), key=lambda kv: kv[1])[:5])"
```

Expected: `OK 300 gold, no leakage`, and the five lowest-agreement fields printed
for inspection — none should be below 0.75.

---

### Acceptance criteria

- [ ] `data/gold/eval_gold.jsonl` has **exactly 300** rows.
- [ ] Every row: `label_source == "human"`, `verified_by_human == true`,
      `verified_at` a valid ISO-8601 UTC timestamp, `gold` passes
      `validate_prediction`, keys in SPEC §3.3 order.
- [ ] Zero `doc_id` overlap with `train.jsonl` and `dev.jsonl`, asserted in code
      (not just in the verify block) and raising on violation.
- [ ] Every gold `doc_id` satisfies `split_for(doc_id) == "eval_pool"`.
- [ ] The review loop is resumable: interrupting after N documents and restarting
      resumes at document N+1 with no lost edits (demonstrated by a test).
- [ ] Enum fields can only be set from a menu of their exact members; a test
      proves free text is rejected.
- [ ] `results/gold_stats.json` reports `teacher_field_agreement` for all 16
      fields and `n_docs_edited`.
- [ ] Sampling is reproducible: `sxl gold sample --seed 1337` twice yields the
      same 330 `doc_id`s.
