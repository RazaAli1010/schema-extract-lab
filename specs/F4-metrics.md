## F4 — Metrics library and scoring CLI

**Goal:** `sxl metrics score --arm <arm>` turns a predictions JSONL into
`results/metrics/<arm>.json` containing `schema_valid_rate`, per-field
exact-match and P/R/F1, and `macro_f1` — computed by one formula over all 16
fields, with no numpy, no sklearn, and no GPU.

**Depends on:** F0 (`schema.py`, `normalize.py`, `io.py`, `config.py`)

*(F4 is implementable immediately after F0 and does not wait for F3 — it is
tested entirely on synthetic fixtures. It only needs real files at run time.)*

---

### Context digest

**Hardware (SPEC §2.1):** laptop, 8 GB RAM, no GPU. The metric formulas in
SPEC §3.5 are deliberately plain arithmetic so this module has **zero numeric
dependencies** — no numpy, no sklearn, no pandas. Keep it that way; it is what
lets the laptop install stay under 100 MB.

**Inputs (SPEC §3.3):**
- predictions — `artifacts/predictions/<arm>.jsonl`, keys in order:
  `{"doc_id", "arm", "raw_output", "parsed", "schema_valid", "latency_ms", "prompt_tokens", "completion_tokens"}`
  where `parsed` is `null` when the output did not parse or did not validate.
- gold — `data/gold/eval_gold.jsonl`, key `gold` holding the 16 fields.

**Output (SPEC §3.3)** — `results/metrics/<arm>.json`:
```json
{"arm": "base_fewshot", "split": "eval_gold", "n": 300,
 "schema_valid_rate": 0.0, "macro_f1": 0.0,
 "per_field": {"title": {"em": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}},
 "generated_at": "...", "git_sha": "..."}
```

**Field taxonomy (F0 `schema.py`):** `FIELD_NAMES` (16, ordered), `ENUM_FIELDS`
(5), `SET_FIELDS` (`required_skills`), `NUMERIC_FIELDS` (3).
**Iterate `FIELD_NAMES`, never `model_fields`.**

**Normalization (F0 `normalize.py`, SPEC §3.5):** `norm_field(field, value)` and
`is_absent(field, value)`. `null`, `"unknown"`, and `[]` are all **absent**.
F4 calls these; it must not re-implement string casing or set logic.

**The metric contract — SPEC §3.5, reproduced in full because it is the whole
feature:**

- **(a)** `schema_valid_rate` = (# predictions that parse **and** validate) / N,
  where **N is always the full split size**.
- **(b) Invalid predictions are wrong, not excluded.** A prediction with
  `parsed == null` contributes zero TP and counts as a miss on every field whose
  gold value is present. It is **never** dropped from the denominator.
- **(c) Per-field TP/FP/FN**, aggregated over all N documents:

  | | gold absent | gold present |
  |---|---|---|
  | **pred absent** | — (ignored) | FN |
  | **pred present, matches** | n/a | TP |
  | **pred present, differs** | FP | FP **and** FN |

  For `required_skills`: `TP += |P ∩ G|`, `FP += |P \ G|`, `FN += |G \ P|`.

  `precision = TP/(TP+FP)`, `recall = TP/(TP+FN)`, `f1 = 2PR/(P+R)`; each `0.0`
  when its denominator is 0.
- **(d)** `per_field[f].em` = fraction of N documents where normalized pred equals
  normalized gold, **including documents where both are absent**, and counting
  invalid predictions as mismatches.
- **(e)** `macro_f1` = arithmetic mean of the 16 `per_field[f].f1` values.

Note the deliberate asymmetry between (c) and (d): a document where both gold and
prediction are absent is a **success for EM** and **invisible to F1**. Reporting
both is what stops a model that answers `null` to everything from looking good.

**Split isolation (SPEC §3.4):** F3 guarantees no leakage; **F4 re-asserts it**.

**Arms (SPEC §3.6):** `ARMS` in `config.py` — `base_fewshot`,
`base_fewshot_constrained`, `lora_ft`, `lora_ft_constrained`, `teacher`.

### Context deltas

F4 implements SPEC §3.5 arithmetic exactly as written, but **adds two keys to the
`results/metrics/<arm>.json` contract** in SPEC §3.3. Apply to SPEC §3.3 before
implementing:

```json
"macro_f1_null_baseline": 0.0,   // macro-F1 of an arm that predicts empty_posting() for all N
"n_missing_predictions": 0       // gold doc_ids absent from the predictions file, scored as invalid
```

F8 consumes both (it renders a `null baseline` row and needs the missing count to
caveat coverage), so they are part of the contract, not internal diagnostics.

---

### Scope

1. **`src/sxl/metrics.py::Counts`** — a tiny dataclass `(tp, fp, fn, em_hits, support)`
   with `+` defined, where `support` = number of documents whose **gold** value is
   present (for `required_skills`, the total number of gold skill elements). This
   is what makes `per_field[f].support` interpretable: a field with support 4 out
   of 300 has an F1 that means almost nothing, and F8 must be able to say so.

2. **`src/sxl/metrics.py::score_field(field, gold_val, pred_val, pred_valid) -> Counts`**
   The single formula. Branch on `field in SET_FIELDS` for the set arithmetic;
   everything else uses the scalar table. When `pred_valid is False`, treat
   `pred_val` as absent **and** count `em_hits = 0` even if gold is also absent —
   an unparseable output is not credited with correctly omitting a field.

   ```python
   def score_field(field, gold_val, pred_val, pred_valid) -> Counts:
       g = norm_field(field, gold_val)
       g_abs = is_absent(field, gold_val)
       if not pred_valid:
           if field in SET_FIELDS:
               return Counts(0, 0, len(g), 0, len(g))
           return Counts(0, 0, 0 if g_abs else 1, 0, 0 if g_abs else 1)
       ...
   ```

3. **`src/sxl/metrics.py::score_arm(gold_rows, pred_rows, arm) -> dict`**
   - build `{doc_id: row}` for both sides
   - **assert the prediction set covers the gold set.** A prediction file missing
     documents is a silent way to inflate every score. Missing doc_ids are
     synthesized as `schema_valid=False, parsed=None` and counted in a
     `n_missing_predictions` field, **not** dropped. Extra prediction doc_ids not
     in gold raise `RuntimeError`.
   - assert every `pred_rows[i]["arm"] == arm`, raise on mismatch (this catches
     the copy-paste error of scoring `lora_ft` predictions as `base_fewshot`)
   - accumulate `Counts` per field over all N gold documents
   - emit the SPEC §3.3 output dict with `per_field` keyed in `FIELD_NAMES` order

4. **`src/sxl/metrics.py::assert_no_leakage(gold_rows) -> None`**
   Re-run F3's check at scoring time: no gold `doc_id` may appear in
   `train.jsonl` or `dev.jsonl`. Raise with the offending ids. Cheap, and it is
   the last gate before a number becomes a claim.

5. **`src/sxl/metrics.py::score_teacher_arm() -> dict`**
   The `teacher` arm has no predictions file — its "prediction" is the teacher
   label already sitting in `eval_pool.jsonl`, and the gold is the human-corrected
   version of the same document in `eval_gold.jsonl`. Synthesize a predictions
   list from `eval_pool` (`schema_valid=True` for all, since F2 only wrote valid
   rows; `latency_ms=None`) and score it through the identical `score_arm` path.
   **Do not fork the metric code for this arm.**

6. **`sxl metrics score` CLI** (replaces the F0 stub):
   `--arm TEXT` (must be in `ARMS`, else exit 2 listing valid values),
   `--pred PATH` (default `PREDICTIONS_DIR/<arm>.jsonl`),
   `--gold PATH` (default `EVAL_GOLD_PATH`),
   `--out PATH` (default `METRICS_DIR/<arm>.json`).
   Prints a human-readable table to stdout — `macro_f1`, `schema_valid_rate`, and
   the per-field rows sorted by **ascending f1** so the weakest fields are read
   first. Exit 1 if the gold file is missing or has ≠ 300 rows.

7. **`sxl metrics compare` CLI** — reads every existing
   `results/metrics/*.json` and prints one row per arm
   (`arm | n | schema_valid | macro_f1`), sorted by `macro_f1` descending. No
   file output; F8 owns the committed table. This exists so a Kaggle session can
   check its result against the others in one command.

8. **Degenerate-baseline guard.** `score_arm` additionally computes
   `macro_f1_null_baseline`: the macro-F1 of a hypothetical arm that returns
   `empty_posting()` for every document. Include it in the output JSON. Any real
   arm scoring at or below it is not extracting anything, and F8 must be able to
   show the floor alongside the results.

---

### Out of scope

- Generating predictions — F5 (baseline arms), F6 (fine-tuned arms). F4 only
  consumes `artifacts/predictions/<arm>.jsonl`.
- Latency, throughput, or cost — F7 owns `results/bench/<arm>.json`. F4 must
  **not** aggregate the `latency_ms` field even though it is present in the
  prediction records; two features writing latency numbers is exactly how the
  headline table ends up self-contradictory.
- The committed headline table `results/tables/headline.md` — F8.
- Significance testing / confidence intervals — deferred. F8 notes that with
  n=300 the standard error on a per-field F1 is roughly ±0.03, so differences
  under ~3 points are not meaningful. *(Production alternative: bootstrap CIs
  over the 300 documents; ~20 lines, deferred only to keep F4 dependency-free.)*

---

### Implementation notes

- **No numpy, no sklearn, no pandas.** Every number here is a ratio of integers.
  Adding sklearn would pull scipy (~90 MB) onto a 5 GB laptop for four
  divisions.
- **Float determinism.** Sum integer counters and divide **once** at the end;
  never accumulate floats across 300 documents. Round to 4 decimal places only in
  the output JSON, never in intermediate math.
- **`support` for `required_skills`** is the count of gold skill *elements*
  across the split, not the count of documents. Document this in the output key's
  docstring — it is the one field where `support` means something different, and
  a reader comparing `support: 1840` for skills against `support: 210` for
  `title` needs to know why.
- **Enum fields and `"unknown"`.** Because `"unknown"` is absent (SPEC §3.5), a
  field like `remote_mode` where 70% of documents are genuinely silent will have
  low `support` and a volatile F1, while its EM stays high. This is correct and
  intended; F8 explains it rather than F4 papering over it.
- **`parsed` may be a valid-looking dict that still fails validation.** Trust the
  `schema_valid` boolean written by F5/F6 (which came from
  `validate_prediction`), but **re-validate defensively** in `score_arm` and
  raise if the two disagree — a mismatch means an upstream feature wrote the
  field by hand.
- **Timestamps** in UTC with a trailing `Z`; `git_sha` from F0's non-raising
  helper.

---

### Test plan

`tests/test_metrics_formula.py` — the golden fixture. **This is the most
important test in the repo**; if a future session "improves" the metric code and
this fails, the metric changed.

Construct 4 documents by hand with known values, write the expected TP/FP/FN and
the expected macro-F1 **as literals in the test file**, with a comment showing
the arithmetic. Cover, at minimum:
- doc 1: perfect prediction (all 16 fields match, some absent on both sides)
- doc 2: prediction valid but 3 fields wrong, including one enum wrong and one
  numeric off by 1.0
- doc 3: `schema_valid=False`, `parsed=None` — must produce FN on every
  present-gold field, zero EM hits, and **must not** be dropped
- doc 4: `required_skills` gold `{python, sql, aws}` vs pred `{python, sql, java}`
  → TP=2, FP=1, FN=1 for that field

`tests/test_metrics_edges.py`
- A field absent in both gold and prediction: EM hit, no TP/FP/FN, `support` 0,
  `f1 == 0.0` (not `nan`, not `1.0`).
- All-absent gold for a field → `precision == recall == f1 == 0.0`, no
  `ZeroDivisionError`.
- `empty_posting()` predicted for all 300 → `macro_f1` equals
  `macro_f1_null_baseline` exactly.
- Case/whitespace differences (`"Senior Engineer "` vs `"senior  engineer"`) count
  as a match — proves normalization is wired in.
- `salary_min` gold `120000` vs pred `120000.0` matches (float cast).

`tests/test_metrics_integrity.py`
- A prediction file missing 5 gold doc_ids: `n == 300`,
  `n_missing_predictions == 5`, and those 5 score as invalid.
- A prediction file with an extra doc_id raises `RuntimeError`.
- A prediction row whose `arm` disagrees with the `--arm` flag raises.
- A prediction row with `schema_valid=True` but a `parsed` that fails
  `validate_prediction` raises.
- `assert_no_leakage` raises when a gold id is planted in a fake `train.jsonl`.

---

### Verify

```bash
pytest -q tests/test_metrics_formula.py tests/test_metrics_edges.py tests/test_metrics_integrity.py

# end-to-end smoke on synthetic data, no GPU needed:
python - <<'PY'
import json, pathlib
from sxl.schema import empty_posting
g=[json.loads(l) for l in open("data/gold/eval_gold.jsonl")]
p=pathlib.Path("artifacts/predictions/_smoke.jsonl"); p.parent.mkdir(parents=True,exist_ok=True)
with p.open("w") as f:
    for r in g:   # perfect oracle: predict the gold back
        f.write(json.dumps({"doc_id":r["doc_id"],"arm":"base_fewshot","raw_output":"",
                            "parsed":r["gold"],"schema_valid":True,"latency_ms":0.0,
                            "prompt_tokens":0,"completion_tokens":0})+"\n")
PY
sxl metrics score --arm base_fewshot --pred artifacts/predictions/_smoke.jsonl --out /tmp/m.json
python -c "import json;d=json.load(open('/tmp/m.json'));assert d['macro_f1']==1.0 or d['macro_f1']>0.999, d['macro_f1'];assert d['schema_valid_rate']==1.0;print('oracle OK', d['macro_f1'])"

sxl metrics score --arm teacher          # real teacher-vs-human number
sxl metrics compare
```

Expected: all metric tests green; the oracle run prints `macro_f1` of 1.0 (any
field where gold is absent for all 300 documents will contribute 0.0 — if so, the
assertion must be relaxed to `> 0.999` **and the low-support field named in the
output**, not silently ignored); the teacher arm produces a real macro-F1 well
under 1.0.

---

### Acceptance criteria

- [ ] `tests/test_metrics_formula.py` passes against hand-computed literal
      expected values for all four fixture documents.
- [ ] An invalid prediction (`parsed: null`) contributes FN on every present-gold
      field and is **not** removed from the denominator — asserted by a test.
- [ ] Predicting the gold back for all 300 documents yields
      `schema_valid_rate == 1.0` and `macro_f1 ≥ 0.999`.
- [ ] Predicting `empty_posting()` for all 300 yields exactly
      `macro_f1_null_baseline`.
- [ ] No `ZeroDivisionError` or `nan` appears in any output for any input,
      including an all-absent field.
- [ ] `results/metrics/<arm>.json` matches the SPEC §3.3 shape exactly, with
      `per_field` containing all 16 `FIELD_NAMES` in order.
- [ ] Scoring raises on: extra prediction doc_ids, an `arm` field mismatch, a
      `schema_valid`/`parsed` disagreement, and gold/train leakage.
- [ ] `sxl metrics score --arm teacher` produces a real number from
      `eval_pool` vs `eval_gold` through the same `score_arm` code path.
- [ ] `import sxl.metrics` pulls in no numpy, scipy, sklearn, or pandas
      (`tests/test_no_torch_import.py` extended to assert this).
