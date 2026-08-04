## F8 — Results aggregation, headline table, and README

**Goal:** `sxl report build` reads every file in `results/` and emits
`results/tables/headline.md` plus a README whose numbers are all machine-derived
— the artifact a hiring manager actually reads, with every claim traceable to a
committed JSON file.

**Depends on:** F4 (`results/metrics/*.json`), F5, F6, F7 (`results/bench/*.json`),
F3 (`results/gold_stats.json`), F2 (`results/teacher_stats.json`)

---

### Context digest

**Hardware (SPEC §2.1):** laptop, 8 GB RAM, no GPU. This feature reads small JSON
files and writes markdown. Nothing here imports torch.

**SPEC §1 — the deliverable that matters is the results table, not the model:**

> | arm | schema-valid % | macro-F1 | p50 ms/doc | $/1k docs |
> |---|---|---|---|---|
>
> Every cell must be produced by a committed script from committed artifacts.
> **No cell is ever typed by hand.**

**SPEC §1.1 — headline claims are hypotheses, not targets.** The pitch numbers
("within 3 F1", "45ms on a T4", "~$0.002/1k docs") are placeholders to be filled
in by measurement. Two are known to be optimistic:
- **45 ms/doc** is not reachable single-stream on a T4 for a ~150-token output
  (~2–4 s is the physical expectation). F7 reports single-stream and amortized
  separately; **F8 must present them as separate columns and must not quote the
  amortized figure as "latency"**.
- **"200x larger model"** — the teacher's parameter count is not public.
  **Never claim a size ratio.** Say "frontier teacher model (`claude-sonnet-5`)".

> If the measured result is unflattering, the measured result ships.

**Input files and their exact shapes (SPEC §3.3):**
- `results/metrics/<arm>.json` → `arm, split, n, schema_valid_rate, macro_f1,
  per_field{em, precision, recall, f1, support}, macro_f1_null_baseline,
  n_missing_predictions, generated_at, git_sha`
- `results/bench/<arm>.json` → `arm, gpu_name, dtype, batch_size, n_docs, warmup,
  p50_ms, p95_ms, mean_ms, throughput_docs_per_s, gpu_hourly_usd,
  cost_per_1k_docs_usd, mean_completion_tokens, generated_at`
- `results/bench/<arm>_sweep.json` → `sweep[], best_batch_size`
- `results/bench/teacher.json` → additionally `"measurement": "api_wall_clock"`
- `results/gold_stats.json` → `teacher_field_agreement`, `n_docs_edited`,
  `edit_rate_by_field`, `n_final`
- `results/teacher_stats.json` → `spend_usd`, `n_ok`, `enum_distribution`
- `results/train_stats.json` → `trainable_pct`, `best_eval_loss`, `peak_vram_gb`,
  `train_runtime_s`, `adapter_repo`
- `results/corpus_stats.json` → `n_kept`, license string

**Arms (SPEC §3.6):** `base_fewshot`, `base_fewshot_constrained`, `lora_ft`,
`lora_ft_constrained`, `teacher`.
`base_fewshot` is *the competitor that matters*. The `*_constrained` arms exist
to show that constrained decoding buys **validity, not accuracy** — SPEC §3.6
calls that gap "the most interesting result in the project", and F8 must present
it as such rather than burying it in a row.

**Metric semantics F8 must explain correctly (SPEC §3.5):**
- Invalid predictions are **counted as wrong, not excluded**; N is always 300.
- `"unknown"`, `null`, and `[]` are all **absent**; a field where most documents
  are genuinely silent has low `support` and a volatile F1 while its EM stays
  high. F8 explains this rather than papering over it.
- `macro_f1_null_baseline` is the floor: an arm at or below it is extracting
  nothing.
- `support` for `required_skills` counts **skill elements**, not documents.

**Honest-tradeoff obligations inherited from SPEC §5.3** — the README must state:
- **vLLM was not used** (Turing sm_75 support is degrading); on an A10/L4/A100
  vLLM would be the correct serving path and **would improve the throughput
  numbers substantially**.
- **Unsloth was not used** (marginal at 1.7B, dependency friction).
- ONNX/GGUF export deferred.
- QLoRA available as fallback; fp16 LoRA was the default.

**Definition of done (SPEC §7):** README states the measured numbers **and names
the honest caveats**: teacher labels are not ground truth outside the 300; one
domain; one seed; single-GPU; no vLLM.

### Context deltas

Addition to `config.py`:
```python
TABLES_DIR = RESULTS/"tables"
```

New committed outputs (extends SPEC §4's `results/` tree):
```
results/tables/headline.md      # the 5-row summary
results/tables/per_field.md     # 16 rows x 4 arms, sorted by lora_ft f1 ascending
results/tables/sweep.md         # batch-size sweep
results/README_FACTS.json       # every number the README interpolates
```

---

### Scope

1. **`src/sxl/report.py::load_results() -> dict`**
   Glob `results/metrics/*.json` and `results/bench/*.json` plus the four stats
   files. **Missing files degrade gracefully to a `—` cell; they do not crash.**
   The report must be buildable mid-project when only the baseline arms exist —
   that is when it is most useful for deciding what to fix next.
   Warn (to stderr) once per missing arm, listing what is absent.

2. **`src/sxl/report.py::build_headline() -> str`** → `results/tables/headline.md`
   One row per arm in a **fixed order** (`base_fewshot`,
   `base_fewshot_constrained`, `lora_ft`, `lora_ft_constrained`, `teacher`), with
   these columns and **these exact headers**:

   | arm | schema-valid % | macro-F1 | Δ vs teacher | p50 ms/doc (batch 1) | amortized ms/doc @ best batch | $/1k docs |

   Rules that are the point of this feature:
   - **`p50 ms/doc (batch 1)` and `amortized ms/doc @ best batch` are separate
     columns and are never merged.** The header text must carry the qualifier.
   - The `teacher` row's latency cells are marked `API` and footnoted as an
     `api_wall_clock` measurement, not comparable to local `generate()`.
   - `Δ vs teacher` = `macro_f1(arm) − macro_f1(teacher)`, signed, 3 decimals.
     **This is the "within N F1" claim** and it is computed, never asserted.
   - Append a `null baseline` row from `macro_f1_null_baseline` so the floor is
     visible.
   - Footnote: `$/1k docs` assumes `gpu_hourly_usd` (read from the bench file,
     not hardcoded) and is derived, not measured.
   - Footnote: n=300; with n=300 the standard error on a per-field F1 is roughly
     ±0.03, so **differences under ~3 points are not meaningful**.

3. **`src/sxl/report.py::build_per_field() -> str`** → `results/tables/per_field.md`
   16 rows × (em, f1, support) per arm, sorted **ascending by `lora_ft` F1** so
   the weakest fields are read first. Flag any field with `support < 30` as
   `low-n` — an F1 computed from 12 populated documents is noise and must be
   labeled, not quoted.

4. **`src/sxl/report.py::build_sweep() -> str`** → `results/tables/sweep.md`
   Batch size × throughput × amortized ms × peak VRAM per arm, with OOM rows
   preserved and marked. This table is what makes the cost column credible.

5. **`src/sxl/report.py::facts() -> dict`** → `results/README_FACTS.json`
   Every scalar the README interpolates, in one place: each arm's
   `schema_valid_rate` / `macro_f1` / `p50_ms` / `cost_per_1k_docs_usd`, the
   `Δ vs teacher`, `n_train`, `n_eval_gold`, teacher `spend_usd`,
   `trainable_pct`, `peak_vram_gb`, `train_runtime_s`, `gpu_name`, the three
   lowest `teacher_field_agreement` fields, and `git_sha`.
   **The README is generated by interpolating this file** — no number is typed
   into markdown by hand (SPEC §1).

6. **`src/sxl/report.py::build_readme() -> str`** — regenerate `README.md`
   between `<!-- BEGIN GENERATED -->` / `<!-- END GENERATED -->` markers, leaving
   hand-written prose outside them untouched. Sections:
   - one-paragraph problem statement and the headline table
   - **the constrained-decoding finding**, stated explicitly: constraining the
     grammar takes `schema_valid_rate` to ~1.0 while `macro_f1` moves by
     `<computed Δ>` — validity is free, accuracy is not (SPEC §3.6)
   - the cost/latency table with the batch-1 vs amortized distinction spelled out
   - reproduction instructions: laptop steps, then "the GPU steps run on Kaggle
     (`GPU T4 x2`); see `notebooks/`" with the ~30 h/week quota noted
   - **Limitations**, mandatory, each with a number where one exists:
     1. teacher labels are **not ground truth** outside the 300 human-verified
        documents; the teacher's own per-field agreement with a human is
        `<lowest three fields and values from gold_stats>`
     2. one domain (job postings), one seed, one hyperparameter configuration —
        no sweep, no confidence intervals
     3. single-reviewer gold set; no inter-annotator agreement
     4. single T4, fp16, **no vLLM** (Turing support degrading); on an
        A10/L4/A100 with vLLM, throughput would be substantially better and the
        cost column would drop
     5. n=300 eval; differences under ~3 F1 points are within noise
     6. `$/1k docs` derives from an assumed `$<gpu_hourly_usd>/h` GPU rate

7. **`src/sxl/report.py::check_claims() -> list[str]`** — the guardrail.
   Scan the generated README for claim patterns and verify each against
   `README_FACTS.json`; return violations and **exit 1** if any:
   - any number appearing next to `ms`, `%`, `F1`, or `$` must match a value in
     `README_FACTS.json` to the printed precision
   - the strings `200x`, `200×`, and any `\d+x larger` pattern → **rejected**
     (SPEC §1.1: never claim a parameter-count ratio)
   - the string `45ms` / `45 ms` → rejected unless it is genuinely the measured
     `p50_ms`
   - a `p50` figure sourced from an amortized-throughput field → rejected
   This is a spec-enforcement test, not decoration: it is what makes SPEC §1.1
   mechanically true instead of aspirational.

8. **`sxl report build` CLI** (replaces the F0 stub) with `--out-dir PATH`,
   `--readme/--no-readme`, `--strict/--no-strict` (default `--strict`, meaning
   `check_claims` failures exit 1). Print the headline table to stdout so a
   mid-project run is a one-command status check.

---

### Out of scope

- Computing any metric, latency, or cost from raw data — F4 and F7 own those.
  **F8 reads `results/*.json` and never recomputes a number.** If a value is
  wrong, the fix belongs in F4 or F7, not here.
- Plots or charts. A markdown table renders on GitHub, in a PDF, and in a
  terminal; a matplotlib dependency would pull ~60 MB onto a 5 GB laptop for
  something a table does better. *(If a chart is later wanted for a blog post,
  generate it from `README_FACTS.json` in a separate throwaway script.)*
- A model card on the HF Hub — nice-to-have; the adapter repo gets a link to this
  README instead.
- Statistical significance testing — F4's out-of-scope note defers bootstrap CIs;
  F8 states the ±0.03 rule of thumb rather than computing intervals.
- Any GPU work.

---

### Implementation notes

- **No new dependencies.** Markdown tables are f-strings. Do not add `tabulate`,
  `jinja2`, or `pandas`.
- **Missing-file tolerance is a feature, not laziness.** The realistic workflow is:
  F5 finishes → run the report → see the baseline → then decide on F6. A report
  that crashes without all five arms is useless exactly when it is most needed.
- **Column alignment.** Right-align numerics, fix decimals per column type
  (`schema_valid_rate` 1dp as a percentage, `macro_f1` 3dp, ms 0dp, dollars 5dp).
  Inconsistent precision across a row reads as carelessness in the one artifact
  that gets read most closely.
- **`Δ vs teacher` will very likely be negative**, and may be well outside 3
  points. That is a legitimate result and the README says so in plain language.
  The interesting framing is not "we matched the teacher" but "we recovered X% of
  the teacher's macro-F1 at Y× lower cost" — and both X and Y are computed.
- **Do not let `check_claims` be disabled in CI.** `--no-strict` exists for local
  iteration only; the `make report` target uses `--strict`.
- **`git_sha` in every table footer**, so a screenshot of the table can be traced
  to the commit that produced it.
- **The teacher's cost column uses standard, not batch, rates** (F7 Scope 6) —
  F8 must not silently substitute the 50%-off batch figure used for *labeling*,
  because a latency-sensitive production deployment cannot use a 24-hour batch
  API. Footnote this.

---

### Test plan

`tests/test_report_tables.py` — against a fixture `results/` tree:
- `build_headline` with all five arms produces 6 data rows (5 arms + null
  baseline) and the exact column headers from Scope 2.
- `build_headline` with only `base_fewshot` present produces `—` cells for the
  rest and does **not** raise; a warning is emitted naming the missing arms.
- `Δ vs teacher` equals a hand-computed difference on the fixture.
- The `teacher` row's latency cells render as `API`, not as a number.
- `per_field` rows are sorted ascending by `lora_ft` f1 and a `support: 12` field
  is flagged `low-n`.

`tests/test_report_claims.py` — the guardrail, and the most important test here:
- A README containing `200x larger` → `check_claims` returns a violation.
- A README containing `45ms` when `p50_ms` is `2841.3` → violation.
- A README quoting `macro_f1` to 3dp matching `README_FACTS.json` → no violation.
- A README quoting `0.812` when facts say `0.798` → violation.
- A README that quotes the amortized figure under a `p50` label → violation.
- `sxl report build --strict` exits 1 when any violation exists.

`tests/test_report_facts.py`
- `README_FACTS.json` contains every key the README template interpolates
  (assert by parsing the template's placeholders and diffing against the facts
  keys — a missing key must fail the test, not render as `{p50_ms}`).

---

### Verify

```bash
sxl report build
cat results/tables/headline.md
cat results/tables/per_field.md | head -20
python - <<'PY'
import json
f=json.load(open("results/README_FACTS.json"))
print("delta vs teacher:", f["delta_vs_teacher"])
print("lora_ft  p50:", f["lora_ft"]["p50_ms"], "ms   amortized:", f["lora_ft"]["amortized_ms_per_doc"], "ms")
print("cost/1k :", f["lora_ft"]["cost_per_1k_docs_usd"])
assert f["lora_ft"]["p50_ms"] != f["lora_ft"]["amortized_ms_per_doc"], "columns conflated"
PY
grep -Ei "200 ?x|45 ?ms" README.md && echo "FORBIDDEN CLAIM PRESENT" && exit 1
sxl report build --strict          # must exit 0
pytest -q tests/test_report_claims.py
```

Expected: the headline table renders with six rows and separate batch-1 and
amortized latency columns; `README_FACTS.json` shows those two values differing
by roughly an order of magnitude; the forbidden-claim grep finds nothing;
`--strict` exits 0.

---

### Acceptance criteria

- [ ] `results/tables/headline.md`, `per_field.md`, `sweep.md`, and
      `README_FACTS.json` are generated by `sxl report build` and committed.
- [ ] Every numeric cell in every table traces to a value in a
      `results/**/*.json` file; a test asserts no literal number is embedded in
      the report source.
- [ ] The headline table has **separate columns** for single-stream `p50 ms/doc
      (batch 1)` and `amortized ms/doc @ best batch`, and a test asserts they are
      not equal for at least one arm.
- [ ] `Δ vs teacher` is computed from `results/metrics/*.json`, signed, and never
      hand-written.
- [ ] The `null baseline` row is present so the floor is visible.
- [ ] `check_claims` rejects `200x`-style parameter-ratio claims and rejects a
      `45ms` figure that is not the measured `p50_ms`; `--strict` exits 1 on any
      violation.
- [ ] `README.md` contains all six Limitations items from Scope 6, each carrying
      a real number where one exists, including the three lowest
      `teacher_field_agreement` fields.
- [ ] The README states plainly that vLLM was not used, why (T4/sm_75), and that
      a modern GPU with vLLM would improve the throughput and cost figures.
- [ ] `sxl report build` succeeds with only a subset of arms present, emitting
      `—` cells and a warning rather than crashing.
- [ ] `import sxl.report` pulls in no torch, numpy, pandas, or matplotlib.
- [ ] `sxl report build --strict` exits 0 on the final committed results.
