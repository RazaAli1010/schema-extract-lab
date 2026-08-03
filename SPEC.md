# SPEC.md — schema-extract-lab (shared context)

> **This file is the source of truth.** Every Claude Code session loads this file
> plus exactly one `specs/F*.md`. Where a feature spec and this file disagree,
> **this file wins** and the feature spec is a bug. Feature specs may only extend
> this file through their **Context deltas** section, which must be applied here
> *before* implementation begins.
>
> Verified against live package indexes on **2026-07-30**. Re-verify before trusting.

---

## 1. Goal

Fine-tune a ~1.7B open model to extract a **strict JSON schema** from unstructured
job postings, and prove with measured numbers that it lands close to a hosted
teacher model (`gpt-4o-mini`) at a fraction of the latency and cost.

The deliverable that matters is a **results table**, not a model. Specifically:

| arm | schema-valid % | macro-F1 | p50 ms/doc | $/1k docs |
|---|---|---|---|---|
| base few-shot (real competitor) | | | | |
| base few-shot + constrained | | | | |
| LoRA fine-tuned | | | | |
| teacher (ceiling) | | | | |

Every cell must be produced by a committed script from committed artifacts. No
cell is ever typed by hand.

### 1.1 Headline claims are hypotheses, not targets

The project pitch contains illustrative numbers ("within 3 F1", "45ms on a T4",
"~$0.002/1k docs"). **These are placeholders to be filled in by measurement, not
goals to hit.** No session may tune a benchmark, pick a batch size, or select a
subset to make a number match the pitch.

Two are already known to be optimistic and must not be assumed:

- **45ms/doc.** A T4 runs a 1.7B model in fp16 at roughly 40–70 tok/s
  single-stream. A ~150-token JSON output is therefore ~2–4 **seconds** at
  `batch_size=1`. Sub-100ms per document is only reachable as an *amortized*
  figure at large batch. F7 must report single-stream p50/p95 **and** amortized
  per-doc cost at the best batch size, clearly labeled as different things.
- **"200x larger model."** The teacher's parameter count is not public. Never
  claim a size ratio. Say "teacher model (`gpt-4o-mini`)". Do not call it a
  *frontier* model either — `gpt-4o-mini` is a small, cheap hosted model, so the
  claim being tested is "can a 1.7B fine-tune match a cheap hosted API model",
  not "…match a frontier model".

If the measured result is unflattering, the measured result ships.

---

## 2. Hardware reality — read this before writing any code

This is the single most important constraint in the project. It is repeated in
every feature spec because violating it wastes hours.

### 2.1 The laptop (development machine)

- **8 GB RAM, ~5 GB free disk, no GPU.**
- The laptop **must never** download model weights, install `torch`, or run
  inference. A single fp16 copy of Qwen3-1.7B is ~3.4 GB and would consume most
  of the free disk.
- `pip install -e .` on the laptop installs the **base** dependency group only
  (§6.1). That group is deliberately torch-free and totals well under 100 MB.
- Anything the laptop does is text-in / text-out on small JSONL files, plus
  network calls to the teacher API.

**Laptop-allowed work:** repo scaffolding, schema definition, corpus fetch and
normalization, teacher API labeling, human verification tooling, metrics
computation, report generation, all unit tests.

**Laptop-forbidden work:** `import torch`, `transformers` model loading,
`AutoModelForCausalLM.from_pretrained`, training, inference, latency benchmarks,
downloading `*.safetensors`.

Every module that imports torch lives under `src/sxl/gpu/` and is imported
**lazily inside functions**, never at module top level, so that `sxl` remains
importable on the laptop. Tests enforce this (§9.2).

### 2.2 Kaggle (all GPU work)

- **Free tier: ~30 GPU-hours/week**, replenished weekly. Budget it — the full
  project fits in roughly 12–16 hours if nothing is wasted.
- Accelerators: **`GPU T4 x2`** (2 × Tesla T4, 16 GB VRAM each) or **`GPU P100`**
  (1 × 16 GB). **This project standardizes on a single T4** so latency numbers
  are comparable across arms. Select `GPU T4 x2` and use `cuda:0` only.
- Host RAM on GPU instances: **~32 GB**. Max session: **~12 h** (interactive
  sessions idle out sooner). Assume any session can die; checkpoint accordingly.
- Disk: **`/kaggle/working` = 20 GB and is persisted** as notebook output.
  **`/kaggle/tmp` ≈ 60 GB scratch, not persisted.** Model weights and HF cache go
  to `/kaggle/tmp`. Only small JSON/JSONL results and the LoRA adapter go to
  `/kaggle/working`.
- Set `HF_HOME=/kaggle/tmp/hf` at the top of every notebook (§6.4).

### 2.3 T4 hardware limits (compute capability 7.5, Turing)

These cause real, confusing failures. All were verified against current upstream
issue trackers on 2026-07-30.

| Constraint | Consequence |
|---|---|
| **No bfloat16** (needs CC ≥ 8.0) | Use `dtype=torch.float16` everywhere. `bf16=False, fp16=True` in `SFTConfig`. Passing bf16 raises `ValueError` or silently downcasts. |
| **No FlashAttention-2** (needs sm80) | Use `attn_implementation="sdpa"`. Do not install `flash-attn`. |
| **No FP8 / NVFP4** | Only fp16 and bitsandbytes NF4 int4 are available. |
| **vLLM on Turing is unreliable** | vLLM has known sm_75 breakage (CUTLASS DSL arch detection, deregistered `TORCH_SDPA`, FA2-only paths). **vLLM is banned from this project.** See §5.3. |

fp16 (not bf16) training is numerically touchier: keep `max_grad_norm=0.3` and
watch for NaN loss. This is expected and is handled in F6.

### 2.4 Moving work between laptop and Kaggle

Code goes **laptop → Kaggle via a public GitHub repo**:

```python
!pip install -q "git+https://github.com/<user>/schema-extract-lab@<commit-sha>"
```

Pin a commit SHA, never a branch — a mid-session push must not change what a
running notebook is executing.

Data goes **laptop → Kaggle as a Kaggle Dataset** named `sxl-data`, containing
`train.jsonl`, `dev.jsonl`, `eval_gold.jsonl`. Mounted read-only at
`/kaggle/input/sxl-data/`. These files are small (a few MB).

Results come **Kaggle → laptop** by downloading the notebook output and
committing the small JSON files into `results/`. **Never** commit weights,
`*.safetensors`, or `.bin` to git.

The LoRA adapter (r=16 on a 1.7B model is roughly 20–40 MB) is published to the
**Hugging Face Hub** under `<hf_user>/qwen3-1.7b-jobpost-lora` and consumed from
there by later notebooks. It is not committed to git.

Notebooks are **thin**: install the package, call one CLI command, save output.
All logic lives in `src/sxl/` and is unit-testable on the laptop.

---

## 3. The contract — schema, splits, arms

Anything in this section is load-bearing. Field names, enum values, file names,
and metric definitions are exact and identical in every feature.

### 3.1 Domain: job postings only

The original brief said "job postings, invoices, **or** clinical notes". v1 ships
**job postings only**, for three reasons: the data is public and permissively
licensed; there is no PHI or de-identification burden (MIMIC-IV needs credentialed
access and cannot be used); and the schema has enough enum, numeric, nullable, and
set-valued variety to make the per-field breakdown interesting.

The pipeline is domain-parameterized (`domain` field on every record, prompts and
schema resolved by domain key) so a second domain is additive, but no feature in
F0–F8 implements one. Adding invoices is a future F9.

### 3.2 The target schema — `JobPosting`

Canonical definition lives in `src/sxl/schema.py`. **16 fields, flat, no nesting.**
Flatness is deliberate: nested lists of objects cannot be scored with a single
clean F1 formula, and the metric contract (§3.5) depends on flatness.

| # | field | type | null? | notes |
|---|---|---|---|---|
| 1 | `title` | `str` | yes | normalized role title |
| 2 | `company` | `str` | yes | |
| 3 | `employment_type` | enum `EmploymentType` | **no** | |
| 4 | `seniority` | enum `Seniority` | **no** | |
| 5 | `remote_mode` | enum `RemoteMode` | **no** | |
| 6 | `location_city` | `str` | yes | primary location only |
| 7 | `location_region` | `str` | yes | state / province |
| 8 | `location_country` | `str` | yes | ISO-3166 alpha-2, uppercase |
| 9 | `salary_min` | `float` | yes | |
| 10 | `salary_max` | `float` | yes | |
| 11 | `salary_currency` | `str` | yes | ISO-4217, uppercase |
| 12 | `salary_period` | enum `SalaryPeriod` | **no** | |
| 13 | `years_experience_min` | `int` | yes | |
| 14 | `education_level` | enum `EducationLevel` | **no** | |
| 15 | `required_skills` | `list[str]` | **no** (may be `[]`) | **SET field** |
| 16 | `posting_date` | `str` | yes | `YYYY-MM-DD` |

**Enum members (exact strings, lowercase snake_case):**

```
EmploymentType : full_time | part_time | contract | internship | temporary | unknown
Seniority      : intern | junior | mid | senior | lead | principal | unknown
RemoteMode     : onsite | hybrid | remote | unknown
SalaryPeriod   : hourly | daily | weekly | monthly | yearly | unknown
EducationLevel : none | high_school | associate | bachelor | master | doctorate | unknown
```

**The null/unknown rule.** Enum fields are **never** `null` — absence is the
member `"unknown"`. (`EducationLevel.none` means "the posting states no formal
education is required"; `unknown` means "the posting does not say". These are
different and must not be conflated.) Non-enum fields use `null` for absence.
`required_skills` uses `[]`. This rule exists so §3.5 can treat every field with
one formula.

**JSON Schema.** `src/sxl/schema.py` exports Draft 2020-12 JSON Schema with
`additionalProperties: false` and **all 16 keys in `required`**. A prediction
missing a key is invalid, not partially credited.

### 3.3 Data files and record shapes

All files are **JSONL, UTF-8, one object per line, keys in the order below**.

`data/raw/docs.jsonl` — output of F1
```json
{"doc_id": "jp_a1b2c3d4e5", "domain": "job_posting", "text": "...", "source": "xanderios/linkedin-job-postings", "char_len": 3412}
```

`doc_id` is `"jp_" + sha256(f"{source}\x00{source_key}")[:10]` — a function of
content, never of position. Sequential ids are **forbidden**: split membership
derives from `doc_id` (§3.4), so a sequence number would silently move documents
between `train` and `eval_gold` the moment upstream reordered its rows.

`data/labeled/{train,dev,eval_pool}.jsonl` and `data/gold/eval_gold.jsonl` — F2, F3
```json
{"doc_id": "jp_a1b2c3d4e5", "domain": "job_posting", "text": "...",
 "gold": { <the 16 JobPosting fields> },
 "label_source": "teacher",
 "teacher_model": "gpt-4o-mini",
 "verified_by_human": false,
 "verified_at": null}
```
In `eval_gold.jsonl`, `label_source` is `"human"` and `verified_by_human` is
`true` for all 300 rows.

`artifacts/predictions/<arm>.jsonl` — F5, F6, F7
```json
{"doc_id": "jp_000001", "arm": "base_fewshot", "raw_output": "...",
 "parsed": { ... } , "schema_valid": true,
 "latency_ms": 2841.3, "prompt_tokens": 1180, "completion_tokens": 143}
```
`parsed` is `null` when the output does not parse or does not validate.

`results/corpus_stats.json` — F1 (committed)
```json
{"source": "xanderios/linkedin-job-postings", "license": "mit",
 "n_seen": 0, "n_dropped_short": 0, "n_dropped_long": 0,
 "n_dropped_dup_id": 0, "n_dropped_dup_prefix": 0, "n_kept": 0,
 "char_len": {"p5": 0, "p50": 0, "p95": 0, "max": 0},
 "split_counts": {"train": 0, "dev": 0, "eval_pool": 0},
 "generated_at": "...", "git_sha": "abc1234"}
```

`results/metrics/<arm>.json` — F4
```json
{"arm": "base_fewshot", "split": "eval_gold", "n": 300,
 "schema_valid_rate": 0.0, "macro_f1": 0.0,
 "per_field": {"title": {"em": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}},
 "generated_at": "2026-07-30T00:00:00Z", "git_sha": "abc1234"}
```

`results/bench/<arm>.json` — F7
```json
{"arm": "lora_ft", "gpu_name": "Tesla T4", "dtype": "float16",
 "batch_size": 1, "n_docs": 200, "warmup": 10,
 "p50_ms": 0.0, "p95_ms": 0.0, "mean_ms": 0.0,
 "throughput_docs_per_s": 0.0, "gpu_hourly_usd": 0.35,
 "cost_per_1k_docs_usd": 0.0, "generated_at": "..."}
```

### 3.4 Splits

Target sizes: **train ≈ 5000**, **dev = 300**, **eval_gold = 300** (F3 samples
**330 candidates** from `eval_pool` and finalizes 300). Total teacher calls ≈ 5700.

The binding constraint on corpus size is the 5% `eval_pool` bucket: it must yield
**≥ 340** documents, so the corpus needs **≥ 7,000**. F1 targets 7,500.
`train` and `dev` are capped at their targets; **`eval_pool` is never capped.**

The corpus source is **`xanderios/linkedin-job-postings`** (MIT, ungated, 33,246
real postings, descriptions of ~3–5.5k chars), fetched by F1. It replaced
`lukebarousse/data_jobs`, which turned out to be metadata-only — 17 short columns
and no description field at all, longest string ~85 chars — so it could not
support extraction from unstructured text. F1 asserts at runtime that the chosen
text column has a median length above `CORPUS_MIN_CHARS`, so that failure mode
crashes loudly rather than yielding an empty corpus.

Split assignment is **deterministic and content-independent of ordering**:

```python
bucket = int(hashlib.sha256(doc_id.encode()).hexdigest()[:8], 16) % 100
# 0-4   -> eval_pool  (5%, F3 samples 300 from here to hand-verify)
# 5-9   -> dev        (5%)
# 10-99 -> train      (90%)
```

The three splits are disjoint by `doc_id`. **`eval_gold` doc_ids never appear in
`train.jsonl` or `dev.jsonl`** — F3 owns enforcing this and F4 owns asserting it
at metric time. Re-running any stage must not move a document between splits.

### 3.5 Metrics — one formula, sixteen fields

Defined once in `src/sxl/metrics.py`. Every arm, every report uses these and only
these. No sklearn (keeps the laptop install light); ~40 lines of arithmetic.

**Normalization before any comparison** (`src/sxl/normalize.py`, deterministic):
- strings: strip → collapse internal whitespace → lowercase
- `location_country`, `salary_currency`: additionally uppercase after strip (compared case-insensitively either way)
- numbers: cast to `float`, compare with `==` (values are money/years, not measurements)
- `posting_date`: compare the literal `YYYY-MM-DD` string
- `required_skills`: normalize each string as above, then compare as a **set**
- `null` and `"unknown"` are both the **absent** marker; `[]` is absent for `required_skills`

**(a) `schema_valid_rate`** = (# predictions that parse as JSON **and** validate
against the JSON Schema) / N. N is always the full split size.

**(b) Invalid predictions are wrong, not excluded.** A prediction that fails to
parse or validate contributes zero true positives and counts as a miss on every
field with a non-absent gold value. It is **never** dropped from the denominator.
This is the difference between an honest number and a self-congratulatory one.

**(c) Per-field TP/FP/FN**, aggregated over all N documents of the split:

| | gold absent | gold present |
|---|---|---|
| **pred absent** | — (true negative, ignored) | FN |
| **pred present, matches** | n/a | TP |
| **pred present, differs** | FP | FP **and** FN |

For `required_skills` (set field), do this per element: `TP += |P ∩ G|`,
`FP += |P \ G|`, `FN += |G \ P|`.

Then `precision = TP/(TP+FP)`, `recall = TP/(TP+FN)`,
`f1 = 2PR/(P+R)`, each `0.0` when the denominator is 0.

**(d) `per_field[f].em`** (exact match) = fraction of the N documents where
normalized pred equals normalized gold, *including* documents where both are
absent, and counting invalid predictions as mismatches. EM and F1 differ mainly
in how they treat the both-absent case; report both.

**(e) `macro_f1`** = arithmetic mean of the 16 `per_field[f].f1` values. Every
field weighs the same regardless of how often it is populated — that is the point
of macro.

### 3.6 The arms

Fixed set. `arm` is a string key used in filenames and records.

| arm | model | prompt | decoding | runs on |
|---|---|---|---|---|
| `base_fewshot` | `Qwen/Qwen3-1.7B` | 3-shot + JSON Schema in system prompt | greedy, unconstrained | Kaggle (F5) |
| `base_fewshot_constrained` | `Qwen/Qwen3-1.7B` | same | Outlines, schema-constrained | Kaggle (F5) |
| `lora_ft` | base + LoRA adapter | minimal instruction, no shots, no schema | greedy, unconstrained | Kaggle (F6) |
| `lora_ft_constrained` | base + LoRA adapter | same as `lora_ft` | Outlines, schema-constrained | Kaggle (F6, optional) |
| `teacher` | `gpt-4o-mini` | production labeling prompt | API default | laptop (F2) |

`base_fewshot` is **the competitor that matters**. If the fine-tune does not beat
it, that is the finding and it gets reported. The `*_constrained` arms exist to
separate two things people conflate: constrained decoding forces
`schema_valid_rate → 1.0` for free, but does nothing for *field accuracy*. Showing
that gap is the most interesting result in the project.

`teacher` is scored on the 300 hand-verified `eval_gold` rows, which measures
teacher-vs-human disagreement. Without it, "within N F1 of the teacher" is
meaningless.

**Prompt/generation constants** (`src/sxl/prompts.py`):
```python
MAX_INPUT_CHARS   = 6000     # truncate doc text, applied identically in all arms
MAX_NEW_TOKENS    = 512
TEMPERATURE       = 0.0      # greedy everywhere; extraction is not creative
N_FEWSHOT         = 3        # fixed exemplars, drawn from train, never from eval_gold
```

### 3.7 Qwen3 thinking mode must be OFF

Qwen3 is a hybrid reasoning model. Left at defaults it emits a `<think>...</think>`
block before the answer, which destroys `schema_valid_rate` for reasons unrelated
to the model's extraction ability and makes the baseline arm look artificially bad.

Every call to `apply_chat_template` in this project passes:
```python
tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                              enable_thinking=False)
```
Training data is rendered with the identical call, so train and inference match.
F5 must verify no `<think>` appears in `base_fewshot` raw outputs and fail loudly
if it does.

---

## 4. Repo layout

```
schema-extract-lab/
├── SPEC.md                      # this file
├── README.md                    # generated/updated by F8
├── specs/F0..F8.md
├── pyproject.toml
├── Makefile
├── .env.example
├── src/sxl/
│   ├── __init__.py
│   ├── config.py                # paths, constants, env
│   ├── schema.py                # JobPosting + enums + JSON Schema export
│   ├── normalize.py             # §3.5 normalization
│   ├── corpus.py                # F1
│   ├── teacher.py               # F2
│   ├── splits.py                # §3.4 hashing
│   ├── verify.py                # F3 human-verification TUI
│   ├── metrics.py               # §3.5
│   ├── prompts.py               # §3.6 constants + builders; F2's teacher prompt
│   │                            # and F5's build_student_prompt / pick_fewshot /
│   │                            # FEWSHOT_IDS live here side by side
│   ├── report.py                # F8
│   ├── cli.py                   # typer app, single entrypoint
│   └── gpu/                     # TORCH-ONLY. never imported on the laptop.
│       ├── __init__.py          # must stay empty of torch imports
│       ├── runner.py            # F5/F6 batched generate()
│       ├── constrained.py       # F5 outlines wrapper
│       ├── train_lora.py        # F6
│       └── bench.py             # F7
├── notebooks/
│   ├── kaggle_baseline.ipynb    # F5
│   ├── kaggle_train_lora.ipynb  # F6
│   └── kaggle_bench.ipynb       # F7
├── data/                        # gitignored (except .gitkeep)
│   ├── raw/docs.jsonl
│   ├── labeled/{train,dev,eval_pool}.jsonl
│   └── gold/eval_gold.jsonl
├── artifacts/                   # gitignored — predictions, scratch
├── results/                     # COMMITTED, small JSON + md tables
│   ├── corpus_stats.json
│   ├── metrics/<arm>.json
│   ├── bench/<arm>.json
│   └── tables/headline.md
└── tests/
```

**CLI contract** — one entrypoint, `sxl`, defined in `src/sxl/cli.py` with typer.
Commands are added by the feature that owns them and never renamed:

```
sxl corpus build                 # F1
sxl teacher label --split train  # F2
sxl gold sample                  # F3
sxl gold verify                  # F3
sxl metrics score --arm <arm>    # F4
sxl gpu predict --arm <arm>      # F5  (Kaggle only)
sxl gpu train                    # F6  (Kaggle only)
sxl gpu bench --arm <arm>        # F7  (Kaggle only)
sxl report build                 # F8
```

`sxl gpu *` subcommands import `sxl.gpu.*` lazily, inside the command body, so
`sxl --help` works on the laptop with no torch installed.

---

## 5. Engineering principles

1. **The laptop never sees a GPU dependency.** §2.1. Enforced by a test.
2. **Reproducible or it didn't happen.** Every artifact records `git_sha`,
   `generated_at`, and the config that produced it. Every random operation takes
   a seed; `SEED = 1337` in `config.py`.
3. **Idempotent stages.** Re-running any command over existing outputs is safe and
   produces byte-identical results, except for timestamps. Teacher labeling is
   cached and resumable by `doc_id` (§F2) — a crashed run must never re-bill.
4. **Every loop has a cap that lives in config.** Retries, API concurrency,
   generation length, spend. No unbounded `while True`.
5. **Fail loudly on contract violations.** Schema drift, split leakage, and
   arm/file mismatches raise, not warn.
6. **Report what you measured.** §1.1.
7. **Pin versions, print versions.** Every Kaggle notebook prints the resolved
   versions of torch/transformers/trl/peft/bitsandbytes into its output as the
   first cell, so a stale Kaggle base image is visible in the artifact.

### 5.3 Explicitly rejected alternatives

Recorded so no session re-litigates them.

- **Unsloth** — rejected. Its headline 2× speedup is ~1.2–1.5× on a 1.7B model,
  and it currently carries a messy dependency negotiation with transformers 5.x
  and vLLM coexistence that has burned people on Kaggle. Not worth spending
  scarce GPU-hours debugging an installer. Plain `peft` + `trl` is enough here.
  *(Production alternative: on a 7B+ model with more GPU budget, Unsloth is worth
  revisiting.)*
- **vLLM** — rejected, §2.3. Turing (sm_75) support is degrading and vLLM pins
  older transformers. Latency measurement uses plain `transformers.generate()`
  with a static KV cache. *(Production alternative: on an A10/L4/A100, vLLM would
  be the correct serving path and would improve the throughput numbers
  substantially — F8 says so in the write-up rather than pretending otherwise.)*
- **QLoRA (4-bit) as the default** — rejected as default, kept as OOM fallback.
  A 1.7B model in fp16 is ~3.4 GB and fits a 16 GB T4 with room for activations;
  4-bit would add quantization error for no headroom benefit. F6 defaults to
  fp16 LoRA.
- **ONNX Runtime / llama.cpp export** — out of scope for v1. It would improve the
  cost table but is a separate project. F8 notes it as future work.
- **MIMIC-IV / real clinical notes** — rejected, §3.1. Credentialed access.

---

## 6. Tech stack — verified versions (2026-07-30)

### 6.1 Laptop (base group) — no torch

```toml
[project]
requires-python = ">=3.11"
dependencies = [
  "pydantic==2.13.4",
  "jsonschema==4.26.0",
  "openai==2.51.0",
  "typer==0.27.0",
  "python-dotenv>=1.0",
  "datasets==5.0.1",      # base, not gpu: F1 streams the corpus on the laptop.
                          # Verified torch-free 2026-07-31: `find_spec("torch")`
                          # is None and no nvidia-* wheels. It pulls pyarrow,
                          # pandas, numpy, fsspec, huggingface_hub, aiohttp.
]
[project.optional-dependencies]
dev = ["pytest==9.1.1", "ruff==0.16.0"]
gpu = [                      # NEVER installed on the laptop
  "torch==2.13.0",
  "transformers==5.14.1",
  "trl==1.5.1",
  "peft==0.20.0",
  "accelerate==1.14.0",
  "bitsandbytes==0.50.0",
  "outlines==1.3.2",
]
```

`metrics.py` deliberately has **no** numpy/sklearn dependency — the formulas in
§3.5 are plain arithmetic, and this keeps the laptop install trivial.

**Corpus constants** (`config.py`, applied from the F1 context deltas):

```python
CORPUS_SOURCES = ("xanderios/linkedin-job-postings",)  # HF ids, priority order
CORPUS_TARGET_N   = 7500
CORPUS_MIN_N      = 7000     # below this, `sxl corpus build` exits non-zero.
# 7000 docs x 5% = ~350 eval_pool, which must exceed F3's 330 candidates.
# 6500 would leave only ~325 and starve F3 -- do not lower these.
CORPUS_MIN_CHARS  = 400      # drop stubs
CORPUS_MAX_CHARS  = 40000    # drop scrape artifacts / concatenated pages
CORPUS_DEDUPE_PREFIX_CHARS = 600   # near-duplicate window
CORPUS_MAX_SCAN   = 200_000  # hard cap on rows read upstream (§5.4)
CORPUS_PEEK_ROWS  = 20       # rows sampled to auto-detect the text column
CORPUS_MIN_FREE_BYTES = 2 * 1024**3
CORPUS_MIN_SPLIT_N = 340     # required in both `dev` and `eval_pool`
HF_CACHE_DIR = DATA / ".hfcache"   # deleted after a successful build
```

**Inference constants** (`config.py`, applied from the F5 context deltas):

```python
GEN_BATCH_SIZE = 8           # inference batch for the F5/F6 prediction runs
KAGGLE_TMP     = Path("/kaggle/tmp")      # ~60 GB scratch, NOT persisted
KAGGLE_WORKING = Path("/kaggle/working")  # 20 GB, persisted as notebook output
```

`HF_HOME`/`HF_HUB_CACHE` are redirected to `HF_CACHE_DIR` **before** `datasets`
is imported — `huggingface_hub` freezes the cache path into module constants at
import time, so setting it afterwards silently fills `~/.cache/huggingface` on a
laptop with 5 GB free. `sxl.corpus` therefore imports `datasets` inside
`fetch_raw` only, and `tests/test_no_torch_import.py` enforces it.

### 6.2 transformers 5.x — breaking changes that will bite

transformers v5 was a hard break from the v4 API that most tutorials and most
model-card snippets still show. Do not write v4 code.

| v4 (wrong now) | v5 (correct) |
|---|---|
| `from_pretrained(..., torch_dtype=torch.float16)` | `from_pretrained(..., dtype=torch.float16)` |
| `from_pretrained(..., load_in_4bit=True)` | pass `quantization_config=BitsAndBytesConfig(...)` |
| `Trainer(tokenizer=tok)` | `Trainer(processing_class=tok)` |
| `TRANSFORMERS_CACHE` env var | `HF_HOME` |
| `transformers-cli` | `transformers` |
| TF / Flax classes | removed entirely; PyTorch only |

Also: **`dtype` now defaults to `"auto"`**, meaning models load in whatever
precision they were saved in rather than fp32. Always pass `dtype` explicitly so
a T4 never receives bf16 weights. v5 requires `huggingface_hub>=1.0`
(httpx-backed, so catch `httpx.HTTPError` not `requests.HTTPError`),
`peft>=0.18`, `bitsandbytes>=0.46.1`.

### 6.3 trl 1.x — breaking changes

| old | current (1.5.1) |
|---|---|
| `SFTConfig(max_seq_length=...)` | `SFTConfig(max_length=...)` |
| `SFTTrainer(tokenizer=...)` | `SFTTrainer(processing_class=...)` |
| `dataset_text_field` + manual packing | still supported; pass pre-rendered `"text"` column |

`loss_type` now defaults to `"chunked_nll"` (≈30% less peak VRAM, no action
needed). `SFTTrainer` accepts `quantization_config=` directly alongside
`peft_config=`. Passing an already-`get_peft_model`-wrapped model **and**
`peft_config` is an error — do one or the other. This project passes a plain
model plus `peft_config=LoraConfig(...)`.

### 6.4 outlines 1.x — the old API is gone

Every `outlines.generate.json(...)` / `outlines.models.transformers(...)` snippet
on the internet is v0 and will `AttributeError` on 1.3.2. Current API:

```python
import outlines
from transformers import AutoModelForCausalLM, AutoTokenizer

model = outlines.from_transformers(
    AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map="cuda:0"),
    AutoTokenizer.from_pretrained(MODEL),
)
raw = model(prompt, JobPosting, max_new_tokens=512)   # returns a JSON *string*
obj = JobPosting.model_validate_json(raw)
```

The call returns a string, not a model instance — validate it yourself. Building
the index for a 16-field schema takes a few seconds and is done **once**, outside
the timing loop.

### 6.5 Model and teacher

- Student: **`Qwen/Qwen3-1.7B`** — 1.7B params (1.4B non-embedding), 28 layers,
  GQA 16 Q-heads / 8 KV-heads, 32,768 context. Hybrid reasoning → §3.7.
  *(Newer Qwen3.5-0.8B/2B models exist. Not used: Qwen3-1.7B is well-supported,
  stable, and matches the project brief. Do not silently substitute.)*
- Fallback student if Qwen3 misbehaves: `meta-llama/Llama-3.2-1B-Instruct`
  (gated — needs an accepted HF license). Qwen3 is preferred precisely because it
  is ungated and works on Kaggle without token gymnastics.
- Teacher: **`gpt-4o-mini`** via the OpenAI **Batch API** (50% discount, 24h
  SLA). Alternative if label quality proves too weak: `gpt-4o`. Whichever is
  used is recorded in `teacher_model` on every row and must be constant within
  a run.

**Budget.** ~5700 docs × ~1.2k input + ~350 output tokens ≈ 6.8M in / 2.0M out.
At `gpt-4o-mini` batch rates ($0.075/$0.30 per M — half the $0.15/$0.60 standard
rates) that is **≈ $1.10**. `MAX_TEACHER_SPEND_USD = 25.0` in `config.py` is a
hard stop, not a guideline (§F2) — it now sits far above the estimate, so it
protects against a runaway loop rather than acting as a live budget ceiling.

### 6.6 GPU-hour budget (of ~30/week)

| feature | est. hours | notes |
|---|---|---|
| F5 baseline (2 arms × 300 docs) | 3–4 | unconstrained is slow; batch it |
| F6 LoRA training (5k rows, 2–3 epochs) | 2–4 | plus 1h of failed first attempts |
| F6 inference (2 arms × 300) | 2–3 | |
| F7 benchmarks (all arms, batch sweep) | 2 | |
| contingency | 4 | assume one session dies |
| **total** | **~13–17** | fits one week with margin |

---

## 7. Definition of done (project level)

- [ ] `results/tables/headline.md` exists, is generated by `sxl report build`, and
      every cell traces to a file in `results/`.
- [ ] All five arms in §3.6 have a `results/metrics/<arm>.json`.
- [ ] `pytest` passes on the laptop with only the base + dev dependency groups
      installed (no torch present).
- [ ] `eval_gold.jsonl` has exactly 300 rows, all `verified_by_human: true`, and
      zero `doc_id` overlap with `train.jsonl` / `dev.jsonl`.
- [ ] README states the measured numbers and names the honest caveats: teacher
      labels are not ground truth outside the 300; one domain; one seed;
      single-GPU; no vLLM.

---

## 8. Feature index and dependency order

| ID | feature | runs on | depends on |
|---|---|---|---|
| F0 | Repo scaffold, schema, config, CLI skeleton | laptop | — |
| F1 | Corpus acquisition & normalization | laptop | F0 |
| F2 | Teacher labeling pipeline (batch, cached, capped) | laptop | F0, F1 |
| F3 | Eval-set sampling + human verification tool | laptop | F0, F2 |
| F4 | Metrics library + scoring CLI | laptop | F0 |
| F5 | Prompted baseline arms on Kaggle | **Kaggle** | F0, F3, F4 |
| F6 | LoRA fine-tune + fine-tuned inference | **Kaggle** | F0, F2, F3, F4 |
| F7 | Latency / throughput / cost benchmark | **Kaggle** | F0, F5, F6 |
| F8 | Results aggregation, headline table, README | laptop | F4, F5, F6, F7 |

---

## 9. Testing

### 9.1 What gets tested
Pure functions with hand-written fixtures: schema validation, normalization,
split hashing, all metric math, prompt rendering, teacher-response parsing. GPU
code is tested only for import-time laptop-safety (§9.2) and by its own
acceptance criteria on Kaggle.

### 9.2 The laptop-safety test (`tests/test_no_torch_import.py`)
Non-negotiable, added in F0:

```python
import subprocess, sys, textwrap

def test_importing_sxl_does_not_import_torch():
    code = textwrap.dedent("""
        import sys
        import sxl, sxl.cli, sxl.metrics, sxl.schema, sxl.teacher
        assert "torch" not in sys.modules, sorted(m for m in sys.modules if "torch" in m)
    """)
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0
```

### 9.3 Metric golden tests
`tests/test_metrics.py` includes a hand-computed fixture: 4 documents, known
TP/FP/FN, expected macro-F1 written out by hand in the test. If a session
"improves" the metric code and this fails, the metric changed — that is the point.
