## F5 — Prompted baseline arms on Kaggle (`base_fewshot`, `base_fewshot_constrained`)

**Goal:** On a Kaggle T4, run un-finetuned `Qwen/Qwen3-1.7B` over the 300
`eval_gold` documents twice — once few-shot with the schema in the prompt, once
with Outlines schema-constrained decoding — and produce two prediction files that
F4 can score. This is the baseline the fine-tune must beat to justify itself.

**Depends on:** F0 (schema, config, io), F3 (`eval_gold.jsonl`), F4 (scoring)

---

### ⚠️ This feature runs on Kaggle, not the laptop

**SPEC §2.1 — the dev laptop has 8 GB RAM, ~5 GB free disk, and no GPU.** It
**must not** install torch, download `Qwen/Qwen3-1.7B` (~3.4 GB fp16), or execute
any function in this feature.

What the laptop does in this feature:
- write `src/sxl/gpu/runner.py`, `src/sxl/gpu/constrained.py`, and
  `notebooks/kaggle_baseline.ipynb`
- run the *unit* tests, which mock the model and never import torch
- push to GitHub and read the results back

What Kaggle does:
- everything that loads a model or generates a token

**Kaggle setup (SPEC §2.2):** Accelerator **`GPU T4 x2`**, use **`cuda:0` only**.
30 GPU-h/week; this feature should cost **3–4 h**. `/kaggle/working` is 20 GB and
persisted; `/kaggle/tmp` is ~60 GB scratch and is not. Sessions cap at ~12 h and
can die at any time.

**T4 limits (SPEC §2.3) — all four apply here:**
- **No bf16.** `dtype=torch.float16` everywhere.
- **No FlashAttention-2.** `attn_implementation="sdpa"`.
- **vLLM is banned** (SPEC §5.3) — Turing support is degrading. Plain
  `transformers.generate()`.
- No FP8/NVFP4.

---

### Context digest

**Arms being produced (SPEC §3.6):**

| arm | prompt | decoding |
|---|---|---|
| `base_fewshot` | 3-shot + JSON Schema in system prompt | greedy, unconstrained |
| `base_fewshot_constrained` | same | Outlines, schema-constrained |

`base_fewshot` is **the competitor that matters**. SPEC §3.6: *"If the fine-tune
does not beat it, that is the finding and it gets reported."* The
`*_constrained` arm exists to separate two things people conflate: constrained
decoding forces `schema_valid_rate → 1.0` for free but does nothing for field
accuracy, and showing that gap is the most interesting result in the project.

**Generation constants (SPEC §3.6, from `config.py`):**
```python
MAX_INPUT_CHARS = 6000    # identical truncation in every arm
MAX_NEW_TOKENS  = 512
TEMPERATURE     = 0.0     # greedy; extraction is not creative
N_FEWSHOT       = 3       # fixed exemplars, drawn from train, NEVER from eval_gold
BASE_MODEL      = "Qwen/Qwen3-1.7B"
```

**Qwen3 thinking mode must be OFF — SPEC §3.7.** Qwen3 is a hybrid reasoning
model; at defaults it emits `<think>...</think>` before the answer, which
destroys `schema_valid_rate` for reasons unrelated to extraction ability and
makes this baseline look artificially bad. Every `apply_chat_template` call:
```python
tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                              enable_thinking=False)
```
**F5 must verify no `<think>` appears in `base_fewshot` raw outputs and fail
loudly if it does.**

**Output shape (SPEC §3.3)** — `artifacts/predictions/<arm>.jsonl`:
```json
{"doc_id": "...", "arm": "base_fewshot", "raw_output": "...",
 "parsed": {...}, "schema_valid": true,
 "latency_ms": 2841.3, "prompt_tokens": 1180, "completion_tokens": 143}
```
`parsed` is `null` when the output does not parse or does not validate.
`schema_valid` comes **only** from `sxl.schema.validate_prediction` (F0).

**Model (SPEC §6.5):** `Qwen/Qwen3-1.7B` — 1.7B params, 28 layers, GQA 16 Q /
8 KV heads, 32,768 context. Ungated (no HF token needed), which is why it was
chosen over Llama-3.2-1B.

**Library versions (SPEC §6.1–6.4), verified 2026-07-30:**
`torch==2.13.0`, `transformers==5.14.1`, `outlines==1.3.2`, `peft==0.20.0`.

### Context deltas

Additions to `config.py`:
```python
GEN_BATCH_SIZE   = 8        # inference batch for F5/F6 prediction runs
KAGGLE_TMP       = Path("/kaggle/tmp")
KAGGLE_WORKING   = Path("/kaggle/working")
```

Addition to `prompts.py` (which F2 created): `build_student_prompt(text, shots)`
and `FEWSHOT_IDS: tuple[str, str, str]` — three `doc_id`s frozen at
implementation time and committed, so the few-shot arm is reproducible.

---

### Scope

1. **`src/sxl/prompts.py::pick_fewshot(train_rows, n) -> list[dict]`** and
   `FEWSHOT_IDS`. Select `n = N_FEWSHOT` exemplars from **`train.jsonl` only**
   — drawing a shot from `eval_gold` or `dev` is test-set contamination.
   Selection is deterministic (`random.Random(SEED)`) and the chosen ids are then
   **hard-coded into `FEWSHOT_IDS` and committed**, so a later corpus rebuild
   cannot silently change the baseline. Prefer exemplars that between them cover
   a populated salary, a non-`unknown` `remote_mode`, and a non-empty
   `required_skills` — a 3-shot prompt whose exemplars are all `null` teaches the
   model to answer `null`.

2. **`src/sxl/prompts.py::build_student_prompt(text, shots) -> list[dict]`**
   Returns chat messages. System message: extraction role + `json.dumps(JSON_SCHEMA, indent=2)`
   + the **null/unknown rule** (SPEC §3.2, `none` vs `unknown` spelled out) +
   "emit only a JSON object, no prose, no code fences". Then `N_FEWSHOT`
   user/assistant pairs, each assistant turn being
   `json.dumps(shot["gold"], separators=(",", ":"))`. Then the user turn with
   `text[:MAX_INPUT_CHARS]`.
   This is **separate from** F2's `build_teacher_prompt` and must not reuse it —
   the student's prompt is part of the measured arm.

3. **`src/sxl/gpu/runner.py::load_model(model_id, adapter=None)`**
   ```python
   import torch
   from transformers import AutoModelForCausalLM, AutoTokenizer
   tok = AutoTokenizer.from_pretrained(model_id)
   tok.padding_side = "left"                     # REQUIRED for batched generate
   if tok.pad_token is None: tok.pad_token = tok.eos_token
   model = AutoModelForCausalLM.from_pretrained(
       model_id,
       dtype=torch.float16,                      # transformers v5: `dtype`, NOT `torch_dtype`
       attn_implementation="sdpa",               # no FA2 on sm_75
       device_map={"": 0},
   ).eval()
   if adapter:                                   # used by F6, not F5
       from peft import PeftModel
       model = PeftModel.from_pretrained(model, adapter).eval()
   ```
   All torch/transformers imports are **inside the function**, never at module
   top level (SPEC §2.1) — but note `runner.py` lives under `src/sxl/gpu/` and is
   never imported by `sxl.cli` at import time anyway.

4. **`src/sxl/gpu/runner.py::generate_batch(model, tok, prompts, max_new_tokens) -> list[dict]`**
   Left-padded batch of `GEN_BATCH_SIZE`, `do_sample=False` (greedy — do **not**
   pass `temperature=0.0`, which transformers rejects alongside `do_sample=False`),
   `pad_token_id=tok.pad_token_id`. Decode only the **completion** slice
   (`out[:, inputs["input_ids"].shape[1]:]`) — decoding the full sequence and
   string-stripping the prompt is fragile and has bitten every project that tried
   it. Return per-item `raw_output`, `prompt_tokens`, `completion_tokens`, and a
   **per-batch** `latency_ms` divided evenly across the batch, clearly documented
   as amortized. Single-document p50/p95 is **F7's job**, not this file's.

5. **`src/sxl/gpu/runner.py::predict_arm(arm, gold_rows, model, tok, constrained) -> list[dict]`**
   Loop batches → generate → `teacher.extract_json` (F2's tolerant parser, reused)
   → `validate_prediction` → build the SPEC §3.3 record. **Checkpoint every
   batch** by appending to `artifacts/predictions/<arm>.partial.jsonl`; on start,
   skip doc_ids already present. Kaggle sessions die; 300 documents at ~3 s each
   is ~15 minutes per arm and losing it to an idle timeout is avoidable.
   Rename `.partial` → final on completion.

6. **`src/sxl/gpu/constrained.py::build_constrained_generator(model, tok)`**
   ```python
   import outlines
   gen = outlines.from_transformers(model, tok)     # outlines 1.x API
   raw = gen(prompt, JobPosting, max_new_tokens=MAX_NEW_TOKENS)   # returns a STRING
   ```
   Every `outlines.generate.json(...)` / `outlines.models.transformers(...)`
   snippet online is the **v0 API and will `AttributeError` on 1.3.2** (SPEC §6.4).
   Build the index **once**, outside any timing loop — it takes seconds for a
   16-field schema. The call returns a JSON string, not a model instance; validate
   it with `validate_prediction` like every other arm so `schema_valid` is
   produced by one code path.

7. **`sxl gpu predict` CLI** (replaces the F0 stub), with the `from sxl.gpu import ...`
   **inside the command body**:
   `--arm [base_fewshot|base_fewshot_constrained]`, `--gold PATH`,
   `--model TEXT` (default `BASE_MODEL`), `--adapter TEXT` (unused in F5, present
   for F6), `--batch-size INT`, `--limit INT` (for smoke runs), `--out PATH`.

8. **`notebooks/kaggle_baseline.ipynb`** — thin, ~6 cells:
   1. `import os; os.environ["HF_HOME"] = "/kaggle/tmp/hf"` **before any HF
      import** — the default cache is on the 20 GB persisted volume and a 3.4 GB
      model plus safetensors staging will fill it.
   2. `!pip install -q "git+https://github.com/<user>/schema-extract-lab@<SHA>"`
      — **pin a commit SHA, never a branch** (SPEC §2.4).
   3. Version banner (SPEC §5.7): print resolved `torch`, `transformers`, `trl`,
      `peft`, `bitsandbytes`, `outlines` versions and
      `torch.cuda.get_device_name(0)` + `torch.cuda.get_device_capability(0)`.
      **Assert capability is `(7, 5)`** so the artifact records that these numbers
      came from a T4 and not from a session that silently got a P100.
   4. `!sxl gpu predict --arm base_fewshot --limit 8` — smoke run first, inspect
      the raw output by eye before spending 15 minutes.
   5. Full runs for both arms.
   6. Copy `artifacts/predictions/*.jsonl` to `/kaggle/working/` and print the
      row counts.

---

### Out of scope

- Any training — F6 owns `train_lora.py`. F5 loads the base model read-only.
- The `lora_ft` / `lora_ft_constrained` arms — F6, using **this same**
  `runner.py::predict_arm` with `--adapter`. F5 must therefore write `predict_arm`
  generically; F6 adds no new generation code.
- Latency/throughput/cost measurement — **F7 exclusively.** The `latency_ms` field
  F5 writes is amortized batch time for debugging only and must never appear in
  a results table (SPEC §3.3 note, F4 out-of-scope note).
- Scoring — F4. F5 produces prediction files and stops.
- Prompt tuning to improve the baseline beyond a fair, competent attempt. A
  deliberately weak baseline is the most common way this genre of project lies.
  One honest schema-in-prompt few-shot setup, documented, and that is the number.

---

### Implementation notes

- **transformers 5.14.1 breaking changes (SPEC §6.2)** — `dtype=` not
  `torch_dtype=`; no `load_in_4bit=` shortcut; `HF_HOME` not `TRANSFORMERS_CACHE`;
  TF/Flax gone. `dtype` now defaults to `"auto"`, so **always pass it explicitly**
  or a T4 may receive bf16 weights and raise.
- **`enable_thinking=False`** on every `apply_chat_template` call (SPEC §3.7).
  Add an explicit post-run assertion: if `"<think>"` appears in any
  `raw_output`, raise — do not "handle" it by stripping, because stripping would
  hide a train/inference template mismatch that F6 would then inherit.
- **`padding_side = "left"`.** Decoder-only batched generation with right padding
  produces garbage for every sequence but the longest. This is the single most
  common batched-inference bug.
- **Greedy decoding:** `do_sample=False` alone. Passing `temperature=0.0` with
  `do_sample=False` triggers a transformers warning-or-error depending on
  version; `TEMPERATURE = 0.0` in config is documentation of intent, not an
  argument to forward.
- **Outlines index build cost** is per-schema, not per-document. Build once and
  reuse across all 300 documents or the constrained arm will take an hour.
- **Expect the constrained arm to be slower per token** (logit masking overhead)
  and to hit `MAX_NEW_TOKENS` more often on `required_skills`, since a grammar
  cannot end a list early if the model wants to keep going. If truncation rate
  exceeds ~5%, record it in the run log — F8 needs to report it as a real cost of
  constrained decoding, not hide it.
- **VRAM:** 1.7B fp16 ≈ 3.4 GB weights + KV cache. `GEN_BATCH_SIZE = 8` at ~1.5k
  prompt tokens is comfortable inside 16 GB. If OOM, halve the batch — do **not**
  switch to 4-bit here, because the baseline arm must use the same precision as
  the fine-tuned arm for the comparison to mean anything.
- **GPU-hour budget:** 2 arms × 300 documents. Unconstrained ≈ 3 s/doc at batch 8
  amortized to well under 1 s; constrained slower. Budget 3–4 h including one
  failed session. Stop the Kaggle session manually when done — idle sessions burn
  quota.

---

### Test plan

Laptop tests only; **no test imports torch or loads a model.**

`tests/test_prompts.py`
- `build_student_prompt` output contains all 16 field names and every enum member
  string.
- The rendered prompt contains the `none` vs `unknown` distinction sentence.
- Document text is truncated at exactly `MAX_INPUT_CHARS`.
- `pick_fewshot` never returns a `doc_id` present in `eval_gold.jsonl` or
  `dev.jsonl` (assert against the real files if present, else a fixture).
- `pick_fewshot` is deterministic across two calls with the same seed.
- The three `FEWSHOT_IDS` between them cover a non-null salary, a non-`unknown`
  `remote_mode`, and a non-empty `required_skills`.

`tests/test_runner_records.py` — with a fake generator returning canned strings:
- A well-formed JSON completion → `schema_valid True`, `parsed` populated, record
  has exactly the 8 SPEC §3.3 keys in order.
- A completion wrapped in ```` ```json ```` fences → parsed via
  `teacher.extract_json`, `schema_valid True`.
- A refusal string → `parsed is None`, `schema_valid False`, and the record is
  still **written** (not skipped) — F4 depends on full coverage.
- A completion containing `<think>` raises.
- Resuming with a `.partial` file containing 5 doc_ids requests only the rest.

`tests/test_gpu_lazy_import.py`
- `import sxl.cli` leaves `torch` out of `sys.modules` even though the `gpu`
  sub-app is registered (extends F0's §9.2 test to cover the new commands).

---

### Verify

**On the laptop:**
```bash
pytest -q
python -c "import sys, sxl.cli; assert 'torch' not in sys.modules; print('lazy OK')"
```

**On Kaggle** (`GPU T4 x2`, `kaggle_baseline.ipynb`):
```bash
!nvidia-smi --query-gpu=name,memory.total --format=csv
!python -c "import torch;print(torch.cuda.get_device_capability(0))"   # must be (7, 5)
!sxl gpu predict --arm base_fewshot --limit 8 --out /kaggle/working/_smoke.jsonl
!head -c 2000 /kaggle/working/_smoke.jsonl
!sxl gpu predict --arm base_fewshot
!sxl gpu predict --arm base_fewshot_constrained
!wc -l artifacts/predictions/base_fewshot.jsonl artifacts/predictions/base_fewshot_constrained.jsonl
!python - <<'PY'
import json
for a in ("base_fewshot","base_fewshot_constrained"):
    R=[json.loads(l) for l in open(f"artifacts/predictions/{a}.jsonl")]
    assert len(R)==300, (a,len(R))
    assert not any("<think>" in r["raw_output"] for r in R), a
    v=sum(r["schema_valid"] for r in R)
    print(a, "valid", v, "/", len(R))
PY
!cp artifacts/predictions/*.jsonl /kaggle/working/
```

**Back on the laptop**, after downloading the two files:
```bash
sxl metrics score --arm base_fewshot
sxl metrics score --arm base_fewshot_constrained
sxl metrics compare
```

Expected: both files have exactly 300 rows and no `<think>`;
`base_fewshot_constrained` shows `schema_valid_rate` at or very near **1.00**
while `base_fewshot` is materially lower; and — the point of the arm — the two
`macro_f1` values are **close**, demonstrating that constraining the grammar buys
validity, not accuracy.

---

### Acceptance criteria

- [ ] `artifacts/predictions/base_fewshot.jsonl` and
      `base_fewshot_constrained.jsonl` each have exactly 300 rows, one per
      `eval_gold` `doc_id`, with the 8 SPEC §3.3 keys in order.
- [ ] Zero occurrences of `<think>` in any `raw_output`; the run raises if any
      appear.
- [ ] `base_fewshot_constrained` achieves `schema_valid_rate ≥ 0.99`.
- [ ] `schema_valid` for every record was produced by
      `sxl.schema.validate_prediction`, not by any local check.
- [ ] The notebook's version banner is present in the saved output and records
      `torch.cuda.get_device_capability(0) == (7, 5)`.
- [ ] No `flash-attn` and no `vllm` in the notebook's install cell.
- [ ] `FEWSHOT_IDS` are committed constants, all three drawn from `train.jsonl`,
      and a test proves none appear in `eval_gold.jsonl` or `dev.jsonl`.
- [ ] Interrupting a run and re-invoking resumes from the `.partial` file without
      regenerating completed documents.
- [ ] `pytest` passes on the laptop with no torch installed, and
      `import sxl.cli` still does not pull torch.
- [ ] Total Kaggle GPU time for this feature is recorded in the notebook output
      and is under 4 hours.
