## F6 — LoRA fine-tune on Kaggle + fine-tuned inference (`lora_ft`, `lora_ft_constrained`)

**Goal:** Train a LoRA adapter for `Qwen/Qwen3-1.7B` on ~5,000 teacher-labeled
job postings on a Kaggle T4, publish the adapter to the HF Hub, and produce
prediction files for the `lora_ft` and `lora_ft_constrained` arms that F4 scores
against the same 300 gold documents the baseline used.

**Depends on:** F0 (schema, config, io), F2 (`train.jsonl`, `dev.jsonl`),
F3 (`eval_gold.jsonl`), F4 (scoring), F5 (`gpu/runner.py`, `gpu/constrained.py`)

---

### ⚠️ This feature runs on Kaggle, not the laptop

**SPEC §2.1 — the dev laptop has 8 GB RAM, ~5 GB free disk, and no GPU.** It
cannot hold the base model (3.4 GB fp16), let alone train. **Do not** attempt a
"tiny local run to check the code" — there is no disk for it. The smoke test is
`--limit 32 --max-steps 5` **on Kaggle**.

Laptop writes `src/sxl/gpu/train_lora.py`, the data-rendering functions, the
notebook, and the mocked unit tests. Kaggle does everything else.

**Kaggle setup (SPEC §2.2):** Accelerator **`GPU T4 x2`**, use **`cuda:0` only**
(single GPU — the latency numbers in F7 are single-T4 and training on two would
not change that, while adding DDP complexity for a 1.7B model that fits on one).
Sessions cap at ~12 h and **can die at any time** — checkpointing is mandatory,
not optional. `/kaggle/tmp` (~60 GB, ephemeral) holds the HF cache and training
checkpoints; `/kaggle/working` (20 GB, persisted) holds only the final adapter
and the prediction JSONLs.

**Budget (SPEC §6.6):** training 2–4 h + inference 2–3 h, of ~30 h/week. Assume
one wasted session. Do not start a full run without a passing 5-step smoke run.

**T4 limits (SPEC §2.3) — decisive for this feature:**
- **No bf16** (needs CC ≥ 8.0). `dtype=torch.float16`, and in `SFTConfig`:
  `fp16=True, bf16=False`. Passing bf16 raises.
- **fp16 training is numerically touchier than bf16.** Keep `max_grad_norm=0.3`
  and watch for NaN loss (SPEC §2.3). This is expected behavior on a T4, not a
  bug in the data.
- **No FlashAttention-2.** `attn_implementation="sdpa"`.
- **vLLM banned** (SPEC §5.3).

---

### Context digest

**Rejected alternatives (SPEC §5.3) — do not re-litigate:**
- **Unsloth: rejected.** ~1.2–1.5× real speedup at 1.7B, and a messy dependency
  negotiation with transformers 5.x / vLLM coexistence that burns scarce GPU
  hours on installer debugging. Use plain `peft` + `trl`.
- **QLoRA 4-bit: rejected as the default, kept as the OOM fallback.** 1.7B fp16
  is ~3.4 GB on a 16 GB card — there is no headroom problem to solve, and 4-bit
  would add quantization error for nothing. **Default is fp16 LoRA.**

**Arms produced (SPEC §3.6):**

| arm | model | prompt | decoding |
|---|---|---|---|
| `lora_ft` | base + adapter | **minimal instruction, no shots, no schema** | greedy, unconstrained |
| `lora_ft_constrained` | base + adapter | same | Outlines-constrained |

The fine-tuned prompt is deliberately *short*: the whole point of fine-tuning is
that the schema moves from the prompt into the weights. Keeping the 1.5k-token
schema in the prompt would erase the latency and cost advantage that is the
project's headline.

**Qwen3 thinking mode OFF — SPEC §3.7:** every `apply_chat_template` call passes
`enable_thinking=False`, **including when rendering training data**, so train and
inference templates match exactly. A mismatch here is silent and ruins everything
downstream.

**Data (SPEC §3.3):** `train.jsonl` / `dev.jsonl` rows have
`{doc_id, domain, text, gold, label_source, teacher_model, verified_by_human, verified_at}`.
**Split isolation (SPEC §3.4):** `eval_gold` doc_ids never appear in train or
dev. F6 must assert this **before** the first optimizer step — discovering
leakage after a 3-hour run is the worst outcome in the project.

**Output (SPEC §3.3):** `artifacts/predictions/<arm>.jsonl` with the 8 keys;
`schema_valid` from `sxl.schema.validate_prediction` only.

**Reuse from F5, do not rewrite:** `gpu/runner.py::load_model(model_id, adapter=...)`,
`::generate_batch`, `::predict_arm`, `gpu/constrained.py::build_constrained_generator`,
`teacher.extract_json`. F6 adds **no new generation code** — it calls
`sxl gpu predict --arm lora_ft --adapter <repo>`.

**Versions (SPEC §6.1–6.3), verified 2026-07-30:** `torch==2.13.0`,
`transformers==5.14.1`, `trl==1.5.1`, `peft==0.20.0`, `datasets==5.0.1`,
`accelerate==1.14.0`, `bitsandbytes==0.50.0`.

### Context deltas

Additions to `config.py`:

```python
LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.05
LORA_TARGET_MODULES = ("q_proj","k_proj","v_proj","o_proj",
                       "gate_proj","up_proj","down_proj")
TRAIN_EPOCHS          = 3
TRAIN_LR              = 2e-4      # LoRA wants ~10x full-FT LR
TRAIN_MAX_LENGTH      = 2048
TRAIN_BATCH_SIZE      = 2
TRAIN_GRAD_ACCUM      = 8         # effective batch 16
TRAIN_WARMUP_RATIO    = 0.03
TRAIN_MAX_GRAD_NORM   = 0.3       # fp16 stability on T4
ADAPTER_REPO          = "<hf_user>/qwen3-1.7b-jobpost-lora"
CHECKPOINT_DIR        = KAGGLE_TMP/"ckpt"
```

New optional dependency in the `gpu` extra: none — `trl`, `peft`, `datasets`,
`accelerate`, `bitsandbytes` are already declared in SPEC §6.1.

---

### Scope

1. **`src/sxl/gpu/train_lora.py::render_example(row, tok) -> str`**
   Render one training row to a single `"text"` string:
   ```python
   msgs = [{"role": "system",    "content": TRAIN_SYSTEM},           # short, ~30 tokens
           {"role": "user",      "content": row["text"][:MAX_INPUT_CHARS]},
           {"role": "assistant", "content": json.dumps(row["gold"], separators=(",", ":"))}]
   return tok.apply_chat_template(msgs, tokenize=False, enable_thinking=False)
   ```
   Note **no `add_generation_prompt`** here (the assistant turn is present) but
   **`enable_thinking=False` is still required**. `TRAIN_SYSTEM` lives in
   `prompts.py` and is the *same string* used by `build_ft_prompt` at inference
   — export one constant and use it in both places so they cannot drift.
   `json.dumps` uses `separators=(",", ":")` and **`sort_keys=False`** so field
   order matches `FIELD_NAMES`; the model learns a fixed key order, which is
   free accuracy.

2. **`src/sxl/prompts.py::build_ft_prompt(text) -> list[dict]`**
   The inference-time counterpart: `TRAIN_SYSTEM` + the user text, with
   `add_generation_prompt=True, enable_thinking=False`. **No shots, no schema
   dump.** A test asserts the rendered prefix is byte-identical to
   `render_example`'s prefix up to the assistant turn.

3. **`src/sxl/gpu/train_lora.py::assert_no_leakage(train, dev, gold) -> None`**
   Called as the **first statement** of `train()`. Raise if any `eval_gold`
   `doc_id` appears in train or dev. Also assert every train row passes
   `validate_prediction(row["gold"])` — training on malformed targets teaches the
   model to emit malformed JSON.

4. **`src/sxl/gpu/train_lora.py::train(...) -> dict`**
   ```python
   from peft import LoraConfig
   from trl import SFTConfig, SFTTrainer

   peft_config = LoraConfig(
       r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
       target_modules=list(LORA_TARGET_MODULES), bias="none", task_type="CAUSAL_LM")

   args = SFTConfig(
       output_dir=str(CHECKPOINT_DIR),
       num_train_epochs=TRAIN_EPOCHS,
       per_device_train_batch_size=TRAIN_BATCH_SIZE,
       gradient_accumulation_steps=TRAIN_GRAD_ACCUM,
       learning_rate=TRAIN_LR,
       lr_scheduler_type="cosine",
       warmup_ratio=TRAIN_WARMUP_RATIO,
       max_grad_norm=TRAIN_MAX_GRAD_NORM,
       fp16=True, bf16=False,                 # T4: no bf16 (SPEC §2.3)
       max_length=TRAIN_MAX_LENGTH,           # trl 1.x: `max_length`, NOT `max_seq_length`
       dataset_text_field="text",
       gradient_checkpointing=True,
       logging_steps=10, eval_strategy="steps", eval_steps=100,
       save_strategy="steps", save_steps=100, save_total_limit=2,
       load_best_model_at_end=True, metric_for_best_model="eval_loss",
       seed=SEED, report_to=[])

   trainer = SFTTrainer(model=BASE_MODEL, args=args,
                        train_dataset=ds_train, eval_dataset=ds_dev,
                        processing_class=tok,      # trl 1.x: NOT `tokenizer=`
                        peft_config=peft_config)
   ```
   Pass a **plain model id plus `peft_config`** — passing an already
   `get_peft_model`-wrapped model *and* `peft_config` is an error in trl 1.x
   (SPEC §6.3).

5. **Crash resilience.** `save_steps=100` with `save_total_limit=2` into
   `/kaggle/tmp/ckpt`, and `train()` accepts `--resume-from-checkpoint auto` which
   picks the newest checkpoint if one exists. Because `/kaggle/tmp` is **not**
   persisted, also copy the newest checkpoint to `/kaggle/working/ckpt_latest/`
   every save (a LoRA checkpoint is tens of MB, well inside the 20 GB budget) so
   a dead session can resume from the notebook's saved output. **State plainly in
   the notebook markdown that `/kaggle/tmp` disappears when the session ends.**

6. **`src/sxl/gpu/train_lora.py::publish(adapter_dir, repo_id)`**
   `model.push_to_hub(repo_id)` + `tok.push_to_hub(repo_id)` using `HF_TOKEN` from
   Kaggle Secrets. Also write `results/train_stats.json` (committed):
   ```json
   {"base_model": "Qwen/Qwen3-1.7B", "adapter_repo": "...",
    "n_train": 0, "n_dev": 0, "epochs": 3, "lora_r": 16,
    "trainable_params": 0, "trainable_pct": 0.0,
    "final_train_loss": 0.0, "best_eval_loss": 0.0,
    "train_runtime_s": 0.0, "gpu_name": "Tesla T4", "dtype": "float16",
    "peak_vram_gb": 0.0, "generated_at": "...", "git_sha": "..."}
   ```
   `peak_vram_gb` from `torch.cuda.max_memory_allocated()`. **The adapter is
   never committed to git** (SPEC §2.4) — the Hub is the artifact store.

7. **Extend `sxl gpu train` CLI** (replaces the F0 stub) with
   `--train PATH`, `--dev PATH`, `--epochs INT`, `--lr FLOAT`, `--limit INT`,
   `--max-steps INT` (smoke runs), `--resume-from-checkpoint TEXT`,
   `--load-in-4bit/--no-load-in-4bit` (the documented OOM fallback; default off).
   When `--load-in-4bit`, build a `BitsAndBytesConfig(load_in_4bit=True,
   bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
   bnb_4bit_use_double_quant=True)` and pass it as `quantization_config=` to
   `SFTTrainer` — **note `compute_dtype` is float16, not bfloat16, on a T4**, and
   transformers v5 removed the `load_in_4bit=True` shortcut on `from_pretrained`
   (SPEC §6.2).

8. **`notebooks/kaggle_train_lora.ipynb`** — thin, ~8 cells:
   1. `os.environ["HF_HOME"] = "/kaggle/tmp/hf"` before any HF import.
   2. `!pip install -q "git+https://github.com/<user>/schema-extract-lab@<SHA>"` —
      **pinned SHA, not a branch**.
   3. Version banner + `assert torch.cuda.get_device_capability(0) == (7, 5)`.
   4. Attach the `sxl-data` Kaggle Dataset; copy `train/dev/eval_gold.jsonl` into
      place; print row counts and assert the no-leakage check passes.
   5. **Smoke:** `!sxl gpu train --limit 32 --max-steps 5` — must complete and
      show a finite, non-NaN loss before proceeding.
   6. Full train; watch for NaN loss (fp16 on T4).
   7. `publish`, then `!sxl gpu predict --arm lora_ft --adapter $ADAPTER_REPO` and
      the same for `lora_ft_constrained`.
   8. Copy adapter + predictions to `/kaggle/working/`; print row counts and
      elapsed GPU time.

---

### Out of scope

- Any change to `gpu/runner.py` or `gpu/constrained.py` beyond bug fixes — F5
  owns them and wrote `load_model(..., adapter=...)` for exactly this use.
- Latency, throughput, or cost benchmarking — **F7 exclusively**. The
  `latency_ms` in F6's prediction records is amortized batch time and must never
  reach a results table.
- Scoring — F4.
- Merging the adapter into the base weights (`merge_and_unload`) — F7 may
  evaluate it as a latency variant; F6 ships the adapter unmerged.
- GGUF / ONNX export, quantized serving — SPEC §5.3, future work.
- Hyperparameter search. One configuration, one seed, reported honestly. A sweep
  on a 30 h/week budget would consume the whole quota for a fraction of a point.
  F8 states this as a limitation.
- DPO/GRPO or any preference training — SFT only.

---

### Implementation notes

- **trl 1.5.1 (SPEC §6.3):** `SFTConfig(max_length=...)` — `max_seq_length` is
  gone. `SFTTrainer(processing_class=tok)` — `tokenizer=` is gone. `loss_type`
  now defaults to `"chunked_nll"` (≈30% less peak VRAM, no action needed).
- **transformers 5.14.1 (SPEC §6.2):** `dtype=` not `torch_dtype=`; no
  `load_in_4bit=` shortcut on `from_pretrained`; `Trainer(processing_class=)` not
  `tokenizer=`; `HF_HOME` not `TRANSFORMERS_CACHE`.
- **Loss masking.** By default SFT computes loss over the whole rendered string,
  including the input document. For extraction that is mostly harmless and often
  mildly helpful, but it dilutes the gradient on the ~350 JSON tokens that
  matter. If `eval_loss` plateaus high, switch to completion-only loss (trl
  supports assistant-only masking via `assistant_only_loss=True` when the chat
  template marks assistant spans). **Verify the flag name against the installed
  trl before using it** — this area of the trl API has moved repeatedly.
- **fp16 NaN.** If loss goes NaN in the first 50 steps: confirm `bf16=False`,
  keep `max_grad_norm=0.3`, drop `TRAIN_LR` to 1e-4, and only then consider
  `--load-in-4bit`. Do **not** "fix" it by switching to bf16 — the T4 cannot.
- **Sequence length.** `TRAIN_MAX_LENGTH=2048` covers a 6,000-char document
  (~1.5k tokens) plus a ~350-token JSON target. Check the actual token-length
  p95 over `train.jsonl` in the notebook and print it; if p95 exceeds 2048,
  raise the cap to 3072 and halve the batch rather than silently truncating
  targets — **a truncated JSON target teaches the model to emit truncated JSON**,
  which would show up as a mysteriously low `schema_valid_rate`.
- **VRAM estimate:** 3.4 GB weights + LoRA optimizer state (tiny) + activations
  at batch 2 × 2048 with gradient checkpointing ≈ 8–11 GB. Comfortable in 16 GB.
- **`trainable_pct`** should land around 0.5–1.5% at r=16 on the seven target
  modules. If it prints 100%, `peft_config` did not attach and you are
  full-fine-tuning a 1.7B model on a T4 — kill the run.
- **Epoch count.** 5,000 examples × 3 epochs at effective batch 16 ≈ 940 steps.
  With `load_best_model_at_end` on `eval_loss`, over-training is bounded. Expect
  ~1.5–3 h.

---

### Test plan

Laptop only; **mock the tokenizer and never import torch.**

`tests/test_render_example.py`
- `render_example` output ends with the JSON target and contains no `<think>`.
- Its prefix is byte-identical to `build_ft_prompt`'s rendering up to the
  assistant turn (the train/inference match assertion).
- The JSON target's keys are in `FIELD_NAMES` order (not alphabetical).
- The rendered prompt does **not** contain the JSON Schema or any few-shot
  exemplar — proving the fine-tuned arm's prompt is short.
- Document text is truncated at `MAX_INPUT_CHARS`.

`tests/test_train_guards.py`
- `assert_no_leakage` raises when a gold `doc_id` is planted in `train.jsonl`,
  and names it.
- `train()` raises before constructing a trainer when a train row's `gold` fails
  `validate_prediction` (verify by monkeypatching the trainer constructor to
  raise if reached).

`tests/test_ft_prompt_len.py`
- Token-free proxy: the `lora_ft` prompt string is at least 5× shorter than the
  `base_fewshot` prompt string for the same document. This is the mechanism
  behind the cost claim and deserves a regression test.

---

### Verify

**On Kaggle** (`GPU T4 x2`, `kaggle_train_lora.ipynb`):
```bash
!python -c "import torch;assert torch.cuda.get_device_capability(0)==(7,5);print('T4 ok')"
!sxl gpu train --limit 32 --max-steps 5          # smoke: finite loss, no NaN, no OOM
!sxl gpu train                                    # full run
!python -c "import json;d=json.load(open('results/train_stats.json'));print(d['trainable_pct'], d['best_eval_loss'], d['peak_vram_gb']);assert 0.1<d['trainable_pct']<5.0"
!sxl gpu predict --arm lora_ft            --adapter $ADAPTER_REPO
!sxl gpu predict --arm lora_ft_constrained --adapter $ADAPTER_REPO
!python - <<'PY'
import json
for a in ("lora_ft","lora_ft_constrained"):
    R=[json.loads(l) for l in open(f"artifacts/predictions/{a}.jsonl")]
    assert len(R)==300, (a, len(R))
    assert not any("<think>" in r["raw_output"] for r in R), a
    print(a, "valid", sum(r["schema_valid"] for r in R), "/300",
          "median completion tokens", sorted(r["completion_tokens"] for r in R)[150])
PY
!cp -r artifacts/predictions/*.jsonl results/train_stats.json /kaggle/working/
```

**Back on the laptop:**
```bash
sxl metrics score --arm lora_ft
sxl metrics score --arm lora_ft_constrained
sxl metrics compare
```

Expected: `trainable_pct` between 0.1 and 5.0; a finite `best_eval_loss`;
300 rows per arm with no `<think>`; and `sxl metrics compare` showing `lora_ft`
above `base_fewshot` on `macro_f1`. **If it is not above, that is the result** —
report it, investigate loss masking and epoch count, and do not quietly weaken
the baseline (SPEC §1.1, F5 out-of-scope).

---

### Acceptance criteria

- [ ] `assert_no_leakage` runs as the first statement of `train()` and raises on
      any `eval_gold` doc_id found in train or dev.
- [ ] A 5-step smoke run completes on a T4 with a finite (non-NaN) loss before any
      full run is started.
- [ ] `results/train_stats.json` reports `trainable_pct` between 0.1% and 5%,
      `dtype == "float16"`, `gpu_name` containing `T4`, and a finite
      `best_eval_loss`.
- [ ] The adapter is published to `ADAPTER_REPO` on the HF Hub and loads via
      `PeftModel.from_pretrained` in a fresh session.
- [ ] No `*.safetensors`, `*.bin`, or checkpoint directory is committed to git.
- [ ] `artifacts/predictions/lora_ft.jsonl` and `lora_ft_constrained.jsonl` each
      have exactly 300 rows matching the `eval_gold` doc_ids, with the 8 SPEC §3.3
      keys in order and zero `<think>` occurrences.
- [ ] `lora_ft_constrained` achieves `schema_valid_rate ≥ 0.99`.
- [ ] The `lora_ft` prompt contains no JSON Schema and no few-shot exemplars, and
      is ≥5× shorter than the `base_fewshot` prompt (regression test).
- [ ] Training and inference render the identical chat prefix with
      `enable_thinking=False` — asserted by a byte-comparison test.
- [ ] No `unsloth`, `flash-attn`, or `vllm` appears in the notebook install cell.
- [ ] Total Kaggle GPU time for this feature is recorded and is under 7 hours.
