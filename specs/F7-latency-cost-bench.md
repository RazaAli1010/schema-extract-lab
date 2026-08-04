## F7 — Latency, throughput, and cost benchmark on a T4

**Goal:** Produce `results/bench/<arm>.json` for every model arm: measured
single-stream p50/p95 milliseconds per document, throughput across a batch-size
sweep, and a derived dollars-per-1k-documents figure. **This is the feature that
makes the project land with hiring managers**, and it is also the feature most
likely to produce a flattering lie if written carelessly.

**Depends on:** F0 (config, io), F5 (`gpu/runner.py`, base arms), F6 (adapter)

---

### ⚠️ This feature runs on Kaggle, not the laptop

**SPEC §2.1 — the dev laptop has 8 GB RAM, ~5 GB free disk, and no GPU.** A
latency benchmark on a machine with no GPU is meaningless; do not even mock one.
The laptop writes `src/sxl/gpu/bench.py`, the cost arithmetic (which *is* unit
testable without torch), and `notebooks/kaggle_bench.ipynb`.

**Kaggle setup (SPEC §2.2):** Accelerator **`GPU T4 x2`**, **use `cuda:0` only**.
Every number in this feature is a single-T4 number and must be labeled as such.
Budget **~2 h** of the ~30 h/week quota (SPEC §6.6).

**Critical:** Kaggle's `GPU T4 x2` and `GPU P100` are different hardware. A
session that silently landed on a P100 would produce numbers that are not
comparable across arms. **Assert `torch.cuda.get_device_capability(0) == (7, 5)`
and record `torch.cuda.get_device_name(0)` in every output file.** All arms must
be benchmarked in the **same session** where possible, and any arm benchmarked in
a different session records that session's GPU name for cross-checking.

---

### Context digest

**SPEC §1.1 — headline claims are hypotheses, not targets.** Reproduced here in
full because this feature is where the temptation lives:

> The project pitch contains illustrative numbers ("within 3 F1", "45ms on a T4",
> "~$0.002/1k docs"). **These are placeholders to be filled in by measurement,
> not goals to hit.** No session may tune a benchmark, pick a batch size, or
> select a subset to make a number match the pitch.
>
> **45ms/doc.** A T4 runs a 1.7B model in fp16 at roughly 40–70 tok/s
> single-stream. A ~150-token JSON output is therefore ~2–4 **seconds** at
> `batch_size=1`. Sub-100ms per document is only reachable as an *amortized*
> figure at large batch. F7 must report single-stream p50/p95 **and** amortized
> per-doc cost at the best batch size, clearly labeled as different things.

**If the measured single-stream p50 is 2,800 ms, the table says 2,800 ms.** The
amortized figure goes in a separate, differently-named column. Conflating them is
the failure this spec exists to prevent.

**Output shape (SPEC §3.3)** — `results/bench/<arm>.json`:
```json
{"arm": "lora_ft", "gpu_name": "Tesla T4", "dtype": "float16",
 "batch_size": 1, "n_docs": 200, "warmup": 10,
 "p50_ms": 0.0, "p95_ms": 0.0, "mean_ms": 0.0,
 "throughput_docs_per_s": 0.0, "gpu_hourly_usd": 0.35,
 "cost_per_1k_docs_usd": 0.0, "generated_at": "..."}
```

**T4 limits (SPEC §2.3):** no bf16 → `float16`; no FlashAttention-2 → `sdpa`;
**vLLM banned** (SPEC §5.3 — Turing support degrading). SPEC §5.3 also requires
the write-up to say plainly that *on an A10/L4/A100, vLLM would be the correct
serving path and would improve these throughput numbers substantially* — F8
quotes that; F7 records enough detail to make it credible.

**Cost constant (SPEC §6.5 / F0 `config.py`):** `T4_HOURLY_USD = 0.35`
(~GCP on-demand T4). This is an **assumption, not a measurement**, and must be
emitted into every bench file so a reader can recompute at their own rate.

**Reuse from F5:** `gpu/runner.py::load_model`, `::generate_batch`,
`prompts.build_student_prompt` / `build_ft_prompt`. F7 adds no new generation
path — benchmarking a *different* code path than the one that produced the
accuracy numbers would make the two tables incomparable.

### Context deltas

Additions to `config.py`:

```python
BENCH_N_DOCS      = 200
BENCH_WARMUP      = 10
BENCH_BATCH_SIZES = (1, 2, 4, 8, 16, 32)
BENCH_REPEATS     = 3            # full repeats of the batch-1 sweep, for variance
```

**Additions to the `results/bench/<arm>.json` contract in SPEC §3.3.** Apply to
SPEC §3.3 before implementing:

```json
"mean_completion_tokens": 0.0,   // latency is mostly a function of output length
"p50_spread_pct": 0.0,           // spread of p50 across BENCH_REPEATS
"cache_implementation": "static", // or "dynamic"; must be identical across arms
"index_build_s": 0.0,            // constrained arms only; one-time, excluded from per-doc latency
"measurement": "local_gpu",      // "api_wall_clock" for the teacher arm
"torch_version": "2.13.0", "transformers_version": "5.14.1"
```

Addition to SPEC §3.3 — a second output file per arm,
`results/bench/<arm>_sweep.json`:
```json
{"arm": "lora_ft", "gpu_name": "Tesla T4",
 "sweep": [{"batch_size": 1, "throughput_docs_per_s": 0.0,
            "amortized_ms_per_doc": 0.0, "peak_vram_gb": 0.0,
            "mean_completion_tokens": 0.0, "oom": false}],
 "best_batch_size": 0, "generated_at": "..."}
```

---

### Scope

1. **`src/sxl/gpu/bench.py::measure_single_stream(model, tok, prompts, ...) -> dict`**
   The honest headline number.
   - `BENCH_WARMUP = 10` untimed iterations first (CUDA context, kernel autotune,
     and the first `sdpa` call are all one-time costs that would otherwise land
     in the first measurement)
   - then `BENCH_N_DOCS = 200` documents, **one at a time**, each timed
     end-to-end: `apply_chat_template` → tokenize → `generate` → decode →
     `extract_json` → `validate_prediction`. **Include the tokenization and
     parsing**; a user of this model pays for them, so the benchmark does too
   - `torch.cuda.synchronize()` immediately before starting and immediately after
     finishing **each** iteration — without this you are timing kernel *launches*,
     not kernel *execution*, and will report an impossibly fast number
   - use `time.perf_counter()`, not `time.time()`
   - report `p50`, `p95`, `mean`, plus `min`/`max` and `stdev`, computed with
     `statistics` (no numpy — SPEC §6.1)
   - also record `mean_completion_tokens`, because **latency here is almost
     entirely a function of output length**, and an arm that emits 400 tokens is
     slow for a reason a reader deserves to see

2. **`src/sxl/gpu/bench.py::measure_sweep(model, tok, prompts) -> dict`**
   For each `b` in `BENCH_BATCH_SIZES`: warm up, run `ceil(BENCH_N_DOCS/b)`
   batches, record wall time, `throughput_docs_per_s = n/elapsed`,
   `amortized_ms_per_doc = 1000/throughput`, and
   `peak_vram_gb` from `torch.cuda.max_memory_allocated()` (reset with
   `torch.cuda.reset_peak_memory_stats()` before each batch size).
   **Catch `torch.cuda.OutOfMemoryError`**, record `{"oom": true}` for that batch
   size, `torch.cuda.empty_cache()`, and continue — do not crash the sweep and
   lose the smaller batch sizes that already succeeded.
   `best_batch_size` = the largest non-OOM batch size, reported as such.

3. **`src/sxl/gpu/bench.py::cost_per_1k(throughput_docs_per_s, hourly_usd) -> float`**
   ```python
   return (1000.0 / throughput_docs_per_s) / 3600.0 * hourly_usd
   ```
   Pure arithmetic, **unit-testable on the laptop with no GPU** — and it must be,
   because this is the number that goes on a résumé.

4. **`src/sxl/gpu/bench.py::run(arm, adapter, ...) -> dict`**
   Orchestrate: load model (fp16, sdpa, `cuda:0`), build the arm's prompts using
   **the same prompt builder that produced that arm's predictions in F5/F6**
   (`build_student_prompt` for `base_*`, `build_ft_prompt` for `lora_ft*`), run
   `measure_single_stream` `BENCH_REPEATS` times and report the **median of the
   three p50s** plus the spread, run `measure_sweep` once, write both output
   files.
   **Assertion:** `torch.cuda.get_device_capability(0) == (7, 5)` — raise
   otherwise. Record `gpu_name`, `dtype`, `torch.__version__`,
   `transformers.__version__`.

5. **Benchmark the constrained arms too.** `base_fewshot_constrained` and
   `lora_ft_constrained` go through Outlines, whose logit masking has real
   per-token overhead. **Build the Outlines index during warmup, outside the
   timed region** (it is a one-time startup cost, not a per-document cost) — but
   **record the index build time separately** as `index_build_s` in the output,
   because it is a real deployment cost and hiding it would be dishonest in the
   opposite direction.

6. **The teacher arm's latency and cost are measured differently and must be
   labeled so.** Write `results/bench/teacher.json` with:
   - `p50_ms` / `p95_ms` from wall-clock timing of ~50 **non-batch** synchronous
     Anthropic API calls (run **on the laptop**, not Kaggle — it is a network
     measurement and needs no GPU)
   - `cost_per_1k_docs_usd` computed from actual token usage at **standard**
     (non-batch) rates, since a latency-sensitive deployment cannot use the 24-hour
     Batch API
   - a `"measurement": "api_wall_clock"` key distinguishing it from the GPU arms'
     `"measurement": "local_gpu"`. Comparing an API round-trip to a local
     `generate()` is apples-to-oranges and the file must say so.

7. **`sxl gpu bench` CLI** (replaces the F0 stub), imports inside the body:
   `--arm TEXT`, `--adapter TEXT`, `--n-docs INT`, `--batch-sizes TEXT`,
   `--hourly-usd FLOAT`, `--repeats INT`, `--out-dir PATH`.
   Plus `sxl bench teacher` (laptop, no GPU import) for Scope 6.

8. **`notebooks/kaggle_bench.ipynb`** — thin, ~6 cells:
   1. `os.environ["HF_HOME"] = "/kaggle/tmp/hf"` before any HF import.
   2. `!pip install -q "git+https://github.com/<user>/schema-extract-lab@<SHA>"` —
      pinned SHA.
   3. Version banner + `assert torch.cuda.get_device_capability(0) == (7, 5)` +
      `nvidia-smi`.
   4. **Benchmark all four GPU arms in one session** so they share hardware,
      thermal state, and driver version.
   5. Print a comparison table to the notebook output.
   6. Copy `results/bench/*.json` to `/kaggle/working/`.

---

### Out of scope

- Any accuracy metric — F4 owns `macro_f1` and `schema_valid_rate`. F7 writes
  **no** accuracy numbers into `results/bench/`, and F4 writes **no** latency
  numbers into `results/metrics/`. Two features writing the same quantity is how
  a headline table ends up contradicting itself (SPEC §3.3, F4 out-of-scope).
- vLLM, TensorRT-LLM, ONNX Runtime, or GGUF serving — SPEC §5.3. F8 names them as
  the obvious production next step; F7 does not implement them.
- Multi-GPU / tensor-parallel measurement — single T4 only, by design.
- Quantized (int8/int4) inference latency — a legitimate follow-up, deferred. If
  the fp16 numbers disappoint, that is the honest v1 result.
- The final markdown table — F8.
- Speculative decoding, KV-cache reuse across documents, or prompt caching —
  none apply to independent single-document extraction.

---

### Implementation notes

- **`torch.cuda.synchronize()` is the whole ballgame.** CUDA kernel launches are
  asynchronous. Timing `generate()` without a synchronize measures how fast Python
  can enqueue work and will produce a number roughly 100× too fast — which would
  *look* like it validated the 45 ms hypothesis. Synchronize before `t0` and
  before `t1`, every iteration.
- **`use_cache=True`** (the default) — confirm it is not disabled anywhere.
  Generation without a KV cache is quadratic and would make every number
  meaningless.
- **Static KV cache.** transformers supports `cache_implementation="static"`,
  which can meaningfully help small models by enabling CUDA-graph-friendly
  shapes. Try it, measure both, and **report whichever you use** in the output
  file as a `cache_implementation` key. Do not silently use the faster one for
  one arm and not another.
- **Batch-1 latency ≈ (completion tokens) / (tokens per second).** Before
  trusting any measurement, sanity-check it: 150 tokens at a plausible 50 tok/s
  is ~3,000 ms. **A measured p50 of 45 ms for a 150-token generation is not a
  triumph — it is a bug**, almost certainly a missing synchronize or a run that
  generated 2 tokens and hit EOS. Assert
  `p50_ms > mean_completion_tokens * 2` (i.e. faster than 500 tok/s is
  implausible on a T4) and raise if violated.
- **`mean_completion_tokens` differences between arms are a real finding.** The
  fine-tuned model should emit tighter JSON (no `<think>`, no preamble, learned
  key order) than the few-shot baseline. If `lora_ft` is faster, this is *why*,
  and F8 should attribute it correctly rather than to the LoRA weights being
  intrinsically quicker — they are not; the adapter adds a small amount of
  compute per token.
- **Adapter overhead.** An unmerged LoRA adds two small matmuls per target module
  per token — typically low single-digit percent. If you also measure a
  `merge_and_unload()` variant, record it as a distinct arm key
  `lora_ft_merged` rather than overwriting `lora_ft`.
- **Variance.** Kaggle GPUs are shared infrastructure and thermally variable.
  `BENCH_REPEATS = 3` and reporting the spread across repeats is the minimum
  honest treatment. If the three p50s differ by more than ~15%, say so in the
  output and in F8 rather than quoting the best one.
- **`T4_HOURLY_USD = 0.35` is an assumption.** Emit it in every file. A reader on
  AWS `g4dn.xlarge` (~$0.53/h on-demand) should be able to rescale in one
  multiplication. Do not present the cost figure as if it were measured.

---

### Test plan

Laptop only, no torch: test the arithmetic and the reporting logic, which is
where the errors that matter actually live.

`tests/test_bench_cost.py`
- `cost_per_1k(10.0, 0.35)` → `(1000/10)/3600*0.35 == 0.009722...`; assert to 6
  decimal places.
- `cost_per_1k(1.0, 0.35)` → `0.09722...`.
- Doubling throughput exactly halves cost.
- `cost_per_1k(0.0, ...)` raises rather than returning `inf`.

`tests/test_bench_stats.py`
- p50/p95 on a hand-written list of 200 latencies match values computed by hand
  (use a list where the answer is obvious, e.g. `list(range(1, 201))`).
- The implausible-speed guard raises for `p50_ms=45, mean_completion_tokens=150`
  and passes for `p50_ms=3000, mean_completion_tokens=150`.

`tests/test_bench_schema.py`
- The dict emitted by the reporting function has exactly the SPEC §3.3
  `results/bench/<arm>.json` keys.
- `gpu_hourly_usd` is always present and equals the value used in the
  `cost_per_1k_docs_usd` computation (assert consistency, so nobody can hand-edit
  one without the other).
- A sweep entry with `oom: true` is preserved in the output and excluded from
  `best_batch_size` selection.

---

### Verify

**On the laptop:**
```bash
pytest -q tests/test_bench_cost.py tests/test_bench_stats.py tests/test_bench_schema.py
sxl bench teacher --n 50          # network-only, no GPU
```

**On Kaggle** (`GPU T4 x2`, `kaggle_bench.ipynb`, all arms in one session):
```bash
!python -c "import torch;assert torch.cuda.get_device_capability(0)==(7,5);print(torch.cuda.get_device_name(0))"
!sxl gpu bench --arm base_fewshot
!sxl gpu bench --arm base_fewshot_constrained
!sxl gpu bench --arm lora_ft            --adapter $ADAPTER_REPO
!sxl gpu bench --arm lora_ft_constrained --adapter $ADAPTER_REPO
!python - <<'PY'
import json, glob
for f in sorted(glob.glob("results/bench/*.json")):
    d=json.load(open(f))
    if "sweep" in d:
        print(f, "best_batch", d["best_batch_size"]); continue
    print(f"{d['arm']:26} p50={d['p50_ms']:8.1f}ms p95={d['p95_ms']:8.1f}ms "
          f"tok={d.get('mean_completion_tokens',0):6.1f} "
          f"thru={d['throughput_docs_per_s']:6.2f}/s ${d['cost_per_1k_docs_usd']:.5f}/1k "
          f"[{d['gpu_name']}]")
    assert d["p50_ms"] > d.get("mean_completion_tokens",0)*2, ("implausible", f)
    assert "T4" in d["gpu_name"], f
PY
!cp results/bench/*.json /kaggle/working/
```

Expected: every arm's `gpu_name` contains `T4`; every `p50_ms` passes the
plausibility guard; the sweep shows throughput rising with batch size until OOM
or saturation. **Expect single-stream p50 in the 1,500–4,000 ms range, not 45 ms**
— and expect amortized per-document time at `best_batch_size` to be roughly an
order of magnitude lower. Both go in the table, separately labeled.

---

### Acceptance criteria

- [ ] `results/bench/<arm>.json` exists for all four GPU arms plus
      `teacher.json`, each with the exact SPEC §3.3 keys.
- [ ] Every GPU bench file records `gpu_name` containing `T4`, `dtype ==
      "float16"`, and was produced in a session where
      `torch.cuda.get_device_capability(0) == (7, 5)` was asserted.
- [ ] `torch.cuda.synchronize()` brackets every timed region — verified by code
      review and by the plausibility guard passing on real numbers.
- [ ] The implausible-speed guard (`p50_ms > 2 × mean_completion_tokens`) is
      implemented, tested, and passes on the real measurements.
- [ ] Single-stream p50/p95 and amortized-at-best-batch throughput are reported as
      **separate, distinctly named** quantities; no file conflates them.
- [ ] `cost_per_1k_docs_usd` is derived by the tested `cost_per_1k` function from
      the `throughput_docs_per_s` and `gpu_hourly_usd` in the same file — a test
      asserts internal consistency.
- [ ] `results/bench/teacher.json` carries `"measurement": "api_wall_clock"` and
      uses **standard**, not batch, API rates.
- [ ] `results/bench/<arm>_sweep.json` records `peak_vram_gb` per batch size and
      preserves OOM entries instead of crashing.
- [ ] `index_build_s` is recorded for the two constrained arms and is excluded
      from per-document latency.
- [ ] `BENCH_REPEATS = 3` repeats were run and the spread across them is reported.
- [ ] No accuracy metric appears anywhere in `results/bench/`.
- [ ] Total Kaggle GPU time for this feature is recorded and is under 2.5 hours.
