# Bench run log (F7 provenance)

The bench artifacts in this directory cannot record everything a reader needs.
`config.git_sha()` returns `"unknown"` on Kaggle — the package is a pip install
with no `.git` — and GPU wall-clock time is a property of the *session*, not of
any one measurement. F7's acceptance criteria require both. This file is where
they live.

## Session 1 — 2026-08-04

| | |
|---|---|
| **GPU time** | **59 min (0.99 h)** — against F7's 2.5 h budget |
| accelerator | Kaggle `GPU T4 x2`, `cuda:0` only |
| device | `Tesla T4`, compute capability `(7, 5)` — asserted at run time |
| torch / transformers | `2.13.0+cu130` / `5.14.1` |
| pinned `COMMIT_SHA` | **`da31903`** *(to confirm against cell 2 of the run — see below)* |
| arms measured | all four GPU arms, one session |
| produced | `{base_fewshot,base_fewshot_constrained,lora_ft,lora_ft_constrained}{,_sweep}.json` |

All four GPU arms ran in this single session, so they share hardware, thermal
state and driver version. That is what makes them comparable with each other;
an arm re-measured later must record its own session here.

**On the pinned SHA.** `teacher.json` carries `git_sha: da31903` because it ran
on the laptop, and the notebook changes that produced this run's parameters
(`--n-docs 10`, `--warmup 5`, batch sweep capped at 8, whole-batch sweep
truncation) all landed in `da31903`. The GPU files themselves cannot confirm it.
If cell 2 held a different SHA, correct this row — provenance that is inferred
rather than read is worth exactly what it costs to check.

### Teacher probe — 2026-08-04, laptop

| | |
|---|---|
| calls | **30**, not the 50 of `config.BENCH_TEACHER_N` |
| model | `gpt-4o-mini` |
| spend | `$0.01441` at standard (non-batch) rates |
| failures | 0 |
| `git_sha` | `da31903` (recorded in the artifact — this ran on the laptop) |

Cut to 30 to limit API spend. `teacher.json` records `n_docs: 30` and
`n_calls_ok: 30`, so the figure is self-documenting. **It has no repeats**, so
its `p50_spread_pct` of `0.0` means none was measured — not that it was stable.

## Sample sizes

The GPU arms are **10 documents × 3 repeats**; `config.BENCH_N_DOCS` is 100.
Reduced to fit remaining quota, uniformly across arms so that no arm is
flattered relative to another. Consequences a reader must not be left to
discover:

- **`p95_ms` is the maximum** at n=10 — nearest rank puts p95 on the 10th of 10.
  `base_fewshot`'s is a 35.8 s outlier against an 8.9 s median.
- **`p50_ms` is a 10-sample median.** `p50_spread_pct` across the three repeats
  is what says whether to trust it; all four arms came in under 6%.
- **The sweep stops at batch 8** because 10 documents cannot fill a wider batch,
  not because throughput saturated there.

## Known reading traps

- **`best_batch_size` is the largest non-OOM size, not the fastest.** On
  `base_fewshot` throughput peaks at batch 4 (0.115 docs/s) and falls at 8
  (0.086) — padding waste and straggler dominance, since a batch decodes until
  its longest sequence finishes. Its `best_cost_per_1k_docs_usd` of $1.13 is
  therefore worse than the $0.84 batch 4 actually achieved. Read the full
  `sweep` array.
- **`teacher.json` is not comparable with the GPU arms** — `api_wall_clock`
  against `local_gpu`, and priced at standard rather than Batch rates.
