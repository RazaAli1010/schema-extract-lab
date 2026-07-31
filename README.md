# schema-extract-lab

Fine-tune a ~1.7B open model to extract a strict JSON schema from unstructured job
postings, and measure how close it lands to a hosted teacher model (`gpt-4o-mini`).

See [SPEC.md](SPEC.md) for the full contract and [specs/](specs/) for per-feature specs.

> This README is a placeholder. **F8** regenerates it with the measured results table
> and the honest caveats (SPEC §7).

## Setup (laptop — no GPU, no torch)

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
pytest -q
sxl --help
```

Do **not** install the `gpu` extra locally (SPEC §2.1) — it is for Kaggle only.

## Status

| ID | feature | state |
|---|---|---|
| F0 | Repo scaffold, schema, config, CLI skeleton | done |
| F1 | Corpus acquisition & normalization | not started |
| F2 | Teacher labeling pipeline | not started |
| F3 | Eval-set sampling + human verification | not started |
| F4 | Metrics library + scoring CLI | not started |
| F5 | Prompted baseline arms (Kaggle) | not started |
| F6 | LoRA fine-tune + inference (Kaggle) | not started |
| F7 | Latency / throughput / cost benchmark (Kaggle) | not started |
| F8 | Results aggregation, headline table, README | not started |
