## F0 — Repo scaffold, schema contract, config, CLI skeleton

**Goal:** A `pip install -e .` on an 8 GB / no-GPU laptop yields a working `sxl`
CLI, an importable `JobPosting` schema that is the single source of truth for all
16 fields, and a passing test suite — with `torch` nowhere in the dependency tree.

**Depends on:** none

---

### Context digest

Everything here is quoted from `SPEC.md` and must be reproduced exactly.

**Hardware (SPEC §2.1):** development laptop has **8 GB RAM, ~5 GB free disk, no
GPU**. This feature must not add any dependency that pulls `torch`, `nvidia-*`,
`transformers`, or CUDA wheels. Total install footprint target: **< 100 MB**.

**The 16 fields (SPEC §3.2)** — exact names, order, types, nullability:

| # | field | type | null? |
|---|---|---|---|
| 1 | `title` | `str` | yes |
| 2 | `company` | `str` | yes |
| 3 | `employment_type` | `EmploymentType` | no |
| 4 | `seniority` | `Seniority` | no |
| 5 | `remote_mode` | `RemoteMode` | no |
| 6 | `location_city` | `str` | yes |
| 7 | `location_region` | `str` | yes |
| 8 | `location_country` | `str` | yes |
| 9 | `salary_min` | `float` | yes |
| 10 | `salary_max` | `float` | yes |
| 11 | `salary_currency` | `str` | yes |
| 12 | `salary_period` | `SalaryPeriod` | no |
| 13 | `years_experience_min` | `int` | yes |
| 14 | `education_level` | `EducationLevel` | no |
| 15 | `required_skills` | `list[str]` | no (may be `[]`) |
| 16 | `posting_date` | `str` (`YYYY-MM-DD`) | yes |

**Enum members (SPEC §3.2), exact lowercase strings:**
```
EmploymentType : full_time | part_time | contract | internship | temporary | unknown
Seniority      : intern | junior | mid | senior | lead | principal | unknown
RemoteMode     : onsite | hybrid | remote | unknown
SalaryPeriod   : hourly | daily | weekly | monthly | yearly | unknown
EducationLevel : none | high_school | associate | bachelor | master | doctorate | unknown
```

**Null/unknown rule (SPEC §3.2):** enum fields are never `null`; absence is
`"unknown"`. `EducationLevel.none` ("posting says no degree required") ≠
`unknown` ("posting is silent"). Non-enum fields use `null`. `required_skills`
uses `[]`.

**Normalization (SPEC §3.5):** strings → strip, collapse internal whitespace,
lowercase. `location_country` and `salary_currency` additionally uppercased after
strip. Numbers → `float`, `==`. `posting_date` → literal string compare.
`required_skills` → normalize each element, compare as a set. `null`,
`"unknown"`, and `[]` are all the **absent** marker.

**Split hashing (SPEC §3.4):**
```python
bucket = int(hashlib.sha256(doc_id.encode()).hexdigest()[:8], 16) % 100
# 0-4 eval_pool | 5-9 dev | 10-99 train
```

**Repo layout (SPEC §4)** and **CLI command names (SPEC §4)** — reproduce
verbatim. `sxl gpu *` subcommands must import `sxl.gpu.*` **lazily inside the
command body**.

**Pinned base deps (SPEC §6.1):** `pydantic==2.13.4`, `jsonschema==4.26.0`,
`anthropic==0.120.2`, `typer==0.27.0`, `python-dotenv>=1.0`, `datasets==5.0.1`;
dev: `pytest==9.1.1`, `ruff==0.16.0`; gpu extra declared but never installed here.

`datasets` sits in **base**, not `gpu`, because F1 streams the corpus on the
laptop. It is the one base dependency that could plausibly drag in torch — F0's
verify block is where that gets caught (see Verify).

**Principles (SPEC §5):** `SEED = 1337`; every artifact carries `git_sha` and
`generated_at`; all caps live in `config.py`.

### Context deltas

none — F0 implements the shared contract, it does not extend it.

---

### Scope

1. **`pyproject.toml`** — hatchling or setuptools backend, `requires-python = ">=3.11"`,
   `src/` layout, the three dependency groups from SPEC §6.1 verbatim, and a
   console script `sxl = "sxl.cli:app"`. Add a `[tool.ruff]` section with
   `line-length = 100`.

2. **`src/sxl/config.py`** — no logic, only constants and resolved paths:
   ```python
   from pathlib import Path
   ROOT = Path(__file__).resolve().parents[2]
   DATA, ARTIFACTS, RESULTS = ROOT/"data", ROOT/"artifacts", ROOT/"results"
   DOCS_PATH       = DATA/"raw"/"docs.jsonl"
   TRAIN_PATH      = DATA/"labeled"/"train.jsonl"
   DEV_PATH        = DATA/"labeled"/"dev.jsonl"
   EVAL_POOL_PATH  = DATA/"labeled"/"eval_pool.jsonl"
   EVAL_GOLD_PATH  = DATA/"gold"/"eval_gold.jsonl"
   PREDICTIONS_DIR = ARTIFACTS/"predictions"
   METRICS_DIR     = RESULTS/"metrics"
   BENCH_DIR       = RESULTS/"bench"

   SEED = 1337
   DOMAIN = "job_posting"
   ARMS = ("base_fewshot", "base_fewshot_constrained",
           "lora_ft", "lora_ft_constrained", "teacher")

   N_TRAIN_TARGET, N_DEV_TARGET, N_EVAL_GOLD = 5000, 300, 300
   MAX_INPUT_CHARS, MAX_NEW_TOKENS, TEMPERATURE, N_FEWSHOT = 6000, 512, 0.0, 3

   TEACHER_MODEL = "claude-sonnet-5"
   MAX_TEACHER_SPEND_USD = 25.0
   TEACHER_MAX_RETRIES = 3

   BASE_MODEL = "Qwen/Qwen3-1.7B"
   T4_HOURLY_USD = 0.35   # ~GCP on-demand T4; source recorded in results/bench/*
   ```
   Plus `def ensure_dirs() -> None` creating every directory above, and
   `def git_sha() -> str` returning `subprocess`-read short SHA or `"unknown"`
   outside a repo (must not raise).

3. **`src/sxl/schema.py`** — the contract module.
   ```python
   class EmploymentType(str, Enum): ...   # 5 enums, members exactly as above
   class JobPosting(BaseModel):
       model_config = ConfigDict(extra="forbid")
       # 16 fields in the table order; enums have no default,
       # nullable fields are `X | None` with no default either —
       # every key must be explicitly present.
   ```
   Also export:
   - `FIELD_NAMES: tuple[str, ...]` — the 16 names, in table order. **Everything
     downstream iterates this, never `JobPosting.model_fields`**, so ordering is
     stable.
   - `ENUM_FIELDS: frozenset[str]` = `{employment_type, seniority, remote_mode, salary_period, education_level}`
   - `SET_FIELDS: frozenset[str]` = `{required_skills}`
   - `NUMERIC_FIELDS: frozenset[str]` = `{salary_min, salary_max, years_experience_min}`
   - `JSON_SCHEMA: dict` — `JobPosting.model_json_schema()` post-processed so that
     `additionalProperties` is `False` and `required` lists all 16 keys.
     **Verify this after generation** — pydantic omits keys from `required` when
     they have defaults, which is why no field gets a default.
   - `validate_prediction(obj: dict) -> JobPosting | None` — returns the model on
     success, `None` on any `ValidationError`. Never raises. This is the single
     function that defines `schema_valid` for the whole project.
   - `empty_posting() -> JobPosting` — all nullables `None`, all enums
     `"unknown"`, `required_skills=[]`. Used as a fixture and as the
     degenerate-baseline reference.

4. **`src/sxl/normalize.py`** — pure, no I/O:
   ```python
   def norm_str(v: str | None) -> str | None          # strip, collapse ws, lower
   def norm_country(v: str | None) -> str | None      # norm_str then upper
   def norm_currency(v: str | None) -> str | None     # norm_str then upper
   def norm_skills(v: list[str] | None) -> frozenset[str]
   def norm_field(field: str, value) -> object        # dispatch on FIELD_NAMES
   def is_absent(field: str, value) -> bool           # None | "unknown" | frozenset()
   ```
   `norm_field` must handle every one of the 16 names and raise `KeyError` on an
   unknown name — silent pass-through would let a typo'd field score as perfect.

5. **`src/sxl/splits.py`**:
   ```python
   def split_for(doc_id: str) -> Literal["train", "dev", "eval_pool"]
   def bucket_for(doc_id: str) -> int   # 0..99, exposed for tests
   ```
   Exactly the SPEC §3.4 formula. No randomness, no seed, no dependence on
   iteration order or corpus size.

6. **`src/sxl/io.py`** — JSONL helpers used by every later feature so nobody
   hand-rolls them:
   ```python
   def read_jsonl(path: Path) -> Iterator[dict]
   def write_jsonl(path: Path, rows: Iterable[dict]) -> int   # atomic: tmp + rename
   def append_jsonl(path: Path, row: dict) -> None            # for resumable F2
   def write_json(path: Path, obj: dict) -> None              # sorted keys, indent=2, trailing \n
   ```
   `write_jsonl` writes to `path.with_suffix(".tmp")` then `os.replace` so a
   killed run never leaves a half-written split.

7. **`src/sxl/cli.py`** — typer app with the SPEC §4 command tree registered as
   stubs. Every non-F0 command raises
   `typer.Exit(code=2)` with a message naming the feature that implements it
   (e.g. `"sxl teacher label is implemented in F2"`). The `gpu` sub-app's callbacks
   contain their `from sxl.gpu import ...` **inside the function body**.
   `sxl --help` and `sxl schema dump` must work with zero optional deps installed.
   Implement one real command now: `sxl schema dump [--out PATH]` writing
   `JSON_SCHEMA` via `write_json`.

8. **Repo hygiene** — `.gitignore` covering `data/`, `artifacts/`, `.env`,
   `*.safetensors`, `*.bin`, `*.pt`, `__pycache__`, `.venv`; `.gitkeep` in each
   ignored dir; `.env.example` with `ANTHROPIC_API_KEY=` and `HF_TOKEN=`;
   a `Makefile` with `install`, `test`, `lint`, `fmt` targets; empty
   `src/sxl/gpu/__init__.py` carrying only a module docstring warning that
   nothing here may be imported at package import time.

---

### Out of scope

- Any corpus download or network call — F1.
- Anything that talks to the Anthropic API — F2.
- Metric arithmetic — F4 (this feature only defines the field taxonomy it needs).
- Any file under `src/sxl/gpu/` beyond the empty `__init__.py` — F5/F6/F7.
- Prompt text — F5 owns `prompts.py`; F0 only puts the four constants in `config.py`.

---

### Implementation notes

- **pydantic 2.13.4.** Use `ConfigDict(extra="forbid")`, not the deprecated inner
  `class Config`. `model_json_schema()` on a `str, Enum` subclass emits a
  `$defs` entry with `enum: [...]` — that is correct and Outlines (F5) consumes
  it directly, so do not flatten `$defs` away.
- **The `required` trap.** pydantic marks a field optional in JSON Schema if it
  has *any* default, including `= None`. Declare nullable fields as
  `title: str | None` with **no default**. Assert `len(JSON_SCHEMA["required"]) == 16`
  at import time — this single assertion prevents a whole class of silent
  scoring bugs downstream.
- **typer 0.27.0.** Sub-apps via `app.add_typer(gpu_app, name="gpu")`. Typer
  imports click; neither pulls torch.
- **`git_sha()` must not raise** — it runs inside Kaggle notebooks where the repo
  is a pip install with no `.git`. Catch `FileNotFoundError` and
  `CalledProcessError`, return `"unknown"`.
- **Do not install the `gpu` extra locally.** `pip install -e ".[dev]"` only.
  Installing `.[gpu]` on the dev laptop will exhaust the 5 GB of free disk.

---

### Test plan

`tests/test_schema.py`
- `JSON_SCHEMA["required"]` has exactly the 16 `FIELD_NAMES`, and
  `additionalProperties is False`.
- `FIELD_NAMES` equals the SPEC §3.2 table order (hard-coded literal in the test).
- `validate_prediction` returns `None` for: a missing key; an extra key; an enum
  value not in its member list; `employment_type: null`; a string in `salary_min`.
- `validate_prediction` accepts `empty_posting().model_dump()`.
- Every enum's member set matches a hard-coded literal in the test.

`tests/test_normalize.py`
- `norm_str("  Senior   DATA Engineer ") == "senior data engineer"`.
- `norm_country(" us ") == "US"`; `norm_currency("usd") == "USD"`.
- `norm_skills(["Python ", "python", "SQL"]) == frozenset({"python", "sql"})`.
- `is_absent` is `True` for `None`, for `"unknown"` on each enum field, and for
  `[]` on `required_skills`; `False` for `"none"` on `education_level`.
- `norm_field("nope", 1)` raises `KeyError`.

`tests/test_splits.py`
- `split_for` is deterministic across calls and process restarts (hard-code two
  known doc_ids and their expected split in the test).
- Over 10,000 synthetic ids, bucket proportions land within ±2 points of 5/5/90.

`tests/test_no_torch_import.py` — verbatim from SPEC §9.2.

`tests/test_cli.py`
- `sxl --help` exits 0.
- `sxl schema dump --out tmp` writes a file whose `required` has 16 entries.
- `sxl teacher label` exits 2 with a message containing `"F2"`.

---

### Verify

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pip list | grep -Ei "^(torch|nvidia|transformers|triton)" && echo "FAIL: gpu dep leaked" && exit 1
du -sh .venv                      # must print well under 300M
pytest -q                         # all green
sxl --help && sxl schema dump --out /tmp/s.json
python -c "import json;d=json.load(open('/tmp/s.json'));assert len(d['required'])==16;assert d['additionalProperties'] is False;print('OK')"
```

Expected: the `grep` finds nothing, `du` is small, pytest is green, and the final
line prints `OK`.

---

### Acceptance criteria

- [ ] `pip install -e ".[dev]"` on a torch-free environment succeeds and the venv
      is under 300 MB.
- [ ] `pip list` shows no `torch`, `nvidia-*`, `transformers`, or `triton`.
- [ ] `tests/test_no_torch_import.py` passes — importing `sxl`, `sxl.cli`,
      `sxl.metrics`, `sxl.schema`, `sxl.teacher` in a fresh interpreter leaves
      `sys.modules` free of any `torch*` entry.
- [ ] `JSON_SCHEMA` has all 16 keys in `required` and `additionalProperties: False`;
      an import-time assertion enforces this.
- [ ] `validate_prediction` rejects each of the five malformed payloads listed in
      the test plan and accepts `empty_posting()`.
- [ ] `split_for` returns the same value for the same `doc_id` across separate
      processes.
- [ ] `sxl --help` lists every command in SPEC §4; each unimplemented one exits 2
      naming its owning feature.
- [ ] `.gitignore` prevents `data/`, `artifacts/`, `*.safetensors`, and `.env`
      from being staged (verify with `git status --porcelain` after touching one
      of each).
