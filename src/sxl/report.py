"""F8 — aggregate `results/**/*.json` into markdown tables and the README.

This module **reads numbers and never computes one**. Every metric, latency and
cost was produced by F4 and F7 and committed; F8's whole job is to arrange them
so a reader can see what happened, including the parts that are unflattering
(SPEC §1.1: *if the measured result is unflattering, the measured result ships*).

Three properties are load-bearing:

* **Missing files degrade to `—`.** The report must build mid-project, when only
  the baseline arms exist — that is exactly when it is most useful for deciding
  what to fix next. Nothing here raises on an absent artifact; `load_results`
  returns warning strings and `cli.py` prints them, matching
  `verify.agreement_warnings` and `teacher.quality_warnings`.
* **Single-stream latency and amortized latency are separate columns, always.**
  SPEC §1.1 exists because the project pitch quoted an amortized-shaped number as
  a latency. `check_claims` rejects that conflation mechanically.
* **No number is typed into markdown.** The README is `README_TEMPLATE` rendered
  from `README_FACTS.json`, and `check_claims` re-reads the result to verify that
  every unit-bearing figure in it traces back to a committed file.

Laptop-only (SPEC §2.1): stdlib plus `sxl.config`/`sxl.io`/`sxl.schema`. No
torch, numpy, pandas or matplotlib, and no new dependency for table formatting —
markdown tables are f-strings.
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from sxl.config import (
    ARMS,
    BASE_MODEL,
    BENCH_DIR,
    CORPUS_STATS_PATH,
    GOLD_AUDIT_PATH,
    GOLD_STATS_PATH,
    HEADLINE_PATH,
    METRICS_DIR,
    N_EVAL_GOLD,
    PER_FIELD_PATH,
    README_FACTS_PATH,
    README_PATH,
    REPORT_LOW_SUPPORT,
    REPORT_NOISE_F1,
    SWEEP_TABLE_PATH,
    TABLES_DIR,
    TEACHER_MODEL,
    TRAIN_STATS_PATH,
    git_sha,
    utc_now,
)
from sxl.io import read_json
from sxl.schema import FIELD_NAMES

__all__ = [
    "BEGIN_MARKER",
    "END_MARKER",
    "README_TEMPLATE",
    "ReportPaths",
    "best_sweep_entry",
    "build_headline",
    "build_per_field",
    "build_readme",
    "build_sweep",
    "check_claims",
    "facts",
    "load_results",
    "splice_readme",
]

#: The cell for anything that has not been measured yet. An em dash, never `0`,
#: never a blank: a reader must be able to tell "not run" from "scored zero".
DASH = "—"

#: The teacher's latency cells. Its `measurement` is `api_wall_clock` — network
#: round-trips from a laptop — which is not comparable with a local `generate()`
#: and so is never rendered as a number in a column of local timings.
API_CELL = "API"

BEGIN_MARKER = "<!-- BEGIN GENERATED -->"
END_MARKER = "<!-- END GENERATED -->"

#: Sourced from the F8 spec, Scope 2. Changing one of these strings changes the
#: artifact a reader screenshots, so the test asserts them literally.
HEADLINE_HEADERS = (
    "arm",
    "schema-valid %",
    "macro-F1",
    "Δ vs teacher",
    "p50 ms/doc (batch 1)",
    "amortized ms/doc @ best batch",
    "$/1k docs",
)

#: The label of the floor row: an arm at or below it extracted nothing.
NULL_BASELINE_ROW = "null baseline"


# --- paths --------------------------------------------------------------------


@dataclass(frozen=True)
class ReportPaths:
    """Every file F8 reads or writes, in one injectable object.

    Mirrors `metrics.MetricsPaths` and `verify.GoldPaths`: passing
    `ReportPaths.in_dir(tmp_path)` makes it structurally impossible for a test to
    overwrite the real `README.md` or the committed `results/tables/`.
    """

    metrics_dir: Path
    bench_dir: Path
    tables_dir: Path
    corpus_stats: Path
    gold_stats: Path
    gold_audit: Path
    teacher_stats: Path
    train_stats: Path
    facts: Path
    readme: Path

    @classmethod
    def default(cls) -> ReportPaths:
        from sxl.config import TEACHER_STATS_PATH

        return cls(
            metrics_dir=METRICS_DIR,
            bench_dir=BENCH_DIR,
            tables_dir=TABLES_DIR,
            corpus_stats=CORPUS_STATS_PATH,
            gold_stats=GOLD_STATS_PATH,
            gold_audit=GOLD_AUDIT_PATH,
            teacher_stats=TEACHER_STATS_PATH,
            train_stats=TRAIN_STATS_PATH,
            facts=README_FACTS_PATH,
            readme=README_PATH,
        )

    @classmethod
    def in_dir(cls, root: Path) -> ReportPaths:
        root = Path(root)
        results = root / "results"
        return cls(
            metrics_dir=results / "metrics",
            bench_dir=results / "bench",
            tables_dir=results / "tables",
            corpus_stats=results / "corpus_stats.json",
            gold_stats=results / "gold_stats.json",
            gold_audit=results / "gold_review_audit.json",
            teacher_stats=results / "teacher_stats.json",
            train_stats=results / "train_stats.json",
            facts=results / "README_FACTS.json",
            readme=root / "README.md",
        )

    def with_results_dir(self, results: Path) -> ReportPaths:
        """Re-root every input, and the facts file, at `results`.

        The tables directory and the README are left alone: `--results-dir` says
        where to *read* from, `--out-dir` says where to write.
        """
        results = Path(results)
        return replace(
            self,
            metrics_dir=results / "metrics",
            bench_dir=results / "bench",
            corpus_stats=results / "corpus_stats.json",
            gold_stats=results / "gold_stats.json",
            gold_audit=results / "gold_review_audit.json",
            teacher_stats=results / "teacher_stats.json",
            train_stats=results / "train_stats.json",
            facts=results / README_FACTS_PATH.name,
        )

    def with_out_dir(self, out_dir: Path) -> ReportPaths:
        """Redirect the generated tables. Inputs and the README are unchanged."""
        return replace(self, tables_dir=Path(out_dir))

    @property
    def headline(self) -> Path:
        return self.tables_dir / HEADLINE_PATH.name

    @property
    def per_field(self) -> Path:
        return self.tables_dir / PER_FIELD_PATH.name

    @property
    def sweep(self) -> Path:
        return self.tables_dir / SWEEP_TABLE_PATH.name


def _resolve(paths: ReportPaths | None) -> ReportPaths:
    return paths if paths is not None else ReportPaths.default()


# --- loading ------------------------------------------------------------------


def load_results(paths: ReportPaths | None = None) -> dict[str, Any]:
    """Read every committed artifact. Absent files are absent keys, never errors.

    Returns `{"metrics": {arm: obj}, "bench": {arm: obj}, "sweep": {arm: obj},
    "stats": {name: obj}, "missing": [str, ...]}`.

    `missing` is prose for a human, one line per absent artifact, which the CLI
    prints to stderr. Library code returns warnings rather than printing them —
    the same split `verify.agreement_warnings` uses.
    """
    p = _resolve(paths)
    metrics: dict[str, Any] = {}
    bench: dict[str, Any] = {}
    sweep: dict[str, Any] = {}
    missing: list[str] = []

    for path in sorted(Path(p.metrics_dir).glob("*.json")):
        obj = read_json(path)
        metrics[obj.get("arm", path.stem)] = obj

    for path in sorted(Path(p.bench_dir).glob("*.json")):
        obj = read_json(path)
        if path.stem.endswith("_sweep"):
            sweep[obj.get("arm", path.stem.removesuffix("_sweep"))] = obj
        else:
            bench[obj.get("arm", path.stem)] = obj

    for arm in ARMS:
        absent = [
            label for label, table in (("metrics", metrics), ("bench", bench)) if arm not in table
        ]
        # The teacher has no batch sweep by construction — it is an API arm, not a
        # `generate()` loop — so its absence is expected and is not reported.
        if arm not in sweep and arm != "teacher":
            absent.append("sweep")
        if absent:
            missing.append(f"{arm}: no {', '.join(absent)} — those cells render {DASH}")

    stats: dict[str, Any] = {}
    for name, path in (
        ("corpus", p.corpus_stats),
        ("gold", p.gold_stats),
        ("gold_audit", p.gold_audit),
        ("teacher", p.teacher_stats),
        ("train", p.train_stats),
    ):
        if Path(path).exists():
            stats[name] = read_json(path)
        else:
            missing.append(f"{name} stats: {path} not found")

    return {"metrics": metrics, "bench": bench, "sweep": sweep, "stats": stats, "missing": missing}


def best_sweep_entry(sweep: dict[str, Any] | None) -> dict[str, Any] | None:
    """The fastest non-OOM row of a batch sweep, or `None`.

    **This deliberately differs from `gpu.bench.best_batch_size`**, which records
    the *largest* batch size that did not OOM. Those are not the same row: in the
    committed `base_fewshot` sweep, batch 4 runs at 8675 ms/doc and batch 8 at
    11583 ms/doc, so the recorded "best" is 34% slower and dearer than the best
    the same sweep actually measured. Publishing it would understate the baseline
    — the arm SPEC §3.6 calls *the competitor that matters* — and flatter the
    fine-tune by comparison. F8 still only picks among committed rows; it
    computes nothing, and the headline table footnotes the divergence.
    """
    if not sweep:
        return None
    usable = [e for e in sweep.get("sweep", []) if not e.get("oom")]
    if not usable:
        return None
    return max(usable, key=lambda e: e.get("throughput_docs_per_s") or 0.0)


# --- formatting ---------------------------------------------------------------

#: How each fact is printed. One canonical rendering per key is what makes
#: "matches to the printed precision" a decidable question in `check_claims`.
#: Units live in column headers and in the README's prose, never in the cell, so
#: every cell is a bare numeral that can be matched against this table.
_KIND_BY_KEY: dict[str, str] = {
    "schema_valid_rate": "pct1",
    "macro_f1": "f3",
    "macro_f1_null_baseline": "f3",
    "delta_vs_teacher": "signed3",
    "noise_f1": "f3",
    "f1": "f3",
    "em": "f3",
    "agreement": "f3",
    "p50_ms": "ms0",
    "p95_ms": "ms0",
    "amortized_ms_per_doc": "ms0",
    "cost_per_1k_docs_usd": "usd5",
    "cost_per_1k_docs_usd_batch1": "usd5",
    "gpu_hourly_usd": "usd2",
    "spend_usd": "usd2",
    "doc_error_rate": "pct1",
    "gold_audit_doc_error_rate": "pct1",
    "trainable_pct": "num2",
    "peak_vram_gb": "num2",
    "train_runtime_h": "num1",
    "throughput_docs_per_s": "num4",
}


def _kind_for(key: str, value: Any) -> str:
    if key in _KIND_BY_KEY:
        return _KIND_BY_KEY[key]
    return "int0" if isinstance(value, int) and not isinstance(value, bool) else "f3"


def _fmt(value: Any, kind: str) -> str:
    """Render one number. `None` is `—` in every kind, so a gap never reads as 0."""
    if value is None:
        return DASH
    if kind == "pct1":
        return f"{value * 100:.1f}"
    if kind == "f3":
        return f"{value:.3f}"
    if kind == "signed3":
        return f"{value:+.3f}"
    if kind == "ms0":
        return f"{value:.0f}"
    if kind == "usd5":
        return f"{value:.5f}"
    if kind == "usd2":
        return f"{value:.2f}"
    if kind == "num1":
        return f"{value:.1f}"
    if kind == "num2":
        return f"{value:.2f}"
    if kind == "num4":
        return f"{value:.4f}"
    if kind == "int0":
        return f"{int(value)}"
    raise ValueError(f"unknown format kind {kind!r}")  # pragma: no cover - programmer error


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _show(key: str, value: Any) -> str:
    """Render `value` the one way `key` is always printed."""
    if value is None:
        return DASH
    if isinstance(value, str):
        return value
    if not _is_number(value):  # structured leaves (the agreement list) print as-is
        return str(value)
    return _fmt(value, _kind_for(key, value))


def _table(headers: tuple[str, ...], aligns: tuple[str, ...], rows: list[tuple[str, ...]]) -> str:
    """A GitHub-flavoured pipe table. No `tabulate` — this is an f-string."""
    rule = tuple("---:" if a == "r" else ":---" for a in aligns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(rule) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def _footer() -> str:
    """`git_sha` under every table, so a screenshot traces to a commit."""
    return f"_generated by `sxl report build` at {utc_now()} from git `{git_sha()}`._"


def _get(obj: dict[str, Any] | None, key: str) -> Any:
    return None if obj is None else obj.get(key)


def _hourly_usd(bench: dict[str, Any]) -> float | None:
    """The GPU rate every bench file echoes. Read, never hardcoded here."""
    return next((b["gpu_hourly_usd"] for b in bench.values() if b.get("gpu_hourly_usd")), None)


def _cost_at(entry: dict[str, Any] | None, hourly_usd: float | None) -> float | None:
    """Cost per 1k documents at one sweep row's measured throughput.

    Sweep *entries* record throughput but not cost — only the file-level `best_*`
    fields carry a cost, and those describe the file's own `best_batch_size`,
    which is not the row this table reports (see `best_sweep_entry`). Reading the
    file-level cost beside a different row's latency would put two batch sizes in
    one line of the table, which is precisely the conflation F8 exists to prevent.

    So the cost is re-derived — from F7's own `cost_per_1k`, never a second copy
    of the formula, applied to a committed throughput. The headline footnote says
    plainly that this column is derived rather than measured. `sxl.gpu.bench` is
    deliberately torch-free at import (`tests/test_gpu_lazy_import.py`), but the
    import stays inside the function so `sxl.report` never reaches into `gpu/`
    just by being imported.
    """
    from sxl.gpu.bench import BenchError, cost_per_1k

    throughput = _get(entry, "throughput_docs_per_s")
    if throughput is None or hourly_usd is None:
        return None
    try:
        return cost_per_1k(throughput, hourly_usd)
    except BenchError:
        # A zero throughput means that row's measurement failed. It has no cost,
        # and inventing one would be worse than an em dash.
        return None


# --- headline -----------------------------------------------------------------


def build_headline(results: dict[str, Any], *, footer: bool = True) -> str:
    """`results/tables/headline.md` — the deliverable SPEC §1 cares about.

    `footer=False` when the table is embedded in the README, which carries its own
    provenance line and does not need a second one.
    """
    metrics, bench, sweeps = results["metrics"], results["bench"], results["sweep"]
    teacher_f1 = _get(metrics.get("teacher"), "macro_f1")

    rows: list[tuple[str, ...]] = []
    for arm in ARMS:
        m, b = metrics.get(arm), bench.get(arm)
        best = best_sweep_entry(sweeps.get(arm))

        macro = _get(m, "macro_f1")
        delta = None if (macro is None or teacher_f1 is None) else macro - teacher_f1

        if arm == "teacher":
            # An API round-trip is not a local generate(); rendering it in this
            # column as a numeral would invite exactly the comparison the
            # measurement cannot support.
            p50_cell = amortized_cell = API_CELL
            cost = _get(b, "cost_per_1k_docs_usd")
        else:
            p50_cell = _show("p50_ms", _get(b, "p50_ms"))
            amortized = _get(best, "amortized_ms_per_doc")
            batch = _show("batch_size", _get(best, "batch_size"))
            amortized_cell = (
                DASH
                if amortized is None
                else f"{_show('amortized_ms_per_doc', amortized)} (bs {batch})"
            )
            # Paired with the amortized column beside it: this is the cost of
            # serving at the batch size that column reports, not at batch 1.
            cost = _cost_at(best, _hourly_usd(bench))
            if cost is None:
                cost = _get(b, "cost_per_1k_docs_usd")

        rows.append(
            (
                f"`{arm}`",
                _show("schema_valid_rate", _get(m, "schema_valid_rate")),
                _show("macro_f1", macro),
                _show("delta_vs_teacher", delta),
                p50_cell,
                amortized_cell,
                _show("cost_per_1k_docs_usd", cost),
            )
        )

    # The floor. Any arm at or below it is extracting nothing, and without the row
    # a reader has no scale against which to read a macro-F1 at all.
    null_baseline = next(
        (m["macro_f1_null_baseline"] for m in metrics.values() if "macro_f1_null_baseline" in m),
        None,
    )
    rows.append(
        (
            NULL_BASELINE_ROW,
            DASH,
            _show("macro_f1_null_baseline", null_baseline),
            DASH,
            DASH,
            DASH,
            DASH,
        )
    )

    table = _table(HEADLINE_HEADERS, ("l", "r", "r", "r", "r", "r", "r"), rows)
    parts = [table, _headline_notes(results)]
    if footer:
        parts.append(_footer())
    return "\n\n".join(parts) + "\n"


def _headline_notes(results: dict[str, Any]) -> str:
    """Footnotes. Every figure in them is read from a file, none is asserted."""
    metrics, bench = results["metrics"], results["bench"]
    any_bench = next(iter(bench.values()), None)
    n = _get(next(iter(metrics.values()), None), "n") or N_EVAL_GOLD
    hourly = _hourly_usd(bench)
    t = bench.get("teacher") or {}

    batch1 = ", ".join(
        f"`{arm}` ${_show('cost_per_1k_docs_usd', b['cost_per_1k_docs_usd'])}"
        for arm, b in bench.items()
        if arm != "teacher" and b.get("cost_per_1k_docs_usd") is not None
    )
    two_columns = (
        "- **The two latency columns are different measurements and are never "
        "merged.** `p50 ms/doc (batch 1)` is single-stream: one document, start to "
        "finish. `amortized ms/doc @ best batch` is wall-clock divided by documents "
        "at the fastest non-OOM batch size in `results/bench/<arm>_sweep.json`. "
        "Only the first is a latency a user would experience."
    )
    best_batch = (
        "- **Best batch = fastest, not largest.** The sweep files' own "
        "`best_batch_size` field records the largest batch that did not OOM, which "
        "for `base_fewshot` is not its fastest. This table selects the "
        "highest-throughput non-OOM row instead; see `sweep.md` for every row."
    )
    cost = (
        f"- **`$/1k docs` is derived, not measured**: throughput at that batch size "
        f"against an assumed ${_show('gpu_hourly_usd', hourly)}/h GPU rate, recorded "
        f"as `gpu_hourly_usd` in every bench file. At batch 1 the same arms cost "
        f"{batch1} per 1k."
    )
    teacher = (
        f"- **The `teacher` row's latency cells read `{API_CELL}`** because its "
        f"`measurement` is `{t.get('measurement', 'api_wall_clock')}` — sequential "
        f"network calls from a laptop, not comparable with a local `generate()`. Its "
        f"cost uses **standard, not Batch API, rates**: a latency-sensitive "
        f"deployment cannot wait on a 24-hour queue, so the half-price Batch figure "
        f"F2 paid for labeling would be the wrong number here."
    )
    noise = (
        f"- **n = {n}** hand-checked evaluation documents. The rule of thumb at that "
        f"size is a standard error near {_show('noise_f1', REPORT_NOISE_F1)} on a "
        f"per-field F1, so **differences smaller than that are not meaningful** and "
        f"should not be ranked."
    )
    sample = (
        f"- Latency figures come from {_show('n_docs', _get(any_bench, 'n_docs'))} timed "
        f"documents × {_show('repeats', _get(any_bench, 'repeats'))} repeats per arm on "
        f"a {_get(any_bench, 'gpu_name') or DASH} at "
        f"`{_get(any_bench, 'dtype') or DASH}`. That is a small sample; the spread "
        f"across repeats is recorded as `p50_spread_pct` in each bench file."
    )
    notes = [two_columns, best_batch, cost if batch1 else "", teacher, noise, sample]
    return "\n".join(nb for nb in notes if nb)


# --- per-field ----------------------------------------------------------------


def build_per_field(results: dict[str, Any]) -> str:
    """`results/tables/per_field.md` — 16 fields, weakest first.

    Sorted ascending by `lora_ft` F1 so the failures are read before the wins.
    `support` is a property of the gold set, identical for every arm, so it is one
    column rather than five copies.
    """
    metrics = results["metrics"]
    present = [a for a in ARMS if a in metrics]
    if not present:
        return f"No metrics found — nothing to break down.\n\n{_footer()}\n"

    support = {f: m["support"] for f, m in metrics[present[0]]["per_field"].items()}
    sort_arm = next((a for a in ("lora_ft", "base_fewshot", *present) if a in metrics), present[0])
    order = sorted(
        FIELD_NAMES,
        key=lambda f: (metrics[sort_arm]["per_field"].get(f, {}).get("f1", 0.0), f),
    )

    headers = ("field", "support", "note")
    aligns = ("l", "r", "l")
    for arm in present:
        headers += (f"{arm} em", f"{arm} f1")
        aligns += ("r", "r")

    rows: list[tuple[str, ...]] = []
    for field in order:
        n_support = support.get(field, 0)
        note = "low-n" if n_support < REPORT_LOW_SUPPORT else ""
        row: tuple[str, ...] = (f"`{field}`", _show("support", n_support), note)
        for arm in present:
            pf = metrics[arm]["per_field"].get(field, {})
            row += (_show("em", pf.get("em")), _show("f1", pf.get("f1")))
        rows.append(row)

    notes = (
        f"Sorted ascending by `{sort_arm}` F1: the weakest fields are the ones worth "
        f"reading.\n\n"
        f"- **`low-n` means the F1 beside it is noise.** Any field with fewer than "
        f"{_show('low_support', REPORT_LOW_SUPPORT)} populated gold values is flagged; "
        f"an F1 computed from a handful of documents moves several points on one "
        f"document and must not be quoted as a result.\n"
        f"- **`support` counts gold values, not documents, for `required_skills`** — "
        f"it is a set field, so its support is skill *elements* summed across the "
        f"split. Comparing it against a scalar field's support compares two units.\n"
        f'- **EM and F1 diverge on quiet fields.** `null`, `"unknown"` and `[]` are '
        f"all the *absent* marker (SPEC §3.5). A field most postings genuinely do not "
        f"mention scores high EM — both sides agree it is absent — while its F1, which "
        f"ignores the both-absent case, stays low and volatile. Both are reported "
        f"because neither alone is honest.\n"
        f"- Invalid predictions are counted as **wrong, not excluded**; the "
        f"denominator is always the full split."
    )
    return "\n\n".join([_table(headers, aligns, rows), notes, _footer()]) + "\n"


# --- sweep --------------------------------------------------------------------


def build_sweep(results: dict[str, Any]) -> str:
    """`results/tables/sweep.md` — what makes the cost column credible."""
    sweeps, bench = results["sweep"], results["bench"]
    present = [a for a in ARMS if a in sweeps]
    if not present:
        return f"No batch sweeps found.\n\n{_footer()}\n"

    hourly = _hourly_usd(bench)
    headers = (
        "arm",
        "batch size",
        "throughput docs/s",
        "amortized ms/doc",
        "$/1k docs",
        "peak VRAM GB",
        "",
    )
    rows: list[tuple[str, ...]] = []
    for arm in present:
        best = best_sweep_entry(sweeps[arm])
        for entry in sweeps[arm].get("sweep", []):
            oom = bool(entry.get("oom"))
            mark = "OOM" if oom else ("**fastest**" if entry is best else "")
            rows.append(
                (
                    f"`{arm}`",
                    _show("batch_size", entry.get("batch_size")),
                    _show("throughput_docs_per_s", entry.get("throughput_docs_per_s")),
                    _show("amortized_ms_per_doc", entry.get("amortized_ms_per_doc")),
                    _show("cost_per_1k_docs_usd", None if oom else _cost_at(entry, hourly)),
                    _show("peak_vram_gb", entry.get("peak_vram_gb")),
                    mark,
                )
            )

    oom_note = (
        "OOM rows are kept, not dropped: where an arm stops scaling is part of the "
        "result, and a sweep with the failures removed would imply headroom that "
        "does not exist."
    )
    fastest_note = (
        "**fastest** marks the row this arm's `$/1k docs` and amortized latency are "
        "read from in `headline.md`. It is not always the sweep file's own "
        "`best_batch_size`, which records the largest non-OOM size instead."
    )
    notes = [oom_note, fastest_note]
    for arm in present:
        note = (sweeps[arm].get("note") or "").strip()
        if note:
            notes.append(f"`{arm}`: {note}")

    body = "\n".join(f"- {n}" for n in notes)
    table = _table(headers, ("l", "r", "r", "r", "r", "r", "l"), rows)
    return f"{table}\n\n{body}\n\n{_footer()}\n"


# --- facts --------------------------------------------------------------------


def facts(results: dict[str, Any]) -> dict[str, Any]:
    """`results/README_FACTS.json` — every scalar the README interpolates.

    The README is this file rendered through `README_TEMPLATE`. Nothing reaches
    the prose that is not first a value here, which is what makes SPEC §1's "no
    cell is ever typed by hand" checkable rather than a promise.
    """
    metrics, bench, sweeps, stats = (
        results["metrics"],
        results["bench"],
        results["sweep"],
        results["stats"],
    )
    teacher_f1 = _get(metrics.get("teacher"), "macro_f1")
    gold, train, corpus, teacher_stats = (
        stats.get("gold", {}),
        stats.get("train", {}),
        stats.get("corpus", {}),
        stats.get("teacher", {}),
    )
    audit = stats.get("gold_audit", {})
    any_bench = next(iter(bench.values()), None)

    hourly = _hourly_usd(bench)

    out: dict[str, Any] = {}
    for arm in ARMS:
        m, b = metrics.get(arm), bench.get(arm)
        best = best_sweep_entry(sweeps.get(arm))
        macro = _get(m, "macro_f1")
        cost_best = _cost_at(best, hourly)
        out[arm] = {
            "schema_valid_rate": _get(m, "schema_valid_rate"),
            "macro_f1": macro,
            "delta_vs_teacher": (
                None if (macro is None or teacher_f1 is None) else macro - teacher_f1
            ),
            "p50_ms": _get(b, "p50_ms"),
            "p95_ms": _get(b, "p95_ms"),
            "amortized_ms_per_doc": _get(best, "amortized_ms_per_doc"),
            "best_batch_size": _get(best, "batch_size"),
            "cost_per_1k_docs_usd": (
                cost_best if cost_best is not None else _get(b, "cost_per_1k_docs_usd")
            ),
            "cost_per_1k_docs_usd_batch1": _get(b, "cost_per_1k_docs_usd"),
        }

    agreement: dict[str, float] = gold.get("teacher_field_agreement", {}) or {}
    lowest = sorted(agreement.items(), key=lambda kv: (kv[1], kv[0]))[:3]
    residual = audit.get("residual_error_sample", {}) or {}

    out["lowest_teacher_agreement"] = [{"field": f, "agreement": v} for f, v in lowest]
    out["lowest_teacher_agreement_str"] = (
        ", ".join(f"`{f}` {_show('agreement', v)}" for f, v in lowest) or DASH
    )
    out["n_eval_gold"] = gold.get("n_final") or _get(next(iter(metrics.values()), None), "n")
    out["n_train"] = train.get("n_train") or teacher_stats.get("n_ok")
    out["n_corpus"] = corpus.get("n_kept")
    out["corpus_source"] = corpus.get("source")
    out["corpus_license"] = corpus.get("license")
    out["teacher_model"] = teacher_stats.get("teacher_model") or TEACHER_MODEL
    out["base_model"] = train.get("base_model") or BASE_MODEL
    out["adapter_repo"] = train.get("adapter_repo")
    out["gpu_name"] = _get(any_bench, "gpu_name") or train.get("gpu_name")
    out["gpu_hourly_usd"] = next(
        (b["gpu_hourly_usd"] for b in bench.values() if b.get("gpu_hourly_usd") is not None), None
    )
    out["dtype"] = _get(any_bench, "dtype") or train.get("dtype")
    out["bench_n_docs"] = _get(any_bench, "n_docs")
    out["bench_repeats"] = _get(any_bench, "repeats")
    out["spend_usd"] = teacher_stats.get("spend_usd_cached") or teacher_stats.get("spend_usd")
    out["trainable_pct"] = train.get("trainable_pct")
    out["peak_vram_gb"] = train.get("peak_vram_gb")
    out["train_runtime_s"] = train.get("train_runtime_s")
    out["train_runtime_h"] = (
        None if train.get("train_runtime_s") is None else train["train_runtime_s"] / 3600
    )
    out["train_epochs"] = train.get("epochs")
    out["best_eval_loss"] = train.get("best_eval_loss")
    out["macro_f1_null_baseline"] = next(
        (m["macro_f1_null_baseline"] for m in metrics.values() if "macro_f1_null_baseline" in m),
        None,
    )
    out["gold_reviewer"] = gold.get("reviewer")
    out["gold_audit_n"] = residual.get("n")
    out["gold_audit_doc_error_rate"] = residual.get("doc_error_rate")
    out["noise_f1"] = REPORT_NOISE_F1
    out["low_support"] = REPORT_LOW_SUPPORT
    out["generated_at"] = utc_now()
    out["git_sha"] = git_sha()
    return out


def _flatten(obj: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """`{"lora_ft": {"macro_f1": ...}}` -> `{"lora_ft__macro_f1": ...}`."""
    flat: dict[str, Any] = {}
    for key, value in obj.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten(value, f"{name}__"))
        else:
            flat[name] = value
    return flat


def display_context(facts_dict: dict[str, Any]) -> dict[str, str]:
    """Every fact flattened and rendered exactly once — the README's namespace.

    Template placeholders are the flattened keys (`{lora_ft__macro_f1}`), so the
    template reads as an explicit bill of materials for its own numbers and a test
    can diff its placeholders against this dict.
    """
    return {k: _show(k.rsplit("__", 1)[-1], v) for k, v in _flatten(facts_dict).items()}


# --- README -------------------------------------------------------------------

README_TEMPLATE = """\
{begin}
<!-- Regenerated by `sxl report build`. Do not edit between these markers:
     every number below is interpolated from results/README_FACTS.json. -->

## What this is

A LoRA fine-tune of `{base_model}` that extracts a strict, flat sixteen-field JSON
schema from unstructured job postings, measured against the hosted teacher model
(`{teacher_model}`) that produced its training labels. The question is narrow and
answerable: **can a small open model, fine-tuned on a few thousand teacher-labeled
examples, do this job closely enough to a cheap hosted API to be worth running
yourself?**

The deliverable is the table below, not the model. Every cell is produced by
`sxl report build` from a committed JSON file under `results/`; none is typed by
hand, and a guardrail (`sxl.report.check_claims`) fails the build if the prose
here quotes a number the artifacts do not contain.

## Results

{headline_table}

**The fine-tune reaches {lora_ft__macro_f1} macro-F1 against the teacher's {teacher__macro_f1}**
— a signed difference of {lora_ft__delta_vs_teacher} — while the unconstrained few-shot
baseline on the same base model sits at {base_fewshot__macro_f1}. Most of the gap between a
prompted small model and the hosted API that taught it is recovered by fine-tuning. It also
serves a thousand documents for ${lora_ft__cost_per_1k_docs_usd} against the teacher's
${teacher__cost_per_1k_docs_usd}, on a single rented {gpu_name}.

### Constrained decoding buys validity, not accuracy

This is the most interesting result here, and it is easy to miss. Constraining the decoder to
the JSON Schema takes schema-validity from {base_fewshot__schema_valid_rate}% to
{base_fewshot_constrained__schema_valid_rate}% on the baseline, and from
{lora_ft__schema_valid_rate}% to {lora_ft_constrained__schema_valid_rate}% on the fine-tune.
**Validity is essentially free.**

Accuracy is not. The same constraint moves macro-F1 to {base_fewshot_constrained__delta_vs_teacher}
from {base_fewshot__delta_vs_teacher} against the teacher on the baseline — a real gain, because
parse failures were being scored as wrong — but on the fine-tune, which already emits valid JSON
almost always, it moves macro-F1 from {lora_ft__macro_f1} to {lora_ft_constrained__macro_f1}: no
improvement, and it costs latency ({lora_ft__p50_ms} ms to {lora_ft_constrained__p50_ms} ms at
batch 1, plus grammar-index build time). Constrained decoding guarantees the *shape* of the
answer. It has nothing to say about whether the answer is right.

### Latency and cost, stated as two different things

Single-stream p50 for the fine-tune is **{lora_ft__p50_ms} ms per document**. The amortized
figure at its fastest batch size (batch {lora_ft__best_batch_size}) is
**{lora_ft__amortized_ms_per_doc} ms per document**. These differ by roughly an order of
magnitude and they answer different questions: the first is what one user waits, the second is
what a thousand-document backlog costs you. This README never quotes the second as a latency.
A ~1.7B model in fp16 on a {gpu_name} generating a ~150-token object takes seconds, not
milliseconds, single-stream; any claim otherwise is an amortized number wearing a latency's
clothes.

Per-field breakdown: [`results/tables/per_field.md`](results/tables/per_field.md).
Batch sweep: [`results/tables/sweep.md`](results/tables/sweep.md).

## How it was built

| stage | what happened |
|:---|:---|
| corpus | {n_corpus} real postings from `{corpus_source}` ({corpus_license}), deduplicated and length-filtered |
| labels | {n_train} training rows labeled by `{teacher_model}` via the Batch API for ${spend_usd} |
| gold | {n_eval_gold} held-out documents reviewed field by field; no gold document appears in training |
| fine-tune | LoRA r=16 on {base_model}, fp16, {train_epochs} epochs, {trainable_pct}% of parameters trainable, {train_runtime_h} h on a {gpu_name}, peak {peak_vram_gb} GB VRAM |
| adapter | [`{adapter_repo}`](https://huggingface.co/{adapter_repo}) |

## Reproducing it

Everything except the four GPU stages runs on a laptop with no GPU and no torch:

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
pytest -q

sxl corpus build                    # fetch and normalize the corpus
sxl teacher label --split train     # costs money; --dry-run projects the spend first
sxl gold sample && sxl gold verify  # build the held-out evaluation set
sxl metrics score --arm teacher     # score any arm that has predictions
sxl report build                    # regenerate every table and this README
```

Do **not** install the `gpu` extra locally — it is Kaggle-only, and a single fp16
copy of the model would fill a small laptop's disk.

The GPU steps (`sxl gpu predict`, `sxl gpu train`, `sxl gpu bench`) run on Kaggle
with the `GPU T4 x2` accelerator, using `cuda:0` only so latency stays comparable
across arms. The notebooks are thin wrappers around those commands and live in
[`notebooks/`](notebooks/). Kaggle's free tier is about 30 GPU-hours a week; the
whole project fits inside one week's quota.

## Limitations

These are the reasons not to over-read the table above.

1. **Teacher labels are not ground truth.** Outside the {n_eval_gold} reviewed documents,
   every training label is `{teacher_model}`'s opinion, and the model is being trained to
   imitate it — including its mistakes. Where the teacher was checked against a reviewer, its
   weakest fields were {lowest_teacher_agreement_str}. The first of those is not a close call:
   the teacher fabricated a plausible date for most postings that never stated one.

2. **One domain, one seed, one configuration.** Job postings only; `SEED = 1337`
   everywhere; a single set of hyperparameters, one LoRA rank, one learning rate.
   No sweep was run and no confidence intervals were computed, so nothing here
   speaks to variance across runs.

3. **The gold set was reviewed as `{gold_reviewer}`, not by a human end to end.** A human
   audited a {gold_audit_n}-document sample of it and found at least one field error in
   {gold_audit_doc_error_rate}% of those documents. That residual error rate is a floor under
   *every* accuracy number in this README — including the teacher's, which is scored against
   the same set — and there is no inter-annotator agreement figure, because there was one
   reviewer.

4. **One GPU, fp16, and no vLLM.** All timings come from a single {gpu_name}
   (Turing, sm_75) running plain `transformers.generate()`. vLLM was deliberately
   not used: its Turing support is degrading, and debugging it would have cost
   GPU-hours the project did not have. Unsloth was skipped for similar reasons —
   its speedup is marginal at this model size and its dependency negotiation is
   not. ONNX Runtime and GGUF export were left out of scope entirely. **On an
   A10, L4 or A100 with vLLM, throughput would be substantially better and the
   `$/1k docs` column would drop accordingly**, so read the cost figures as an
   upper bound for this serving path, not as the best achievable.

5. **The evaluation set is {n_eval_gold} documents.** At that size the standard error on a
   per-field F1 is around {noise_f1}, so **differences smaller than about three points are not
   meaningful**, and the per-field table flags any field with fewer than {low_support}
   populated gold values as `low-n`. The latency numbers rest on a smaller sample still:
   {bench_n_docs} timed documents × {bench_repeats} repeats per arm.

6. **`$/1k docs` is an assumption, not a measurement.** It divides measured throughput by an
   assumed ${gpu_hourly_usd}/h rate for the GPU. A different rate, a spot instance, or a colder
   start moves the column without anything about the model changing. The teacher's cost is
   priced at standard API rates rather than the half-price Batch rates used for labeling,
   because a latency-sensitive deployment cannot wait on a 24-hour queue.

_Facts file: [`results/README_FACTS.json`](results/README_FACTS.json). Generated at
{generated_at} from git `{git_sha}`._
{end}"""


def build_readme(facts_dict: dict[str, Any], headline: str) -> str:
    """Render the generated block. Every number arrives via `facts_dict`."""
    context = display_context(facts_dict)
    context["headline_table"] = headline.strip()
    context["begin"], context["end"] = BEGIN_MARKER, END_MARKER
    return README_TEMPLATE.format_map(context)


def splice_readme(existing: str, generated: str) -> str:
    """Replace the marked block, leaving hand-written prose outside it untouched.

    On a README with no markers yet — which is every README before F8 first runs —
    the block is appended, so nothing a human wrote is ever discarded.
    """
    start, end = existing.find(BEGIN_MARKER), existing.find(END_MARKER)
    if start == -1 or end == -1 or end < start:
        return existing.rstrip("\n") + "\n\n" + generated.rstrip("\n") + "\n"
    head = existing[:start]
    tail = existing[end + len(END_MARKER) :]
    return head + generated.rstrip("\n") + tail


def generated_block(readme: str) -> str:
    """Just the machine-written part, or `""`. What `check_claims` inspects."""
    start, end = readme.find(BEGIN_MARKER), readme.find(END_MARKER)
    if start == -1 or end == -1 or end < start:
        return ""
    return readme[start : end + len(END_MARKER)]


# --- the guardrail ------------------------------------------------------------

_NUM = r"\d+(?:\.\d+)?"

#: A number must start a word to be a claim. Without this, the `50` in the column
#: label `p50 ms/doc` reads as a fifty-millisecond measurement — which is both a
#: false positive and, ironically, exactly the number SPEC §1.1 disowns.
_START = r"(?<![\w.])"

#: A number is a *claim* when it carries one of these units. Bare digits — spec
#: section references, LoRA ranks, list markers, `sm_75` — are not claims and are
#: not scanned; a figure in milliseconds, percent, dollars or F1 is.
_CLAIM_PATTERNS = (
    re.compile(rf"{_START}({_NUM})\s*%"),
    re.compile(rf"{_START}({_NUM})\s*ms\b"),
    re.compile(rf"\$\s*({_NUM})"),
    re.compile(rf"{_START}([+-]?{_NUM})\s*(?:macro-)?F1\b"),
)

#: SPEC §1.1: the teacher's parameter count is not public, so no size ratio may
#: ever be claimed — not "200x", not any multiple.
_RATIO_PATTERNS = (
    re.compile(r"\b200\s*[x×]", re.IGNORECASE),
    re.compile(rf"\b{_NUM}\s*[x×]\s*(?:larger|bigger|smaller)", re.IGNORECASE),
)

#: The pitch's illustrative latency. Physically unreachable single-stream for a
#: ~1.7B model on a T4, so it may appear only if it is genuinely a measured p50.
_FORTY_FIVE_MS = re.compile(r"\b45\s*ms\b", re.IGNORECASE)

#: SPEC §1.1 again: `gpt-4o-mini` is a small, cheap hosted model. Calling it
#: frontier restates the claim the project exists to avoid making.
_FRONTIER = re.compile(r"\bfrontier\b", re.IGNORECASE)


def _claim_numbers(text: str) -> set[str]:
    found: set[str] = set()
    for pattern in _CLAIM_PATTERNS:
        found.update(m.group(1).lstrip("+") for m in pattern.finditer(text))
    return found


def _allowed_numbers(facts_dict: dict[str, Any]) -> set[str]:
    """Every number the facts file authorizes, at the precision F8 prints it.

    Numeric leaves contribute their one canonical rendering. String leaves — the
    pre-composed ones like `lowest_teacher_agreement_str` — contribute whatever
    claim-shaped numbers they already contain, since the string itself is a
    committed fact.
    """
    allowed: set[str] = set()
    for key, value in _flatten(facts_dict).items():
        if value is None:
            continue
        if not _is_number(value):
            allowed |= set(re.findall(_NUM, str(value)))
            continue
        allowed.add(_show(key.rsplit("__", 1)[-1], value).lstrip("+"))
        # The same fact printed to a different precision is still that fact: a
        # ratio may be quoted as a ratio or as a percentage, and a millisecond
        # figure rounded to whole units is not a different measurement.
        allowed.add(_fmt(value, "f3").lstrip("+"))
        allowed.add(_fmt(value, "ms0"))
        allowed.add(_fmt(value * 100, "num1"))
    return allowed


def check_claims(readme: str, facts_dict: dict[str, Any]) -> list[str]:
    """Verify the generated block against the facts file. Returns violations.

    This is a spec-enforcement mechanism, not decoration: it is what makes SPEC
    §1.1 mechanically true instead of aspirational. `sxl report build --strict`
    exits 1 on any violation.
    """
    block = generated_block(readme)
    if not block:
        return [f"no generated block: {BEGIN_MARKER} / {END_MARKER} markers not found"]

    violations: list[str] = []
    allowed = _allowed_numbers(facts_dict)

    for pattern in _RATIO_PATTERNS:
        for m in pattern.finditer(block):
            violations.append(
                f"parameter-ratio claim {m.group(0)!r}: the teacher's size is not "
                f"public and no ratio may be claimed (SPEC §1.1)"
            )

    p50s = {
        _show("p50_ms", arm_facts["p50_ms"])
        for arm_facts in facts_dict.values()
        if isinstance(arm_facts, dict) and arm_facts.get("p50_ms") is not None
    }
    if _FORTY_FIVE_MS.search(block) and "45" not in p50s:
        violations.append(
            "'45 ms' appears but is not a measured p50 — that figure is the pitch's "
            "placeholder, not a result (SPEC §1.1)"
        )

    if _FRONTIER.search(block):
        violations.append(
            f"'frontier' appears: {facts_dict.get('teacher_model', TEACHER_MODEL)} is a "
            f"small, cheap hosted model and must not be described as frontier (SPEC §1.1)"
        )

    for number in sorted(_claim_numbers(block)):
        if number not in allowed:
            violations.append(
                f"{number!r} is quoted with a unit but is not a value in README_FACTS.json"
            )

    violations += _check_p50_not_amortized(block, facts_dict)
    return violations


def _check_p50_not_amortized(block: str, facts_dict: dict[str, Any]) -> list[str]:
    """Reject an amortized figure presented under a `p50` label.

    The two are an order of magnitude apart, and conflating them is the specific
    error SPEC §1.1 was written to prevent. A line that names both measurements is
    exempt — it is drawing the distinction, not hiding it.
    """
    amortized = {
        arm: _show("amortized_ms_per_doc", f["amortized_ms_per_doc"])
        for arm, f in facts_dict.items()
        if isinstance(f, dict) and f.get("amortized_ms_per_doc") is not None
    }
    violations = []
    for line in block.splitlines():
        low = line.lower()
        if "p50" not in low or "amortized" in low:
            continue
        for arm, value in amortized.items():
            if re.search(rf"\b{re.escape(value)}\s*ms\b", line):
                violations.append(
                    f"line labels {value} ms as a p50, but that is {arm}'s amortized "
                    f"figure: {line.strip()!r}"
                )
    return violations


def template_placeholders() -> set[str]:
    """The names `README_TEMPLATE` interpolates. Used by the facts test."""
    return {
        name
        for _, name, _, _ in string.Formatter().parse(README_TEMPLATE)
        if name and name not in {"begin", "end", "headline_table"}
    }
