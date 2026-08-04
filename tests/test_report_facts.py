"""F8 — `README_FACTS.json` is the README's only source of numbers (SPEC §1).

"No cell is ever typed by hand" is checkable rather than aspirational only if the
template and the facts file cannot drift apart. These tests are that check.
"""

from __future__ import annotations

import re

from _fakes import write_results_tree
from sxl.config import REPORT_LOW_SUPPORT, REPORT_NOISE_F1
from sxl.report import (
    README_TEMPLATE,
    build_headline,
    build_readme,
    display_context,
    facts,
    load_results,
    template_placeholders,
)


def _facts(tmp_path):
    return facts(load_results(write_results_tree(tmp_path)))


def test_every_template_placeholder_exists_in_the_facts_file(tmp_path):
    """A missing key must fail here, not render as a literal `{p50_ms}` for a reader."""
    context = display_context(_facts(tmp_path))
    missing = template_placeholders() - set(context)
    assert missing == set(), f"template interpolates keys the facts file lacks: {sorted(missing)}"


def test_the_facts_file_carries_every_scalar_the_spec_names(tmp_path):
    f = _facts(tmp_path)

    for arm in ("base_fewshot", "base_fewshot_constrained", "lora_ft", "teacher"):
        for key in (
            "schema_valid_rate",
            "macro_f1",
            "delta_vs_teacher",
            "p50_ms",
            "amortized_ms_per_doc",
            "cost_per_1k_docs_usd",
            "cost_per_1k_docs_usd_batch1",
        ):
            assert key in f[arm], f"{arm}.{key}"

    for key in (
        "n_train",
        "n_eval_gold",
        "spend_usd",
        "trainable_pct",
        "peak_vram_gb",
        "train_runtime_s",
        "gpu_name",
        "gpu_hourly_usd",
        "macro_f1_null_baseline",
        "lowest_teacher_agreement",
        "gold_reviewer",
        "git_sha",
    ):
        assert key in f, key


def test_the_three_lowest_teacher_agreement_fields_are_recorded_lowest_first(tmp_path):
    """Limitation 1 quotes these, and quoting the *best* three would invert it."""
    f = _facts(tmp_path)
    assert [e["field"] for e in f["lowest_teacher_agreement"]] == [
        "posting_date",
        "company",
        "title",
    ]
    assert "`posting_date` 0.200" in f["lowest_teacher_agreement_str"]


def test_p50_and_amortized_are_stored_as_separate_facts(tmp_path):
    """The Verify step of the F8 spec asserts exactly this: columns not conflated."""
    lora = _facts(tmp_path)["lora_ft"]
    assert lora["p50_ms"] != lora["amortized_ms_per_doc"]
    assert lora["cost_per_1k_docs_usd"] != lora["cost_per_1k_docs_usd_batch1"]


def test_the_template_embeds_no_measurement_of_its_own():
    """The mechanical form of "no number is typed into markdown by hand".

    Digits that are not measurements — SPEC section references, `sm_75`,
    `Qwen3-1.7B`, list markers — are fine. A digit carrying a unit is not: those
    must arrive through a placeholder, so after the placeholders are stripped
    there must be no unit-bearing number left anywhere in the template.
    """
    stripped = re.sub(r"\{[^}]*\}", "", README_TEMPLATE)
    claims = re.findall(r"(?<![\w.])(\d+(?:\.\d+)?)\s*(%|ms\b|F1\b)", stripped)
    claims += [(m, "$") for m in re.findall(r"\$\s*(\d+(?:\.\d+)?)", stripped)]
    assert claims == [], f"literal measurements in README_TEMPLATE: {claims}"


def test_the_readme_states_all_six_limitations_with_their_numbers(tmp_path):
    """SPEC §7's definition of done: the measured numbers *and* the caveats."""
    results = load_results(write_results_tree(tmp_path))
    f = facts(results)
    readme = build_readme(f, build_headline(results, footer=False))
    body = readme[readme.index("## Limitations") :]

    for n in range(1, 7):
        assert f"\n{n}. " in body, f"limitation {n} is missing"

    # 1: teacher labels are not ground truth, with the weakest fields named.
    assert "not ground truth" in body and "`posting_date` 0.200" in body
    # 2: one domain, one seed, one configuration.
    assert "SEED = 1337" in body and "confidence intervals" in body
    # 3: the gold set is model-verified, with the audit's own error rate.
    assert "model_verified" in body
    assert "28-document sample" in body and "25.0%" in body
    assert "inter-annotator agreement" in body
    # 4: single GPU, fp16, and why vLLM was not used.
    assert "no vLLM" in body and "sm_75" in body
    assert "Unsloth" in body and "GGUF" in body
    assert "throughput would be substantially better" in body
    # 5: n=300 and the noise floor.
    assert "300 documents" in body
    assert f"{REPORT_NOISE_F1:.3f}" in body and f"{REPORT_LOW_SUPPORT}" in body
    # 6: the cost column is an assumption.
    assert "$0.36/h" in body and "standard API rates" in body


def test_the_readme_names_the_teacher_without_claiming_a_size_ratio(tmp_path):
    """SPEC §1.1 twice over: the right model name, and no `Nx larger` claim."""
    results = load_results(write_results_tree(tmp_path))
    readme = build_readme(facts(results), build_headline(results, footer=False))

    assert "gpt-4o-mini" in readme
    assert "claude-sonnet" not in readme, "specs/F8-report.md names the wrong teacher"
    assert not re.search(r"\d+\s*[x×]\s*larger", readme)
    assert "frontier" not in readme.lower()


def test_the_readme_explains_that_constraining_buys_validity_not_accuracy(tmp_path):
    """SPEC §3.6 calls this the most interesting result in the project."""
    results = load_results(write_results_tree(tmp_path))
    readme = build_readme(facts(results), build_headline(results, footer=False))

    assert "Constrained decoding buys validity, not accuracy" in readme
    assert "Validity is\nessentially free." in readme or "essentially free" in readme
