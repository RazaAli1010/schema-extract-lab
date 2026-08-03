"""Schema-constrained decoding with Outlines (F5's `base_fewshot_constrained` arm).

**Kaggle only.** `import outlines` happens inside the function body so this
module stays importable on the laptop (SPEC §2.1).

This arm exists to separate two things people conflate: constrained decoding
forces `schema_valid_rate` to ~1.0 for free, but does nothing for *field
accuracy*. Showing that gap is the most interesting result in the project (SPEC
§3.6), which is why the constrained path validates through the same
`validate_prediction` as every other arm rather than assuming the grammar makes
validation redundant. A grammar guarantees well-formedness against the schema it
was built from; it does not guarantee the run wired up the schema you think it
did, and `schema_valid` must mean the same thing in all five arms.

**The API here is outlines 1.x.** Every `outlines.generate.json(...)` /
`outlines.models.transformers(...)` snippet on the internet is v0 and raises
`AttributeError` on 1.3.2 (SPEC §6.4).
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

from sxl.config import MAX_NEW_TOKENS
from sxl.gpu.runner import GenerateFn, Messages, render_prompt
from sxl.schema import JobPosting


def build_constrained_generator(
    model,
    tok,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> GenerateFn:
    """Return a `GenerateFn` that decodes under the `JobPosting` grammar.

    The index is built **once, here**, outside any loop: construction is
    per-schema and takes seconds for the 16-field object, so rebuilding it per
    document would turn a 20-minute arm into an hour-long one (F5 §Implementation
    notes).

    Generation is **sequential**, one document per call. Outlines' batched entry
    points on 1.x are less well-trodden than the single-prompt path, and a
    failure 200 documents into a run costs Kaggle GPU quota that this project
    budgets tightly (SPEC §6.6). The unconstrained arm keeps its batching; the
    constrained arm trades throughput for a path that works.

    Because there is no batch to amortize over, `latency_ms` here is genuine
    per-document wall time -- but it is still *not* a benchmark number. F7 owns
    latency exclusively, with warmup and percentiles this function does not do.
    """
    import outlines

    generator = outlines.from_transformers(model, tok)

    def _generate(batch: Sequence[Messages]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for messages in batch:
            prompt = render_prompt(tok, messages)

            started = time.perf_counter()
            # Returns a JSON *string*, not a model instance -- validating it is
            # the caller's job, and `predict_arm` does it through
            # `validate_prediction` like every other arm.
            raw = generator(prompt, JobPosting, max_new_tokens=max_new_tokens)
            elapsed_ms = (time.perf_counter() - started) * 1000.0

            results.append(
                {
                    "raw_output": raw,
                    "prompt_tokens": len(tok(prompt)["input_ids"]),
                    "completion_tokens": len(tok(raw)["input_ids"]),
                    "latency_ms": elapsed_ms,
                }
            )
        return results

    return _generate
