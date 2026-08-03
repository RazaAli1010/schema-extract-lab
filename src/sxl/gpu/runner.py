"""Batched generation and prediction-record assembly (F5, reused verbatim by F6).

**Kaggle only for anything that touches a model.** Every torch / transformers /
peft import lives *inside* a function body, so this module stays importable on
the 8 GB laptop with no torch installed (SPEC §2.1). That is not cosmetic:
`predict_arm` -- the part with all the contract logic in it -- is unit-tested on
the laptop by injecting a fake `generate_fn`, and it can only be tested there if
importing this file is free.

The split is deliberate:

- `load_model` / `generate_batch` are thin, torch-y, and untestable off-GPU.
- `predict_arm` is pure Python over an injected `generate_fn`, and owns every
  rule the downstream metrics depend on: the SPEC §3.3 record shape, `parsed`
  and `schema_valid` coming from one code path, invalid predictions being
  *written* rather than dropped, `<think>` being fatal, and checkpoint/resume.

F6 adds no generation code. It calls `load_model(..., adapter=...)` and this same
`predict_arm`.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sxl.config import (
    GEN_BATCH_SIZE,
    MAX_NEW_TOKENS,
    PREDICTIONS_DIR,
)
from sxl.io import append_jsonl, read_jsonl, write_jsonl
from sxl.prompts import build_student_prompt
from sxl.schema import validate_prediction
from sxl.teacher import extract_json

#: SPEC §3.3, in order. Written as a literal so the key order is the code, not a
#: convention someone has to remember; `_record` asserts against it.
RECORD_KEYS = (
    "doc_id",
    "arm",
    "raw_output",
    "parsed",
    "schema_valid",
    "latency_ms",
    "prompt_tokens",
    "completion_tokens",
)

#: One document's prompt, as chat messages.
Messages = Sequence[dict[str, str]]

#: A `generate_fn` maps a batch of message-lists to one result dict per item,
#: carrying `raw_output`, `prompt_tokens`, `completion_tokens`, `latency_ms`.
#:
#: Messages cross this boundary rather than rendered strings because the chat
#: template belongs to the tokenizer, which lives on the GPU side of the split --
#: that keeps `enable_thinking=False` (SPEC §3.7) in exactly one place and lets a
#: laptop test drive `predict_arm` with a fake that never sees a tokenizer.
#: `make_generate_fn` and `constrained.build_constrained_generator` both produce
#: one of these, which is why `predict_arm` needs no idea which arm it is running.
GenerateFn = Callable[[Sequence[Messages]], list[dict[str, Any]]]

#: `predict_arm`'s prompt builder: document text -> chat messages. F5 passes
#: `build_student_prompt` bound to the frozen exemplars; F6's `lora_ft` arm passes
#: its own minimal, shot-free builder and reuses everything else in this file.
PromptFn = Callable[[str], Messages]


class RunnerError(RuntimeError):
    """A contract violation during a prediction run. Fatal, never warned (SPEC §5.5)."""


@dataclass(frozen=True)
class PredictPaths:
    """Where a prediction run reads and writes. Injectable, like `TeacherPaths`.

    On Kaggle the package is a pip install, so `config.ROOT` points into
    site-packages and the defaults below are wrong -- the notebook passes explicit
    paths for all three (see the note in `config.py`).
    """

    predictions: Path

    @classmethod
    def default(cls) -> PredictPaths:
        return cls(predictions=PREDICTIONS_DIR)

    @classmethod
    def in_dir(cls, root: Path) -> PredictPaths:
        return cls(predictions=Path(root) / "predictions")

    def pred_for(self, arm: str) -> Path:
        """The final file F4 scores. Matches `metrics.MetricsPaths.pred_for`."""
        return self.predictions / f"{arm}.jsonl"


def partial_path(out_path: Path) -> Path:
    """The checkpoint file for `out_path`.

    Built with `with_name`, not `with_suffix`: `Path("base_fewshot.jsonl")`
    .with_suffix(".partial.jsonl") is a ValueError-free but wrong-looking
    round-trip, and being explicit here keeps `.partial.jsonl` a single concept.
    """
    out_path = Path(out_path)
    return out_path.with_name(f"{out_path.stem}.partial.jsonl")


# --- torch-only from here to `predict_arm` -----------------------------------


def load_model(model_id: str, adapter: str | None = None):
    """Load the student model and tokenizer for single-T4 fp16 inference.

    Every constant here is a T4 (sm_75) constraint from SPEC §2.3, not a
    preference:

    - `dtype=torch.float16` -- Turing has no bfloat16. transformers v5 renamed the
      argument from `torch_dtype`, and `dtype` now defaults to `"auto"`, which
      would happily hand a T4 the bf16 weights the checkpoint was saved in.
    - `attn_implementation="sdpa"` -- FlashAttention-2 needs sm80.
    - `device_map={"": 0}` -- the accelerator is `GPU T4 x2` but the project
      standardizes on one card so latency is comparable across arms.
    - `padding_side = "left"` -- decoder-only batched generation with right
      padding produces garbage for every sequence but the longest. This is the
      single most common batched-inference bug.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16,
        attn_implementation="sdpa",
        device_map={"": 0},
    ).eval()

    if adapter:  # F6's `lora_ft` arms; unused by F5
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter).eval()

    return model, tok


def render_prompt(tok, messages: Sequence[dict[str, str]]) -> str:
    """Apply the chat template with **thinking mode off** (SPEC §3.7).

    The single call site in the project. Qwen3 is a hybrid reasoning model and at
    defaults emits `<think>...</think>` before the answer, which would destroy
    `schema_valid_rate` for reasons that have nothing to do with the model's
    extraction ability and make this baseline look artificially bad. F6 renders
    its training data through this same function so train and inference cannot
    drift apart.
    """
    return tok.apply_chat_template(
        list(messages),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def generate_batch(
    model,
    tok,
    prompts: Sequence[str],
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> list[dict[str, Any]]:
    """Greedily complete already-rendered `prompts` as one left-padded batch.

    `do_sample=False` **alone**: passing `temperature=0.0` beside it is rejected
    by transformers. `config.TEMPERATURE = 0.0` documents the intent; it is not
    an argument to forward.

    `latency_ms` is the batch wall time divided evenly across the batch. It is
    **amortized, for debugging only**, and must never reach a results table --
    single-document p50/p95 is F7's job exclusively (SPEC §3.3).
    """
    import torch

    inputs = tok(list(prompts), return_tensors="pt", padding=True).to(model.device)
    prompt_width = inputs["input_ids"].shape[1]

    started = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
        )
    if torch.cuda.is_available():  # generate() is async; time it honestly
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    # Slice the completion off by position. Decoding the whole sequence and
    # string-stripping the prompt back off is fragile -- the template's own
    # special tokens survive the round trip -- and has bitten every project that
    # has tried it.
    completions = out[:, prompt_width:]

    results: list[dict[str, Any]] = []
    per_item_ms = elapsed_ms / max(len(prompts), 1)
    for i in range(len(prompts)):
        row = completions[i]
        # Count real tokens, not padded width: `prompt_tokens` from the attention
        # mask, `completion_tokens` up to the first pad. Otherwise every short
        # answer in the batch reports the longest answer's length.
        n_completion = int((row != tok.pad_token_id).sum().item())
        results.append(
            {
                "raw_output": tok.decode(row, skip_special_tokens=True),
                "prompt_tokens": int(inputs["attention_mask"][i].sum().item()),
                "completion_tokens": n_completion,
                "latency_ms": per_item_ms,
            }
        )
    return results


def make_generate_fn(model, tok, max_new_tokens: int = MAX_NEW_TOKENS) -> GenerateFn:
    """Bind a model/tokenizer into the `GenerateFn` `predict_arm` expects.

    This is where messages become strings, via `render_prompt` -- the one call
    site of the chat template in the unconstrained path.
    """

    def _generate(batch: Sequence[Messages]) -> list[dict[str, Any]]:
        prompts = [render_prompt(tok, messages) for messages in batch]
        return generate_batch(model, tok, prompts, max_new_tokens=max_new_tokens)

    return _generate


# --- torch-free from here: the tested core -----------------------------------


def _record(doc_id: str, arm: str, gen: dict[str, Any]) -> dict[str, Any]:
    """Assemble one SPEC §3.3 prediction record.

    `schema_valid` is `validate_prediction(...) is not None` and nothing else --
    `metrics._check_prediction_rows` re-derives it and raises if a file disagrees,
    so a local "looks like JSON to me" check here would fail scoring outright.
    """
    raw = gen["raw_output"]
    if "<think>" in raw:
        # Fail loudly rather than stripping. A `<think>` block means
        # `enable_thinking=False` did not take, which is a train/inference
        # template mismatch that F6 would silently inherit; stripping it here
        # would hide exactly the bug the check exists to catch (SPEC §3.7).
        raise RunnerError(
            f"{doc_id}: model emitted a <think> block. `enable_thinking=False` is not taking "
            "effect -- fix the chat template rather than stripping the output, or F6 inherits "
            "the mismatch."
        )

    posting = validate_prediction(extract_json(raw))
    record = {
        "doc_id": doc_id,
        "arm": arm,
        "raw_output": raw,
        "parsed": posting.model_dump(mode="json") if posting is not None else None,
        "schema_valid": posting is not None,
        "latency_ms": float(gen["latency_ms"]),
        "prompt_tokens": int(gen["prompt_tokens"]),
        "completion_tokens": int(gen["completion_tokens"]),
    }
    if tuple(record) != RECORD_KEYS:  # pragma: no cover - structural guard
        raise RunnerError(f"record key order drifted from SPEC §3.3: {tuple(record)}")
    return record


def _batched(items: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def predict_arm(
    arm: str,
    gold_rows: Sequence[dict[str, Any]],
    generate_fn: GenerateFn,
    out_path: Path,
    *,
    prompt_fn: PromptFn,
    batch_size: int = GEN_BATCH_SIZE,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> dict[str, Any]:
    """Run `arm` over `gold_rows` and write `out_path`. Resumable.

    Checkpoints to `<out>.partial.jsonl` after every batch and skips doc_ids
    already present there on start: 300 documents is ~15 minutes per arm and
    Kaggle sessions idle out without warning, so losing a completed run to a
    timeout is avoidable and therefore not acceptable.

    Returns a summary dict. `n_truncated` matters for the constrained arm: a
    grammar cannot end `required_skills` early, so truncation is a real cost of
    constrained decoding that F8 reports rather than hides (F5 §Implementation
    notes).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    partial = partial_path(out_path)

    done: dict[str, dict[str, Any]] = {}
    if partial.exists():
        for row in read_jsonl(partial):
            if row.get("arm") != arm:
                raise RunnerError(
                    f"{partial} holds arm {row.get('arm')!r} but this run is {arm!r}. "
                    "Delete the partial file or point --out somewhere else; resuming across "
                    "arms would blend two measurements into one file."
                )
            done[row["doc_id"]] = row

    todo = [row for row in gold_rows if row["doc_id"] not in done]
    for batch in _batched(todo, max(batch_size, 1)):
        generated = generate_fn([prompt_fn(row["text"]) for row in batch])
        if len(generated) != len(batch):
            raise RunnerError(
                f"generate_fn returned {len(generated)} outputs for {len(batch)} prompts"
            )
        for row, gen in zip(batch, generated, strict=True):
            record = _record(row["doc_id"], arm, gen)
            done[record["doc_id"]] = record
            append_jsonl(partial, record)

    # Emit in gold order so the file diffs cleanly and F4 sees one row per
    # eval_gold doc_id regardless of how many resumes it took to get here.
    ordered = [done[row["doc_id"]] for row in gold_rows]
    n = write_jsonl(out_path, ordered)
    partial.unlink(missing_ok=True)

    n_valid = sum(r["schema_valid"] for r in ordered)
    n_truncated = sum(r["completion_tokens"] >= max_new_tokens for r in ordered)
    return {
        "arm": arm,
        "n": n,
        "n_valid": n_valid,
        "n_invalid": n - n_valid,
        "n_truncated": n_truncated,
        "n_resumed": n - len(todo),
        "mean_completion_tokens": (
            round(sum(r["completion_tokens"] for r in ordered) / n, 1) if n else 0.0
        ),
    }


def fewshot_prompt_fn(shots: Sequence[dict[str, Any]]) -> PromptFn:
    """The `PromptFn` of both `base_fewshot` arms: the schema-in-prompt few-shot setup."""

    def _build(text: str) -> Messages:
        return build_student_prompt(text, shots)

    return _build
