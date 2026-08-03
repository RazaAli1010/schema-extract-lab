"""LoRA fine-tuning of the student model (F6). **Kaggle only for `train`.**

Every `torch` / `transformers` / `trl` / `peft` / `datasets` import lives *inside*
a function body, so this module stays importable on the 8 GB laptop with no torch
installed (SPEC §2.1). `datasets` is on that list for a second reason as well:
`huggingface_hub` freezes its cache path into module constants at import time, so
an early import would pin the cache before the notebook has set `HF_HOME`.

The split mirrors `runner.py`, and for the same reason. Everything that encodes a
contract -- the training text, the split-isolation assertion, checkpoint
resolution -- is torch-free and unit-tested on the laptop with a fake tokenizer.
Only `train` itself, which is a thin arrangement of trl objects, is untestable
off-GPU.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from sxl.config import (
    ADAPTER_DIR,
    BASE_MODEL,
    CHECKPOINT_DIR,
    CHECKPOINT_MIRROR_DIR,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    LORA_TARGET_MODULES,
    SEED,
    TRAIN_BATCH_SIZE,
    TRAIN_EPOCHS,
    TRAIN_EVAL_STEPS,
    TRAIN_GRAD_ACCUM,
    TRAIN_LEN_SAMPLE,
    TRAIN_LOGGING_STEPS,
    TRAIN_LR,
    TRAIN_MAX_GRAD_NORM,
    TRAIN_MAX_LENGTH,
    TRAIN_MAX_TRAINABLE_PCT,
    TRAIN_MIN_TRAINABLE_PCT,
    TRAIN_SAVE_TOTAL_LIMIT,
    TRAIN_STATS_PATH,
    TRAIN_WARMUP_RATIO,
    git_sha,
    utc_now,
)
from sxl.gpu.runner import render_prompt
from sxl.io import read_jsonl, write_json
from sxl.prompts import build_ft_prompt
from sxl.schema import FIELD_NAMES, validate_prediction

#: `results/train_stats.json`, in order (F6 §Scope-6). Written as a literal for
#: the same reason `runner.RECORD_KEYS` is: the key order is the code.
STATS_KEYS = (
    "base_model",
    "adapter_repo",
    "n_train",
    "n_dev",
    "epochs",
    "lora_r",
    "trainable_params",
    "trainable_pct",
    "final_train_loss",
    "best_eval_loss",
    "train_runtime_s",
    "gpu_name",
    "n_gpus",
    "dtype",
    "peak_vram_gb",
    "generated_at",
    "git_sha",
    # Beyond the spec's list. Each one answers a question that would otherwise
    # need the notebook output, which is not committed: was this the 4-bit
    # fallback, did the targets fit, and how many optimizer steps actually ran.
    "load_in_4bit",
    "max_length",
    "token_p95",
    "effective_batch_size",
    "n_steps",
    "lr",
    "seed",
)

_CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")


class TrainError(RuntimeError):
    """A contract violation in the fine-tune. Fatal, never warned (SPEC §5.5)."""


# --- torch-free: the tested core ---------------------------------------------


def render_example(row: dict[str, Any], tok) -> str:
    """One training row as a single `"text"` string: inference prefix, target, EOS.

    Built as `render_prompt(...) + target` rather than by handing the assistant
    turn to `apply_chat_template`, because Qwen3's template renders those two
    situations **differently**: an assistant message inside the message list gets
    no thinking block, while `add_generation_prompt=True, enable_thinking=False`
    opens the turn *and* emits an empty `<think></think>`. Rendering training data
    the message-list way would teach the model to answer at a position it never
    occupies at inference -- a mismatch with no symptom except a mysteriously bad
    arm. Concatenation makes the prefix byte-identical **by construction**, which
    is what `tests/test_render_example.py` asserts.

    The target is compact JSON with `sort_keys=False`, so the key order is
    `FIELD_NAMES` order (SPEC §3.2). The model learns one fixed key order, which
    is free accuracy at inference.
    """
    gold = row["gold"]
    if tuple(gold) != FIELD_NAMES:
        raise TrainError(
            f"{row.get('doc_id')}: gold keys are not FIELD_NAMES order: {tuple(gold)}. "
            "Training on drifted keys teaches the model the wrong output shape."
        )

    prefix = render_prompt(tok, build_ft_prompt(row["text"]))
    eos = getattr(tok, "eos_token", None) or "<|im_end|>"
    if prefix.endswith(eos):
        raise TrainError(
            "the rendered prompt already ends with EOS -- the chat template changed and "
            "appending a target here would train the model to answer after the end of turn."
        )
    target = json.dumps(gold, ensure_ascii=False, separators=(",", ":"))
    return f"{prefix}{target}{eos}"


def assert_no_leakage(
    train_rows: Sequence[dict[str, Any]],
    dev_rows: Sequence[dict[str, Any]],
    gold_rows: Sequence[dict[str, Any]],
) -> None:
    """Split isolation (SPEC §3.4) and target validity, before the first step.

    Two different contract violations, one function, because both share a
    property: discovering either one *after* a three-hour run is the worst
    outcome available in this project (F6 §Scope-3).

    - Any `eval_gold` doc_id appearing in train or dev is test-set contamination,
      and every number downstream of it is worthless.
    - Any row whose `gold` fails `validate_prediction` teaches the model to emit
      malformed JSON. That surfaces much later as a low `schema_valid_rate` which
      looks exactly like a decoding problem and is not one.
    """
    gold_ids = {row["doc_id"] for row in gold_rows}

    for name, rows in (("train", train_rows), ("dev", dev_rows)):
        leaked = sorted(gold_ids.intersection(row["doc_id"] for row in rows))
        if leaked:
            raise TrainError(
                f"split leakage: {len(leaked)} eval_gold doc_id(s) present in {name}: "
                f"{_sample(leaked)}. SPEC §3.4 requires the splits to be disjoint; "
                "fix the splits rather than the assertion."
            )

    seen: set[str] = set()
    repeated: set[str] = set()
    for row in train_rows:
        (repeated if row["doc_id"] in seen else seen).add(row["doc_id"])
    if repeated:
        dupes = sorted(repeated)
        raise TrainError(f"train has {len(dupes)} duplicate doc_id(s): {_sample(dupes)}")

    for name, rows in (("train", train_rows), ("dev", dev_rows)):
        bad = sorted(r["doc_id"] for r in rows if validate_prediction(r.get("gold")) is None)
        if bad:
            raise TrainError(
                f"{len(bad)} {name} row(s) have a `gold` that fails the schema: {_sample(bad)}. "
                "Training on malformed targets teaches the model to emit malformed JSON."
            )


def _sample(items: Sequence[str], n: int = 10) -> str:
    head = list(items[:n])
    return ", ".join(head) + (f" (+{len(items) - n} more)" if len(items) > n else "")


def token_length_stats(
    rows: Sequence[dict[str, Any]],
    tok,
    sample: int = TRAIN_LEN_SAMPLE,
) -> dict[str, int]:
    """p50/p95/max token length of the rendered examples over a sample of `rows`.

    Answers the one question that decides `TRAIN_MAX_LENGTH`: if p95 exceeds the
    cap, the fix is to raise the cap and halve the batch, **not** to let the
    trainer truncate -- a truncated JSON target teaches the model to emit
    truncated JSON (F6 §Implementation notes).

    The sample is a stride, not a draw: reproducible without a seed, and it spans
    the whole file rather than its first `n` rows, which on a `doc_id`-sorted file
    would be a biased slice.
    """
    if not rows:
        return {"n_sampled": 0, "p50": 0, "p95": 0, "max": 0}

    stride = max(1, len(rows) // max(sample, 1))
    picked = list(rows[::stride])[:sample]
    lengths = sorted(len(tok(render_example(row, tok))["input_ids"]) for row in picked)
    return {
        "n_sampled": len(lengths),
        "p50": _percentile(lengths, 0.50),
        "p95": _percentile(lengths, 0.95),
        "max": lengths[-1],
    }


def _percentile(sorted_values: Sequence[int], q: float) -> int:
    """Nearest-rank percentile. No numpy -- `metrics.py` sets that precedent."""
    if not sorted_values:
        return 0
    idx = min(len(sorted_values) - 1, max(0, math.ceil(q * len(sorted_values)) - 1))
    return int(sorted_values[idx])


def effective_batch(batch_size: int, grad_accum: int, n_gpu: int = 1, world_size: int = 1) -> int:
    """The real number of examples behind one optimizer step.

    `n_gpu * world_size` covers both shapes without double-counting: DataParallel
    is one process over several cards (`n_gpu=2, world_size=1`), DDP is several
    processes over one card each (`n_gpu=1, world_size=2`).

    Recorded rather than assumed because Kaggle's `GPU T4 x2` makes both cards
    visible and `Trainer` wraps the model in DataParallel on its own initiative.
    A run that silently trains at batch 32 while `train_stats.json` claims 16 is
    not reproducible, and the discrepancy is invisible in the loss curve.
    """
    return batch_size * grad_accum * max(1, n_gpu) * max(1, world_size)


def harvest_losses(
    metrics: dict[str, Any],
    log_history: Sequence[dict[str, Any]],
    best_metric: float | None = None,
) -> tuple[float, float]:
    """`(final_train_loss, best_eval_loss)` from what the Trainer reports.

    `metrics["train_loss"]` first, `log_history` only as the fallback. The other
    order looks equivalent and is not: `log_history` gains a `"loss"` entry every
    `logging_steps` steps, so a run shorter than that -- exactly the 5-step smoke
    run this project mandates -- logs nothing at all, and scanning it first yields
    NaN for a perfectly healthy run. `train()` then refuses to write the adapter
    because it believes fp16 diverged. `metrics` is populated on every run.

    `best_metric` is only set when `load_best_model_at_end` is on, which the smoke
    path turns off, so `eval_loss` falls back to the minimum seen in the history.
    """
    final_train_loss = metrics.get("train_loss")
    if final_train_loss is None:
        final_train_loss = next(
            (entry["loss"] for entry in reversed(log_history) if "loss" in entry), float("nan")
        )

    best_eval_loss = best_metric
    if best_eval_loss is None:
        eval_losses = [e["eval_loss"] for e in log_history if "eval_loss" in e]
        if not eval_losses:
            eval_losses = [metrics["eval_loss"]] if "eval_loss" in metrics else []
        best_eval_loss = min(eval_losses) if eval_losses else float("nan")

    return float(final_train_loss), float(best_eval_loss)


def newest_checkpoint(directory: Path | str) -> Path | None:
    """The highest-numbered `checkpoint-N` in `directory`, or None.

    By parsed step number, never lexically or by mtime: sorted as strings
    `checkpoint-900` beats `checkpoint-1000`, and mtimes are equal often enough
    on a copy to be unreliable.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return None
    found = [
        (int(m.group(1)), child)
        for child in directory.iterdir()
        if child.is_dir() and (m := _CHECKPOINT_RE.match(child.name))
    ]
    return max(found)[1] if found else None


def resolve_resume(
    value: str,
    output_dir: Path | str,
    mirror_dir: Path | str | None = None,
) -> str | None:
    """Turn `--resume-from-checkpoint` into a path Trainer accepts, or None.

    `"auto"` prefers a live checkpoint in `output_dir`, then falls back to the
    persisted mirror -- which is the whole point of the mirror, since `/kaggle/tmp`
    does not survive the session that wrote it (SPEC §2.2). The mirror is staged
    back into `output_dir` first, because Trainer resumes relative to its own
    output directory.
    """
    value = (value or "").strip()
    if not value or value.lower() == "none":
        return None

    output_dir = Path(output_dir)
    if value.lower() != "auto":
        path = Path(value)
        if not path.is_dir():
            raise TrainError(f"--resume-from-checkpoint {value} is not a directory")
        return str(path)

    live = newest_checkpoint(output_dir)
    if live is not None:
        return str(live)

    mirrored = newest_checkpoint(mirror_dir) if mirror_dir else None
    if mirrored is None:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    staged = output_dir / mirrored.name
    if not staged.is_dir():
        shutil.copytree(mirrored, staged)
    return str(staged)


def write_train_stats(stats: dict[str, Any], path: Path | str = TRAIN_STATS_PATH) -> Path:
    """Write `results/train_stats.json` with the keys in `STATS_KEYS` order."""
    missing = [k for k in STATS_KEYS if k not in stats]
    if missing:  # pragma: no cover - structural guard
        raise TrainError(f"train_stats is missing keys: {missing}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, {k: stats[k] for k in STATS_KEYS}, sort_keys=False)
    return path


# --- torch from here ----------------------------------------------------------


def build_dataset(rows: Sequence[dict[str, Any]], tok):
    """A `datasets.Dataset` with the single `"text"` column `SFTConfig` expects."""
    from datasets import Dataset  # inside: HF cache path freeze + laptop safety

    return Dataset.from_list([{"text": render_example(row, tok)} for row in rows])


def _mirror_callback(dest: Path):
    """A `TrainerCallback` copying the newest checkpoint into `dest` on every save.

    `/kaggle/tmp` is ~60 GB but is **not** persisted (SPEC §2.2): when the session
    dies the checkpoints die with it. `/kaggle/working` is the notebook's saved
    output, so a LoRA checkpoint mirrored there (tens of MB, optimizer state
    included) is what makes `--resume-from-checkpoint auto` work across sessions.

    Copy to staging, then swap, so a session killed mid-copy leaves the previous
    good mirror in place rather than a half-written directory that a later resume
    would choke on.
    """
    from transformers import TrainerCallback

    dest = Path(dest)

    class MirrorLatestCheckpoint(TrainerCallback):
        def on_save(self, args, state, control, **_kw):
            src = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            if not src.is_dir():
                return control
            staging = dest.with_name(dest.name + ".staging")
            shutil.rmtree(staging, ignore_errors=True)
            shutil.copytree(src, staging)
            shutil.rmtree(dest, ignore_errors=True)
            os.replace(staging, dest)
            return control

    return MirrorLatestCheckpoint()


def _build_trainer(model_arg, args, ds_train, ds_dev, tok, peft_config):
    """The single trl touch point.

    A separate module-level function purely so `tests/test_train_guards.py` can
    monkeypatch it and prove `train()` raises *before* a trainer is ever
    constructed -- an assertion that would be untestable if this were inline.
    """
    from trl import SFTTrainer

    return SFTTrainer(
        model=model_arg,
        args=args,
        train_dataset=ds_train,
        eval_dataset=ds_dev,
        processing_class=tok,  # trl 1.x: `tokenizer=` is gone (SPEC §6.3)
        peft_config=peft_config,  # plain model + peft_config, never both wrapped
    )


def _quantized_model(model_id: str):
    """The documented OOM fallback (SPEC §5.3), built here rather than by trl.

    Where `quantization_config` belongs in trl 1.5.1 is not something the laptop
    can verify, so this path builds the model itself and hands the object over --
    which works whichever kwarg trl accepts. `bnb_4bit_compute_dtype` is
    **float16**, not bfloat16: a T4 has no bf16 (SPEC §2.3). transformers v5 also
    removed the `load_in_4bit=True` shortcut on `from_pretrained` (SPEC §6.2).
    """
    import torch
    from peft import prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb,
        dtype=torch.float16,
        attn_implementation="sdpa",
        device_map={"": 0},
    )
    return prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)


def _upcast_trainable_params(model) -> None:
    """Put the LoRA parameters in fp32.

    With fp16 base weights the adapters are created in fp16 too, and `Trainer`'s
    `GradScaler.unscale_()` refuses fp16 gradients outright ("Attempting to
    unscale FP16 gradients"). Upcasting only the trainable tensors costs a few MB
    and is what every working fp16-LoRA recipe does; `prepare_model_for_kbit_training`
    already handles it on the 4-bit path, and doing it twice is harmless.
    """
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.float()


def train(
    *,
    train_path: Path | str,
    dev_path: Path | str,
    gold_path: Path | str,
    model_id: str = BASE_MODEL,
    output_dir: Path | str = CHECKPOINT_DIR,
    adapter_dir: Path | str = ADAPTER_DIR,
    mirror_dir: Path | str | None = CHECKPOINT_MIRROR_DIR,
    stats_path: Path | str = TRAIN_STATS_PATH,
    epochs: int = TRAIN_EPOCHS,
    lr: float = TRAIN_LR,
    max_length: int = TRAIN_MAX_LENGTH,
    batch_size: int = TRAIN_BATCH_SIZE,
    grad_accum: int = TRAIN_GRAD_ACCUM,
    limit: int = 0,
    max_steps: int = 0,
    resume_from_checkpoint: str = "",
    load_in_4bit: bool = False,
    push_to: str = "",
    seed: int = SEED,
    echo: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Fine-tune a LoRA adapter and write `results/train_stats.json`.

    The order of the body is the contract: every guard that can fire before a GPU
    is touched fires before a GPU is touched, and nothing here imports torch until
    after `assert_no_leakage` has passed.
    """
    train_rows = list(read_jsonl(Path(train_path)))
    dev_rows = list(read_jsonl(Path(dev_path)))
    gold_rows = list(read_jsonl(Path(gold_path)))

    # First, on the FULL splits -- `--limit` is a smoke-run convenience and must
    # not be able to hide contamination by slicing it out of view.
    assert_no_leakage(train_rows, dev_rows, gold_rows)
    echo(
        f"[train] no leakage: {len(train_rows)} train / {len(dev_rows)} dev / {len(gold_rows)} gold"
    )

    smoke = max_steps > 0
    if limit:
        train_rows = train_rows[:limit]
        if smoke:
            # 300 evals every few steps costs more than the smoke run itself.
            dev_rows = dev_rows[: max(8, limit // 4)]
    if not train_rows:
        raise TrainError(f"{train_path} has no rows")

    import torch
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import SFTConfig

    tok = AutoTokenizer.from_pretrained(model_id)
    # NOT `padding_side = "left"`. `runner.load_model` sets that for batched
    # *generation*; using it here would left-pad the training labels and teach the
    # model to predict pad tokens.
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    len_stats = token_length_stats(train_rows, tok)
    echo(f"[train] token length over {len_stats['n_sampled']} rows: {len_stats}")
    if len_stats["p95"] > max_length:
        echo(
            f"[train] WARNING p95 {len_stats['p95']} > max_length {max_length}: targets will be "
            "truncated. Re-run with --max-length 3072 --batch-size 1. A truncated JSON target "
            "teaches the model to emit truncated JSON."
        )

    ds_train = build_dataset(train_rows, tok)
    ds_dev = build_dataset(dev_rows, tok)
    # Printed so the saved notebook output carries proof of what was trained on.
    sample = ds_train[0]["text"]
    echo(f"[train] example head: {sample[:220]!r}")
    echo(f"[train] example tail: {sample[-160:]!r}")

    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=list(LORA_TARGET_MODULES),
        bias="none",
        task_type="CAUSAL_LM",
    )

    # With `--max-steps 5` there is no eval and no checkpoint, so
    # `load_best_model_at_end` would have no best checkpoint to load and Trainer
    # would raise at the end of a run that otherwise succeeded.
    eval_steps = min(TRAIN_EVAL_STEPS, max_steps) if smoke else TRAIN_EVAL_STEPS
    args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        max_steps=max_steps or -1,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=TRAIN_WARMUP_RATIO,
        max_grad_norm=TRAIN_MAX_GRAD_NORM,
        fp16=True,
        bf16=False,  # T4 is compute capability 7.5: no bf16 (SPEC §2.3)
        max_length=max_length,  # trl 1.x: `max_seq_length` is gone (SPEC §6.3)
        dataset_text_field="text",
        gradient_checkpointing=True,
        # Reentrant checkpointing over a frozen base model is the classic
        # "none of the inputs have requires_grad" crash.
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=TRAIN_LOGGING_STEPS,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=TRAIN_SAVE_TOTAL_LIMIT,
        load_best_model_at_end=not smoke,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=seed,
        data_seed=seed,
        report_to=[],
        # transformers v5 defaults `dtype` to "auto", which hands a T4 whatever
        # precision the checkpoint was saved in. `fp16=True` above is autocast,
        # not weight dtype -- these are different things (SPEC §6.2).
        model_init_kwargs=(
            None
            if load_in_4bit
            else {
                "dtype": "float16",
                "attn_implementation": "sdpa",
                "device_map": {"": 0},
            }
        ),
    )

    model_arg = _quantized_model(model_id) if load_in_4bit else model_id
    trainer = _build_trainer(model_arg, args, ds_train, ds_dev, tok, peft_config)
    _upcast_trainable_params(trainer.model)

    trainable, total = trainer.model.get_nb_trainable_parameters()
    trainable_pct = 100.0 * trainable / max(total, 1)
    echo(f"[train] trainable {trainable:,} / {total:,} = {trainable_pct:.3f}%")
    if not TRAIN_MIN_TRAINABLE_PCT < trainable_pct < TRAIN_MAX_TRAINABLE_PCT:
        raise TrainError(
            f"trainable_pct {trainable_pct:.3f}% is outside "
            f"({TRAIN_MIN_TRAINABLE_PCT}, {TRAIN_MAX_TRAINABLE_PCT}). Near 100% means "
            "`peft_config` never attached and this run is full-fine-tuning a 1.7B model "
            "on a T4 -- it would take days and OOM first."
        )

    if mirror_dir:
        trainer.add_callback(_mirror_callback(Path(mirror_dir)))

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    resume = resolve_resume(resume_from_checkpoint, output_dir, mirror_dir)
    if resume:
        echo(f"[train] resuming from {resume}")
    result = trainer.train(resume_from_checkpoint=resume)

    final_train_loss, best_eval_loss = harvest_losses(
        result.metrics, trainer.state.log_history, trainer.state.best_metric
    )

    # From the Trainer, not from our own arguments: on Kaggle's `GPU T4 x2` both
    # cards are visible and Trainer wraps the model in DataParallel unasked, which
    # doubles the examples per optimizer step. Reading it back is the difference
    # between recording what ran and recording what we asked for.
    n_gpu = max(1, int(getattr(trainer.args, "n_gpu", 1)))
    world_size = max(1, int(getattr(trainer.args, "world_size", 1)))
    n_gpus = n_gpu * world_size
    batch_total = effective_batch(batch_size, grad_accum, n_gpu, world_size)
    if n_gpus > 1:
        echo(
            f"[train] {n_gpus} GPUs in use -> effective batch {batch_total}, not "
            f"{batch_size * grad_accum}. Recorded in train_stats.json; F7's latency "
            "numbers are single-T4 and are unaffected."
        )

    # A NaN loss under fp16 is the documented T4 hazard (SPEC §2.3). Refuse to
    # produce an adapter from it: a silently broken adapter costs a full inference
    # run to discover.
    if not math.isfinite(float(final_train_loss)):
        raise TrainError(
            f"final train loss is {final_train_loss} -- fp16 training diverged. Confirm "
            "bf16=False, keep max_grad_norm=0.3, then retry with --lr 1e-4 and only then "
            "--load-in-4bit. Never switch to bf16: the T4 cannot do it."
        )

    adapter_dir = Path(adapter_dir)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(adapter_dir))
    tok.save_pretrained(str(adapter_dir))
    echo(f"[train] adapter saved to {adapter_dir}")

    if push_to:
        echo(f"[train] published {publish(adapter_dir, push_to)}")

    stats = {
        "base_model": model_id,
        "adapter_repo": push_to or "",
        "n_train": len(train_rows),
        "n_dev": len(dev_rows),
        "epochs": epochs,
        "lora_r": LORA_R,
        "trainable_params": int(trainable),
        "trainable_pct": round(trainable_pct, 4),
        "final_train_loss": round(float(final_train_loss), 6),
        "best_eval_loss": round(float(best_eval_loss), 6),
        "train_runtime_s": round(float(result.metrics.get("train_runtime", 0.0)), 2),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "n_gpus": n_gpus,
        # float16 even on the 4-bit path: it is the compute dtype, and
        # `load_in_4bit` below carries the quantization fact separately.
        "dtype": "float16",
        "peak_vram_gb": round(
            (torch.cuda.max_memory_allocated() / 1024**3) if torch.cuda.is_available() else 0.0, 3
        ),
        "generated_at": utc_now(),
        "git_sha": git_sha(),
        "load_in_4bit": load_in_4bit,
        "max_length": max_length,
        "token_p95": len_stats["p95"],
        "effective_batch_size": batch_total,
        "n_steps": int(trainer.state.global_step),
        "lr": lr,
        "seed": seed,
    }
    write_train_stats(stats, stats_path)
    echo(f"[train] wrote {stats_path}")
    return stats


def publish(adapter_dir: Path | str, repo_id: str, *, token: str | None = None) -> str:
    """Upload the saved adapter directory to the Hugging Face Hub.

    The Hub is this project's artifact store; the adapter is never committed to
    git (SPEC §2.4). `upload_folder` over a `save_pretrained` directory is exactly
    what `PeftModel.from_pretrained(base, repo_id)` consumes, and unlike
    `model.push_to_hub` it does not need the live model still resident on a GPU
    that is about to run inference.
    """
    if "<" in repo_id or "/" not in repo_id:
        raise TrainError(
            f"{repo_id!r} is not a Hub repo id. config.ADAPTER_REPO ships as a placeholder -- "
            "set it, or pass --push-to <hf_user>/<repo>."
        )
    adapter_dir = Path(adapter_dir)
    if not adapter_dir.is_dir():
        raise TrainError(f"{adapter_dir} does not exist; nothing to publish")

    from huggingface_hub import HfApi

    api = HfApi(token=token or os.environ.get("HF_TOKEN"))
    api.create_repo(repo_id, exist_ok=True, repo_type="model")
    api.upload_folder(folder_path=str(adapter_dir), repo_id=repo_id, repo_type="model")
    return f"https://huggingface.co/{repo_id}"
