"""TORCH-ONLY package. Never imported on the development laptop (SPEC §2.1).

Nothing may be imported at package-import time — this file must stay free of any
`torch`, `transformers`, `peft`, `trl`, or `outlines` import, and so must every
`from sxl.gpu import ...` outside a function body. `sxl.cli` imports the
submodules here **lazily, inside the command body**, so that `sxl --help` works on
a machine with no torch installed. `tests/test_no_torch_import.py` enforces this.

Submodules (added by their owning features, all Kaggle-only):
    runner.py       F5/F6  batched generate()
    constrained.py  F5     outlines wrapper
    train_lora.py   F6     LoRA fine-tune
    bench.py        F7     latency / throughput / cost
"""
