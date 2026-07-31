"""Placeholder — the metric library is implemented in F4 (SPEC §3.5).

This module exists in F0 only because the mandatory laptop-safety test
(SPEC §9.2, `tests/test_no_torch_import.py`) imports `sxl.metrics` by name. It
must stay importable and torch-free.

F4 implements, here and nowhere else: `schema_valid_rate`, per-field TP/FP/FN,
`em`, and `macro_f1` — plain arithmetic, no numpy, no sklearn.
"""

from __future__ import annotations

__all__: list[str] = []
