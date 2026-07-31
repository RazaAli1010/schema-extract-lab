"""Placeholder — the teacher labeling pipeline is implemented in F2 (SPEC §6.5).

This module exists in F0 only because the mandatory laptop-safety test
(SPEC §9.2, `tests/test_no_torch_import.py`) imports `sxl.teacher` by name. It
must stay importable and torch-free.

F2 implements: Anthropic Batch API labeling, caching and resume by `doc_id` (a
crashed run must never re-bill), and the hard `MAX_TEACHER_SPEND_USD` stop.
"""

from __future__ import annotations

__all__: list[str] = []
