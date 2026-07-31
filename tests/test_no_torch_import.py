"""SPEC §9.2 — non-negotiable. The laptop never acquires a GPU dependency.

Verbatim from the spec, plus a check that `sxl.gpu` itself is importable without
torch (F5/F6/F7 must keep their torch imports inside function bodies).
"""

import subprocess
import sys
import textwrap


def test_importing_sxl_does_not_import_torch():
    code = textwrap.dedent("""
        import sys
        import sxl, sxl.cli, sxl.corpus, sxl.metrics, sxl.schema, sxl.teacher
        assert "torch" not in sys.modules, sorted(m for m in sys.modules if "torch" in m)
    """)
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


def test_importing_sxl_corpus_does_not_import_datasets():
    """F1's `datasets` import must stay inside `fetch_raw`.

    Not style: `huggingface_hub` freezes the cache path into module constants at
    import time, so hoisting the import would make `_set_hf_cache()` a silent
    no-op and fill `~/.cache/huggingface` on a laptop with 5 GB free (SPEC §2.1).
    """
    code = textwrap.dedent("""
        import sys
        import sxl.corpus, sxl.cli
        assert "datasets" not in sys.modules, sorted(
            m for m in sys.modules if m.startswith(("datasets", "huggingface_hub"))
        )
    """)
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


def test_importing_the_gpu_package_does_not_import_torch():
    code = textwrap.dedent("""
        import sys
        import sxl.gpu
        assert "torch" not in sys.modules, sorted(m for m in sys.modules if "torch" in m)
    """)
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


def test_torch_is_not_installed_at_all():
    """The dev environment must never have had `.[gpu]` installed (SPEC §2.1)."""
    import importlib.util

    assert importlib.util.find_spec("torch") is None, "torch is installed on the laptop"
