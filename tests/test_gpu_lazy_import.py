"""The `sxl.gpu` modules must stay importable with no torch installed.

Extends SPEC §9.2 to the modules F5 added. `tests/test_no_torch_import.py` proves
the *package* `sxl.gpu` is clean; this proves its submodules are, which is what
lets `tests/test_runner_records.py` exercise `predict_arm` on the laptop at all.

Each check runs in a subprocess: once torch is in `sys.modules` for this
interpreter, an in-process assertion proves nothing about import order.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

MODULES = ["sxl.gpu", "sxl.gpu.runner", "sxl.gpu.constrained", "sxl.gpu.train_lora"]


def run_snippet(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize("module", MODULES)
def test_importing_a_gpu_module_does_not_import_torch(module):
    result = run_snippet(f"""
        import sys
        import {module}
        assert "torch" not in sys.modules, sorted(m for m in sys.modules if "torch" in m)
        assert "transformers" not in sys.modules
        assert "peft" not in sys.modules
        assert "outlines" not in sys.modules
        assert "trl" not in sys.modules
        # `datasets` for a second reason beyond weight: huggingface_hub freezes
        # its cache path into module constants at import time, so importing it
        # before the notebook sets HF_HOME pins the cache to the wrong volume.
        assert "datasets" not in sys.modules
    """)
    assert result.returncode == 0, result.stderr


def test_importing_the_cli_with_the_gpu_subapp_registered_does_not_import_torch():
    # The F5 command body holds `from sxl.gpu.runner import ...`; typer registering
    # it must not execute that import (SPEC §2.1, §4).
    result = run_snippet("""
        import sys
        import sxl.cli
        assert "torch" not in sys.modules, sorted(m for m in sys.modules if "torch" in m)
        assert "sxl.gpu.runner" not in sys.modules
    """)
    assert result.returncode == 0, result.stderr


def test_gpu_predict_help_works_without_torch():
    # `sxl --help` and `sxl gpu predict --help` are the laptop's only legitimate
    # contact with this command; both must exit 0 with no torch present.
    result = run_snippet("""
        import sys
        from typer.testing import CliRunner
        from sxl.cli import app

        out = CliRunner().invoke(app, ["gpu", "predict", "--help"])
        assert out.exit_code == 0, out.output
        assert "torch" not in sys.modules
    """)
    assert result.returncode == 0, result.stderr


def test_gpu_train_help_works_without_torch():
    result = run_snippet("""
        import sys
        from typer.testing import CliRunner
        from sxl.cli import app

        out = CliRunner().invoke(app, ["gpu", "train", "--help"])
        assert out.exit_code == 0, out.output
        assert "torch" not in sys.modules
        assert "trl" not in sys.modules
    """)
    assert result.returncode == 0, result.stderr
