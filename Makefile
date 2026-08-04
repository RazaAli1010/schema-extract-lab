.PHONY: install test lint fmt report

# NEVER add the gpu extra here — SPEC §2.1: the laptop must not acquire torch.
install:
	uv pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check src tests

fmt:
	ruff format src tests

# --strict is not optional here: it is what enforces SPEC §1.1, that no claim in
# the README outruns the committed artifacts. `--no-strict` is for local
# iteration only and must never reach CI.
report:
	sxl report build --strict
