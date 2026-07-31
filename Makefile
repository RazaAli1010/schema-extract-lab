.PHONY: install test lint fmt

# NEVER add the gpu extra here — SPEC §2.1: the laptop must not acquire torch.
install:
	uv pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check src tests

fmt:
	ruff format src tests
