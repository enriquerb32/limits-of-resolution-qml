.PHONY: all install dev lint format typecheck test test-fast test-quantum check figs trace traceability paper clean

PY ?= python

all: lint typecheck test-fast

install:
	$(PY) -m pip install -e .

dev:
	$(PY) -m pip install -e ".[dev]"
	pre-commit install

lint:
	ruff check .
	ruff format --check src tests

format:
	ruff check --fix .
	ruff format src tests

typecheck:
	mypy

# Library unit tests and regressions against archived results (seconds, NumPy/SciPy only)
test-fast:
	$(PY) -m pytest tests/unit tests/regression src --doctest-modules --cov --cov-fail-under=100

# Variational-circuit physics and gradient checks (needs PennyLane and PyTorch)
test-quantum:
	$(PY) -m pytest -m quantum

# Everything, including slow end-to-end reproduction checks
test:
	$(PY) -m pytest

figs:
	$(PY) scripts/make_paper_figs.py

trace: traceability
traceability:
	$(PY) scripts/build_traceability.py

check: figs traceability
	$(PY) scripts/make_partition_table.py
	$(PY) scripts/verify_parameter_counts.py
	@echo "All reproduction checks passed successfully."

paper:
	cd paper && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode supp.tex

clean:
	rm -rf out/ .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
