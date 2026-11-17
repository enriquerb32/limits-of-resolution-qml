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

# Figures of the v2 manuscript: forest plot from the v2 summary, known-truth simulation (data-independent, v1 archive)
figs:
	$(PY) scripts/make_forest_plot.py
	$(PY) scripts/make_resolution_plot.py
	$(PY) scripts/make_paper_figs.py --results results_v1 --only sim

# Claims of the v2 manuscript (and of the v1 manuscript, tag v1, against results_v1/)
trace: traceability
traceability:
	$(PY) scripts/build_traceability_v2.py
	$(PY) scripts/build_traceability.py

# Every number in the main-text tables of paper/main.tex against the archived v2 results
manuscript:
	$(PY) -m pytest tests/manuscript -q

check: figs traceability manuscript
	$(PY) scripts/make_partition_table.py
	$(PY) scripts/verify_parameter_counts.py
	@echo "All reproduction checks passed successfully."

# main and supp cross-reference each other (xr), so each needs a second pass after the other
paper:
	cd paper && for i in 1 2; do pdflatex -interaction=nonstopmode -halt-on-error main && \
		pdflatex -interaction=nonstopmode -halt-on-error supp || exit 1; done

clean:
	rm -rf out/ .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
