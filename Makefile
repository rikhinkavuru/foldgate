# Single documented entrypoint set. Prefer `pixi run <task>` (see pixi.toml);
# these targets wrap the same commands for non-pixi users.
.PHONY: help setup features experiments paper test lint clean
PY ?= PYTHONPATH=src .venv/bin/python

help:
	@echo "Targets:"
	@echo "  setup        create the venv + install deps (uv)"
	@echo "  features     build the processed delivered-pose table from RNP"
	@echo "  experiments  run E1-E6 (writes results/*.json + results/figures/*.png)"
	@echo "  test         run the test suite"
	@echo "  lint         ruff check"

setup:
	uv venv --python 3.12 .venv
	uv pip install --python .venv/bin/python numpy pandas scikit-learn matplotlib scipy pyarrow crepes crepes-weighted pytest

features:
	$(PY) -m experiments.build_features

experiments: features
	$(PY) -m experiments.e1_iid_validity
	$(PY) -m experiments.e2_exchangeability_break
	$(PY) -m experiments.e3_shift_repair
	$(PY) -m experiments.e3b_weighted_repair
	$(PY) -m experiments.e3c_combined_conditional
	$(PY) -m experiments.e4_selective_utility
	$(PY) -m experiments.e5_ablations
	$(PY) -m experiments.e6_downstream

paper:
	$(PY) paper/build_pdf.py paper/moml2026_shortpaper.md

test:
	$(PY) -m pytest -q

lint:
	.venv/bin/ruff check src tests || true

clean:
	rm -rf results/checkpoints results/**/*.pkl
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
