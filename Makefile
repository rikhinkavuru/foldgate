# Single documented entrypoint set. Prefer `pixi run <task>` (see pixi.toml);
# these targets wrap the same commands for non-pixi users.
.PHONY: help setup download features experiments figures paper test lint clean repro
PY ?= PYTHONPATH=src .venv/bin/python

help:
	@echo "Targets:"
	@echo "  setup        create the venv + install deps (uv)"
	@echo "  download     fetch the RNP tabular artifacts into data/raw (~52 MB, no GPU)"
	@echo "  features     build the processed delivered-pose table from RNP"
	@echo "  experiments  run E1-E21 (writes results/*.json + results/figures/*.png)"
	@echo "  figures      (re)build the summary figures from results/*.json"
	@echo "  paper        build the MoML short-paper PDF"
	@echo "  test         run the test suite"
	@echo "  lint         ruff check src/ experiments/ tests/"
	@echo "  repro        download -> features -> experiments -> figures -> test"

setup:
	uv venv --python 3.12 .venv
	uv pip install --python .venv/bin/python numpy pandas scikit-learn matplotlib scipy pyarrow crepes crepes-weighted pytest ruff

download:
	$(PY) scripts/download_data.py

features:
	$(PY) -m experiments.build_features

# W1: cross-model + intra-model pose-agreement features from the 39.5 GB structure tarball.
# Download prediction_files.tar.gz (Zenodo 18366081) into data/raw first; see docs/DATA_CARD.md.
pose-features:
	$(PY) -m experiments.build_pose_features
	$(PY) -m experiments.build_features   # re-join pose features into the delivered table

experiments: features
	$(PY) -m experiments.e1_iid_validity
	$(PY) -m experiments.e2_exchangeability_break
	$(PY) -m experiments.e3_shift_repair
	$(PY) -m experiments.e3b_weighted_repair
	$(PY) -m experiments.e3c_combined_conditional
	$(PY) -m experiments.e4_selective_utility
	$(PY) -m experiments.e5_ablations
	$(PY) -m experiments.e6_downstream
	$(PY) -m experiments.e6b_interaction_recovery
	$(PY) -m experiments.e7_shift_axes
	$(PY) -m experiments.e8_interface_task
	$(PY) -m experiments.e9_continuous_risk
	$(PY) -m experiments.e10_foldbench
	$(PY) -m experiments.e11_baselines
	$(PY) -m experiments.e12_reliability_drift
	$(PY) -m experiments.e13_loto_validity
	$(PY) -m experiments.e14_disagreement_strata
	$(PY) -m experiments.e15_foldbench_transfer
	$(PY) -m experiments.e16_selective_screening
	$(PY) -m experiments.e17_worst_subpop
	$(PY) -m experiments.e18_localized
	$(PY) -m experiments.e19_shift_decomp
	$(PY) -m experiments.e20_screening_broad
	$(PY) -m experiments.e21_affinity_selective
	$(MAKE) figures

figures:
	$(PY) -m experiments.make_figures

paper:
	$(PY) paper/build_pdf.py paper/moml2026_shortpaper.md

test:
	$(PY) -m pytest -q

lint:
	.venv/bin/ruff check src/ experiments/ tests/

repro: download experiments test
	@echo "reproduction complete: results/*.json + results/figures/*.png"

clean:
	rm -rf results/checkpoints results/**/*.pkl
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
