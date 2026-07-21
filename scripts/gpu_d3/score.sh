#!/usr/bin/env bash
# Self-score every predicted pose: extract native confidence + compute pocket-aligned,
# symmetry-corrected ligand-RMSD vs the deposited RCSB ligand. Writes the small
# results/gpu_d3/scored.csv you copy back. Uses the torch-free scorer venv.
set -euo pipefail
cd "$(dirname "$0")"
venv_score/bin/python selfscore.py \
  --manifests manifests/posebusters_v2.csv manifests/plinder_subset.csv \
  --boltz_root results/gpu_d3/boltz2 \
  --chai_root  results/gpu_d3/chai \
  --gt_cache   results/gpu_d3/ground_truth \
  --out        results/gpu_d3/scored.csv
