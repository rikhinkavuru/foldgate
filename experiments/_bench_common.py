"""Shared config + helpers for the b1-b4 synthetic theorem-validation scripts.

These sit next to experiments/_common.py (which is RNP-specific) but stay
self-contained: the synthetic benchmark needs only foldgate.bench.{synth,
certificates} and torch-free numpy/scipy. JSON lands in results/bench/, small
PNGs in results/figures/bench/. Every run logs the resolved config + git SHA, to
match the repo's reproducibility convention.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

RESBENCH = ROOT / "results" / "bench"
FIGBENCH = ROOT / "results" / "figures" / "bench"

ALPHA = 0.20          # max tolerated error rate among accepted (repo default)
DELTA = 0.10          # certificate failure probability
N_SEEDS = 300         # seeds per config (BENCHMARK_SPEC 2.5)


def git_sha() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT))
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(obj)
    payload.setdefault("meta", {})
    payload["meta"].update({"git_sha": git_sha()})
    path.write_text(json.dumps(payload, indent=2, default=float))


def bootstrap_ci(vals, ci: float = 0.95, n_boot: int = 2000, seed: int = 0):
    """Percentile bootstrap CI on the mean of a 1-D sample."""
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (float("nan"), float("nan"))
    g = np.random.default_rng(seed)
    means = v[g.integers(0, v.size, size=(n_boot, v.size))].mean(axis=1)
    lo = float(np.quantile(means, (1 - ci) / 2))
    hi = float(np.quantile(means, 1 - (1 - ci) / 2))
    return lo, hi
