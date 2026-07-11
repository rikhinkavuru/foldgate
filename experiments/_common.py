"""Shared config + loaders for the experiment scripts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PROCESSED = ROOT / "data" / "processed" / "rnp_delivered.parquet"
FIGDIR = ROOT / "results" / "figures"
RESDIR = ROOT / "results"

ALPHA = 0.20          # max tolerated error rate among accepted poses
DELTA = 0.10          # RCPS failure probability
CONF = "ranking_score"  # primary native-confidence signal
MIN_METHOD_N = 1200   # methods with at least this many delivered poses


def load_delivered() -> pd.DataFrame:
    if not PROCESSED.exists():
        raise SystemExit("run `python -m experiments.build_features` first")
    return pd.read_parquet(PROCESSED)


def methods_with_enough(df: pd.DataFrame, n: int = MIN_METHOD_N) -> list[str]:
    counts = df.groupby("method").size()
    return sorted(counts[counts >= n].index.tolist())


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float))


def rng(seed: int = 20260710) -> np.random.Generator:
    return np.random.default_rng(seed)
