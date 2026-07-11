"""Selective-prediction metrics: risk-coverage curve, AURC, gate evaluation.

Conventions: ``scores`` is confidence (higher = more likely correct);
``correct`` is 1 iff the delivered pose is within 2 A. Selective risk is the
error rate among accepted predictions; coverage is the accepted fraction.
"""

from __future__ import annotations

import numpy as np


def risk_coverage_curve(scores: np.ndarray, correct: np.ndarray):
    """Return (coverage, selective_risk) as we accept the top-k most confident."""
    scores = np.asarray(scores, dtype=float)
    err = 1 - np.asarray(correct, dtype=int)
    order = np.argsort(-scores)
    err_sorted = err[order]
    n = len(scores)
    k = np.arange(1, n + 1)
    coverage = k / n
    selective_risk = np.cumsum(err_sorted) / k
    return coverage, selective_risk


def aurc(scores: np.ndarray, correct: np.ndarray) -> float:
    """Area under the risk-coverage curve (lower = better confidence ranking)."""
    coverage, selective_risk = risk_coverage_curve(scores, correct)
    return float(np.trapezoid(selective_risk, coverage))


def evaluate_gate(scores: np.ndarray, correct: np.ndarray, tau: float | None) -> dict:
    """Realized coverage + selective risk for an accept-iff-(score>=tau) gate."""
    scores = np.asarray(scores, dtype=float)
    correct = np.asarray(correct, dtype=int)
    if tau is None:
        return {"tau": None, "coverage": 0.0, "selective_risk": float("nan"), "n_accept": 0, "n": len(scores)}
    accept = scores >= tau
    n_acc = int(accept.sum())
    risk = float(1 - correct[accept].mean()) if n_acc > 0 else float("nan")
    return {
        "tau": float(tau),
        "coverage": n_acc / len(scores),
        "selective_risk": risk,
        "n_accept": n_acc,
        "n": len(scores),
    }


def conditional_coverage(
    scores: np.ndarray, correct: np.ndarray, strata: np.ndarray, tau: float | None
) -> dict:
    """Per-stratum realized selective risk + coverage for a single global tau."""
    scores = np.asarray(scores, dtype=float)
    correct = np.asarray(correct, dtype=int)
    strata = np.asarray(strata)
    out = {}
    for g in np.unique(strata):
        m = strata == g
        out[int(g)] = evaluate_gate(scores[m], correct[m], tau)
    return out


def bootstrap_ci(
    stat_fn, *arrays, n_boot: int = 1000, ci: float = 0.90, seed: int = 0
):
    """Percentile bootstrap CI for a statistic over paired arrays."""
    rng = np.random.default_rng(seed)
    n = len(arrays[0])
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        v = stat_fn(*[np.asarray(a)[idx] for a in arrays])
        if v is not None and np.isfinite(v):
            vals.append(v)
    if not vals:
        return (float("nan"), float("nan"))
    lo = float(np.quantile(vals, (1 - ci) / 2))
    hi = float(np.quantile(vals, 1 - (1 - ci) / 2))
    return lo, hi
