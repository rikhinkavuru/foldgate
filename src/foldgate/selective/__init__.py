"""foldgate.selective -- the accept/abstain gate and selective-prediction metrics.

Risk-coverage curve, AURC, per-stratum conditional coverage, bootstrap CIs.
Baseline to beat: native-confidence thresholding.
"""

from .metrics import (
    aurc,
    bootstrap_ci,
    conditional_coverage,
    evaluate_gate,
    risk_coverage_curve,
)

__all__ = [
    "risk_coverage_curve",
    "aurc",
    "evaluate_gate",
    "conditional_coverage",
    "bootstrap_ci",
]
