"""foldgate.selective -- the accept/abstain gate and selective-prediction metrics.

Risk-coverage curve, AURC, per-stratum conditional coverage, bootstrap CIs.
Baseline to beat: native-confidence thresholding.
"""

from .enrichment import (
    active_retention_curve,
    bedroc,
    enrichment_factor,
    log_auc,
    random_abstention_ef,
    roc_auc,
    selective_enrichment_curve,
)
from .metrics import (
    aurc,
    bootstrap_ci,
    clopper_pearson,
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
    "clopper_pearson",
    "enrichment_factor",
    "bedroc",
    "roc_auc",
    "log_auc",
    "selective_enrichment_curve",
    "active_retention_curve",
    "random_abstention_ef",
]
