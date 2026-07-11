"""foldgate.conformal -- coverage/risk guarantees, robust to training-similarity shift.

Now: distribution-free selective risk control (RCPS, `risk.py`). Next: weighted
covariate-shift CP (crepes-weighted) and Mondrian group-conditional (crepes),
plus the thin wrapper that fuses them keyed on training-set similarity.
Every method should report where the guarantee becomes vacuous.
"""

from .risk import (
    continuous_risk_threshold,
    hb_upper_bound,
    ltt_threshold,
    naive_threshold,
    rcps_threshold,
)
from .weighted import estimate_weights, weighted_threshold

__all__ = [
    "ltt_threshold",
    "rcps_threshold",
    "naive_threshold",
    "continuous_risk_threshold",
    "hb_upper_bound",
    "estimate_weights",
    "weighted_threshold",
]
