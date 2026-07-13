"""foldgate.conformal -- coverage/risk guarantees, robust to training-similarity shift.

Shipped in-house (no heavy conformal dependency): distribution-free selective risk
control (LTT / RCPS, `risk.py`); weighted covariate-shift CP and the concept-shift
diagnostic (`weighted.py`); group-conditional (Mondrian) calibration, applied inline in
the experiment drivers; label-free worst-subpopulation (CVaR) certificates (`robust.py`).
Every method reports where the guarantee becomes vacuous.
"""

from .localized import (
    default_bandwidth,
    kish_ess,
    localized_threshold,
    rlcp_quantile,
    sweep_error_target,
)
from .risk import (
    continuous_risk_threshold,
    hb_upper_bound,
    ltt_threshold,
    naive_threshold,
    rcps_threshold,
    wsr_betting_pvalue,
)
from .robust import (
    chi2_dro_risk_ucb,
    clopper_pearson_upper,
    cvar_binary_ucb,
    cvar_cdf_band_ucb,
    error_rate_ucb,
    robustness_radius,
    simultaneous_certificate,
    worst_subpopulation_certificate,
)
from .shift_decomp import shift_decomposition
from .weighted import (
    concept_shift_diagnostic,
    effective_n,
    estimate_weights,
    estimate_weights_cv,
    weighted_ltt_threshold,
    weighted_threshold,
)

__all__ = [
    "ltt_threshold",
    "rcps_threshold",
    "naive_threshold",
    "continuous_risk_threshold",
    "wsr_betting_pvalue",
    "hb_upper_bound",
    "estimate_weights",
    "estimate_weights_cv",
    "effective_n",
    "weighted_threshold",
    "weighted_ltt_threshold",
    "concept_shift_diagnostic",
    "worst_subpopulation_certificate",
    "cvar_binary_ucb",
    "cvar_cdf_band_ucb",
    "error_rate_ucb",
    "clopper_pearson_upper",
    "chi2_dro_risk_ucb",
    "robustness_radius",
    "simultaneous_certificate",
    "rlcp_quantile",
    "localized_threshold",
    "sweep_error_target",
    "default_bandwidth",
    "kish_ess",
    "shift_decomposition",
]
