"""Weighted conformal selective-risk control under covariate shift.

Group-conditional calibration (Mondrian) needs labels in the deployment stratum.
Weighted conformal instead reweights the *labelled source* (calibration) points
by the likelihood ratio  w(x) = p_target(x) / p_source(x)  so that a threshold
certified on the source is valid on the (unlabelled) target -- the harder, more
useful setting when you cannot label the novel-chemotype regime you deploy on
(Tibshirani, Foygel Barber, Candes, Ramdas 2019).

The likelihood ratio is estimated by a probabilistic source-vs-target classifier
on the novelty features:  w = c/(1-c) * (n_source/n_target).  Because the weights
are estimated, the resulting coverage is approximate -- weight-estimation error
is the known Achilles heel -- so we (a) report sensitivity to the weight model
and (b) keep group-conditional conformal as the rigorous fallback.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


def estimate_weights(
    cal_features: np.ndarray,
    target_features: np.ndarray,
    clip: float = 20.0,
) -> np.ndarray:
    """Likelihood-ratio weights for calibration points under covariate shift.

    Fits a logistic source(0)-vs-target(1) classifier on the novelty features and
    returns w_i = c_i/(1-c_i) * (n_cal/n_target), clipped for stability.
    """
    cal_features = np.asarray(cal_features, dtype=float)
    target_features = np.asarray(target_features, dtype=float)
    if cal_features.ndim == 1:
        cal_features = cal_features[:, None]
        target_features = target_features[:, None]

    X = np.vstack([cal_features, target_features])
    z = np.r_[np.zeros(len(cal_features)), np.ones(len(target_features))]
    keep = ~np.isnan(X).any(axis=1)
    clf = LogisticRegression(max_iter=1000).fit(X[keep], z[keep])

    c = clf.predict_proba(np.nan_to_num(cal_features, nan=np.nanmedian(cal_features)))[:, 1]
    c = np.clip(c, 1e-6, 1 - 1e-6)
    w = (c / (1 - c)) * (len(cal_features) / max(len(target_features), 1))
    return np.clip(w, 1.0 / clip, clip)


def _effective_n(w: np.ndarray) -> float:
    s = w.sum()
    return float(s * s / np.sum(w * w)) if np.sum(w * w) > 0 else 0.0


def weighted_threshold(
    scores: np.ndarray,
    correct: np.ndarray,
    weights: np.ndarray,
    alpha: float,
    delta: float = 0.1,
    coverage_grid: np.ndarray | None = None,
    min_eff: float = 20.0,
    finite_sample_slack: bool = False,
) -> float | None:
    """Largest accept set whose weighted selective risk is <= alpha on the target.

    Default (finite_sample_slack=False) is the standard weighted-conformal plug-in:
    the reweighted empirical risk estimates the target risk and we certify when it
    is <= alpha. This gives approximate (not finite-sample-delta) coverage, which is
    the accepted trade-off for weighted conformal under estimated weights; the
    finite-sample guarantee is provided instead by group-conditional calibration.
    Set finite_sample_slack=True to add a conservative Hoeffding slack on the
    effective sample size.
    """
    scores = np.asarray(scores, dtype=float)
    err = 1 - np.asarray(correct, dtype=int)
    w = np.asarray(weights, dtype=float)
    if coverage_grid is None:
        coverage_grid = np.arange(0.05, 1.0 + 1e-9, 0.01)

    last_valid = None
    for c in coverage_grid:
        tau = float(np.quantile(scores, np.clip(1.0 - c, 0.0, 1.0)))
        acc = scores >= tau
        if not acc.any():
            continue
        w_acc, err_acc = w[acc], err[acc]
        denom = w_acc.sum()
        n_eff = _effective_n(w_acc)
        if denom <= 0 or n_eff < min_eff:
            continue
        r_w = float((w_acc * err_acc).sum() / denom)
        est = r_w + (np.sqrt(np.log(1.0 / delta) / (2.0 * n_eff)) if finite_sample_slack else 0.0)
        if est <= alpha:
            last_valid = tau
        else:
            break
    return last_valid
