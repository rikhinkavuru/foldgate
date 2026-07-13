"""Weighted conformal selective-risk control under covariate shift.

Group-conditional calibration (Mondrian) needs labels in the deployment stratum.
Weighted conformal instead reweights the *labelled source* (calibration) points by
the likelihood ratio  w(x) = p_target(x) / p_source(x)  so that a threshold certified
on the source is valid on the (unlabelled) target -- the harder, more useful setting
when you cannot label the novel-chemotype regime you deploy on (Tibshirani, Foygel
Barber, Candes, Ramdas 2019).

Two certifiers live here:

* ``weighted_threshold`` -- the plug-in Hajek estimator of the target selective risk.
  Fast, but only *approximate* coverage (no finite-sample 1 - delta), because the
  reweighted empirical mean is a point estimate.

* ``weighted_ltt_threshold`` -- an importance-weighted Learn-then-Test gate. It tests
  the reweighted-source null  E_source[w (L - alpha) 1_accept] >= 0, which equals the
  target null  R_target(tau) >= alpha  under covariate shift, with a WSR betting
  p-value on the rescaled weighted losses (Almeida et al. 2025, "High Probability Risk
  Control Under Covariate Shift", PMLR v266). Fixed-sequence testing over the coverage
  grid then gives an *exact finite-sample*  P(R_target(tau_hat) <= alpha) >= 1 - delta,
  the risk-control analogue of Tibshirani-2019 -- but *conditional on the weights being
  correct*. Under estimated weights the guarantee is exact only up to weight-estimation
  error, so we (a) estimate weights out-of-fold with a calibrated classifier, (b) report
  n_eff and a weight-model sensitivity sweep, (c) test the covariate-shift assumption
  itself (``concept_shift_diagnostic``), and (d) keep group-conditional conformal as the
  rigorous fallback when overlap is poor.
"""

from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from .risk import wsr_betting_pvalue


def estimate_weights(
    cal_features: np.ndarray,
    target_features: np.ndarray,
    clip: float = 20.0,
) -> np.ndarray:
    """Likelihood-ratio weights for calibration points under covariate shift.

    Fits a logistic source(0)-vs-target(1) classifier on the novelty features and
    returns w_i = c_i/(1-c_i) * (n_cal/n_target), clipped for stability. In-sample
    variant kept for backward compatibility; prefer ``estimate_weights_cv``, which
    cross-fits and probability-calibrates the classifier so the weights are unbiased.
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


def estimate_weights_cv(
    cal_features: np.ndarray,
    target_features: np.ndarray,
    clip: float = 20.0,
    n_splits: int = 5,
    method: str = "isotonic",
    seed: int = 0,
) -> np.ndarray:
    """Cross-fitted, probability-calibrated likelihood-ratio weights.

    The density ratio  w = p_target / p_source  is estimated from a source-vs-target
    classifier as  w = c/(1-c) * (n_cal/n_target). Two correctness requirements the
    in-sample estimator misses:

      1. **Out-of-fold** -- each calibration point's weight is predicted by a classifier
         that never saw it, so downstream weighted risk is not optimistically biased by
         the classifier overfitting the calibration set.
      2. **Calibrated probabilities** -- ``CalibratedClassifierCV`` (isotonic/sigmoid)
         makes c a genuine probability; an uncalibrated classifier yields biased weights.

    Falls back to a single calibrated fit if a novelty stratum is too thin to K-fold.
    """
    cal = np.asarray(cal_features, dtype=float)
    tgt = np.asarray(target_features, dtype=float)
    if cal.ndim == 1:
        cal, tgt = cal[:, None], tgt[:, None]
    med = np.nanmedian(np.vstack([cal, tgt]), axis=0)
    cal = np.where(np.isnan(cal), med, cal)
    tgt = np.where(np.isnan(tgt), med, tgt)

    X = np.vstack([cal, tgt])
    z = np.r_[np.zeros(len(cal)), np.ones(len(tgt))].astype(int)
    ratio = len(cal) / max(len(tgt), 1)
    c_cal = np.full(len(cal), 0.5)

    n_splits = int(min(n_splits, np.bincount(z).min()))
    base = LogisticRegression(max_iter=1000)
    if n_splits < 2:
        clf = CalibratedClassifierCV(base, method=method, cv=3).fit(X, z)
        c_cal = clf.predict_proba(cal)[:, 1]
    else:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for train_idx, test_idx in skf.split(X, z):
            clf = CalibratedClassifierCV(base, method=method, cv=3)
            clf.fit(X[train_idx], z[train_idx])
            # score only the held-out calibration rows (z == 0) in this fold
            cal_mask = test_idx < len(cal)
            rows = test_idx[cal_mask]
            c_cal[rows] = clf.predict_proba(X[test_idx][cal_mask])[:, 1]

    c_cal = np.clip(c_cal, 1e-6, 1 - 1e-6)
    w = (c_cal / (1 - c_cal)) * ratio
    return np.clip(w, 1.0 / clip, clip)


def effective_n(w: np.ndarray) -> float:
    """Kish effective sample size  (sum w)^2 / sum w^2  of a weight vector."""
    w = np.asarray(w, dtype=float)
    s2 = float(np.sum(w * w))
    return float(w.sum() ** 2 / s2) if s2 > 0 else 0.0


_effective_n = effective_n  # internal alias


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
    the reweighted empirical risk estimates the target risk and we certify when it is
    <= alpha. This gives approximate (not finite-sample-delta) coverage, the accepted
    trade-off for a point estimate under estimated weights. For the finite-sample
    certificate use ``weighted_ltt_threshold``; the rigorous label-based fallback is
    group-conditional calibration.
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
        n_eff = effective_n(w_acc)
        if denom <= 0 or n_eff < min_eff:
            continue
        r_w = float((w_acc * err_acc).sum() / denom)
        est = r_w + (np.sqrt(np.log(1.0 / delta) / (2.0 * n_eff)) if finite_sample_slack else 0.0)
        if est <= alpha:
            last_valid = tau
        else:
            break
    return last_valid


def weighted_ltt_threshold(
    scores: np.ndarray,
    correct: np.ndarray,
    weights: np.ndarray,
    alpha: float,
    delta: float = 0.1,
    clip_ceiling: float = 20.0,
    coverage_grid: np.ndarray | None = None,
    min_eff: float = 20.0,
) -> float | None:
    """Importance-weighted Learn-then-Test gate with a finite-sample certificate.

    For each candidate accept set A(tau) we test  H0: R_target(tau) >= alpha. Under
    covariate shift  R_target(tau) >= alpha  <=>  E_{source|A}[w (L - alpha)] >= 0. We
    rescale the accepted weighted losses  g_i = w_i (L_i - alpha) in [-B*alpha, B*(1-alpha)]
    to  h_i = (g_i + B*alpha)/B in [0,1]  (B = clip_ceiling = max possible weight), for
    which  E[h] >= alpha  <=>  E[g] >= 0. A WSR betting p-value on {h_i} at target alpha
    then certifies R_target(tau) < alpha at level delta; the fixed-sequence walk over the
    pre-specified coverage grid controls the family-wise error, giving
    P(R_target(tau_hat) <= alpha) >= 1 - delta -- exact and finite-sample, *conditional on
    the weights*. Heavy clipping (large B) flattens h and weakens power, the honest cost
    of the finite-sample guarantee; when it certifies nothing, use the Mondrian fallback.

    Returns tau (accept iff score >= tau), or None if nothing is certifiable.
    """
    scores = np.asarray(scores, dtype=float)
    err = 1 - np.asarray(correct, dtype=int)
    w = np.asarray(weights, dtype=float)
    B = float(clip_ceiling)
    if coverage_grid is None:
        coverage_grid = np.arange(0.05, 1.0 + 1e-9, 0.01)

    last_valid = None
    for c in coverage_grid:
        tau = float(np.quantile(scores, np.clip(1.0 - c, 0.0, 1.0)))
        acc = scores >= tau
        if not acc.any():
            continue
        w_acc, err_acc = w[acc], err[acc]
        if effective_n(w_acc) < min_eff:
            continue
        g = w_acc * (err_acc - alpha)
        h = np.clip((g + B * alpha) / B, 0.0, 1.0)
        if wsr_betting_pvalue(h, alpha, delta) <= delta:
            last_valid = tau
        else:
            break
    return last_valid


def concept_shift_diagnostic(
    conf_source: np.ndarray,
    correct_source: np.ndarray,
    conf_target: np.ndarray,
    correct_target: np.ndarray,
    n_bins: int = 5,
) -> dict:
    """Test the covariate-shift assumption: does P(correct | confidence) move?

    Weighted conformal is exact only under *pure* covariate shift, where the confidence
    -> correctness map P(Y | s) is stable and only the marginal P(s) changes. If the
    novel regime also degrades the map (concept / label shift), reweighting controls an
    aligned distribution and the true target risk can exceed alpha. We bin confidence and
    compare per-bin correctness rates between source and target; a large gap flags concept
    shift and argues for the group-conditional certificate instead. Returns per-bin rates,
    the max absolute gap, and a weighted mean gap.
    """
    cs = np.asarray(conf_source, dtype=float)
    ys = np.asarray(correct_source, dtype=int)
    ct = np.asarray(conf_target, dtype=float)
    yt = np.asarray(correct_target, dtype=int)
    edges = np.quantile(np.concatenate([cs, ct]), np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    bins = []
    gaps, wts = [], []
    for lo, hi in zip(edges[:-1], edges[1:], strict=False):
        ms = (cs >= lo) & (cs < hi)
        mt = (ct >= lo) & (ct < hi)
        ps = float(ys[ms].mean()) if ms.any() else float("nan")
        pt = float(yt[mt].mean()) if mt.any() else float("nan")
        gap = abs(ps - pt) if (ms.any() and mt.any()) else float("nan")
        bins.append({"lo": float(lo), "hi": float(hi), "n_source": int(ms.sum()),
                     "n_target": int(mt.sum()), "p_correct_source": ps,
                     "p_correct_target": pt, "abs_gap": gap})
        if np.isfinite(gap):
            gaps.append(gap)
            wts.append(int(mt.sum()))
    max_gap = float(np.nanmax(gaps)) if gaps else float("nan")
    mean_gap = float(np.average(gaps, weights=wts)) if gaps else float("nan")
    return {"bins": bins, "max_abs_gap": max_gap, "mean_abs_gap_target_weighted": mean_gap}
