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

# ---- multiplicity-control constants (see docs/theory/MULTIPLICITY_SPEC.md) ----
# K is the number of co-folding methods carried through every "all models" claim
# (af3, boltz1, boltz1x, chai, protenix in the current RNP delivery). At load time
# it also equals len(methods_with_enough(df)); we pin it as a constant so scripts
# that import _common share one definition of the family size.
K = 5
DELTA_JOINT = DELTA / K   # Bonferroni union-bound per-model level for a JOINT certificate (=0.02)


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


# =============================================================================
# Multiplicity control for the 5-model claim families
# =============================================================================
# The paper makes two statistically opposite kinds of "all five models" claims and
# each needs its own control (docs/theory/MULTIPLICITY_SPEC.md):
#
#   * Certificate / validity claims ("realized risk <= alpha with prob >= 1-delta")
#     are for-all upper bounds whose violations accumulate. A JOINT guarantee splits
#     delta by the union bound: certify each model at DELTA_JOINT = DELTA / K = 0.02.
#     No step-down helps a required conjunction of certificates, so Bonferroni is
#     both correct and, at K=5, negligibly conservative.
#
#   * Significance / discovery claims ("effect > 0") are there-exists rejections:
#       - "the effect holds for EVERY model" is an intersection-union test
#         (Berger 1982): require each per-model p <= level and the joint Type-I
#         error is already controlled at level, with no penalty. Use iut_all.
#       - "drift is present SOMEWHERE in this grid" inflates Type-I and needs FWER
#         (holm / romano_wolf_stepdown) or FDR (bh).
#
#   * A "flat" verdict is a non-rejection and cannot prove a null. Upgrade it to a
#     TOST equivalence test against a practical margin (tost_equivalence).


def holm(pvals) -> np.ndarray:
    """Holm (1979) step-down FWER-adjusted p-values.

    Holm, S. (1979). A Simple Sequentially Rejective Multiple Test Procedure.
    Scand. J. Statist. 6:65-70.

    Given m raw p-values, sort ascending p_(1) <= ... <= p_(m) and set the adjusted
    value of the j-th smallest to  max_{i<=j} min( (m - i + 1) * p_(i), 1 ),  where
    the running maximum enforces monotonicity along the sorted order. Reject a
    hypothesis at family-wise level `level` iff its adjusted p-value is <= level.
    Holm controls the FWER at `level` under arbitrary dependence and is uniformly
    more powerful than plain Bonferroni. Use it for a per-model "which models show
    the effect" discovery table, not for a required conjunction (that is an IUT).

    Returns adjusted p-values in the ORIGINAL input order.
    """
    p = np.asarray(pvals, dtype=float).ravel()
    m = p.size
    if m == 0:
        return p.copy()
    order = np.argsort(p, kind="mergesort")          # ascending, stable
    factors = np.arange(m, 0, -1, dtype=float)        # m, m-1, ..., 1
    stepped = np.maximum.accumulate(factors * p[order])
    stepped = np.minimum(stepped, 1.0)
    out = np.empty(m, dtype=float)
    out[order] = stepped
    return out


def bh(pvals, q: float = DELTA) -> np.ndarray:
    """Benjamini-Hochberg (1995) FDR step-up rejection mask at level `q`.

    Benjamini, Y. & Hochberg, Y. (1995). Controlling the False Discovery Rate: a
    Practical and Powerful Approach to Multiple Testing. JRSS-B 57:289-300.

    Sort p ascending p_(1) <= ... <= p_(m), find the largest k with
    p_(k) <= (k / m) * q, and reject every hypothesis with p <= p_(k). This controls
    the expected proportion of false discoveries among rejections at q under
    independence or positive dependence (PRDS), so the shared-reference E12 grid
    needs no Benjamini-Yekutieli log-factor. FDR is the right control when the claim
    is the collective "drift is elevated on structural novelty" rather than a
    per-cell guarantee.

    Returns a boolean rejection mask in the ORIGINAL input order.
    """
    p = np.asarray(pvals, dtype=float).ravel()
    m = p.size
    if m == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p, kind="mergesort")
    sorted_p = p[order]
    thresh = (np.arange(1, m + 1, dtype=float) / m) * q
    below = sorted_p <= thresh
    reject = np.zeros(m, dtype=bool)
    if below.any():
        kmax = np.max(np.where(below)[0])             # largest passing rank (0-indexed)
        crit = sorted_p[kmax]
        reject[order[: kmax + 1]] = True              # step-up: reject all up to crit
        # ties at the critical value are all rejected (crit is p_(k))
        reject[p <= crit] = True
    return reject


def romano_wolf_stepdown(boot_matrix, stat_hat, se, level: float = DELTA) -> dict:
    """Romano-Wolf (2005) step-down max-t bootstrap: dependence-aware FWER control.

    Romano, J.P. & Wolf, M. (2005). Stepwise Multiple Testing as Formalized Data
    Snooping. Econometrica 73:1237-1282.

    One-sided test per cell of  H0: theta_cell <= 0  against  H1: theta_cell > 0
    (positive reliability drift). Inputs:
      boot_matrix : (S cells, B reps) bootstrap draws of the per-cell estimate,
      stat_hat    : (S,) observed per-cell point estimates,
      se          : (S,) per-cell standard errors (bootstrap std is the natural choice).

    Studentize each cell,  t_cell = stat_hat / se, and build the null distribution by
    RECENTERING every bootstrap draw at its own estimate,
    t*_{cell,b} = (boot_{cell,b} - stat_hat_cell) / se_cell.  Order cells by observed t
    descending. Step down: at step j take the max of t* over the still-active cells
    {(j), (j+1), ...} per bootstrap rep, and read the adjusted p-value of cell (j) as
    the tail mass of that max distribution at or above t_(j); a running maximum keeps
    the adjusted p-values monotone along the order. Because the max is taken over the
    SAME resampled complexes, the procedure captures the strong positive dependence in
    the E12 grid (cells share the S0 reference and the underlying complexes) and is
    uniformly at least as powerful as Holm or Bonferroni, with equality only under
    independence. Under the complete null with subset pivotality this controls the
    FWER at `level`.

    Returns a dict with:
      t_stat       : (S,) studentized statistics,
      p_adjusted   : (S,) step-down FWER-adjusted one-sided p-values, original order,
      reject       : (S,) bool, p_adjusted <= level,
      crit_value   : single-step (1-level) quantile of the full-family max-t null
                     distribution, a studentized simultaneous critical value; the
                     conservative simultaneous band is stat_hat +/- crit_value * se,
                     and a cell clears the one-sided band when t_stat >= crit_value,
      level        : the FWER level used.
    """
    boot = np.asarray(boot_matrix, dtype=float)
    stat = np.asarray(stat_hat, dtype=float).ravel()
    se = np.asarray(se, dtype=float).ravel()
    if boot.ndim != 2:
        raise ValueError("boot_matrix must be 2-D (cells x B)")
    S, B = boot.shape
    if stat.size != S or se.size != S:
        raise ValueError("stat_hat and se must have length = number of cells (rows of boot_matrix)")
    if S == 0:
        return {"t_stat": np.zeros(0), "p_adjusted": np.zeros(0),
                "reject": np.zeros(0, dtype=bool), "crit_value": float("nan"), "level": level}

    se_safe = np.where(se > 0, se, np.finfo(float).tiny)
    t_obs = stat / se_safe
    tstar = (boot - stat[:, None]) / se_safe[:, None]          # (S, B) recentered null

    order = np.argsort(-t_obs, kind="mergesort")               # most to least significant
    adj_sorted = np.empty(S, dtype=float)
    running = 0.0
    for j in range(S):
        active = order[j:]
        maxnull = tstar[active, :].max(axis=0)                 # (B,) max-t over active cells
        raw = (1.0 + np.count_nonzero(maxnull >= t_obs[order[j]])) / (1.0 + B)
        running = max(running, raw)                            # step-down monotonicity
        adj_sorted[j] = running

    p_adjusted = np.empty(S, dtype=float)
    p_adjusted[order] = adj_sorted

    maxnull_all = tstar.max(axis=0)                            # single-step full-family max-t
    crit_value = float(np.quantile(maxnull_all, 1.0 - level))
    return {
        "t_stat": t_obs,
        "p_adjusted": p_adjusted,
        "reject": p_adjusted <= level,
        "crit_value": crit_value,
        "level": float(level),
    }


def iut_all(pvals, level: float = DELTA) -> bool:
    """Intersection-union test for the conjunction "the effect holds for ALL cells".

    Berger, R.L. (1982). Multiparameter Hypothesis Testing and Acceptance Sampling.
    Technometrics 24:295-300.

    To reject the union null (claim every alternative holds), require each per-cell
    p-value to individually reject at `level`. The IUT of a conjunction is valid at
    `level` with NO multiplicity penalty: max_c p_c <= level already controls the
    joint Type-I error at level. This is the correct read of an "excludes zero for
    every model" / "for all five models" claim.

    Returns True iff max(pvals) <= level (an empty family returns True, vacuously).
    """
    p = np.asarray(pvals, dtype=float).ravel()
    if p.size == 0:
        return True
    return bool(np.nanmax(p) <= level)


def tost_equivalence(boot_or_samples, margin: float, alpha: float = 0.05) -> bool:
    """Two-one-sided-tests (TOST) equivalence against a symmetric +/- margin.

    Schuirmann (1987); the equivalence-testing standard for turning a "flat" or "no
    effect" verdict into a positive claim, since failing to reject a null never
    proves it. Given a bootstrap / resample distribution of the point estimate
    theta_hat (each entry an estimate of the same scalar), reject the non-equivalence
    null  |theta| >= margin  at level `alpha` when BOTH one-sided tests reject, i.e.
    when the (1 - 2*alpha) percentile interval of the bootstrap lies strictly inside
    (-margin, +margin). With alpha = 0.05 this is the [5, 95] percentile interval
    contained in +/- margin, matching the one-sided 0.05 convention used elsewhere.
    A wide interval fails even when centered at zero, which is the point: equivalence
    requires the estimate to be both near zero and precise.

    Returns True iff the equivalence null is rejected (the effect is practically zero).
    """
    x = np.asarray(boot_or_samples, dtype=float).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0 or not (margin > 0):
        return False
    lo = float(np.quantile(x, alpha))
    hi = float(np.quantile(x, 1.0 - alpha))
    return bool(lo > -margin and hi < margin)
