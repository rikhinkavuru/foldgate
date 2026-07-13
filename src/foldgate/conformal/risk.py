"""Distribution-free selective risk control (RCPS) for the accept/abstain gate.

We accept a delivered pose when its confidence ``s`` clears a threshold ``tau``.
We want the *largest* accept set whose error rate (fraction with RMSD > 2 A) is
provably <= alpha. RCPS (Bates, Angelopoulos et al., JACM 2021) picks tau via a
Hoeffding-Bentkus upper confidence bound on the risk and a fixed-sequence walk,
giving  P(selective risk <= alpha) >= 1 - delta,  finite-sample, distribution-free.

The whole thesis lives here: a tau calibrated on low-novelty data is only valid
under exchangeability. Transfer it to a high-novelty test stratum and the
guarantee breaks (realized risk > alpha) -- that is E2.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import binom


def _hb_pvalue(r_hat: float, n: int, R: float) -> float:
    """Hoeffding-Bentkus p-value for H0: true risk >= R, given empirical r_hat."""
    if R <= r_hat:
        return 1.0
    hoeffding = np.exp(-2.0 * n * (R - r_hat) ** 2)
    k = int(np.ceil(n * r_hat))
    bentkus = np.e * binom.cdf(k, n, R)
    return float(min(hoeffding, bentkus))


def hb_upper_bound(r_hat: float, n: int, delta: float) -> float:
    """(1 - delta) HB upper confidence bound on a [0,1]-bounded mean."""
    if n == 0 or r_hat >= 1.0:
        return 1.0
    if _hb_pvalue(r_hat, n, 1.0) >= delta:
        return 1.0
    lo, hi = r_hat, 1.0
    for _ in range(64):
        mid = 0.5 * (lo + hi)
        if _hb_pvalue(r_hat, n, mid) >= delta:
            lo = mid
        else:
            hi = mid
    return hi


def _fixed_sequence(ucb: np.ndarray, s_sorted: np.ndarray, alpha: float, min_accept: int):
    """Largest contiguous top-k accept set (from k=min_accept) with ucb[k] <= alpha.

    Implements the RCPS/LTT fixed-sequence walk: start at the most conservative
    accept set (highest confidence) and grow it until the risk bound is first
    violated. Returns tau = s_sorted[k_last-1], or None if none valid.
    """
    n = len(ucb)
    if n < min_accept:
        return None
    ok = ucb <= alpha                     # ok[i] corresponds to accepting top-(i+1)
    start = min_accept - 1
    if not ok[start]:
        return None
    viol = np.where(~ok[start:])[0]
    last_idx = (n - 1) if len(viol) == 0 else (start + viol[0] - 1)
    return float(s_sorted[last_idx])


def ltt_threshold(
    scores: np.ndarray,
    correct: np.ndarray,
    alpha: float,
    delta: float = 0.1,
    coverage_grid: np.ndarray | None = None,
    min_accept: int = 20,
) -> float | None:
    """Largest accept set (smallest tau) with a certified selective risk <= alpha.

    Learn-then-Test with fixed-sequence testing (Angelopoulos, Bates, Candes,
    Jordan, Lei 2021). Candidate accept sets are ordered by a pre-specified,
    data-independent coverage grid (smallest set first). Each hypothesis
    H0(tau): P(error | score >= tau) >= alpha is tested with an exact binomial
    p-value  P(Bin(n_accept, alpha) <= errors_observed)  at level delta; we grow
    the accept set while H0 keeps being rejected and stop at the first failure.
    Fixed-sequence testing controls the family-wise error at delta without a
    Bonferroni penalty, so it is far more powerful than testing every threshold
    independently, while still giving  P(selective risk <= alpha) >= 1 - delta.

    The binomial test conditions on the accept count, the correct finite-sample
    tool for error-among-accepted (a full-sample Hoeffding UCB over-penalises
    small accept sets and can certify nothing).

    scores: confidence, higher = more likely correct. correct: 1 iff RMSD <= 2 A.
    Returns tau (accept iff score >= tau), or None if nothing is certifiable.
    """
    scores = np.asarray(scores, dtype=float)
    correct = np.asarray(correct, dtype=int)
    if coverage_grid is None:
        coverage_grid = np.arange(0.05, 1.0 + 1e-9, 0.01)  # ascending coverage

    last_valid = None
    for c in coverage_grid:
        tau = float(np.quantile(scores, np.clip(1.0 - c, 0.0, 1.0)))
        accept = scores >= tau
        n_acc = int(accept.sum())
        if n_acc < min_accept:
            continue
        errors = int((1 - correct[accept]).sum())
        if binom.cdf(errors, n_acc, alpha) <= delta:
            last_valid = tau                 # certified; try to grow the set
        else:
            break                            # fixed-sequence: stop at first failure
    return last_valid


# Backward-compatible alias; the selective-risk certifier is LTT (binomial).
rcps_threshold = ltt_threshold


def _empirical_bernstein_ucb(x: np.ndarray, delta: float) -> float:
    """(1 - delta) empirical-Bernstein upper bound on a [0,1] mean (Maurer-Pontil).

    Variance-adaptive: tighter than Hoeffding when the losses concentrate near 0
    (most accepted poses have small RMSD), which is exactly this setting. Kept as a
    closed-form fallback; the default certifier below is the (uniformly tighter)
    WSR betting bound.
    """
    n = len(x)
    if n < 2:
        return 1.0
    mean = float(x.mean())
    var = float(x.var(ddof=1))
    log_term = np.log(2.0 / delta)
    return mean + np.sqrt(2.0 * var * log_term / n) + 7.0 * log_term / (3.0 * (n - 1))


def wsr_betting_pvalue(losses: np.ndarray, target: float, delta: float) -> float:
    """Waudby-Smith & Ramdas (JRSSB 2024) betting p-value for H0: E[L] >= target.

    L must lie in [0,1]. Builds the hedged capital process with a predictable-
    plug-in empirical-Bernstein bet  lambda_i = sqrt(2 log(1/delta) / (n * sigma2_{i-1}))
    (truncated to keep the wealth non-negative) that pays off  1 + lambda_i*(target - L_i).
    Under H0 the wealth K_t is a non-negative supermartingale (E[target - L] <= 0),
    so p = 1 / max_t K_t is a valid p-value by Ville's inequality: reject H0 (certify
    mean loss < target at level 1 - delta) when p <= delta. Uses the whole sample and
    is variance-adaptive, so it dominates Hoeffding-Bentkus and the Maurer-Pontil bound
    without the worst-case-binomial slack. delta appears only through the bet size, so
    validity holds for any delta in (0,1).
    """
    L = np.clip(np.asarray(losses, dtype=float), 0.0, 1.0)
    n = len(L)
    if n == 0:
        return 1.0
    m = float(np.clip(target, 1e-9, 1.0 - 1e-9))
    log1d = np.log(1.0 / delta)
    lam_cap = 0.5 / (1.0 - m)  # keeps 1 + lam*(m - L) >= 0.5 since (m - L) >= -(1 - m)

    # Vectorised predictable plug-ins: everything at step i depends only on j < i.
    idx = np.arange(n)
    prefix_x = np.concatenate([[0.0], np.cumsum(L)])            # prefix_x[i] = sum_{j<i} L_j
    muhat = (0.5 + prefix_x[:-1]) / (idx + 1)                   # plug-in mean before i
    v = (L - muhat) ** 2
    prefix_v = np.concatenate([[0.0], np.cumsum(v)])
    sigma2 = (0.25 + prefix_v[:-1]) / (idx + 1)                 # plug-in variance before i
    lam = np.minimum(np.sqrt(2.0 * log1d / (n * sigma2)), lam_cap)
    log_factors = np.log(1.0 + lam * (m - L))
    log_k = np.cumsum(log_factors)                             # log K_t, t = 1..n
    k_max = float(np.exp(max(log_k.max(), 0.0)))               # include the empty product K_0 = 1
    return float(min(1.0, 1.0 / k_max))


def continuous_risk_threshold(
    scores: np.ndarray,
    loss: np.ndarray,
    target: float,
    delta: float = 0.1,
    coverage_grid: np.ndarray | None = None,
    min_accept: int = 20,
    bound: str = "wsr",
) -> float | None:
    """Largest accept set with a certified mean bounded loss <= target.

    For a continuous, [0,1]-bounded per-pose loss (e.g. min(RMSD, cap)/cap), pick the
    threshold so the mean loss among accepted is <= target with confidence 1 - delta.
    This is the continuous-RMSD analogue of the binary selective-risk gate; loss must
    be scaled into [0,1] by the caller. Uses the same pre-specified coverage-grid
    fixed-sequence walk as the binary gate, so the family-wise 1 - delta certificate
    carries over.

    bound:
      "wsr"       (default) WSR betting p-value -- variance-adaptive, uses the whole
                  sample, uniformly tighter than the alternatives; certifies non-trivial
                  coverage where a distribution-free Hoeffding bound certifies almost none.
      "bernstein" Maurer-Pontil empirical-Bernstein closed-form UCB.
      "hoeffding" conservative distribution-free UCB.
      "binomial"  exact binomial tail for a 0/1 loss -- the degenerate case that must
                  reproduce the E1 binary gate (used only for the equivalence check).
    """
    scores = np.asarray(scores, dtype=float)
    loss = np.clip(np.asarray(loss, dtype=float), 0.0, 1.0)
    if coverage_grid is None:
        coverage_grid = np.arange(0.05, 1.0 + 1e-9, 0.01)

    last_valid = None
    for c in coverage_grid:
        tau = float(np.quantile(scores, np.clip(1.0 - c, 0.0, 1.0)))
        acc = scores >= tau
        n_acc = int(acc.sum())
        if n_acc < min_accept:
            continue
        la = loss[acc]
        if bound == "wsr":
            certified = wsr_betting_pvalue(la, target, delta) <= delta
        elif bound == "binomial":
            errors = int(np.rint(la.sum()))       # la must be 0/1 for this mode
            certified = binom.cdf(errors, n_acc, target) <= delta
        elif bound == "bernstein":
            certified = _empirical_bernstein_ucb(la, delta) <= target
        else:
            ucb = float(la.mean()) + np.sqrt(np.log(1.0 / delta) / (2.0 * n_acc))
            certified = ucb <= target
        if certified:
            last_valid = tau
        else:
            break
    return last_valid


def naive_threshold(
    scores: np.ndarray, correct: np.ndarray, alpha: float, min_accept: int = 20
) -> float | None:
    """Native-confidence baseline: largest accept set with *empirical* risk <= alpha.

    No finite-sample correction -- this is the practitioner's threshold that the
    conformal gate must beat, and the one that silently under-controls under shift.
    """
    scores = np.asarray(scores, dtype=float)
    err = 1 - np.asarray(correct, dtype=int)
    order = np.argsort(-scores)
    s_sorted = scores[order]
    k = np.arange(1, len(scores) + 1)
    r_hat = np.cumsum(err[order]) / k
    return _fixed_sequence(r_hat, s_sorted, alpha, min_accept)
