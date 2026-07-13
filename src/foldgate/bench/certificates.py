"""Distributionally-robust certificates for worst-case selective risk (T3).

Two label-free / finite-sample certificates on the worst-stratum selective risk,
both reused from or matched to src/foldgate/conformal/ so the general benchmark
and the co-folding pipeline share one implementation of each bound:

1. Worst-stratum RCPS upper-confidence bound. For each stratum, a (1 - delta/K)
   Hoeffding-Bentkus UCB on the accepted error rate (reusing conformal.robust);
   the union bound over K strata makes  P(max_k R_k <= U) >= 1 - delta. This is
   the finite-sample certificate: it needs in-stratum labels but no target law.

2. f-divergence / CVaR ball over the K-stratum simplex. Given the oracle (or
   plug-in) per-stratum risks R_k and accept rates a_k, the certified worst risk
   is  sup_{q : D(q || p_cal) <= rho} R_mix(tau; q),  a linear-fractional program
   over the simplex intersected with a divergence ball. Solved exactly (KL or
   chi-square ball) by scipy.optimize; a chi-square closed form using the
   sqrt(2 rho Var) constant (THEOREM_RECONCILED D5) is provided as the fast
   companion. The certificate covers R_mix(tau; p_tgt) exactly once the radius rho
   reaches the true tilt D(p_tgt || p_cal).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from foldgate.conformal.robust import error_rate_ucb

_TINY = 1e-12


# --------------------------------------------------------------------------- #
# 1. Worst-stratum RCPS UCB (Hoeffding-Bentkus, union bound over strata)
# --------------------------------------------------------------------------- #
def worst_stratum_rcps_ucb(
    per_stratum: list[tuple[int, int]],
    delta: float,
    method: str = "hb",
    union_bound: bool = True,
) -> dict:
    """(1 - delta) upper bound on the worst-stratum accepted error rate.

    per_stratum: list of (errors_k, n_accept_k) over the accepted calibration
    points in each stratum. Each stratum gets a (1 - delta/K) UCB (union bound);
    U = max_k UCB_k then satisfies  P(max_k R_k <= U) >= 1 - delta.

    method: "hb" Hoeffding-Bentkus (the RCPS bound), "cp" Clopper-Pearson (tighter
    exact binomial), "wsr" betting bound. Empty strata contribute a trivial UCB 1.
    """
    k = len(per_stratum)
    if k == 0:
        return {"U": float("nan"), "per_stratum_ucb": [], "delta_per_stratum": delta}
    d = delta / k if union_bound else delta
    ucbs = []
    for errors, n in per_stratum:
        if n == 0:
            ucbs.append(1.0)
        else:
            ucbs.append(float(error_rate_ucb(int(errors), int(n), d, method)))
    return {
        "U": float(max(ucbs)),
        "per_stratum_ucb": ucbs,
        "delta_per_stratum": float(d),
        "argmax_stratum": int(np.argmax(ucbs)),
    }


# --------------------------------------------------------------------------- #
# 2. f-divergence / CVaR ball certificate over the stratum simplex
# --------------------------------------------------------------------------- #
def _R_mix_from_q(q: np.ndarray, accept_rate: np.ndarray, risk: np.ndarray) -> float:
    w = q * accept_rate
    denom = w.sum()
    if denom < _TINY:
        return 0.0
    return float((w * risk).sum() / denom)


def dro_ball_certificate(
    risk: np.ndarray,
    accept_rate: np.ndarray,
    p_cal: np.ndarray,
    rho: float,
    divergence: str = "kl",
) -> dict:
    """sup_{q: D(q || p_cal) <= rho} R_mix over the K-stratum simplex (exact).

    R_mix(q) = sum_k q_k a_k R_k / sum_k q_k a_k is linear-fractional in q, so the
    worst case over a convex divergence ball is a small quasiconvex program; we
    solve it with SLSQP from several starts (p_cal, p_tgt-agnostic vertices, and
    the risk-argmax vertex) and take the best. Exact for KL and chi-square balls.

    Guarantee used by the caller: p_tgt is feasible whenever rho >= D(p_tgt||p_cal),
    so the certificate then upper-bounds R_mix(tau; p_tgt).
    """
    risk = np.nan_to_num(np.asarray(risk, dtype=float), nan=0.0)
    accept_rate = np.asarray(accept_rate, dtype=float)
    p_cal = np.asarray(p_cal, dtype=float)
    K = len(risk)

    if divergence == "kl":
        def div(q):
            m = q > 0
            return float(np.sum(q[m] * np.log(q[m] / np.clip(p_cal[m], _TINY, None))))
    elif divergence == "chi2":
        def div(q):
            return float(np.sum((q - p_cal) ** 2 / np.clip(p_cal, _TINY, None)))
    else:
        raise ValueError(f"divergence must be 'kl' or 'chi2', got {divergence!r}")

    def neg_obj(q):
        return -_R_mix_from_q(q, accept_rate, risk)

    cons = (
        {"type": "eq", "fun": lambda q: q.sum() - 1.0},
        {"type": "ineq", "fun": lambda q: rho - div(q)},
    )
    bounds = [(0.0, 1.0)] * K

    # Starting points: p_cal, uniform, and each pure-vertex nudged toward the ball.
    starts = [p_cal.copy(), np.full(K, 1.0 / K)]
    hi = int(np.argmax(risk))
    v = 0.9 * np.eye(K)[hi] + 0.1 * p_cal
    starts.append(v / v.sum())

    best_val, best_q = _R_mix_from_q(p_cal, accept_rate, risk), p_cal.copy()
    for q0 in starts:
        res = minimize(neg_obj, q0, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})
        if res.success or res.status in (0, 9):
            q = np.clip(res.x, 0.0, None)
            q = q / q.sum() if q.sum() > 0 else p_cal
            if div(q) <= rho + 1e-6:
                val = _R_mix_from_q(q, accept_rate, risk)
                if val > best_val:
                    best_val, best_q = val, q
    return {
        "certified_worst_risk": float(np.clip(best_val, 0.0, 1.0)),
        "q_star": [float(x) for x in best_q],
        "rho": float(rho),
        "divergence": divergence,
        "baseline_R_mix": float(_R_mix_from_q(p_cal, accept_rate, risk)),
    }


def chi2_closed_form_certificate(
    risk: np.ndarray,
    accept_rate: np.ndarray,
    p_cal: np.ndarray,
    rho: float,
) -> float:
    """First-order chi-square-ball certificate  R_mix + sqrt(2 rho Var) (D5).

    Computed on the accept-mass distribution pi_k = p_cal_k a_k / sum_j p_cal_j a_j,
    which is the distribution R_mix averages over. The sqrt(2 rho Var) constant is
    the standard Duchi-Namkoong / Cauchois first-order chi-square dual (THEOREM
    RECONCILED D5). This is a fast companion to dro_ball_certificate(divergence=
    'chi2'); it is a linearization, so it can slightly under- or over-shoot the
    exact program for large rho.
    """
    risk = np.nan_to_num(np.asarray(risk, dtype=float), nan=0.0)
    accept_rate = np.asarray(accept_rate, dtype=float)
    p_cal = np.asarray(p_cal, dtype=float)
    w = p_cal * accept_rate
    if w.sum() < _TINY:
        return float("nan")
    pi = w / w.sum()
    mean = float((pi * risk).sum())
    var = float((pi * (risk - mean) ** 2).sum())
    return float(np.clip(mean + np.sqrt(2.0 * max(rho, 0.0) * var), 0.0, 1.0))
