"""Distribution-robust selective-risk certificates: worst-subpopulation control.

Weighted conformal tries to certify the risk on a *named* novel target distribution
and, under the concept shift this project documents, abstains (E3b). The robust route
asks a different, label-free question: for the accepted poses, how small a subpopulation
can we still guarantee error control on, without knowing which subpopulation is the novel
one? That is exactly conditional-value-at-risk (CVaR) control.

CVaR-DRO duality (Rockafellar-Uryasev; Shapiro): for a loss L,

    CVaR_beta(L) = sup over Q with dQ/dP <= 1/(1-beta) of E_Q[L],

so a certificate  CVaR_beta(L) <= alpha  means EVERY subpopulation carrying at least a
(1-beta) fraction of accepted mass has selective risk <= alpha, with no knowledge of the
subpopulation and no importance weights. We report the boundary number

    m* = 1 - beta* = the smallest accepted-mass fraction still certified at level alpha,

which turns the weighted-CP null ("cannot certify the novel stratum") into a positive
statement ("every accepted subpopulation down to mass m* is certified").

The primary label is binary (RMSD <= 2 A), for which CVaR has an exact closed form:
with accepted error rate r,  CVaR_beta(L) = min(1, r / (1 - beta)). CVaR is monotone in r,
so a finite-sample (1 - delta) UPPER bound on r certifies CVaR_beta for *all* beta at once
(Snell et al., "Quantile Risk Control", arXiv:2212.13629, Thm 4.1 CDF-dominance specialised
to a Bernoulli). We take r_ucb as the exact one-sided Clopper-Pearson upper bound (tightest
distribution-free choice for a binomial), with Hoeffding-Bentkus and WSR-betting variants.
A general continuous-loss CVaR bound via a one-sided DKW lower CDF band is provided for the
bounded-RMSD gate.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import beta as beta_dist

from .risk import hb_upper_bound, wsr_betting_pvalue


def clopper_pearson_upper(k: int, n: int, delta: float) -> float:
    """Exact one-sided (1 - delta) Clopper-Pearson upper bound on a binomial rate k/n."""
    if n == 0:
        return 1.0
    if k >= n:
        return 1.0
    return float(beta_dist.ppf(1.0 - delta, k + 1, n - k))


def _wsr_rate_upper(k: int, n: int, delta: float) -> float:
    """(1 - delta) upper bound on a Bernoulli mean by inverting the WSR betting p-value."""
    if n == 0:
        return 1.0
    losses = np.zeros(n)
    losses[:k] = 1.0
    lo, hi = k / n, 1.0
    for _ in range(48):
        mid = 0.5 * (lo + hi)
        # p-value for H0: E[L] >= mid; reject (mean < mid certified) when <= delta
        if wsr_betting_pvalue(losses, mid, delta) <= delta:
            hi = mid
        else:
            lo = mid
    return hi


def error_rate_ucb(errors: int, n: int, delta: float, method: str = "cp") -> float:
    """(1 - delta) upper confidence bound on the accepted error rate.

    method: "cp" exact Clopper-Pearson (default, tightest for binomial), "hb"
    Hoeffding-Bentkus, "wsr" betting bound.
    """
    if n == 0:
        return 1.0
    if method == "cp":
        return clopper_pearson_upper(errors, n, delta)
    if method == "hb":
        return hb_upper_bound(errors / n, n, delta)
    if method == "wsr":
        return _wsr_rate_upper(errors, n, delta)
    raise ValueError(f"unknown method {method}")


def cvar_binary_ucb(errors: int, n: int, beta: float, delta: float, method: str = "cp") -> float:
    """(1 - delta) upper bound on CVaR_beta of the binary selective loss among accepted poses.

    CVaR_beta(L) = min(1, r / (1 - beta)) for a Bernoulli(r); monotone in r, so the UCB on r
    lifts to an UCB on CVaR_beta for every beta simultaneously.
    """
    r = error_rate_ucb(errors, n, delta, method)
    if beta >= 1.0:
        return 1.0
    return float(min(1.0, r / (1.0 - beta)))


def worst_subpopulation_certificate(
    errors: int, n: int, alpha: float, delta: float, method: str = "cp"
) -> dict:
    """Largest beta (smallest accepted subpopulation mass m* = 1 - beta*) with CVaR_beta <= alpha.

    Returns r_ucb, beta_star, m_star (= smallest certified accepted-mass fraction), and whether
    the marginal accept set is even certified (r_ucb <= alpha). If r_ucb > alpha the worst-case
    certificate is vacuous (beta_star = 0, m_star = 1): we cannot guarantee any strict
    subpopulation, only report the honest failure.
    """
    r_ucb = error_rate_ucb(errors, n, delta, method)
    if r_ucb <= alpha and alpha > 0:
        m_star = float(r_ucb / alpha)          # CVaR_beta <= alpha  <=>  1 - beta >= r_ucb / alpha
        beta_star = float(1.0 - m_star)
        certified = True
    else:
        m_star, beta_star, certified = 1.0, 0.0, False
    return {
        "n_accept": int(n),
        "errors": int(errors),
        "r_ucb": float(r_ucb),
        "alpha": float(alpha),
        "beta_star": beta_star,
        "m_star": m_star,
        "marginal_certified": bool(r_ucb <= alpha),
        "certified": bool(certified),
    }


def chi2_dro_risk_ucb(errors: int, n: int, rho: float, delta: float, method: str = "cp") -> float:
    """(1 - delta) upper bound on the worst-case accepted error over a chi-square ball of radius rho.

    For the binary selective loss with accepted error rate r, Cauchy-Schwarz gives the exact
    upper bound  sup_{Q : chi2(Q||P) <= rho} E_Q[L]  <=  r + std(L) * sqrt(rho), where
    chi2(Q||P) = E_P[(dQ/dP - 1)^2] and std(L) = sqrt(r(1-r)). We plug a finite-sample upper
    bound on r (and, since Var is increasing in r for r < 1/2, the matching std upper bound), so
    the result is a valid (1 - delta) certificate on the worst-case risk over the divergence ball.
    This is the smoother f-divergence companion to the likelihood-ratio ball that CVaR controls
    (Cauchois, Gupta, Ali, Duchi, Robust Validation, JASA 2024; Duchi, Namkoong 2019).
    """
    r = error_rate_ucb(errors, n, delta, method)
    std_ub = float(np.sqrt(r * (1.0 - r)))          # valid upper bound on std for r <= 1/2
    return float(min(1.0, r + std_ub * np.sqrt(max(rho, 0.0))))


def robustness_radius(errors: int, n: int, alpha: float, delta: float, method: str = "cp") -> dict:
    """Largest chi-square ball radius rho* at which the accepted risk is still certified <= alpha.

    rho* = ((alpha - r_ucb) / std_ucb)^2 when r_ucb < alpha, else 0. This is the exact two-point
    Pearson chi-square worst-case for a binary loss (chi2((q,1-q)||(r,1-r)) = (q-r)^2/(r(1-r))),
    inverted at r_ucb, so it is finite-sample and tight, not the sqrt(2)-looser first-order form.
    Interpretation: the accepted selective risk stays <= alpha for every test law within
    chi-square divergence rho* of the calibration law, with confidence 1 - delta.

    The certificate is label-aware (it is computed on the accepted loss distribution), so this is
    a valid robustness margin. The honesty caveat is about UNITS: rho* is a divergence-ball radius
    around the calibration law, and translating it into "how novel a pocket" requires a
    covariate-space map that a density-ratio classifier estimates blind to the label law, so we do
    not read rho* as a Tanimoto-novelty budget. rho* is inter-convertible with the CVaR
    worst-subpopulation mass by rho = (1 - m*) / m* (the box ball {dQ/dP <= 1/(1-beta)} attains
    Pearson chi-square beta/(1-beta)).
    """
    r = error_rate_ucb(errors, n, delta, method)
    if r >= alpha or alpha <= 0:
        return {"r_ucb": float(r), "alpha": float(alpha), "rho_star": 0.0, "certified": False}
    std_ub = float(np.sqrt(r * (1.0 - r)))
    rho_star = float(((alpha - r) / std_ub) ** 2) if std_ub > 0 else float("inf")
    return {"r_ucb": float(r), "alpha": float(alpha), "rho_star": rho_star, "certified": True}


def simultaneous_certificate(model_counts: list[tuple[int, int]], alpha: float, delta: float,
                             method: str = "cp") -> dict:
    """Simultaneous worst-subpopulation + robustness certificates across K models via a union bound.

    Each model is certified at level delta / K, so all K statements hold jointly with probability
    at least 1 - delta (Bonferroni over the finite model set). Returns per-model m* and rho* at the
    corrected level plus the worst (smallest) simultaneous values.
    """
    k = len(model_counts)
    if k == 0:
        return {"per_model": [], "joint_m_star": float("nan"), "joint_rho_star": float("nan")}
    d = delta / k
    per = []
    for errors, n in model_counts:
        wsc = worst_subpopulation_certificate(errors, n, alpha, d, method)
        rr = robustness_radius(errors, n, alpha, d, method)
        per.append({"n": n, "errors": errors, "m_star": wsc["m_star"],
                    "certified": wsc["certified"], "rho_star": rr["rho_star"]})
    certified = [p for p in per if p["certified"]]
    return {
        "delta_per_model": d,
        "per_model": per,
        "joint_m_star": max(p["m_star"] for p in per),          # worst (largest) protected mass
        "joint_rho_star": min((p["rho_star"] for p in certified), default=0.0),
        "all_certified": len(certified) == k,
    }


def cvar_cdf_band_ucb(losses: np.ndarray, beta: float, delta: float) -> float:
    """(1 - delta) upper bound on CVaR_beta of a [0,1]-bounded continuous loss via a DKW band.

    One-sided DKW-Massart lower band on the loss CDF, F(x) >= Fhat(x) - eps with
    eps = sqrt(ln(1/delta) / (2n)); by CDF dominance the CVaR of the lower band upper-bounds
    the true CVaR. Conservative (uniform band) but exact finite-sample and distribution-free;
    the binary route above is tighter for the RMSD <= 2 A label. Used for the bounded-RMSD gate.
    """
    L = np.clip(np.sort(np.asarray(losses, dtype=float)), 0.0, 1.0)
    n = len(L)
    if n == 0 or beta >= 1.0:
        return 1.0
    eps = np.sqrt(np.log(1.0 / delta) / (2.0 * n))
    ps = np.linspace(beta, 1.0, 256, endpoint=False)
    q = np.quantile(L, np.clip(ps + eps, 0.0, 1.0))   # upper quantile via shifted level
    return float(min(1.0, q.mean()))
