"""Synthetic generator with a known closed-form P(Y=1 | s, nu).

This is the general (non-co-folding) checker for the impossibility/achievability
theorem. Every example reduces to a triple (s, y, nu):

  s  in [0,1]   a base classifier's self-reported confidence,
  y  in {0,1}   the selective label, y = 1 means the answer is CORRECT/acceptable,
  nu            the novelty coordinate, here a discrete stratum index k = 1..K.

The generator has three isolated knobs (BENCHMARK_SPEC 2.1):

  D   concept-shift slope: on novel strata the true correctness probability sits
      below the reported confidence by D * nu in logit units. D = 0 makes the
      confidence a perfectly calibrated correctness probability (pi == s).
  E   covariate-shift-in-score: E > 0 makes novel strata systematically lower
      confidence with calibration (P(y|s)) intact.
  tilt (via c_cal, c_tgt): shifts the nu-marginal between calibration and target;
      the closed-form tilt magnitude is T = KL(p_tgt || p_cal).

Because pi(z, nu) is closed form, every theorem quantity has a ground-truth
reference computed by scipy quadrature over the logistic-normal (Section 2.3):
per-stratum accept rate, selective risk R_k(tau), oracle threshold tau_k*, and
the mixture risk R_mix. These are cross-checked against a large Monte-Carlo draw
in tests/test_bench.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import integrate
from scipy.optimize import brentq
from scipy.special import expit, logit
from scipy.stats import norm

_TINY = 1e-12
# logit range for tau clipping; tau in (sigmoid(-12), sigmoid(12)).
_T_LO, _T_HI = -12.0, 12.0
# root-find bracket: kept where the accept mass stays numerically resolvable so
# R_mix / R_k do not underflow to NaN at the top of the search.
_T_SEARCH_HI = 8.0


@dataclass(frozen=True)
class SynthParams:
    """Closed-form generator parameters. Defaults from BENCHMARK_SPEC 2.1/2.3.

    With the defaults (beta0=0, beta1=1) and D=0, pi(z, nu) = sigmoid(z) = s
    exactly, so s is a perfectly calibrated correctness probability identical
    across strata (the concept-shift-free control).
    """

    K: int = 8
    D: float = 1.0          # concept-shift slope (reliability decay on novel strata)
    E: float = 0.0          # covariate shift in score given stratum
    sigma_s: float = 1.3    # logit-score noise sd
    m0: float = 0.6         # base score logit level (accept rate near 0.6)
    beta0: float = 0.0
    beta1: float = 1.0
    c_cal: float = 3.0      # calibration nu-marginal concentration on seen strata
    c_tgt: float = 0.0      # target nu-marginal (smaller/negative -> more novel mass)
    Dq: float = 0.0         # TARGET-only extra concept drift: label law differs P vs Q.
    #                         Dq = 0 keeps eta_Q == eta_P (the shared-label control used by
    #                         b1-b4). Dq > 0 makes the target correctness prob sit an extra
    #                         Dq*nu below calibration at fixed score, so the accept-region
    #                         concept gap Delta_bar_A = E_Q[eta_Q - eta_P | accept] > 0. This
    #                         is the regime Theorem 1(c)/(d) is about (weighted CP certifies
    #                         the calibration risk but realizes it plus Delta_bar_A).
    eps_floor: float = 0.0  # irreducible two-sided error floor: pi in [eps, 1-eps]. eps > alpha
    #                         makes a stratum uncertifiable at any positive coverage (the RNP
    #                         no-analog S4 tail), so even in-stratum recalibration must abstain.

    def nu(self) -> np.ndarray:
        """Novelty levels nu_k = (k-1)/(K-1) in [0,1] for k = 1..K."""
        if self.K == 1:
            return np.array([0.0])
        return np.arange(self.K, dtype=float) / (self.K - 1)

    def mu(self) -> np.ndarray:
        """Per-stratum score-logit mean mu_k = m0 - E * nu_k."""
        return self.m0 - self.E * self.nu()

    def p_cal(self) -> np.ndarray:
        return _stratum_probs(self.K, self.c_cal, self.nu())

    def p_tgt(self) -> np.ndarray:
        return _stratum_probs(self.K, self.c_tgt, self.nu())

    def tilt_kl(self) -> float:
        """T = KL(p_tgt || p_cal), the closed-form nu-marginal tilt magnitude."""
        return kl_divergence(self.p_tgt(), self.p_cal())


def _stratum_probs(K: int, c: float, nu: np.ndarray) -> np.ndarray:
    """p(k) proportional to exp(-c * nu_k), normalized over the K-simplex."""
    w = np.exp(-c * nu)
    return w / w.sum()


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p || q) over a finite simplex, in nats."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / np.clip(q[mask], _TINY, None))))


def chi2_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Pearson chi-square divergence chi2(p || q) = sum_k (p_k - q_k)^2 / q_k."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    return float(np.sum((p - q) ** 2 / np.clip(q, _TINY, None)))


def _d_eff(params: SynthParams, population: str) -> float:
    """Effective concept-drift slope: D for calibration, D + Dq for the target law."""
    return params.D + (params.Dq if population == "tgt" else 0.0)


def pi_correct(
    z: np.ndarray | float, nu_k: float, params: SynthParams, population: str = "cal"
):
    """True correctness probability pi(z, nu) for the given label law.

    pi = eps + (1 - 2 eps) * sigmoid(beta0 + beta1 z - D_eff nu), with D_eff = D for
    the calibration law and D + Dq for the target law. The floor eps bounds pi in
    [eps, 1 - eps] so the error 1 - pi cannot fall below eps. With the defaults
    (Dq = 0, eps = 0) this is sigmoid(beta0 + beta1 z - D nu), identical for cal and tgt.
    """
    raw = expit(params.beta0 + params.beta1 * np.asarray(z) - _d_eff(params, population) * nu_k)
    e = params.eps_floor
    return e + (1.0 - 2.0 * e) * raw


# --------------------------------------------------------------------------- #
# Sampling API
# --------------------------------------------------------------------------- #
def sample(
    n: int,
    params: SynthParams,
    population: str = "cal",
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Draw n examples from the calibration ("cal") or target ("tgt") law.

    The two laws differ ONLY in the nu-marginal (p_cal vs p_tgt); the score model
    P(s | nu) and the label model P(y | s, nu) are shared. Returns a frame with
    columns: k (0-based stratum index), nu, z (score logit), s (confidence in
    [0,1]), pi (true correctness prob), y (selective label, 1 = correct).
    """
    if rng is None:
        rng = np.random.default_rng()
    if population == "cal":
        p = params.p_cal()
    elif population == "tgt":
        p = params.p_tgt()
    else:
        raise ValueError(f"population must be 'cal' or 'tgt', got {population!r}")

    nu_levels = params.nu()
    mu = params.mu()
    k = rng.choice(params.K, size=n, p=p)
    nu = nu_levels[k]
    z = mu[k] + params.sigma_s * rng.standard_normal(n)
    s = expit(z)
    pi = pi_correct(z, nu, params, population=population)
    y = (rng.random(n) < pi).astype(int)
    return pd.DataFrame({"k": k, "nu": nu, "z": z, "s": s, "pi": pi, "y": y})


# --------------------------------------------------------------------------- #
# Oracle (closed-form / quadrature) API -- BENCHMARK_SPEC 2.3
# --------------------------------------------------------------------------- #
def _tau_to_t(tau: float) -> float:
    tau = float(np.clip(tau, expit(_T_LO), expit(_T_HI)))
    return float(logit(tau))


def oracle_accept_rate(tau: float, k: int, params: SynthParams) -> float:
    """P(s >= tau | stratum k) = 1 - Phi((t - mu_k) / sigma_s), t = logit(tau)."""
    t = _tau_to_t(tau)
    mu_k = params.mu()[k]
    return float(norm.sf(t, loc=mu_k, scale=params.sigma_s))


def oracle_selective_risk(
    tau: float, k: int, params: SynthParams, population: str = "cal"
) -> float:
    """R_k(tau) = E[1 - pi(z, nu_k) | z >= t, stratum k] via scipy quadrature.

    Numerator integrates the error density (1 - pi) times the score density over
    the accept region z >= t; denominator is the accept mass. The label law is the
    calibration law by default; pass population="tgt" for the drifted target law
    (D + Dq). Returns NaN when the accept mass is negligible (threshold above
    essentially all score mass).
    """
    t = _tau_to_t(tau)
    nu_k = params.nu()[k]
    mu_k = params.mu()[k]
    sig = params.sigma_s

    accept = float(norm.sf(t, loc=mu_k, scale=sig))
    if accept < 1e-9:
        return float("nan")

    def integrand(z):
        err = 1.0 - pi_correct(z, nu_k, params, population=population)
        return err * norm.pdf(z, loc=mu_k, scale=sig)

    num, _ = integrate.quad(integrand, t, np.inf, limit=200)
    return float(np.clip(num / accept, 0.0, 1.0))


def oracle_tau_star(
    k: int, params: SynthParams, alpha: float, population: str = "cal"
) -> float:
    """Smallest tau with R_k(tau) <= alpha (oracle per-stratum threshold).

    R_k(tau) is monotone decreasing in tau (a higher threshold conditions on
    higher-confidence, lower-error scores). Returns 0.0 if accept-all already
    satisfies alpha, or NaN if no threshold controls the stratum (for example a
    target stratum whose error floor eps exceeds alpha).
    """
    r_lo = oracle_selective_risk(expit(_T_LO), k, params, population)  # accept ~everything
    if not np.isnan(r_lo) and r_lo <= alpha:
        return 0.0
    r_hi = oracle_selective_risk(expit(_T_SEARCH_HI), k, params, population)
    if np.isnan(r_hi) or r_hi > alpha:
        return float("nan")

    def f(t):
        return oracle_selective_risk(expit(t), k, params, population) - alpha

    # bracket in logit space; f decreasing so f(_T_LO) > 0, f(high) < 0.
    t_star = brentq(f, _T_LO, _T_SEARCH_HI, xtol=1e-6, rtol=1e-8, maxiter=200)
    return float(expit(t_star))


def oracle_R_mix(
    tau: float, p: np.ndarray, params: SynthParams, population: str = "cal"
) -> float:
    """R_mix(tau; p) = accept-mass-weighted average of R_k(tau) over strata.

    R_mix = sum_k p_k a_k R_k / sum_k p_k a_k, with a_k the per-stratum accept
    rate. This is the pooled selective risk when strata are drawn from p. The
    accept rates depend only on the score model; the per-stratum risks R_k use the
    requested label law (population).
    """
    p = np.asarray(p, dtype=float)
    a = np.array([oracle_accept_rate(tau, k, params) for k in range(params.K)])
    r = np.array([oracle_selective_risk(tau, k, params, population) for k in range(params.K)])
    r = np.nan_to_num(r, nan=0.0)
    w = p * a
    denom = w.sum()
    if denom < 1e-12:
        return float("nan")
    return float(np.clip((w * r).sum() / denom, 0.0, 1.0))


def oracle_coverage_threshold(c: float, p: np.ndarray, params: SynthParams) -> float:
    """Threshold tau with pooled accept rate sum_k p_k P(s>=tau|k) = c.

    Label-independent (uses only the score model), so it pins the accept region for
    the coverage-pinned statements of Theorem 1 without reference to any label law.
    """
    p = np.asarray(p, dtype=float)

    def cov(t):
        a = np.array([oracle_accept_rate(expit(t), k, params) for k in range(params.K)])
        return float((p * a).sum())

    if cov(_T_LO) <= c:
        return float(expit(_T_LO))
    if cov(_T_HI) >= c:
        return float(expit(_T_HI))

    def f(t):
        return cov(t) - c

    # cov decreasing in t; f(_T_LO) > 0, f(_T_HI) < 0.
    t_c = brentq(f, _T_LO, _T_HI, xtol=1e-7, rtol=1e-9, maxiter=200)
    return float(expit(t_c))


def oracle_marginal_threshold(
    p: np.ndarray, params: SynthParams, alpha: float, population: str = "cal"
) -> float:
    """Population marginal threshold tau_inf solving R_mix(tau; p) = alpha.

    This is the threshold a MARGINAL (pooled) method converges to. Under concept
    shift it under-controls the novel strata: max_k R_k(tau_inf) exceeds alpha.
    """
    r_lo = oracle_R_mix(expit(_T_LO), p, params, population)
    if not np.isnan(r_lo) and r_lo <= alpha:
        return 0.0
    r_hi = oracle_R_mix(expit(_T_SEARCH_HI), p, params, population)
    if np.isnan(r_hi) or r_hi > alpha:
        return float("nan")

    def f(t):
        return oracle_R_mix(expit(t), p, params, population) - alpha

    t_star = brentq(f, _T_LO, _T_SEARCH_HI, xtol=1e-6, rtol=1e-8, maxiter=200)
    return float(expit(t_star))


def oracle_impossibility_gap(params: SynthParams, alpha: float) -> dict:
    """Analytic impossibility gap Delta(D, T) = max_k R_k(tau_inf) - alpha.

    tau_inf is the population marginal threshold on the CALIBRATION nu-marginal.
    Delta is the worst-stratum excess risk that no marginal threshold can avoid.
    Delta increases in D and in the tilt T, and Delta -> 0 as D -> 0 (T1).
    """
    p_cal = params.p_cal()
    tau_inf = oracle_marginal_threshold(p_cal, params, alpha)
    if np.isnan(tau_inf):
        return {"tau_inf": float("nan"), "worst_risk": float("nan"),
                "delta": float("nan"), "per_stratum_risk": [float("nan")] * params.K}
    r_k = np.array([oracle_selective_risk(tau_inf, k, params) for k in range(params.K)])
    worst = float(np.nanmax(r_k))
    return {
        "tau_inf": float(tau_inf),
        "worst_risk": worst,
        "delta": float(worst - alpha),
        "per_stratum_risk": [float(x) for x in r_k],
        "tilt_kl": params.tilt_kl(),
    }


def oracle_accept_rates(tau: float, params: SynthParams) -> np.ndarray:
    return np.array([oracle_accept_rate(tau, k, params) for k in range(params.K)])


def oracle_selective_risks(
    tau: float, params: SynthParams, population: str = "cal"
) -> np.ndarray:
    return np.array(
        [oracle_selective_risk(tau, k, params, population) for k in range(params.K)]
    )


def oracle_concept_gap(c: float, params: SynthParams) -> dict:
    """Accept-region concept gap Delta_bar_A at target coverage c (Theorem 1).

    Pins the accept region to coverage c on the TARGET nu-marginal (label-free),
    then returns R_ref = target-covariate risk under the CALIBRATION label law,
    R_Q = the same under the TARGET label law, and Delta_bar_A = R_Q - R_ref. This
    is the quantity weighted CP cannot remove: it certifies R_ref but realizes R_Q.
    """
    p_tgt = params.p_tgt()
    tau_c = oracle_coverage_threshold(c, p_tgt, params)
    r_ref = oracle_R_mix(tau_c, p_tgt, params, population="cal")
    r_q = oracle_R_mix(tau_c, p_tgt, params, population="tgt")
    return {
        "coverage": float(c),
        "tau_c": float(tau_c),
        "R_ref": float(r_ref),
        "R_Q": float(r_q),
        "delta_bar_A": float(r_q - r_ref),
    }
