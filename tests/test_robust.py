"""Finite-sample validity checks for the worst-subpopulation (CVaR) certificates."""

from __future__ import annotations

import numpy as np

from foldgate.conformal.robust import (
    chi2_dro_risk_ucb,
    cvar_binary_ucb,
    cvar_cdf_band_ucb,
    error_rate_ucb,
    robustness_radius,
    simultaneous_certificate,
    worst_subpopulation_certificate,
)


def test_error_rate_ucb_covers():
    """(1 - delta) upper bound on a Bernoulli rate covers the truth at >= 1 - delta."""
    rng = np.random.default_rng(0)
    n, r_true, delta = 300, 0.12, 0.10
    cover = np.mean([
        error_rate_ucb(int((rng.random(n) < r_true).sum()), n, delta, "cp") >= r_true
        for _ in range(1500)
    ])
    assert cover >= 1 - delta - 0.02


def test_cvar_binary_ucb_covers():
    """The CVaR_beta upper bound covers the true CVaR of a Bernoulli at >= 1 - delta."""
    rng = np.random.default_rng(1)
    n, r_true, delta, beta = 300, 0.12, 0.10, 0.5
    true_cvar = min(1.0, r_true / (1 - beta))
    cover = np.mean([
        cvar_binary_ucb(int((rng.random(n) < r_true).sum()), n, beta, delta, "cp") >= true_cvar
        for _ in range(1500)
    ])
    assert cover >= 1 - delta - 0.02


def test_cvar_monotone_in_beta():
    """CVaR_beta is non-decreasing in beta (a smaller subpopulation is at least as risky)."""
    vals = [cvar_binary_ucb(30, 300, b, 0.10, "cp") for b in (0.0, 0.25, 0.5, 0.75)]
    assert all(a <= b + 1e-9 for a, b in zip(vals[:-1], vals[1:], strict=False))


def test_worst_subpop_certificate_formula():
    """m* = r_ucb / alpha when the marginal set is certified; vacuous otherwise."""
    # low error rate -> strong (small m*) certificate
    c = worst_subpopulation_certificate(errors=10, n=1000, alpha=0.20, delta=0.10)
    assert c["certified"] and 0.0 < c["m_star"] < 1.0
    assert abs(c["m_star"] - c["r_ucb"] / 0.20) < 1e-9
    # high error rate -> vacuous
    v = worst_subpopulation_certificate(errors=400, n=1000, alpha=0.20, delta=0.10)
    assert not v["certified"] and v["m_star"] == 1.0 and v["beta_star"] == 0.0


def test_chi2_dro_ucb_covers_population_worstcase():
    """The chi-square DRO bound covers the true two-point Pearson worst-case at >= 1 - delta."""
    rng = np.random.default_rng(3)
    n, r_true, delta, rho = 400, 0.12, 0.10, 0.5
    pop_wc = min(1.0, r_true + np.sqrt(rho * r_true * (1 - r_true)))  # exact 2-point Pearson
    cover = np.mean([
        chi2_dro_risk_ucb(int((rng.random(n) < r_true).sum()), n, rho, delta) >= pop_wc
        for _ in range(1500)
    ])
    assert cover >= 1 - delta - 0.02


def test_robustness_radius_inversion_and_vacuity():
    """rho* inverts the worst-case exactly at alpha, and is 0 when r_ucb >= alpha."""
    c = robustness_radius(errors=50, n=1000, alpha=0.20, delta=0.10)
    assert c["certified"] and c["rho_star"] > 0
    # plugging rho* back into the worst-case returns alpha
    r = c["r_ucb"]
    assert abs((r + np.sqrt(c["rho_star"] * r * (1 - r))) - 0.20) < 1e-6
    v = robustness_radius(errors=400, n=1000, alpha=0.20, delta=0.10)
    assert not v["certified"] and v["rho_star"] == 0.0


def test_simultaneous_union_bound():
    """Joint m* is the worst per-model value, computed at the corrected level delta / K."""
    counts = [(10, 400), (20, 500), (5, 300)]
    s = simultaneous_certificate(counts, alpha=0.20, delta=0.10)
    assert abs(s["delta_per_model"] - 0.10 / 3) < 1e-9
    assert s["joint_m_star"] == max(p["m_star"] for p in s["per_model"])


def test_cvar_cdf_band_upper_bounds_continuous():
    """The DKW-band CVaR bound covers the true continuous CVaR at >= 1 - delta."""
    rng = np.random.default_rng(2)
    n, delta, beta = 400, 0.10, 0.5
    # Beta(2,5) losses on [0,1]; true CVaR_0.5 estimated on a large sample
    big = rng.beta(2, 5, 400000)
    thr = np.quantile(big, beta)
    true_cvar = big[big >= thr].mean()
    cover = np.mean([
        cvar_cdf_band_ucb(rng.beta(2, 5, n), beta, delta) >= true_cvar
        for _ in range(600)
    ])
    assert cover >= 1 - delta - 0.03
