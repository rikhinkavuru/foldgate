"""Numerical validation of the general (non-co-folding) theorem benchmark.

Checks the closed-form oracle against Monte Carlo (C1), the D=0 calibration
control (pi == s), the impossibility monotonicity (T1), and certificate coverage
(T3): the worst-stratum RCPS UCB covers at rate >= 1 - delta, and the
f-divergence ball certificate covers the target risk once its radius reaches the
true tilt. Real-data loaders are exercised but skip cleanly when offline.
"""

from __future__ import annotations

import numpy as np
import pytest

from foldgate.bench import certificates as C
from foldgate.bench import realdata as RD
from foldgate.bench import synth as S
from foldgate.bench.synth import SynthParams


# --------------------------------------------------------------------------- #
# C1 -- closed-form oracle matches a large Monte-Carlo draw
# --------------------------------------------------------------------------- #
def test_oracle_matches_monte_carlo():
    p = SynthParams(K=5, D=1.5, E=1.0, c_cal=2.0, c_tgt=0.0)
    rng = np.random.default_rng(0)
    n = 1_500_000
    df = S.sample(n, p, "cal", rng)
    tau = 0.6
    for k in range(p.K):
        sub_k = df[df.k == k]
        acc = sub_k[sub_k.s >= tau]
        n_acc = len(acc)
        if n_acc < 500:
            continue
        # accept rate: SE from the full stratum count
        a_mc = (sub_k.s >= tau).mean()
        a_or = S.oracle_accept_rate(tau, k, p)
        se_a = np.sqrt(a_mc * (1 - a_mc) / len(sub_k))
        assert abs(a_mc - a_or) < 5 * se_a + 1e-3

        # selective risk: SE from the accepted count (BENCHMARK_SPEC C1)
        r_mc = 1.0 - acc.y.mean()
        r_or = S.oracle_selective_risk(tau, k, p)
        se_r = np.sqrt(max(r_mc * (1 - r_mc), 1e-6) / n_acc)
        assert abs(r_mc - r_or) < 5 * se_r + 1e-3, (k, r_mc, r_or, se_r)


def test_oracle_R_mix_matches_monte_carlo():
    p = SynthParams(K=6, D=1.0, E=0.5, c_cal=3.0, c_tgt=-1.0)
    rng = np.random.default_rng(1)
    df = S.sample(1_500_000, p, "tgt", rng)
    tau = 0.65
    acc = df[df.s >= tau]
    r_mc = 1.0 - acc.y.mean()
    r_or = S.oracle_R_mix(tau, p.p_tgt(), p)
    se = np.sqrt(r_mc * (1 - r_mc) / len(acc))
    assert abs(r_mc - r_or) < 5 * se + 1e-3


# --------------------------------------------------------------------------- #
# D = 0 control -- the confidence is a perfectly calibrated correctness prob
# --------------------------------------------------------------------------- #
def test_D0_makes_s_calibrated():
    p = SynthParams(K=8, D=0.0, E=1.5, c_cal=3.0)
    rng = np.random.default_rng(2)
    df = S.sample(500_000, p, "cal", rng)
    # pi(z, nu) == sigmoid(z) == s exactly when D=0, for every stratum.
    assert float(np.abs(df.pi - df.s).max()) < 1e-9

    # and empirically, P(correct | s) tracks s across the whole [0,1] range.
    edges = np.linspace(0.1, 0.9, 9)
    for lo, hi in zip(edges[:-1], edges[1:], strict=False):
        m = (df.s >= lo) & (df.s < hi)
        if m.sum() < 2000:
            continue
        emp = df.y[m].mean()
        mid = df.s[m].mean()
        assert abs(emp - mid) < 0.02, (lo, hi, emp, mid)


def test_D0_no_impossibility_gap():
    # T1: Delta(D, T) -> 0 as D -> 0, and grows monotonically in D.
    alpha = 0.20
    deltas = []
    for D in [0.0, 0.5, 1.0, 2.0]:
        p = SynthParams(K=8, D=D, E=0.0, c_cal=3.0, c_tgt=0.0)
        g = S.oracle_impossibility_gap(p, alpha)
        deltas.append(g["delta"])
    assert abs(deltas[0]) < 1e-3          # D=0: no gap
    assert all(np.diff(deltas) > 0)        # strictly increasing in D
    assert deltas[-1] > 0.20               # strong concept shift -> large gap


def test_tau_star_hits_alpha():
    # By construction R_k(tau_k*) == alpha for a controllable stratum.
    p = SynthParams(K=6, D=1.5, E=0.5, c_cal=3.0)
    alpha = 0.20
    for k in range(p.K):
        t = S.oracle_tau_star(k, p, alpha)
        if np.isnan(t) or t == 0.0:
            continue
        r = S.oracle_selective_risk(t, k, p)
        assert abs(r - alpha) < 1e-3, (k, t, r)


# --------------------------------------------------------------------------- #
# T3 -- certificate coverage on synthetic
# --------------------------------------------------------------------------- #
def test_worst_stratum_rcps_validity():
    p = SynthParams(K=6, D=1.5, E=0.5, c_cal=3.0)
    alpha, delta = 0.20, 0.10
    # certify at the oracle per-stratum thresholds; the true worst risk is known.
    taus = []
    for k in range(p.K):
        t = S.oracle_tau_star(k, p, alpha)
        taus.append(0.999 if np.isnan(t) else max(t, 1e-4))
    r_true = np.nan_to_num(
        np.array([S.oracle_selective_risk(taus[k], k, p) for k in range(p.K)]), nan=0.0
    )
    worst_true = float(r_true.max())

    n_trials, hits = 400, 0
    for seed in range(n_trials):
        rng = np.random.default_rng(1000 + seed)
        df = S.sample(3000, p, "cal", rng)
        per = []
        for k in range(p.K):
            acc = df[(df.k == k) & (df.s >= taus[k])]
            per.append((int((1 - acc.y).sum()), len(acc)))
        U = C.worst_stratum_rcps_ucb(per, delta, "hb", True)["U"]
        hits += worst_true <= U
    validity = hits / n_trials
    assert validity >= 1 - delta, validity


def test_dro_ball_covers_target_at_true_tilt():
    p = SynthParams(K=6, D=1.5, E=1.0, c_cal=3.0, c_tgt=0.0)
    alpha = 0.20
    tau = S.oracle_marginal_threshold(p.p_cal(), p, alpha)
    assert not np.isnan(tau)
    acc = S.oracle_accept_rates(tau, p)
    rk = S.oracle_selective_risks(tau, p)
    r_tgt = S.oracle_R_mix(tau, p.p_tgt(), p)
    kl = S.kl_divergence(p.p_tgt(), p.p_cal())

    # below the true tilt the certificate need not cover; at/above it must.
    cert_below = C.dro_ball_certificate(rk, acc, p.p_cal(), 0.5 * kl, "kl")[
        "certified_worst_risk"
    ]
    cert_at = C.dro_ball_certificate(rk, acc, p.p_cal(), kl, "kl")[
        "certified_worst_risk"
    ]
    cert_above = C.dro_ball_certificate(rk, acc, p.p_cal(), 1.5 * kl, "kl")[
        "certified_worst_risk"
    ]
    assert cert_at >= r_tgt - 1e-4
    assert cert_above >= r_tgt - 1e-4
    assert cert_above >= cert_at >= cert_below       # monotone in rho
    # at rho=0 the certificate is exactly R_mix(cal) = alpha (tau solves it).
    cert0 = C.dro_ball_certificate(rk, acc, p.p_cal(), 0.0, "kl")[
        "certified_worst_risk"
    ]
    assert abs(cert0 - alpha) < 1e-3


def test_chi2_closed_form_nonnegative_slack():
    # the sqrt(2 rho Var) closed form must add non-negative slack over R_mix(cal).
    p = SynthParams(K=8, D=2.0, E=0.0, c_cal=3.0)
    tau = 0.7
    acc = S.oracle_accept_rates(tau, p)
    rk = S.oracle_selective_risks(tau, p)
    base = C.dro_ball_certificate(rk, acc, p.p_cal(), 0.0, "chi2")[
        "certified_worst_risk"
    ]
    for rho in [0.05, 0.2, 0.5]:
        cf = C.chi2_closed_form_certificate(rk, acc, p.p_cal(), rho)
        assert cf >= base - 1e-9


# --------------------------------------------------------------------------- #
# Real-data loaders -- exercise, but skip cleanly when offline
# --------------------------------------------------------------------------- #
def test_grouped_split_is_leakage_free():
    rng = np.random.default_rng(0)
    group = np.array([0, 0, 0, 1, 1, 1, 1, 2, 2, 2])
    train, cal, test = RD.grouped_split(group, source_groups=0, rng=rng, cal_frac=0.5)
    assert not np.any(train & cal)
    assert not np.any(train & test)
    assert not np.any(cal & test)
    assert train.sum() == 3                      # all of group 0
    assert (cal | test).sum() == 7               # groups 1 and 2 split into cal/test


def test_electricity_triple_or_skip():
    try:
        tri = RD.electricity_triple(n_blocks=4)
    except RD.SkipDataset as e:
        pytest.skip(f"electricity unavailable: {e}")
    assert {"s", "y", "nu", "split"}.issubset(tri.columns)
    assert tri.s.between(0, 1).all()
    assert set(tri.y.unique()).issubset({0, 1})
    assert tri.nu.nunique() >= 2                  # multiple temporal strata


def test_acs_income_triple_or_skip():
    # small states keep the per-state download tractable while still spanning a
    # real spatial shift (source RI -> target VT, WY).
    try:
        tri = RD.acs_income_triple(
            source_states=("RI",), target_states=("VT", "WY"), max_rows=40_000
        )
    except RD.SkipDataset as e:
        pytest.skip(f"ACS unavailable: {e}")
    assert {"s", "y", "nu", "split"}.issubset(tri.columns)
    assert tri.s.between(0, 1).all()
    assert set(tri.y.unique()).issubset({0, 1})
    assert tri.nu.nunique() >= 2                  # multiple target-state strata
    assert (tri.split == "cal").any() and (tri.split == "test").any()


# --------------------------------------------------------------------------- #
# Genuine P-vs-Q concept drift (Dq) and the irreducible error floor (eps) -- b7
# --------------------------------------------------------------------------- #
def test_concept_gap_zero_without_drift():
    # Dq = 0 keeps eta_Q == eta_P, so the accept-region concept gap is ~0.
    p = SynthParams(K=8, D=1.0, Dq=0.0, c_cal=3.0, c_tgt=0.0)
    g = S.oracle_concept_gap(0.5, p)
    assert abs(g["delta_bar_A"]) < 1e-6
    assert abs(g["R_Q"] - g["R_ref"]) < 1e-6


def test_concept_gap_positive_with_drift_and_matches_sample():
    # Dq > 0 opens a positive accept-region concept gap that weighted CP cannot see,
    # and a target-law draw realizes R_Q = R_ref + Delta_bar (Theorem 1a).
    p = SynthParams(K=8, D=1.0, Dq=2.0, c_cal=3.0, c_tgt=0.0)
    g = S.oracle_concept_gap(0.5, p)
    assert g["delta_bar_A"] > 0.05
    assert abs((g["R_ref"] + g["delta_bar_A"]) - g["R_Q"]) < 1e-9
    rng = np.random.default_rng(3)
    df = S.sample(600_000, p, "tgt", rng)
    acc = df.s.to_numpy() >= g["tau_c"]
    realized = 1.0 - df.y.to_numpy()[acc].mean()
    se = np.sqrt(realized * (1 - realized) / acc.sum())
    assert abs(realized - g["R_Q"]) < 5 * se + 2e-3


def test_error_floor_makes_novel_stratum_uncertifiable():
    # eps_floor > alpha means no threshold controls the novel stratum: tau* is NaN
    # and the achievable risk never drops below the floor (the no-analog tail).
    p = SynthParams(K=8, D=1.0, Dq=1.0, eps_floor=0.30, c_cal=3.0, c_tgt=0.0)
    assert np.isnan(S.oracle_tau_star(p.K - 1, p, alpha=0.20, population="tgt"))
    assert S.oracle_selective_risk(0.999, p.K - 1, p, "tgt") >= 0.30 - 1e-6
