"""Validity + sanity tests for the conformal / selective-prediction core.

The key test asserts the finite-sample guarantee empirically: on i.i.d. data the
LTT gate keeps the true selective risk <= alpha with probability >= 1 - delta.
"""

from __future__ import annotations

import numpy as np
import pytest

from foldgate.conformal import (
    concept_shift_diagnostic,
    continuous_risk_threshold,
    effective_n,
    estimate_weights_cv,
    hb_upper_bound,
    ltt_threshold,
    naive_threshold,
    weighted_ltt_threshold,
    wsr_betting_pvalue,
)
from foldgate.conformal.localized import (
    _synthetic_coverage_check,
    localized_threshold,
)
from foldgate.selective import aurc, clopper_pearson, evaluate_gate, risk_coverage_curve


def _synthetic(rng, n, beta=2.5):
    """Scores in [0,1]; P(correct) increases with score (an informative gate)."""
    s = rng.uniform(0, 1, n)
    p = 1 / (1 + np.exp(-beta * (2 * s - 1)))
    y = (rng.uniform(0, 1, n) < p).astype(int)
    return s, y


def test_hb_upper_bound_properties():
    assert hb_upper_bound(0.1, 100, 0.1) >= 0.1
    assert hb_upper_bound(0.1, 100, 0.1) <= 1.0
    # more data -> tighter (smaller) bound at fixed r_hat
    assert hb_upper_bound(0.1, 1000, 0.1) < hb_upper_bound(0.1, 50, 0.1)
    assert hb_upper_bound(0.99, 20, 0.1) <= 1.0


def test_ltt_controls_risk_on_iid():
    """Over many calibration draws, true risk of the accept region <= alpha
    at least (1 - delta) of the time (finite-sample LTT guarantee)."""
    rng = np.random.default_rng(0)
    alpha, delta = 0.2, 0.1
    n_trials, n_cal, n_eval = 300, 1500, 20000
    held = 0
    accepted_any = 0
    for _ in range(n_trials):
        s_cal, y_cal = _synthetic(rng, n_cal)
        tau = ltt_threshold(s_cal, y_cal, alpha=alpha, delta=delta)
        if tau is None:
            held += 1  # abstaining trivially satisfies the risk constraint
            continue
        accepted_any += 1
        # "true" risk of {s >= tau}: estimate on a large fresh sample
        s_ev, y_ev = _synthetic(rng, n_eval)
        r = evaluate_gate(s_ev, y_ev, tau)
        if not np.isfinite(r["selective_risk"]) or r["selective_risk"] <= alpha:
            held += 1
    assert accepted_any > n_trials * 0.5, "gate should usually accept something"
    assert held / n_trials >= 1 - delta - 0.03, f"guarantee held only {held/n_trials:.3f}"


def test_finite_fold_indicator_is_downward_biased():
    """The RNP 'realized risk <= alpha on a finite fold' fraction under-estimates P(true risk <= alpha).

    Demonstrates (not asserts) the E1 defense: with a tight certifier the true risk sits at alpha,
    so a finite test fold crosses alpha ~half the time even when the guarantee holds. The
    large-sample (near-true-risk) indicator meets 1 - delta; the small-fold indicator sits below it.
    """
    rng = np.random.default_rng(20)
    alpha, delta = 0.2, 0.1
    small, large = 1500, 40000
    held_small = held_large = trials = 0
    for _ in range(300):
        s_cal, y_cal = _synthetic(rng, 1500)
        tau = ltt_threshold(s_cal, y_cal, alpha=alpha, delta=delta)
        if tau is None:
            continue
        trials += 1
        ss, ys = _synthetic(rng, small)
        sl, yl = _synthetic(rng, large)
        rs = evaluate_gate(ss, ys, tau)["selective_risk"]
        rl = evaluate_gate(sl, yl, tau)["selective_risk"]
        held_small += (not np.isfinite(rs)) or rs <= alpha
        held_large += (not np.isfinite(rl)) or rl <= alpha
    f_small, f_large = held_small / trials, held_large / trials
    assert f_large >= 1 - delta - 0.03           # true-risk guarantee holds
    assert f_small < f_large - 0.02              # finite-fold indicator is downward-biased


def test_ltt_is_more_conservative_than_naive():
    rng = np.random.default_rng(1)
    s, y = _synthetic(rng, 2000)
    tau_ltt = ltt_threshold(s, y, alpha=0.2, delta=0.1)
    tau_naive = naive_threshold(s, y, alpha=0.2)
    assert tau_ltt is not None and tau_naive is not None
    assert tau_ltt >= tau_naive - 1e-9  # LTT accepts no more than the naive rule


def test_aurc_ranking():
    rng = np.random.default_rng(2)
    s, y = _synthetic(rng, 5000, beta=4.0)
    perfect = y + rng.uniform(0, 1e-6, len(y))       # near-perfect ordering
    random = rng.uniform(0, 1, len(y))
    assert aurc(perfect, y) < aurc(s, y) < aurc(random, y) + 0.05
    base_err = 1 - y.mean()
    assert abs(aurc(random, y) - base_err) < 0.03      # random gate ~ base error rate


def test_risk_coverage_shape():
    rng = np.random.default_rng(3)
    s, y = _synthetic(rng, 1000)
    cov, risk = risk_coverage_curve(s, y)
    assert cov[0] > 0 and abs(cov[-1] - 1.0) < 1e-9
    assert abs(risk[-1] - (1 - y.mean())) < 1e-9       # full coverage = base error


def test_wsr_betting_pvalue_valid_and_powerful():
    """False-certification rate at the null boundary <= delta; high power below it."""
    rng = np.random.default_rng(10)
    delta, target, n = 0.10, 0.20, 400
    false_cert = sum(
        wsr_betting_pvalue((rng.random(n) < target).astype(float), target, delta) <= delta
        for _ in range(1500)
    )
    assert false_cert / 1500 <= delta + 0.02
    power = sum(
        wsr_betting_pvalue((rng.random(n) < 0.10).astype(float), target, delta) <= delta
        for _ in range(300)
    )
    assert power / 300 > 0.9


def test_continuous_binomial_reproduces_binary_ltt():
    """Degenerate check: a 0/1 loss with the binomial bound == the binary LTT gate."""
    rng = np.random.default_rng(11)
    s, y = _synthetic(rng, 3000, beta=3.0)
    loss01 = (1 - y).astype(float)
    tau_c = continuous_risk_threshold(s, loss01, target=0.2, delta=0.1, bound="binomial")
    tau_b = ltt_threshold(s, y, alpha=0.2, delta=0.1)
    assert (tau_c is None and tau_b is None) or np.isclose(tau_c, tau_b)


def test_continuous_gate_wsr_dominates_hoeffding():
    """WSR certifies at least as much coverage as the distribution-free Hoeffding bound."""
    rng = np.random.default_rng(12)
    s, y = _synthetic(rng, 4000, beta=3.0)
    rmsd = np.abs(rng.normal(0, 1, len(s))) * np.exp(-1.5 * s) * 2.0
    loss = np.minimum(rmsd, 4.0) / 4.0
    tau_w = continuous_risk_threshold(s, loss, 0.30, delta=0.1, bound="wsr")
    tau_h = continuous_risk_threshold(s, loss, 0.30, delta=0.1, bound="hoeffding")
    cov_w = float((s >= tau_w).mean()) if tau_w is not None else 0.0
    cov_h = float((s >= tau_h).mean()) if tau_h is not None else 0.0
    assert cov_w >= cov_h - 1e-9


def test_continuous_gate_controls_true_risk_iid():
    """The WSR continuous gate keeps the true mean bounded-loss <= target >= 1 - delta."""
    rng = np.random.default_rng(13)
    alpha_loss, delta = 0.30, 0.10
    held, accepted = 0, 0
    for _ in range(200):
        s, _ = _synthetic(rng, 1500)
        loss = np.clip(rng.beta(2, 5, len(s)) * (1.4 - s), 0, 1)  # loss falls as score rises
        tau = continuous_risk_threshold(s, loss, alpha_loss, delta=delta, bound="wsr")
        if tau is None:
            held += 1
            continue
        accepted += 1
        s_ev = rng.uniform(0, 1, 20000)
        loss_ev = np.clip(rng.beta(2, 5, len(s_ev)) * (1.4 - s_ev), 0, 1)
        acc = s_ev >= tau
        if not acc.any() or loss_ev[acc].mean() <= alpha_loss:
            held += 1
    assert accepted > 100
    assert held / 200 >= 1 - delta - 0.03


def _shift_data(rng, n, mu):
    """Pure covariate shift: X ~ N(mu,1); P(Y|X)=sigmoid(1.5 X) is stable; score=X+noise."""
    x = rng.normal(mu, 1.0, n)
    s = x + rng.normal(0, 0.3, n)
    y = (rng.random(n) < 1 / (1 + np.exp(-1.5 * x))).astype(int)
    return x, s, y


def test_weighted_ltt_controls_target_risk_with_true_weights():
    """With correct (density-ratio) weights, weighted-LTT controls the TARGET risk."""
    rng = np.random.default_rng(14)
    # B must cover the max density ratio; a moderate clip keeps the betting powerful.
    alpha, delta, mu, B = 0.20, 0.10, 0.5, 4.0
    held, accepted = 0, 0
    for _ in range(200):
        xc, sc, yc = _shift_data(rng, 1500, 0.0)          # labelled source
        w = np.exp(mu * xc - 0.5 * mu * mu)               # exact N(mu,1)/N(0,1) ratio
        w = np.clip(w, 1.0 / B, B)
        tau = weighted_ltt_threshold(sc, yc, w, alpha=alpha, delta=delta, clip_ceiling=B)
        if tau is None:
            held += 1
            continue
        accepted += 1
        _, se, ye = _shift_data(rng, 20000, mu)           # fresh target
        acc = se >= tau
        if not acc.any() or (1 - ye[acc]).mean() <= alpha:
            held += 1
    assert accepted > 100
    assert held / 200 >= 1 - delta - 0.04


def test_concept_shift_diagnostic_detects_shift():
    rng = np.random.default_rng(15)
    s = rng.uniform(0, 1, 4000)
    y_stable = (rng.random(4000) < s).astype(int)          # same map
    y_shifted = (rng.random(4000) < np.clip(s - 0.3, 0, 1)).astype(int)  # degraded map
    s2 = rng.uniform(0, 1, 4000)
    y2 = (rng.random(4000) < s2).astype(int)
    no_shift = concept_shift_diagnostic(s, y_stable, s2, y2)["mean_abs_gap_target_weighted"]
    shift = concept_shift_diagnostic(s, y_stable, s, y_shifted)["mean_abs_gap_target_weighted"]
    assert shift > no_shift + 0.1


def test_estimate_weights_cv_no_shift_is_near_one():
    """With source and target from the same distribution, weights concentrate near 1."""
    rng = np.random.default_rng(16)
    cal = rng.normal(0, 1, (1500, 2))
    tgt = rng.normal(0, 1, (1500, 2))
    w = estimate_weights_cv(cal, tgt, clip=10.0, seed=0)
    assert abs(float(np.median(w)) - 1.0) < 0.25
    assert effective_n(w) > 0.7 * len(w)          # low weight variance under no shift


def test_estimate_weights_cv_upweights_shifted_region():
    """Under a mean shift, calibration points near the target get larger weights."""
    rng = np.random.default_rng(17)
    cal = rng.normal(0.0, 1.0, (2000, 1))
    tgt = rng.normal(1.5, 1.0, (2000, 1))
    w = estimate_weights_cv(cal, tgt, clip=20.0, seed=0)
    hi = cal[:, 0] > 1.0
    lo = cal[:, 0] < -1.0
    assert w[hi].mean() > w[lo].mean()            # target-like (high) points weighted up


def test_rlcp_restores_marginal_coverage():
    """RLCP hits the 1 - alpha marginal target; plain non-randomized localization misses.

    Randomizing the kernel centre plus keeping the +inf test mass is what restores
    exact finite-sample marginal validity (Hore & Barber 2024). The self-check
    refreshes calibration each trial, so this is a genuine marginal-coverage
    estimate, not a conditional-on-one-calibration-set artefact.
    """
    res = _synthetic_coverage_check(seed=0)
    cov = res["coverage"]
    target = res["target"]
    assert cov["rlcp"] >= target - 0.006, f"RLCP under target: {cov['rlcp']:.3f}"
    assert cov["plain"] < target, f"plain localization should under-cover: {cov['plain']:.3f}"
    assert cov["rlcp"] > cov["plain"] + 0.008     # randomization + mass buys back coverage


def test_localized_threshold_adapts_and_abstains():
    """The localized gate lifts tau as alpha tightens and abstains on undefined coords."""
    rng = np.random.default_rng(4)
    n = 4000
    coords = rng.uniform(0, 1, n)                  # novelty axis, higher = more familiar
    p = 1 / (1 + np.exp(-6.0 * (coords - 0.25)))   # error rises toward the novel end
    s = coords + rng.normal(0, 0.1, n)             # confidence tracks familiarity
    y = (rng.uniform(0, 1, n) < p).astype(int)

    q = np.array([0.6, 0.85])                       # moderate- and low-error queries
    tau_loose = localized_threshold(s, y, coords, q, alpha=0.30,
                                    generator=np.random.default_rng(0))
    tau_tight = localized_threshold(s, y, coords, q, alpha=0.10,
                                    generator=np.random.default_rng(0))
    both = np.isfinite(tau_loose) & np.isfinite(tau_tight)
    assert both.any()
    # tightening the error target never accepts more (tau at least as high)
    assert np.all(tau_tight[both] >= tau_loose[both] - 1e-9)
    # at the moderate-error query the tighter target must strictly raise the bar
    assert tau_tight[0] > tau_loose[0] + 0.1

    # a query with no localizer coordinate abstains (NaN)
    tau_nan = localized_threshold(s, y, coords, np.array([np.nan]), alpha=0.2,
                                  generator=np.random.default_rng(1))
    assert np.isnan(tau_nan[0])


def test_shift_decomp_identity_and_realized_gap():
    """Gap_total = Gap_concept + Gap_covariate, and equals R_target - R_source."""
    from foldgate.conformal.shift_decomp import shift_decomposition

    rng = np.random.default_rng(0)
    n = 4000
    s_src = rng.uniform(0, 1, n)
    y_src = (rng.random(n) < np.clip(0.3 + 0.5 * s_src, 0, 1)).astype(int)
    s_tgt = rng.uniform(0, 1, n)
    y_tgt = (rng.random(n) < np.clip(0.1 + 0.5 * s_tgt, 0, 1)).astype(int)  # worse map
    d = shift_decomposition(s_src, y_src, s_tgt, y_tgt, tau=0.0, n_bins=5, seed=0)
    assert abs(d["gap_total"] - (d["gap_concept"] + d["gap_covariate"])) < 1e-9
    assert abs(d["gap_total"] - (d["R_target"] - d["R_source"])) < 1e-9
    assert d["ci"][0] <= d["ci"][1]
    assert d["floor_lower"] <= d["ci"][1] + 1e-9


def test_shift_decomp_synthetic_calibration():
    """Covariate-shift null covers zero near the nominal rate; concept shift is detected.

    The (1 - delta) interval covers zero about 90% of the time under the covariate
    null, so we check the rate across seeds rather than assert one draw. Pure concept
    shift is always flagged non-vacuous.
    """
    from foldgate.conformal.shift_decomp import _synthetic_check

    k = 12
    cov_covers = sum(_synthetic_check(seed=sd)["covariate_only"]["covers_zero"] for sd in range(k))
    con_detect = sum(_synthetic_check(seed=sd)["concept_only"]["concept_nonvacuous"] for sd in range(k))
    assert cov_covers >= 0.75 * k, f"covariate null covered zero only {cov_covers}/{k}"
    assert con_detect == k, f"concept shift detected only {con_detect}/{k}"


def test_clopper_pearson_bracket():
    lo, hi = clopper_pearson(50, 100, ci=0.90)
    assert lo < 0.5 < hi and 0.0 <= lo < hi <= 1.0
    assert clopper_pearson(0, 10)[0] == 0.0
    assert clopper_pearson(10, 10)[1] == 1.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
