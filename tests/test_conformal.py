"""Validity + sanity tests for the conformal / selective-prediction core.

The key test asserts the finite-sample guarantee empirically: on i.i.d. data the
LTT gate keeps the true selective risk <= alpha with probability >= 1 - delta.
"""

from __future__ import annotations

import numpy as np
import pytest

from foldgate.conformal import hb_upper_bound, ltt_threshold, naive_threshold
from foldgate.selective import aurc, evaluate_gate, risk_coverage_curve


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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
