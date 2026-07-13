"""Tests for the multiplicity-control utilities in experiments._common.

Covers the five claim-family tools:
  * holm  -- Holm (1979) step-down FWER-adjusted p-values (textbook vector),
  * bh    -- Benjamini-Hochberg (1995) FDR step-up mask (textbook vectors),
  * romano_wolf_stepdown -- Romano-Wolf (2005) max-t step-down; FWER is held at or
    below the nominal level over a simulated complete null with dependence,
  * iut_all -- Berger (1982) intersection-union logic for "holds for all K",
  * tost_equivalence -- TOST on clearly-equivalent vs clearly-different samples.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments._common import (  # noqa: E402
    DELTA,
    DELTA_JOINT,
    K,
    bh,
    holm,
    iut_all,
    romano_wolf_stepdown,
    tost_equivalence,
)


def test_constants():
    assert K == 5
    assert DELTA == pytest.approx(0.10)
    assert DELTA_JOINT == pytest.approx(0.02)


def test_holm_textbook_vector():
    """Hand-worked Holm example; matches the standard step-down output."""
    p = np.array([0.01, 0.04, 0.03, 0.005])
    # sorted 0.005,0.01,0.03,0.04 * (4,3,2,1) = 0.02,0.03,0.06,0.04
    # running max -> 0.02,0.03,0.06,0.06 ; unsorted back to input order:
    expected = np.array([0.03, 0.06, 0.06, 0.02])
    adj = holm(p)
    assert np.allclose(adj, expected, atol=1e-12)
    # rejections at 0.05 are exactly the two smallest raw p-values
    assert list(np.where(adj <= 0.05)[0]) == [0, 3]
    # adjusted p is never below raw p and never above 1
    assert np.all(adj >= p - 1e-12) and np.all(adj <= 1.0)


def test_holm_monotone_along_sorted_order():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 25)
    adj = holm(p)
    srt = adj[np.argsort(p)]
    assert np.all(np.diff(srt) >= -1e-12)          # non-decreasing along sorted p
    # single-hypothesis Holm is the identity
    assert holm([0.037])[0] == pytest.approx(0.037)


def test_bh_textbook_simple():
    """Largest k with p_(k) <= (k/m) q defines the cutoff; reject up to it."""
    p = np.array([0.005, 0.02, 0.04, 0.5, 0.9])
    mask = bh(p, q=0.05)
    assert list(mask) == [True, True, False, False, False]


def test_bh_stepup_property():
    """A p-value that fails its own threshold is still rejected if a later one passes."""
    p = np.array([0.005, 0.04, 0.028, 0.9, 0.008])
    # sorted 0.005,0.008,0.028,0.04,0.9 vs thresholds 0.01,0.02,0.03,0.04,0.05
    # all of the first four pass -> cutoff at 0.04 -> reject the four smallest
    mask = bh(p, q=0.05)
    assert list(mask) == [True, True, True, False, True]
    assert mask.sum() == 4


def test_bh_rejects_at_least_as_many_as_holm():
    rng = np.random.default_rng(1)
    # a mix of clear signal and null p-values
    p = np.concatenate([rng.uniform(0, 0.01, 8), rng.uniform(0, 1, 20)])
    holm_rej = holm(p) <= DELTA
    bh_rej = bh(p, q=DELTA)
    assert bh_rej.sum() >= holm_rej.sum()

    # empty input is handled
    assert bh(np.array([]), q=0.1).size == 0
    assert holm(np.array([])).size == 0


def test_iut_all_logic():
    assert iut_all([0.01, 0.02, 0.005, 0.04, 0.03], level=0.05) is True
    assert iut_all([0.01, 0.20, 0.005], level=0.05) is False
    # boundary: exactly at level rejects (<=)
    assert iut_all([0.05, 0.01], level=0.05) is True
    # empty conjunction is vacuously true
    assert iut_all([], level=0.05) is True


def test_tost_equivalence_clear_cases():
    rng = np.random.default_rng(2)
    margin = 0.05
    equivalent = rng.normal(0.0, 0.008, 4000)      # tight and centred at zero
    different = rng.normal(0.15, 0.010, 4000)       # mean far beyond the margin
    wide = rng.normal(0.0, 0.060, 4000)             # centred but imprecise
    assert tost_equivalence(equivalent, margin) is True
    assert tost_equivalence(different, margin) is False
    assert tost_equivalence(wide, margin) is False   # equivalence needs precision too


def test_tost_equivalence_edges():
    # zero margin can never be declared equivalent; empty sample is not equivalent
    assert tost_equivalence(np.zeros(100), margin=0.0) is False
    assert tost_equivalence(np.array([]), margin=0.05) is False
    # a one-sided-margin miss (mean just under +margin but CI spills over) fails
    rng = np.random.default_rng(3)
    x = rng.normal(0.045, 0.02, 5000)               # 95th pct ~0.078 > 0.05
    assert tost_equivalence(x, margin=0.05) is False


def _null_cells(rng, n, S, rho):
    """One dataset of S mean-zero cells sharing a common factor (positive dependence)."""
    common = rng.normal(0.0, 1.0, n)
    z = rng.normal(0.0, 1.0, (n, S))
    return rho * common[:, None] + np.sqrt(1.0 - rho * rho) * z   # (n, S), true mean 0


def _boot_stats(rng, x, B):
    """Point estimate, bootstrap matrix (S x B), and bootstrap se for a cell matrix."""
    n = x.shape[0]
    stat = x.mean(axis=0)
    idx = rng.integers(0, n, (B, n))
    boot = x[idx].mean(axis=1).T                    # (S, B)
    se = boot.std(axis=1)
    return stat, boot, se


def test_romano_wolf_controls_fwer_on_dependent_null():
    """Over many complete-null datasets, P(any rejection) <= level (dependence-aware).

    All cells are true nulls sharing a common factor (rho=0.5), mimicking the E12
    grid where cells share the S0 reference and the same complexes. Romano-Wolf must
    hold the family-wise error at or below the nominal level.
    """
    rng = np.random.default_rng(20260712)
    level = 0.10
    # Romano-Wolf is asymptotic; at this sample/bootstrap size the estimate sits at
    # nominal, with the Monte-Carlo SE at sims=1500 about 0.008.
    S, n, B, sims, rho = 6, 250, 600, 1500, 0.5
    rw_hits = 0
    holm_hits = 0
    for _ in range(sims):
        x = _null_cells(rng, n, S, rho)
        stat, boot, se = _boot_stats(rng, x, B)
        res = romano_wolf_stepdown(boot, stat, se, level=level)
        rw_hits += bool(np.any(res["p_adjusted"] <= level))
        t = stat / np.where(se > 0, se, np.inf)
        p_one = 1.0 - norm.cdf(t)                    # one-sided normal p for Holm
        holm_hits += bool(np.any(holm(p_one) <= level))
    fwer_rw = rw_hits / sims
    fwer_holm = holm_hits / sims
    # report through the assertion messages so the numbers show up on failure/-rs
    assert fwer_rw <= level + 0.03, f"RW FWER {fwer_rw:.3f} (Holm {fwer_holm:.3f})"
    assert fwer_holm <= level + 0.03, f"Holm FWER {fwer_holm:.3f}"
    # not vacuously conservative: it does not reject the null essentially always-off
    assert fwer_rw >= 0.02, f"RW FWER implausibly low {fwer_rw:.3f}"
    print(f"\nFWER(dependent null, rho={rho}): romano_wolf={fwer_rw:.3f}  holm={fwer_holm:.3f}  "
          f"(level={level}, sims={sims}, S={S}, B={B})")


def test_romano_wolf_controls_fwer_independent_null():
    """FWER also held under independence (no common factor)."""
    rng = np.random.default_rng(7)
    level = 0.10
    S, n, B, sims = 8, 200, 400, 600
    rw_hits = 0
    for _ in range(sims):
        x = rng.normal(0.0, 1.0, (n, S))
        stat, boot, se = _boot_stats(rng, x, B)
        res = romano_wolf_stepdown(boot, stat, se, level=level)
        rw_hits += bool(np.any(res["p_adjusted"] <= level))
    fwer = rw_hits / sims
    assert fwer <= level + 0.045, f"RW FWER (independent) {fwer:.3f}"
    print(f"\nFWER(independent null): romano_wolf={fwer:.3f} (level={level}, sims={sims})")


def test_romano_wolf_detects_a_real_effect():
    """Power + shape: a large single-cell effect is rejected; the band is one-sided."""
    rng = np.random.default_rng(1)
    S, n, B, rho = 6, 300, 500, 0.4
    effects = np.array([0.6, 0.0, 0.0, 0.0, 0.0, 0.0])
    x = effects[None, :] + _null_cells(rng, n, S, rho)
    stat, boot, se = _boot_stats(rng, x, B)
    res = romano_wolf_stepdown(boot, stat, se, level=DELTA)
    j0 = int(np.argmax(stat))
    assert j0 == 0                                   # cell 0 carries the signal
    assert res["p_adjusted"][0] <= DELTA             # detected after adjustment
    assert res["reject"][0]
    assert res["crit_value"] > 0                     # one-sided simultaneous band
    # a rejected cell clears the studentized simultaneous critical value
    assert res["t_stat"][0] >= res["crit_value"] - 1e-9
    # adjusted p-values are valid probabilities
    assert np.all(res["p_adjusted"] >= 0) and np.all(res["p_adjusted"] <= 1)


def test_romano_wolf_shape_and_edge():
    empty = romano_wolf_stepdown(np.zeros((0, 10)), np.array([]), np.array([]))
    assert empty["p_adjusted"].size == 0
    with pytest.raises(ValueError):
        romano_wolf_stepdown(np.zeros(5), np.zeros(5), np.zeros(5))   # boot must be 2-D


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-s"]))
