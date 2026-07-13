"""Unit tests for the W2 screening-enrichment metrics + selective-screening analysis."""

from __future__ import annotations

import numpy as np
import pytest

from foldgate.selective import (
    active_retention_curve,
    bedroc,
    enrichment_factor,
    random_abstention_ef,
    roc_auc,
    selective_enrichment_curve,
)


def _screen(rng, n=2000, n_act=100, sep=2.0):
    """A screen where actives score higher on average (sep controls separability)."""
    labels = np.zeros(n, int)
    labels[:n_act] = 1
    scores = rng.normal(0, 1, n) + labels * sep
    return scores, labels


def test_enrichment_factor_and_auc():
    rng = np.random.default_rng(0)
    s, y = _screen(rng, sep=3.0)
    assert enrichment_factor(s, y, 0.01) > 5          # strong early enrichment
    assert 0.9 < roc_auc(s, y) <= 1.0
    # random scores -> EF ~ 1, AUC ~ 0.5
    r = rng.normal(0, 1, len(y))
    assert enrichment_factor(r, y, 0.01) < 3
    assert abs(roc_auc(r, y) - 0.5) < 0.1


def test_bedroc_bounds():
    rng = np.random.default_rng(1)
    s, y = _screen(rng, sep=6.0)
    perfect = y + rng.uniform(0, 1e-6, len(y))
    assert bedroc(perfect, y) > 0.95                  # near-perfect early recognition
    assert 0.0 <= bedroc(rng.normal(0, 1, len(y)), y) < 0.3   # random is small


def test_selective_screening_lifts_enrichment():
    """A reliability signal correlated with correctness should raise selective EF above random abstention."""
    rng = np.random.default_rng(2)
    s, y = _screen(rng, n=3000, n_act=150, sep=1.2)
    # reliability: high for confidently-correct calls; decoys scoring high are unreliable
    reliability = (s * (2 * y - 1)) + rng.normal(0, 0.5, len(y))
    curve = selective_enrichment_curve(s, reliability, y, coverages=(1.0, 0.5), frac=0.01)
    ef_full = curve[0]["ef_at_frac"]
    ef_half = curve[1]["ef_at_frac"]
    assert ef_half >= ef_full - 1e-9                  # abstaining does not hurt (usually helps)
    rand_mean, _, rand_hi = random_abstention_ef(s, y, coverage=0.5, frac=0.01)
    assert ef_half >= rand_mean - 1e-9                # beats/matches random abstention at matched coverage


def test_active_retention_decoys_drop_faster():
    rng = np.random.default_rng(3)
    s, y = _screen(rng, n=3000, n_act=150, sep=1.2)
    reliability = (s * (2 * y - 1)) + rng.normal(0, 0.5, len(y))
    ret = active_retention_curve(reliability, y, coverages=(0.5,))
    assert ret[0]["decoy_retained"] < ret[0]["active_retained"]   # gate removes decoys faster


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
