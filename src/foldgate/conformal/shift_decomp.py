"""Concept-vs-covariate decomposition of the target selective-risk gap.

E3b showed the weighted-LTT gate abstains on novel-pocket strata instead of
certifying them. The reason this project keeps stating: the shift is not pure
covariate shift on the confidence score. When the score-to-correctness map
P(correct | s) itself moves on the novel regime, importance reweighting on the
score cannot restore the target selective risk, because reweighting can only
re-match the score marginal P(s) and leaves the per-score label map untouched.

This module turns that empirical observation into a scoped, confidence-interval'd
PROPOSITION. Restricted to the accept region {confidence >= tau}, decompose the
realized selective-risk gap between a familiar SOURCE and a novel TARGET as

    Gap_total(tau)   = R_target(tau) - R_source(tau)                     (realized)
                     = A_source - A_target                               (accuracies)
    Gap_concept(tau) = sum_bins m_target(b) * (p_source(b) - p_target(b))
                     = A_score-transported - A_target
    Gap_covariate    = Gap_total - Gap_concept
                     = sum_bins (m_source(b) - m_target(b)) * p_source(b)

where, within score bin b of the accept region,
    p_source(b)  = P(correct | s in b, source),
    p_target(b)  = P(correct | s in b, target),
    m_target(b)  = fraction of accepted target mass in bin b (sum_b m_target = 1),
    A_source     = P(correct | source, accept),
    A_target     = P(correct | target, accept),
    A_score-transported = sum_b m_target(b) p_source(b) = the accuracy a caller
                   would PREDICT for the target by transporting the source label
                   map onto the target score distribution, i.e. the best any
                   covariate reweighting on the score can do.

Then  Gap_concept = (1 - A_target) - (1 - A_score-transported)
                  = R_target(tau) - R_source-reweighted-to-target(tau),
the residual target selective risk that remains after the source is optimally
reweighted on the score to match the target score marginal. This is the accept-
region, score-space analogue of the irreducible joint-error (lambda) term in the
Ben-David et al. 2010 domain-adaptation bound, and it is exactly the term
weighted conformal assumes away: Tibshirani, Foygel Barber, Candes and Ramdas
2019 prove weighted CP is valid under PURE covariate shift, i.e. Gap_concept = 0.

PROPOSITION (under one explicit assumption). If the covariate available to the
reweighting is the confidence score s (as it is for the weighted-CP gate here)
and P(correct | s) is homogeneous within each score bin, then Gap_concept is the
part of Gap_total that no covariate reweighting on the score can close. Its
finite-sample lower confidence bound is therefore a certified floor on the target
selective risk that score-based weighted CP cannot repair, and it certifies that
the group-conditional route (which spends target labels rather than reweighting)
is necessary rather than merely convenient.

What is and is NOT guaranteed:

  * Gap_concept carries a finite-sample confidence interval (a two-level
    nonparametric bootstrap over the accepted source and target samples, which
    propagates both the source label-map uncertainty and the target sample and
    mass uncertainty). ``concept_nonvacuous`` is True iff that interval excludes
    zero. This is the certified, reportable quantity.
  * The proposition holds MEASURED ON THE SCORE. A reweighting that used a richer
    covariate than s could in principle close more of the gap; on the confidence
    score, which is what the weighted-CP gate reweights, it cannot. Binning also
    approximates P(correct | s) as bin-wise constant. Both are stated as
    assumptions, not proven, so this is a scoped proposition rather than an
    impossibility theorem.
  * Gap_covariate is DESCRIPTIVE only. It is the residual Gap_total - Gap_concept,
    it depends on the binning, and it carries no coverage certificate. Do not
    report it as a guarantee.
  * On a thin extreme stratum (S4, n ~ 76 per model) the bootstrap interval is
    honestly wide and typically covers zero: the floor is vacuous there and we
    say so, rather than over-claiming a certificate the sample cannot support.
"""

from __future__ import annotations

import numpy as np


def _accept_bins(
    conf_source: np.ndarray,
    conf_target: np.ndarray,
    tau: float,
    n_bins: int,
) -> np.ndarray:
    """Quantile bin edges over the pooled accept-region scores {s >= tau}.

    Edges are fixed once from the pooled accepted source and target scores so the
    bootstrap resamples share a common, data-independent binning. The outer edges
    are opened to +-inf so every accepted point lands in a bin.
    """
    pooled = np.concatenate(
        [conf_source[conf_source >= tau], conf_target[conf_target >= tau]]
    )
    if pooled.size == 0:
        return np.array([-np.inf, np.inf])
    edges = np.quantile(pooled, np.linspace(0.0, 1.0, n_bins + 1))
    edges = np.unique(edges)          # collapse ties so digitize stays monotone
    if edges.size < 2:
        edges = np.array([pooled.min(), pooled.max() + 1e-9])
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def _bin_index(scores: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Bin index in [0, n_bins-1] for each score under fixed edges."""
    idx = np.digitize(scores, edges[1:-1], right=False)
    return idx.astype(int)


def _source_map(bin_s: np.ndarray, corr_s: np.ndarray, n_bins: int) -> np.ndarray:
    """Per-bin source correctness rate p_source(b), with a global-mean fallback.

    A bin with no source points in a given (bootstrap) resample is imputed with
    the overall accepted-source accuracy, which keeps the transported accuracy
    well-defined without inventing bin-local structure.
    """
    cnt = np.bincount(bin_s, minlength=n_bins).astype(float)
    hit = np.bincount(bin_s, weights=corr_s.astype(float), minlength=n_bins)
    fallback = float(corr_s.mean()) if corr_s.size else 0.0
    with np.errstate(invalid="ignore", divide="ignore"):
        p = np.where(cnt > 0, hit / np.maximum(cnt, 1.0), fallback)
    return p


def shift_decomposition(
    conf_source: np.ndarray,
    correct_source: np.ndarray,
    conf_target: np.ndarray,
    correct_target: np.ndarray,
    tau: float,
    n_bins: int = 5,
    delta: float = 0.10,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict:
    """Decompose the accept-region selective-risk gap into concept + covariate parts.

    Restricted to the accept region {confidence >= tau}, returns the signed
    concept-shift term Gap_concept with a finite-sample confidence interval, the
    realized total gap Gap_total = R_target(tau) - R_source(tau), and the
    descriptive covariate-repairable remainder Gap_covariate.

    Sign convention. All gaps are error-rate (risk) differences target-minus-source.
    Gap_concept > 0 means the source is more accurate than the target WITHIN matched
    score bins: the confidence overstates correctness on the novel regime, and that
    excess target risk survives any score reweighting.

    The Gap_concept confidence interval is a two-level nonparametric bootstrap. Each
    resample draws accepted source points (rebuilding the per-bin source map
    p_source) and, independently, accepted target points (rebuilding both the target
    mass m_target and the target accuracy), so the interval propagates source
    label-map uncertainty together with target sample and mass uncertainty. Bin
    edges are held fixed across resamples. ``ci`` is the two-sided (1 - delta)
    percentile interval; ``floor_lower`` is the one-sided (1 - delta) lower bound,
    the certified floor on the concept gap.

    Guarantees. Gap_concept and its interval are the certified output, under the
    proposition stated in the module docstring (reweighting is on the score; label
    map bin-wise constant). Gap_covariate is descriptive and uncertified. See the
    module docstring for the scope and the thin-stratum caveat.

    Parameters
    ----------
    conf_source, correct_source : source (familiar) confidence and 0/1 correctness.
    conf_target, correct_target : target (novel) confidence and 0/1 correctness.
    tau        : accept threshold; a pose is accepted iff its confidence >= tau.
    n_bins     : quantile bins over the pooled accept-region scores.
    delta      : interval failure probability (0.10 -> 90% interval / lower bound).
    n_boot     : bootstrap resamples.
    seed       : RNG seed for the bootstrap.
    """
    cs = np.asarray(conf_source, dtype=float)
    ys = np.asarray(correct_source, dtype=int)
    ct = np.asarray(conf_target, dtype=float)
    yt = np.asarray(correct_target, dtype=int)

    acc_s = cs >= tau
    acc_t = ct >= tau
    n_acc_s = int(acc_s.sum())
    n_acc_t = int(acc_t.sum())

    base = {
        "tau": float(tau),
        "n_bins": int(n_bins),
        "delta": float(delta),
        "n_accept_source": n_acc_s,
        "n_accept_target": n_acc_t,
    }
    if n_acc_s == 0 or n_acc_t == 0:
        base.update(
            gap_total=float("nan"),
            gap_concept=float("nan"),
            gap_covariate=float("nan"),
            R_source=float("nan"),
            R_target=float("nan"),
            ci=[float("nan"), float("nan")],
            floor_lower=float("nan"),
            concept_nonvacuous=False,
            note="empty accept region on at least one side",
        )
        return base

    cs_a, ys_a = cs[acc_s], ys[acc_s]
    ct_a, yt_a = ct[acc_t], yt[acc_t]

    edges = _accept_bins(cs, ct, tau, n_bins)
    nb = len(edges) - 1
    bin_s = _bin_index(cs_a, edges)
    bin_t = _bin_index(ct_a, edges)

    # Point estimates. A_* are accuracies among accepted; risks R_* = 1 - A_*.
    p_source = _source_map(bin_s, ys_a, nb)
    a_source = float(ys_a.mean())                       # A_source
    a_target = float(yt_a.mean())                       # A_target
    a_transported = float(p_source[bin_t].mean())       # sum_b m_target(b) p_source(b)

    r_source = 1.0 - a_source
    r_target = 1.0 - a_target
    gap_total = a_source - a_target                     # R_target - R_source
    gap_concept = a_transported - a_target              # residual after score reweighting
    gap_covariate = a_source - a_transported            # Gap_total - Gap_concept

    # Two-level bootstrap for the Gap_concept interval.
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=float)
    boot_ref = np.empty(n_boot, dtype=float)
    ys_a_f = ys_a.astype(float)
    yt_a_f = yt_a.astype(float)
    for b in range(n_boot):
        si = rng.integers(0, n_acc_s, n_acc_s)
        p_src_b = _source_map(bin_s[si], ys_a_f[si], nb)
        ti = rng.integers(0, n_acc_t, n_acc_t)
        boot[b] = float(np.mean(p_src_b[bin_t[ti]] - yt_a_f[ti]))
        # R_ref itself, on the same resample: the reference certificate a covariate reweighting
        # of the score would report. Bootstrapped here so downstream users can compare a certified
        # bound AGAINST R_ref without treating R_ref as if it were known exactly.
        boot_ref[b] = 1.0 - float(np.mean(p_src_b[bin_t[ti]]))

    lo = float(np.quantile(boot, delta / 2.0))
    hi = float(np.quantile(boot, 1.0 - delta / 2.0))
    floor_lower = float(np.quantile(boot, delta))       # one-sided (1 - delta) lower bound
    nonvacuous = bool(lo > 0.0 or hi < 0.0)
    r_ref = 1.0 - a_transported
    r_ref_upper = float(np.quantile(boot_ref, 1.0 - delta))   # one-sided (1 - delta) UPPER bound

    # Per-bin table for auditability (point-estimate quantities).
    bins = []
    m_t = np.bincount(bin_t, minlength=nb).astype(float)
    m_t = m_t / m_t.sum() if m_t.sum() > 0 else m_t
    for bidx in range(nb):
        ms = bin_s == bidx
        mt = bin_t == bidx
        ps = float(ys_a[ms].mean()) if ms.any() else float("nan")
        pt = float(yt_a[mt].mean()) if mt.any() else float("nan")
        bins.append({
            "lo": float(edges[bidx]),
            "hi": float(edges[bidx + 1]),
            "n_source": int(ms.sum()),
            "n_target": int(mt.sum()),
            "p_correct_source": ps,
            "p_correct_target": pt,
            "m_target": float(m_t[bidx]),
            "signed_contribution": float(m_t[bidx] * (ps - pt)) if (ms.any() and mt.any()) else float("nan"),
        })

    base.update(
        R_source=r_source,
        R_target=r_target,
        gap_total=float(gap_total),
        gap_concept=float(gap_concept),
        gap_covariate=float(gap_covariate),
        ci=[lo, hi],
        floor_lower=floor_lower,
        concept_nonvacuous=nonvacuous,
        boot_mean=float(boot.mean()),
        R_ref=float(r_ref),
        R_ref_upper=r_ref_upper,
        bins=bins,
    )
    return base


def _synthetic_check(seed: int = 0) -> dict:
    """Validity check: pure covariate shift covers zero; pure concept shift is detected.

    Both cases share support on the score, the overlap condition weighted CP itself
    needs, so the test probes the estimator rather than a degenerate no-overlap
    regime. Scores sit in five tight clusters that the quantile bins align to, so
    within a bin the score is effectively constant and there is no map-curvature
    bias to confound the concept null.

    PURE COVARIATE SHIFT. Same label map P(correct | s) on both sides, only the mass
    P(s) over the score clusters differs (target mass pushed toward low-score
    clusters). Gap_concept should be indistinguishable from zero: its interval
    covers zero.

    PURE CONCEPT SHIFT. Same mass P(s), but the target label map is uniformly worse
    at every score (correctness probability dropped by a constant). Gap_concept
    should be detected as non-vacuous and positive.
    """
    rng = np.random.default_rng(seed)
    n = 8000
    tau = 0.0                       # accept everything; the shift lives above tau
    levels = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    p_map = np.array([0.45, 0.55, 0.65, 0.75, 0.85])   # source P(correct | cluster)
    jitter = 0.03                   # << 0.2 cluster spacing, so bins stay aligned
    mass_src = np.array([0.10, 0.15, 0.20, 0.25, 0.30])
    mass_tgt = np.array([0.30, 0.25, 0.20, 0.15, 0.10])
    mass_src = mass_src / mass_src.sum()
    mass_tgt = mass_tgt / mass_tgt.sum()

    def draw(mass, pmap):
        idx = rng.choice(len(levels), size=n, p=mass)
        s = levels[idx] + rng.normal(0.0, jitter, n)
        y = (rng.random(n) < np.clip(pmap[idx], 0.0, 1.0)).astype(int)
        return s, y

    # Covariate only: same label map, different cluster mass.
    cs, ys = draw(mass_src, p_map)
    ct, yt = draw(mass_tgt, p_map)
    cov = shift_decomposition(cs, ys, ct, yt, tau=tau, n_bins=5, delta=0.10, seed=seed)

    # Concept only: same cluster mass, target map dropped by a constant.
    cs2, ys2 = draw(mass_src, p_map)
    ct2, yt2 = draw(mass_src, p_map - 0.25)
    con = shift_decomposition(cs2, ys2, ct2, yt2, tau=tau, n_bins=5, delta=0.10, seed=seed)

    return {
        "covariate_only": {
            "gap_concept": cov["gap_concept"],
            "ci": cov["ci"],
            "concept_nonvacuous": cov["concept_nonvacuous"],
            "covers_zero": not cov["concept_nonvacuous"],
        },
        "concept_only": {
            "gap_concept": con["gap_concept"],
            "ci": con["ci"],
            "floor_lower": con["floor_lower"],
            "concept_nonvacuous": con["concept_nonvacuous"],
        },
    }


if __name__ == "__main__":
    res = _synthetic_check()
    cov, con = res["covariate_only"], res["concept_only"]
    print("shift_decomposition synthetic validity check\n")
    print("PURE COVARIATE SHIFT (expect CI to cover 0, concept vacuous):")
    print(f"  gap_concept = {cov['gap_concept']:+.4f}  ci = "
          f"[{cov['ci'][0]:+.4f}, {cov['ci'][1]:+.4f}]  "
          f"covers_zero = {cov['covers_zero']}")
    print("\nPURE CONCEPT SHIFT (expect CI above 0, concept non-vacuous):")
    print(f"  gap_concept = {con['gap_concept']:+.4f}  ci = "
          f"[{con['ci'][0]:+.4f}, {con['ci'][1]:+.4f}]  "
          f"floor_lower = {con['floor_lower']:+.4f}  "
          f"nonvacuous = {con['concept_nonvacuous']}")
    ok = cov["covers_zero"] and con["concept_nonvacuous"]
    print(f"\nvalidity check {'PASSED' if ok else 'FAILED'}")
