"""Randomly-localized conformal (RLCP) for a novelty-adaptive accept threshold.

Group-conditional (Mondrian) calibration repairs coverage on novel strata but
pays for it with a piecewise-constant threshold: every pose in a stratum shares
one tau, and the stratum edges are an arbitrary discretisation of a continuous
novelty axis. RLCP replaces the discretisation with a threshold that slides
smoothly along novelty, so a pose that sits at the familiar edge of a stratum is
not held to the same bar as one at its novel edge.

The construction is Hore & Barber, "Conformal prediction with local weights:
randomization enables robust guarantees" (arXiv:2310.07850, JRSSB 2024), which
fixes Guan's Localized CP (arXiv:2106.08460). Localized CP weights each
calibration score by a kernel centred on the test point and takes the weighted
quantile; centring the kernel deterministically on the test point breaks the
exchangeability that split conformal relies on, so its finite-sample coverage is
only approximate. RLCP draws a RANDOM reference point Xtilde from the kernel
around the test point, weights calibration scores by w_i = K_h(X_i, Xtilde),
keeps the +infinity point mass on the test score (weight K_h(x*, Xtilde)), and
takes that weighted quantile. Randomising the centre restores a symmetry across
the n+1 points, which buys back EXACT finite-sample MARGINAL coverage while the
kernel still concentrates the quantile on the test point's neighbourhood.

What is and is not guaranteed (read before citing):

* Exact MARGINAL coverage 1 - alpha, finite-sample, via the randomisation. This
  is the Hore-Barber theorem and it is what the synthetic self-check validates.
* APPROXIMATE neighbourhood-conditional coverage: the kernel concentrates the
  guarantee locally, and it tightens toward pointwise-conditional as the
  bandwidth shrinks under smoothness of the score distribution in the localizer.
  Exact distribution-free POINTWISE conditional coverage is impossible (Barber,
  Candes, Ramdas, Tibshirani 2021, arXiv:1903.04684), so we never claim it.
* RLCP is a statement about prediction-SET coverage of the score quantile. Our
  gate reads the same randomly-localized weighted quantile machinery to set a
  smoothly-varying accept threshold tau(novelty), then reports realized selective
  risk EMPIRICALLY. We do not claim a new finite-sample selective-RISK theorem
  beyond the RLCP marginal-coverage guarantee.

The localizer coordinate here is ``ligand_similarity`` (max Tanimoto to the
training set, higher = more familiar). Where the kernel goes thin -- deep in the
no-analog tail, or on rows whose similarity is undefined -- the gate returns NaN
rather than a threshold, so the reader sees exactly where the certificate is
vacuous instead of a silently over-confident number.
"""

from __future__ import annotations

import numpy as np


def _kernel_weights(anchor: float, coords: np.ndarray, h: float, kernel: str) -> np.ndarray:
    """Kernel weight K_h(anchor, coords) on the 1-D localizer axis."""
    d = np.abs(coords - anchor)
    if kernel == "box":
        return (d <= h).astype(float)
    if kernel == "gaussian":
        return np.exp(-0.5 * (d / h) ** 2)
    raise ValueError(f"unknown kernel {kernel!r}")


def _draw_anchor(query: float, h: float, kernel: str, generator: np.random.Generator) -> float:
    """Sample the RLCP reference point Xtilde ~ K_h(., query).

    The randomisation density must equal (up to normalisation) the kernel used as
    the weight, and the kernel must be translation-invariant so its normaliser
    does not depend on ``query``; both hold for the box and Gaussian kernels here.
    A box kernel draws uniformly on [query - h, query + h]; a Gaussian draws
    N(query, h^2). Returning ``query`` unchanged recovers deterministic (Guan)
    localization, which the self-check uses as the non-randomized comparator.
    """
    if kernel == "box":
        return float(query + generator.uniform(-h, h))
    if kernel == "gaussian":
        return float(query + generator.normal(0.0, h))
    raise ValueError(f"unknown kernel {kernel!r}")


def kish_ess(w: np.ndarray) -> float:
    """Kish effective sample size (sum w)^2 / sum w^2 of a weight vector.

    This is the honest local sample count behind a kernel-weighted estimate; when
    it drops below a floor the neighbourhood is too thin to certify anything.
    """
    w = np.asarray(w, dtype=float)
    s2 = float(np.sum(w * w))
    return float(w.sum() ** 2 / s2) if s2 > 0 else 0.0


def default_bandwidth(coords: np.ndarray, scale: float = 1.0, floor: float = 1e-3) -> float:
    """Silverman rule-of-thumb bandwidth for the 1-D localizer, times ``scale``.

    h = 0.9 * min(std, IQR / 1.34) * n^(-1/5). Data-driven so the neighbourhood
    width tracks the spread of the novelty axis rather than a hand-set constant;
    ``scale`` widens or narrows it and ``floor`` guards a degenerate spread.
    """
    x = np.asarray(coords, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        return floor
    std = float(np.std(x))
    q75, q25 = np.percentile(x, [75, 25])
    iqr = float(q75 - q25)
    spread = min(std, iqr / 1.34) if iqr > 0 else std
    if spread <= 0:
        spread = std if std > 0 else floor
    return max(float(scale * 0.9 * spread * n ** (-0.2)), floor)


def rlcp_quantile(
    cal_scores: np.ndarray,
    cal_coords: np.ndarray,
    query_coord: float,
    level: float,
    h: float,
    kernel: str = "gaussian",
    generator: np.random.Generator | None = None,
    randomize: bool = True,
    infinity_mass: bool = True,
) -> float:
    """Randomly-localized weighted ``level``-quantile of calibration scores.

    Draws the RLCP reference point (or fixes it at ``query_coord`` when
    ``randomize=False``), weights calibration scores by the kernel, appends the
    +infinity point mass for the test point (weight K_h(query, Xtilde)) unless
    ``infinity_mass=False``, and returns the smallest score whose cumulative
    normalized weight reaches ``level``. Returns +inf when the mass above the top
    finite score still falls short of ``level`` (the prediction set is the whole
    line, the honest "cannot localize" outcome).

    With ``randomize=True`` and ``infinity_mass=True`` this is the Hore-Barber
    estimator and a test point drawn exchangeably with the calibration set is
    covered with probability >= level, exactly and marginally. The two comparator
    flags exist only so the self-check can show that dropping either ingredient
    lets coverage fall below the target.
    """
    if generator is None:
        generator = np.random.default_rng()
    s = np.asarray(cal_scores, dtype=float)
    x = np.asarray(cal_coords, dtype=float)
    keep = np.isfinite(s) & np.isfinite(x)
    s, x = s[keep], x[keep]
    if len(s) == 0 or not np.isfinite(query_coord):
        return np.inf

    anchor = _draw_anchor(query_coord, h, kernel, generator) if randomize else float(query_coord)
    w = _kernel_weights(anchor, x, h, kernel)
    w_star = float(_kernel_weights(anchor, np.array([query_coord]), h, kernel)[0]) if infinity_mass else 0.0

    total = float(w.sum()) + w_star
    if total <= 0:
        return np.inf
    order = np.argsort(s, kind="mergesort")
    s_sorted = s[order]
    cum = np.cumsum(w[order]) / total          # the +inf mass sits above every finite score
    idx = int(np.searchsorted(cum, level, side="left"))
    if idx >= len(s_sorted):
        return np.inf                          # only the +inf point mass reaches ``level``
    return float(s_sorted[idx])


def _weighted_fixed_sequence(
    scores: np.ndarray,
    err: np.ndarray,
    w: np.ndarray,
    alpha: float,
    min_accept_eff: float,
) -> float | None:
    """Largest weighted accept set (smallest tau) with local weighted error <= alpha.

    The weighted analogue of the fixed-sequence walk in ``risk.naive_threshold``:
    order poses by confidence, start at the most conservative accept set that
    carries at least ``min_accept_eff`` effective mass, and grow it (lower tau)
    while the kernel-weighted error stays <= alpha, stopping at the first
    violation. Returns tau (accept iff score >= tau), or None if the most
    confident admissible set already exceeds alpha.
    """
    order = np.argsort(-scores, kind="mergesort")
    s_sorted = scores[order]
    w_sorted = w[order]
    e_sorted = err[order]

    cum_w = np.cumsum(w_sorted)
    cum_werr = np.cumsum(w_sorted * e_sorted)
    cum_w2 = np.cumsum(w_sorted * w_sorted)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(cum_w > 0, cum_werr / cum_w, 1.0)
        ess_acc = np.where(cum_w2 > 0, cum_w * cum_w / cum_w2, 0.0)

    admissible = np.where(ess_acc >= min_accept_eff)[0]
    if len(admissible) == 0:
        return None
    start = int(admissible[0])
    if r[start] > alpha:
        return None                                   # cannot certify even the top set
    viol = np.where(r[start:] > alpha)[0]
    last = (len(s_sorted) - 1) if len(viol) == 0 else (start + int(viol[0]) - 1)
    return float(s_sorted[last])


def localized_threshold(
    cal_scores: np.ndarray,
    cal_correct: np.ndarray,
    cal_coords: np.ndarray,
    query_coords: np.ndarray,
    alpha: float,
    h: float | None = None,
    kernel: str = "gaussian",
    generator: np.random.Generator | None = None,
    min_eff: float = 40.0,
    min_accept_eff: float = 20.0,
    randomize: bool = True,
    bandwidth_scale: float = 1.0,
) -> np.ndarray:
    """Novelty-adaptive accept threshold tau(x) for each query novelty value.

    For each query coordinate x* we draw an RLCP random anchor, weight the
    calibration poses by K_h on the novelty axis, and grow the accept set while
    the locally-weighted error stays <= alpha (``_weighted_fixed_sequence``). The
    result is a threshold that slides smoothly along novelty: accept a pose at
    novelty x* iff its confidence >= tau(x*).

    Returns tau per query, with NaN wherever the local kernel is too thin to
    stand on (effective sample size < ``min_eff``), the query coordinate is
    undefined (e.g. a no-analog pose with no Tanimoto), or no admissible accept
    set holds error at alpha. NaN means abstain, and it is the honest signal that
    the certificate is vacuous there rather than a quietly over-confident tau.

    Guarantee: the randomisation gives exact marginal coverage of the score
    quantile (see ``rlcp_quantile`` and the self-check); the selective-risk
    behaviour of this gate is a locally-reweighted plug-in that the experiments
    validate empirically, not a finite-sample theorem.
    """
    if generator is None:
        generator = np.random.default_rng()
    s = np.asarray(cal_scores, dtype=float)
    y = np.asarray(cal_correct, dtype=int)
    x = np.asarray(cal_coords, dtype=float)
    keep = np.isfinite(s) & np.isfinite(x)
    s, y, x = s[keep], y[keep], x[keep]
    err = 1 - y

    if h is None:
        h = default_bandwidth(x, scale=bandwidth_scale)

    queries = np.atleast_1d(np.asarray(query_coords, dtype=float))
    out = np.full(len(queries), np.nan)
    if len(s) == 0:
        return out

    for j, xq in enumerate(queries):
        if not np.isfinite(xq):
            continue                                   # no localizer coordinate -> abstain
        anchor = _draw_anchor(xq, h, kernel, generator) if randomize else float(xq)
        w = _kernel_weights(anchor, x, h, kernel)
        if kish_ess(w) < min_eff:
            continue                                   # kernel too thin here -> abstain
        tau = _weighted_fixed_sequence(s, err, w, alpha, min_accept_eff)
        if tau is not None:
            out[j] = tau
    return out


def sweep_error_target(
    cal_scores: np.ndarray,
    cal_correct: np.ndarray,
    cal_coords: np.ndarray,
    query_coords: np.ndarray,
    alphas: np.ndarray,
    h: float | None = None,
    kernel: str = "gaussian",
    seed: int = 0,
    **kwargs,
) -> dict[float, np.ndarray]:
    """Localized threshold for a range of certified-error targets alpha.

    Returns {alpha: tau_per_query}. A tighter alpha lifts tau (accept fewer poses)
    and widens the region where the gate abstains; sweeping it traces the
    risk-coverage trade-off the localized gate offers at each novelty value. Each
    alpha uses its own seeded generator so the sweep is reproducible.
    """
    x = np.asarray(cal_coords, dtype=float)
    if h is None:
        h = default_bandwidth(x[np.isfinite(x)], scale=kwargs.get("bandwidth_scale", 1.0))
    out: dict[float, np.ndarray] = {}
    for i, a in enumerate(np.atleast_1d(alphas)):
        gen = np.random.default_rng(seed + i)
        out[float(a)] = localized_threshold(
            cal_scores, cal_correct, cal_coords, query_coords, float(a),
            h=h, kernel=kernel, generator=gen, **kwargs,
        )
    return out


def _synthetic_coverage_check(seed: int = 0, n_trials: int = 8000, n_cal: int = 400) -> dict:
    """Marginal-coverage self-check: RLCP holds 1 - alpha, plain localization misses.

    The RLCP guarantee is marginal over the calibration draw as well as the test
    draw, so each trial refreshes the calibration set and scores one fresh test
    point; coverage is pooled over trials. The conditional score distribution
    varies along the localizer: X ~ U(0,1) and V | X ~ N(0, sigma(X)^2) with
    sigma(X) = 0.15 + 1.6 X, so the novel tail is far more dispersed than the
    familiar end. We form the one-sided set {V <= q(X)} at three quantile rules:

      rlcp         randomized anchor + the +inf test mass  (Hore-Barber)
      det_mass     deterministic anchor + the +inf test mass  (Guan-style centre)
      plain        deterministic anchor, no test mass  (the common naive plug-in)

    A narrow bandwidth (small effective local sample) makes the missing finite-
    sample correction bite: rlcp lands at or above the target while plain, the
    non-randomized plug-in, falls below it.
    """
    rng = np.random.default_rng(seed)
    alpha = 0.10
    level = 1.0 - alpha
    h = 0.04

    def sigma(x: np.ndarray) -> np.ndarray:
        return 0.15 + 1.6 * x

    modes = {
        "rlcp": dict(randomize=True, infinity_mass=True),
        "det_mass": dict(randomize=False, infinity_mass=True),
        "plain": dict(randomize=False, infinity_mass=False),
    }
    cov = {name: 0 for name in modes}
    gen = np.random.default_rng(seed + 999)
    for _ in range(n_trials):
        x_cal = rng.uniform(0, 1, n_cal)
        v_cal = rng.normal(0, 1, n_cal) * sigma(x_cal)
        xt = float(rng.uniform(0, 1))
        vt = float(rng.normal(0, 1) * sigma(np.array([xt]))[0])
        for name, cfg in modes.items():
            q = rlcp_quantile(v_cal, x_cal, xt, level, h, kernel="gaussian",
                              generator=gen, **cfg)
            cov[name] += int(vt <= q)
    coverage = {name: c / n_trials for name, c in cov.items()}
    return {"alpha": alpha, "target": level, "coverage": coverage,
            "n_cal": n_cal, "n_trials": n_trials}


if __name__ == "__main__":
    res = _synthetic_coverage_check()
    print(f"RLCP marginal-coverage self-check (target 1 - alpha = {res['target']:.2f}, "
          f"n_cal={res['n_cal']}, n_trials={res['n_trials']})\n")
    for name, c in res["coverage"].items():
        flag = "OK  >= target" if c >= res["target"] - 0.005 else "MISS < target"
        print(f"  {name:9s} coverage = {c:.3f}   {flag}")
