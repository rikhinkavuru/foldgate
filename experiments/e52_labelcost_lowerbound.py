"""E52 -- information-theoretic label-cost lower bound (reviewer B2).

The certification frontier goes to zero on the novel strata not because the gate
is badly designed but because there are too few labelled poses there to earn any
distribution-free certificate at all. This script makes that a *quantity*.

An exact binomial (LTT) certificate accepts a top-confidence region of size n with
`errors` observed errors and certifies selective risk <= alpha at confidence 1-delta
iff

    binom.cdf(errors, n, alpha) <= delta.

For an accept region whose error rate is r (so errors ~ r*n), the smallest n that can
certify *any* non-zero coverage is a property of (r, alpha, delta) alone:

    n_g(r) = min { n : binom.cdf(round(r*n), n, alpha) <= delta }.

Two closed forms bracket the numeric search and are what a practitioner computes for
their own (alpha, delta):

  * r = 0 floor (a perfect accept region):  binom.cdf(0, n, alpha) = (1-alpha)^n,
        so  n_g(0) = ceil( ln(delta) / ln(1 - alpha) )  -- the absolute minimum number
        of labels below which NOTHING is certifiable, no matter how clean the region.
  * Chernoff / relative-entropy bound for 0 < r < alpha:
        n_g(r) ~ ceil( ln(1/delta) / KL(r || alpha) ),
        KL(r||alpha) = r*ln(r/alpha) + (1-r)*ln((1-r)/(1-alpha)),
        the binary relative entropy. As r -> 0 this collapses to the r=0 floor.

At alpha=0.20, delta=0.10 the floor is 11 labels; a realistic novel-stratum error rate
pushes it higher, and we compare n_g to the empirical per-stratum sizes (~76 whole, ~38
per calibration half) to show where the strata are label-starved.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import binom

from experiments._common import (
    ALPHA,
    CONF,
    DELTA,
    RESDIR,
    load_delivered,
    methods_with_enough,
    save_json,
)

STRAT = "novelty_stratum"
MIN_ACCEPT = 20          # LTT default minimum certifiable accept set
N_CAP = 200_000          # numeric search ceiling


def kl_binary(r: float, a: float) -> float:
    """Binary relative entropy KL(r || a) in nats; +inf outside (0,1) edge cases."""
    if r < 0 or r > 1 or a <= 0 or a >= 1:
        return float("nan")
    if r == 0:
        return -math.log(1 - a)
    if r == 1:
        return -math.log(a)
    return r * math.log(r / a) + (1 - r) * math.log((1 - r) / (1 - a))


def n_g_closed_form(r: float, alpha: float, delta: float) -> float:
    """Relative-entropy label-cost lower bound ceil(ln(1/delta)/KL(r||alpha))."""
    if r >= alpha:
        return float("inf")
    kl = kl_binary(r, alpha)
    if not np.isfinite(kl) or kl <= 0:
        return float("inf")
    return math.ceil(math.log(1.0 / delta) / kl)


def n_g_exact(r: float, alpha: float, delta: float, cap: int = N_CAP) -> float:
    """Smallest n with binom.cdf(round(r*n), n, alpha) <= delta (round-half errors)."""
    if r >= alpha:
        return float("inf")
    for n in range(1, cap + 1):
        errors = int(round(r * n))
        if binom.cdf(errors, n, alpha) <= delta:
            return n
    return float("inf")


def best_operating_error(s: np.ndarray, y: np.ndarray, min_accept: int) -> tuple:
    """Lowest error rate over top-confidence prefixes of size >= min_accept.

    Sort by confidence descending; the cleanest reasonably sized accept region is the
    best operating point a within-stratum gate could target. Returns (r_best, k_best).
    """
    n = len(s)
    if n == 0:
        return float("nan"), 0
    order = np.argsort(-s)
    err_sorted = (1 - y[order]).astype(float)
    cum_err = np.cumsum(err_sorted)
    k0 = min(min_accept, n)
    ks = np.arange(k0, n + 1)
    rates = cum_err[k0 - 1:] / ks
    j = int(np.argmin(rates))
    return float(rates[j]), int(ks[j])


def run() -> dict:
    df = load_delivered()
    methods = methods_with_enough(df)

    floor_r0 = math.ceil(math.log(DELTA) / math.log(1 - ALPHA))

    out = {
        "_meta": {
            "alpha": ALPHA,
            "delta": DELTA,
            "min_accept": MIN_ACCEPT,
            "floor_r0": int(floor_r0),
            "floor_r0_formula": "ceil( ln(delta) / ln(1 - alpha) )",
            "closed_form": "n_g(r) = ceil( ln(1/delta) / KL(r || alpha) ),  "
                           "KL(r||a) = r ln(r/a) + (1-r) ln((1-r)/(1-a))",
            "interpretation": (
                "n_g is the minimum labelled poses in a stratum below which an exact "
                "binomial certificate cannot certify any non-zero coverage at (alpha, "
                "delta). floor_r0 assumes a perfect (zero-error) accept region; a "
                "positive stratum error rate raises the requirement."
            ),
        },
        "models": {},
    }

    for m in methods:
        sub = df[df.method == m].dropna(subset=[CONF, STRAT]).reset_index(drop=True)
        s = sub[CONF].to_numpy()
        y = sub["correct"].to_numpy().astype(int)
        strat = sub[STRAT].to_numpy().astype(int)
        rows = {}
        for k in sorted(np.unique(strat).tolist()):
            in_k = strat == k
            n_k = int(in_k.sum())
            yk = y[in_k]
            sk = s[in_k]
            base_err = float(1 - yk.mean()) if n_k else float("nan")
            r_best, k_best = best_operating_error(sk, yk, MIN_ACCEPT)

            rows[str(k)] = {
                "n_stratum": n_k,
                "n_calib_half": n_k // 2,           # e2 uses a 50/50 cal/test split
                "base_error": base_err,
                "best_region_error": r_best,
                "best_region_size": k_best,
                # min labels to certify at the "accept-all-in-stratum" error rate
                "n_g_at_base_exact": n_g_exact(base_err, ALPHA, DELTA),
                "n_g_at_base_closed": n_g_closed_form(base_err, ALPHA, DELTA),
                # min labels to certify at the cleanest achievable operating point
                "n_g_at_best_exact": n_g_exact(r_best, ALPHA, DELTA),
                "n_g_at_best_closed": n_g_closed_form(r_best, ALPHA, DELTA),
                # can this stratum's actual size ever clear the floor at its best point?
                "label_starved_at_best": bool(n_k < n_g_exact(r_best, ALPHA, DELTA)),
                "label_starved_at_base": bool(n_k < n_g_exact(base_err, ALPHA, DELTA)),
            }
        out["models"][m] = rows
    return out


def main() -> None:
    res = run()
    save_json(res, RESDIR / "e52_labelcost_lowerbound.json")
    meta = res["_meta"]
    print("E52 -- information-theoretic label-cost lower bound")
    print(f"alpha={meta['alpha']}, delta={meta['delta']}")
    print(f"absolute floor (r=0, perfect region): n_g = {meta['floor_r0']}  "
          f"[{meta['floor_r0_formula']}]")
    print(f"practitioner closed form:  {meta['closed_form']}\n")
    print(f"{'model':>9} {'S':>2} {'n':>5} {'n/2':>4} {'base_err':>8} "
          f"{'r_best':>7} {'ng_base':>8} {'ng_best':>8} {'starved':>8}")
    for m, rows in res["models"].items():
        for k, r in rows.items():
            ngb = r["n_g_at_base_exact"]
            ngbest = r["n_g_at_best_exact"]
            print(f"{m:>9} {k:>2} {r['n_stratum']:>5} {r['n_calib_half']:>4} "
                  f"{r['base_error']:>8.3f} {r['best_region_error']:>7.3f} "
                  f"{str(ngb):>8} {str(ngbest):>8} "
                  f"{'YES' if r['label_starved_at_best'] else '-':>8}")
        print()


if __name__ == "__main__":
    main()
