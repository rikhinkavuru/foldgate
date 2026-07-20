"""E28 -- label-cost curve: how much certified coverage does n_g in-stratum labels buy?

Addition III.1 (also answers R1.8, R3.5). The repair story rests on "a median of ~38
in-stratum target-labels certifies the gate". This turns that single number into a
design curve: for each governed model and each ligand-novelty stratum g in {S0..S3}
(S4 no-analog is uncertifiable and skipped), sweep a label budget n_g and read the
certified deploy-frozen coverage the budget buys.

Protocol (per model, per stratum g, per budget n_g):
  * work on the one-pose-per-target subset within g, native `ranking_score`;
  * over R=200 draws, sample n_g targets without replacement (with replacement across
    draws), certify the gate on the draw with exact-binomial LTT at alpha=0.20,
    delta=0.10, freeze tau, and measure CERTIFIED coverage = fraction of the FULL
    stratum accepted at that tau (0 if the draw certifies nothing);
  * report median and [5,95] band of certified coverage and the fraction of draws that
    certified anything.

Then a descriptive rate check per stratum: pool (n_g, median_cov) across models and
report the correlation of median_cov with 1/sqrt(n_g) and a monotonicity statement,
i.e. whether coverage rises with labels and roughly tracks a 1 - c/sqrt(n_g) shape.
Honest and descriptive only; no fitted guarantee is claimed.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from experiments._common import (
    ALPHA,
    DELTA,
    RESDIR,
    load_delivered,
    methods_with_enough,
    rng,
    save_json,
)
from foldgate.conformal import ltt_threshold
from foldgate.scores import DEFAULT_FEATURES, ScoreCombiner

R = 200                                   # draws per (model, stratum, budget)
BUDGETS = [5, 10, 20, 40, 80, "all"]      # label budgets n_g
STRATA = [0, 1, 2, 3]                     # ligand-novelty S0..S3; S4 (no-analog) skipped


def _one_pose(df):
    """One pose per (system_id, method): the top-ranked pose = independent target label."""
    return df.sort_values("ranking_score", ascending=False).drop_duplicates(
        ["system_id", "method"]
    )


def _curve_for_stratum(scores_full, correct_full, generator):
    """Return the list of per-budget certified-coverage summaries for one stratum pool."""
    n_pool = scores_full.size
    out = []
    for budget in BUDGETS:
        n_g = n_pool if budget == "all" else int(budget)
        if n_pool < n_g:
            continue  # stratum too thin for this budget
        covs = np.empty(R, dtype=float)
        for r in range(R):
            idx = generator.choice(n_pool, size=n_g, replace=False)
            tau = ltt_threshold(
                scores_full[idx], correct_full[idx], ALPHA, delta=DELTA
            )
            if tau is None:
                covs[r] = 0.0
            else:
                # deploy frozen tau on the FULL stratum -> certified coverage
                covs[r] = float(np.mean(scores_full >= tau))
        out.append(
            {
                "n_g": (n_pool if budget == "all" else int(budget)),
                "budget": budget,
                "median_cov": float(np.median(covs)),
                "cov_lo": float(np.quantile(covs, 0.05)),
                "cov_hi": float(np.quantile(covs, 0.95)),
                "frac_certified": float(np.mean(covs > 0.0)),
                "n_pool": int(n_pool),
            }
        )
    return out


def _combined_pool(m_sub, split_seed):
    """Fit ScoreCombiner ONCE on a target-disjoint fit half; return the pool half with
    frozen out-of-sample combined scores.

    m_sub is one-pose-per-target for a single method, so system_id is unique per row and
    the GroupShuffleSplit is a clean 50/50 target split. The combiner is fit only on
    fit_targets (all strata pooled) and never sees a pool target, so subsampling the pool
    later reuses one frozen predictor with no per-draw refit and no fit/eval leakage.
    """
    grp = m_sub["system_id"].to_numpy()
    gss = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=split_seed)
    (fit_idx, pool_idx), = gss.split(m_sub, groups=grp)
    fit_df = m_sub.iloc[fit_idx]
    pool_df = m_sub.iloc[pool_idx].copy()
    comb = ScoreCombiner(features=DEFAULT_FEATURES).fit(
        fit_df, fit_df["correct"].to_numpy(dtype=int)
    )
    pool_df["combined_score"] = comb.predict(pool_df)
    return pool_df


def main():
    df = load_delivered()
    one = _one_pose(df)
    methods = methods_with_enough(df)
    gen = rng()

    curves = {}
    for method in methods:
        m_sub = one[one.method == method]
        per_stratum = {}
        for g in STRATA:
            g_sub = m_sub[m_sub.novelty_stratum == g]
            scores_full = g_sub["ranking_score"].to_numpy(dtype=float)
            correct_full = g_sub["correct"].to_numpy(dtype=int)
            if scores_full.size == 0:
                continue
            per_stratum[f"S{g}"] = _curve_for_stratum(
                scores_full, correct_full, gen
            )
        curves[method] = per_stratum

    # ---- COMBINED-score variant: the paper's actual repair signal ----
    # Fit the combiner once per model on a target-disjoint half, then run the identical
    # label-cost subsampling on the frozen out-of-sample combined scores of the pool half.
    gen_c = rng(20260711)
    curves_combined = {}
    for method in methods:
        m_sub = one[one.method == method]
        pool_df = _combined_pool(m_sub, split_seed=0)
        per_stratum = {}
        for g in STRATA:
            g_sub = pool_df[pool_df.novelty_stratum == g]
            scores_full = g_sub["combined_score"].to_numpy(dtype=float)
            correct_full = g_sub["correct"].to_numpy(dtype=int)
            if scores_full.size == 0:
                continue
            per_stratum[f"S{g}"] = _curve_for_stratum(
                scores_full, correct_full, gen_c
            )
        curves_combined[method] = per_stratum

    # ---- descriptive rate check: coverage vs n_g, pooled across models per stratum ----
    rate_check = {}
    for g in STRATA:
        key = f"S{g}"
        ns, covs = [], []
        for method in methods:
            for pt in curves[method].get(key, []):
                ns.append(pt["n_g"])
                covs.append(pt["median_cov"])
        ns = np.asarray(ns, dtype=float)
        covs = np.asarray(covs, dtype=float)
        entry = {"n_points": int(ns.size)}
        if ns.size >= 3 and np.std(covs) > 0:
            inv_sqrt = 1.0 / np.sqrt(ns)
            # 1 - c/sqrt(n): coverage should fall (roughly linearly) with 1/sqrt(n_g),
            # so a strong NEGATIVE correlation with 1/sqrt(n_g) supports the shape.
            entry["corr_median_cov_vs_inv_sqrt_n"] = float(
                np.corrcoef(covs, inv_sqrt)[0, 1]
            )
            entry["corr_median_cov_vs_n"] = float(np.corrcoef(covs, ns)[0, 1])
            # monotone in the pooled per-budget MEAN of median_cov?
            budget_means = []
            for b in [5, 10, 20, 40, 80]:
                vals = covs[ns == b]
                if vals.size:
                    budget_means.append((b, float(np.mean(vals))))
            seq = [v for _, v in budget_means]
            entry["budget_mean_median_cov"] = {
                str(b): v for b, v in budget_means
            }
            entry["monotone_nondecreasing_in_budget"] = bool(
                all(seq[i] <= seq[i + 1] + 1e-9 for i in range(len(seq) - 1))
            )
        rate_check[key] = entry

    result = {
        "meta": {
            "experiment": "e28_label_cost_curve",
            "alpha": ALPHA,
            "delta": DELTA,
            "R_draws": R,
            "budgets": [str(b) for b in BUDGETS],
            "ltt_min_accept": 20,
            "note": (
                "certified_coverage = fraction of FULL stratum accepted at the LTT tau "
                "certified on a random n_g-label draw (deploy frozen); 0 if the draw "
                "certifies nothing. LTT default min_accept=20 floors budgets below ~20 "
                "labels at zero certified coverage."
            ),
            "methods": methods,
            "strata": [f"S{g}" for g in STRATA],
            "combined_note": (
                "curves_combined uses ScoreCombiner(DEFAULT_FEATURES) fit ONCE per model "
                "on a target-disjoint 50/50 GroupShuffleSplit(system_id) fit half; scores "
                "are out-of-sample predictions on the pool half. Label-cost subsampling and "
                "certified coverage are measured within each stratum's POOL (roughly half "
                "the native pool size, so the 'all' budget is smaller here)."
            ),
        },
        "curves": curves,
        "curves_combined": curves_combined,
        "rate_check": rate_check,
    }
    out_path = RESDIR / "e28_label_cost_curve.json"
    save_json(result, out_path)

    # ---- console summary: AF3 per-stratum curve ----
    print(f"wrote {out_path}")
    print("\nAF3 median certified coverage by (budget, stratum):")
    print("  budget  " + "  ".join(f"S{g}".rjust(7) for g in STRATA))
    af3 = curves["af3"]
    for budget in BUDGETS:
        cells = []
        for g in STRATA:
            hit = next(
                (p for p in af3.get(f"S{g}", []) if p["budget"] == budget), None
            )
            if hit is None:
                cells.append("   -   ")
            else:
                cells.append(f"{hit['median_cov']:.3f}".rjust(7))
        label = str(budget).rjust(6)
        print(f"  {label}  " + "  ".join(cells))
    print("\nAF3 COMBINED-score median certified coverage by (budget, stratum):")
    print("  budget  " + "  ".join(f"S{g}".rjust(7) for g in STRATA))
    af3c = curves_combined["af3"]
    for budget in BUDGETS:
        cells = []
        for g in STRATA:
            hit = next(
                (p for p in af3c.get(f"S{g}", []) if p["budget"] == budget), None
            )
            if hit is None:
                cells.append("   -   ")
            else:
                cells.append(f"{hit['median_cov']:.3f}".rjust(7))
        print(f"  {str(budget).rjust(6)}  " + "  ".join(cells))

    print("\nrate check (corr of median_cov vs 1/sqrt(n_g), pooled across models):")
    for g in STRATA:
        e = rate_check[f"S{g}"]
        c = e.get("corr_median_cov_vs_inv_sqrt_n")
        mono = e.get("monotone_nondecreasing_in_budget")
        print(f"  S{g}: corr={c:+.3f}  monotone_in_budget={mono}" if c is not None
              else f"  S{g}: (insufficient points)")


if __name__ == "__main__":
    main()
