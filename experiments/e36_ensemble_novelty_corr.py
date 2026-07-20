"""E36 -- does the combined score secretly leak training-set novelty? (reviewer R3.7)

The paper positions the combined score as novelty-BLIND: training-set novelty (nu)
never enters the score directly and is used only downstream, at calibration time, to
key the group-conditional / weighted repair. That framing places the construction in
the pure-impossibility regime -- you cannot certify a fixed nu-blind score uniformly
across the novelty axis without paying coverage on the novel strata.

But the score is built from ensemble-disagreement and cross-model-agreement features
(ens_ranking_std, ens_ranking_range, ens_iptm_std, xmodel_iptm_mean, xmodel_iptm_std).
If any of those tracks nu, the score is quietly a novelty-AWARE re-score, and re-scoring
on nu is exactly the escape hatch from the impossibility (you are allowed to move mass
as a function of the shift variable). That would be a STRENGTHENING of the result if
disclosed, and a silent contradiction of the exclusion claim if not.

So we measure it honestly. For each governed model we compute the Spearman correlation
(90% bootstrap CI, 1000 row-resamples) between ligand_novelty and each candidate score
input, separating the ensemble/cross-model combiner features from the physicochemical
controls. We rank by magnitude, take the max |rho| over the combiner features, and check
it against a substantiality threshold |rho| >= 0.30.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from experiments._common import (
    RESDIR,
    load_delivered,
    methods_with_enough,
    rng,
    save_json,
)

NOVELTY = "ligand_novelty"

# the features the combined score is actually built from (the leak surface)
COMBINER_FEATURES = [
    "ens_ranking_std",
    "ens_ranking_range",
    "ens_iptm_std",
    "xmodel_iptm_mean",
    "xmodel_iptm_std",
]
# native + physicochemical controls: not part of the ensemble combiner, shown for
# calibration of what a "large" |rho| looks like (physicochem can legitimately track nu)
NATIVE_FEATURES = ["iface_iptm", "ranking_score"]
PHYSCHEM_FEATURES = [
    "ligand_molecular_weight",
    "ligand_num_heavy_atoms",
    "pb_valid",
]
ALL_FEATURES = COMBINER_FEATURES + NATIVE_FEATURES + PHYSCHEM_FEATURES

N_BOOT = 1000
CI_LOW, CI_HIGH = 5.0, 95.0   # 90% interval
SUBSTANTIAL = 0.30


def _spearman_ci(x: np.ndarray, y: np.ndarray, g: np.random.Generator) -> dict:
    """Point Spearman rho + 90% bootstrap CI over row resamples. NaN dropped pairwise."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = x.size
    if n < 10 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return {"rho": float("nan"), "abs_rho": float("nan"),
                "ci_low": float("nan"), "ci_high": float("nan"), "n": int(n)}
    rho = float(spearmanr(x, y).correlation)
    boot = np.empty(N_BOOT, dtype=float)
    for b in range(N_BOOT):
        idx = g.integers(0, n, size=n)
        xb, yb = x[idx], y[idx]
        if np.ptp(xb) == 0 or np.ptp(yb) == 0:
            boot[b] = np.nan
            continue
        boot[b] = spearmanr(xb, yb).correlation
    boot = boot[np.isfinite(boot)]
    lo = float(np.percentile(boot, CI_LOW)) if boot.size else float("nan")
    hi = float(np.percentile(boot, CI_HIGH)) if boot.size else float("nan")
    return {"rho": rho, "abs_rho": abs(rho), "ci_low": lo, "ci_high": hi, "n": int(n)}


def run() -> dict:
    df = load_delivered()
    methods = methods_with_enough(df)
    g = rng()
    out = {"features": {"combiner": COMBINER_FEATURES,
                        "native": NATIVE_FEATURES,
                        "physicochemical": PHYSCHEM_FEATURES},
           "substantial_threshold": SUBSTANTIAL,
           "n_boot": N_BOOT,
           "models": {}}

    global_max_abs = 0.0
    global_max_where = None

    for m in methods:
        sub = df[df.method == m]
        nu = sub[NOVELTY].to_numpy(dtype=float)
        per_feat = {}
        for f in ALL_FEATURES:
            per_feat[f] = _spearman_ci(sub[f].to_numpy(dtype=float), nu, g)

        # ranked |rho| over the combiner (leak-surface) features
        ranked = sorted(
            [(f, per_feat[f]["abs_rho"]) for f in COMBINER_FEATURES
             if np.isfinite(per_feat[f]["abs_rho"])],
            key=lambda kv: kv[1], reverse=True,
        )
        combiner_abs = [per_feat[f]["abs_rho"] for f in COMBINER_FEATURES
                        if np.isfinite(per_feat[f]["abs_rho"])]
        physchem_abs = [per_feat[f]["abs_rho"] for f in PHYSCHEM_FEATURES
                        if np.isfinite(per_feat[f]["abs_rho"])]

        max_abs = max(combiner_abs) if combiner_abs else float("nan")
        argmax = ranked[0][0] if ranked else None
        if np.isfinite(max_abs) and max_abs > global_max_abs:
            global_max_abs = max_abs
            global_max_where = {"model": m, "feature": argmax,
                                "rho": per_feat[argmax]["rho"],
                                "ci_low": per_feat[argmax]["ci_low"],
                                "ci_high": per_feat[argmax]["ci_high"]}

        out["models"][m] = {
            "n": int(sub.shape[0]),
            "per_feature": per_feat,
            "combiner_ranked_abs_rho": [{"feature": f, "abs_rho": a} for f, a in ranked],
            "median_abs_rho_combiner": float(np.median(combiner_abs)) if combiner_abs else float("nan"),
            "median_abs_rho_physicochemical": float(np.median(physchem_abs)) if physchem_abs else float("nan"),
            "max_abs_rho_combiner": float(max_abs),
            "argmax_combiner_feature": argmax,
            "combiner_leaks_novelty": bool(np.isfinite(max_abs) and max_abs >= SUBSTANTIAL),
        }

    leaks = bool(global_max_abs >= SUBSTANTIAL)
    med_comb = float(np.median([out["models"][m]["median_abs_rho_combiner"]
                                for m in methods]))
    med_phys = float(np.median([out["models"][m]["median_abs_rho_physicochemical"]
                                for m in methods]))
    out["summary"] = {
        "max_abs_rho_any_combiner_feature": float(global_max_abs),
        "max_location": global_max_where,
        "combiner_leaks_novelty_substantial": leaks,
        "median_abs_rho_combiner_across_models": med_comb,
        "median_abs_rho_physicochemical_across_models": med_phys,
        "regime": ("ACHIEVABILITY (score is partly a novelty-aware re-score -- disclose)"
                   if leaks else
                   "IMPOSSIBILITY (exclusion claim holds; combiner does not leak novelty)"),
    }
    return out


def main() -> None:
    res = run()
    save_json(res, RESDIR / "e36_ensemble_novelty_corr.json")

    print("E36 -- combiner-feature vs training-set-novelty leakage  "
          f"(threshold |rho|>={SUBSTANTIAL}, 90% boot CI, {N_BOOT} reps)\n")
    for m, r in res["models"].items():
        print(f"[{m}]  n={r['n']}  "
              f"median|rho| combiner={r['median_abs_rho_combiner']:.3f}  "
              f"physicochem={r['median_abs_rho_physicochemical']:.3f}  "
              f"leaks={r['combiner_leaks_novelty']}")
        for row in r["combiner_ranked_abs_rho"]:
            f = row["feature"]
            pf = r["per_feature"][f]
            print(f"      {f:22s} |rho|={row['abs_rho']:.3f}  "
                  f"rho={pf['rho']:+.3f} [{pf['ci_low']:+.3f},{pf['ci_high']:+.3f}]  n={pf['n']}")
        print()

    s = res["summary"]
    loc = s["max_location"]
    print("SUMMARY")
    print(f"  max |rho| over any combiner feature = {s['max_abs_rho_any_combiner_feature']:.3f}"
          + (f"  ({loc['model']} / {loc['feature']}, rho={loc['rho']:+.3f} "
             f"[{loc['ci_low']:+.3f},{loc['ci_high']:+.3f}])" if loc else ""))
    print(f"  median |rho| combiner across models       = {s['median_abs_rho_combiner_across_models']:.3f}")
    print(f"  median |rho| physicochemical across models = {s['median_abs_rho_physicochemical_across_models']:.3f}")
    print(f"  substantial leak (|rho|>={SUBSTANTIAL})? {s['combiner_leaks_novelty_substantial']}")
    print(f"  regime: {s['regime']}")


if __name__ == "__main__":
    main()
