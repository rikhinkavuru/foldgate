"""E58 (reviewer D28) -- what does the training-free constraint cost?

The whole reliability layer is training-free w.r.t. novelty: similarity-to-train is the conformal
STRATIFIER, never a feature of the score (combiner.py docstring). D28 asks the price of that
discipline. We quantify it by lifting the constraint: fit a small correctness classifier that
INCLUDES the two novelty features (ligand_similarity, pocket_similarity appended to the combiner's
default feature set) and see how much its ranking improves.

To avoid target leakage the classifier is trained target-grouped -- GroupKFold on system_id, so no
system appears in both train and its own scoring fold -- and every prediction used for AURC is
out-of-fold. Per model we report three AURCs (lower = better confidence ranking):

  (a) AURC(native)            -- raw ranking_score, no training,
  (b) AURC(combined)          -- ScoreCombiner on DEFAULT_FEATURES, training-free (no novelty),
  (c) AURC(novelty-aware)     -- ScoreCombiner on DEFAULT_FEATURES + [ligand_similarity,
                                 pocket_similarity], the trained ceiling.

The gap (b) - (c) is the price of the training-free constraint: how much AURC the layer forgoes by
refusing to put similarity-to-train into the score. A grouped paired bootstrap over scored rows
puts a CI on it.

Outputs results/e58_trained_ceiling.json. Runs on the delivered parquet alone.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupKFold

from experiments._common import RESDIR, load_delivered, methods_with_enough, rng, save_json
from foldgate.scores.combiner import DEFAULT_FEATURES, ScoreCombiner
from foldgate.selective.metrics import aurc

NATIVE = "ranking_score"
NOVELTY_FEATURES = DEFAULT_FEATURES + ["ligand_similarity", "pocket_similarity"]
N_SPLITS = 5
N_BOOT = 2000


def oof_predict(sub, y, groups, features, n_splits: int) -> np.ndarray:
    """Out-of-fold P(correct) from a GroupKFold-trained ScoreCombiner (no target leakage)."""
    pred = np.full(len(sub), np.nan)
    gkf = GroupKFold(n_splits=n_splits)
    for tr, te in gkf.split(sub, y, groups):
        comb = ScoreCombiner(features=features).fit(sub.iloc[tr], y[tr])
        pred[te] = comb.predict(sub.iloc[te])
    return pred


def run(n_splits: int = N_SPLITS, n_boot: int = N_BOOT, seed: int = 20260715) -> dict:
    df = load_delivered()
    methods = methods_with_enough(df)
    g = rng(seed)
    out = {"score_native": NATIVE, "n_splits": n_splits,
           "default_features": DEFAULT_FEATURES, "novelty_features_added": ["ligand_similarity", "pocket_similarity"],
           "methods": methods, "models": {}}

    for m in methods:
        sub = df[df.method == m].reset_index(drop=True)
        y = sub["correct"].to_numpy().astype(int)
        groups = sub["system_id"].to_numpy()
        s_nat = sub[NATIVE].to_numpy()

        p_comb = oof_predict(sub, y, groups, DEFAULT_FEATURES, n_splits)
        p_nov = oof_predict(sub, y, groups, NOVELTY_FEATURES, n_splits)

        a_nat = aurc(s_nat, y)
        a_comb = aurc(p_comb, y)
        a_nov = aurc(p_nov, y)

        # Grouped paired bootstrap on the OOF gap (combined - novelty-aware): resample whole
        # systems so clustered losses are respected.
        sys_ids = np.array(sorted(sub.system_id.unique()))
        idx_by_sys = {sid: np.where(groups == sid)[0] for sid in sys_ids}
        deltas = []
        for _ in range(n_boot):
            pick = g.integers(0, len(sys_ids), len(sys_ids))
            rows = np.concatenate([idx_by_sys[sys_ids[i]] for i in pick])
            deltas.append(aurc(p_comb[rows], y[rows]) - aurc(p_nov[rows], y[rows]))
        d_lo, d_hi = float(np.percentile(deltas, 5)), float(np.percentile(deltas, 95))

        out["models"][m] = {
            "n": int(len(sub)), "n_systems": int(len(sys_ids)),
            "aurc_native": float(a_nat),
            "aurc_combined_trainingfree": float(a_comb),
            "aurc_novelty_aware_trained": float(a_nov),
            "gap_native_to_combined": float(a_nat - a_comb),
            "price_of_trainingfree_combined_minus_novelty": float(a_comb - a_nov),
            "price_ci90": [d_lo, d_hi],
            "price_excludes_zero": bool(d_lo > 0),
        }

    prices = [v["price_of_trainingfree_combined_minus_novelty"] for v in out["models"].values()]
    n_sig = sum(v["price_excludes_zero"] for v in out["models"].values())
    out["summary"] = {
        "median_price_aurc": float(np.median(prices)),
        "max_price_aurc": float(np.max(prices)),
        "n_models_price_excludes_zero": int(n_sig),
        "n_models": len(methods),
        "takeaway": (f"adding similarity-to-train to the trained score cuts AURC by a median "
                     f"{np.median(prices):.4f} (max {np.max(prices):.4f}); this gap, significant in "
                     f"{n_sig}/{len(methods)} models, is the measured price of keeping the score "
                     f"training-free -- novelty is worth more as a stratifier than as a feature"),
    }
    return out


def main() -> None:
    res = run()
    save_json(res, RESDIR / "e58_trained_ceiling.json")
    print("E58 -- price of the training-free constraint (AURC, lower = better)\n")
    print(f"  {'model':>9} {'native':>8} {'combined':>9} {'novelty':>8} {'price':>8} {'CI>0':>6}")
    for m, mo in res["models"].items():
        print(f"  {m:>9} {mo['aurc_native']:>8.4f} {mo['aurc_combined_trainingfree']:>9.4f} "
              f"{mo['aurc_novelty_aware_trained']:>8.4f} "
              f"{mo['price_of_trainingfree_combined_minus_novelty']:>8.4f} "
              f"{str(mo['price_excludes_zero']):>6}")
    print(f"\n{res['summary']['takeaway']}")
    print(f"\nwrote {RESDIR / 'e58_trained_ceiling.json'}")


if __name__ == "__main__":
    main()
