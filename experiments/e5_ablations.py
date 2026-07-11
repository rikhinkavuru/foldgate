"""E5 ablations -- robustness of the findings.

(a) RMSD-threshold sensitivity: does the E2 break and the E4 combined-vs-native
    AURC gain survive changing the 2 A correctness convention?
(b) Feature ablation: which signals drive the combined score's gain over native
    confidence? Cumulative AURC as feature groups are added.

Both address reviewer questions directly rather than deferring them.
"""

from __future__ import annotations

import numpy as np

from experiments._common import DELTA, RESDIR, load_delivered, rng, save_json
from foldgate.conformal import ltt_threshold
from foldgate.scores import ScoreCombiner
from foldgate.selective import aurc, conditional_coverage, evaluate_gate

THRESHOLDS = [1.5, 2.0, 2.5, 3.0]
FEATURE_SETS = {
    "native only": ["ranking_score"],
    "+ interface ipTM": ["ranking_score", "iface_iptm"],
    "+ PoseBusters": ["ranking_score", "iface_iptm", "pb_valid"],
    "+ ensemble spread": ["ranking_score", "iface_iptm", "pb_valid", "ens_ranking_std", "ens_ranking_range", "ens_iptm_std"],
    "+ cross-model agreement": ["ranking_score", "iface_iptm", "pb_valid", "ens_ranking_std",
                                "ens_ranking_range", "ens_iptm_std", "xmodel_iptm_mean",
                                "xmodel_iptm_std", "xmodel_n_models"],
    "+ ligand difficulty (all)": ["ranking_score", "iface_iptm", "pb_valid", "ens_ranking_std",
                                  "ens_ranking_range", "ens_iptm_std", "xmodel_iptm_mean",
                                  "xmodel_iptm_std", "xmodel_n_models", "ligand_molecular_weight",
                                  "ligand_num_rot_bonds", "ligand_num_heavy_atoms"],
}


def threshold_sweep(df, m="af3", n_repeats=120):
    g = rng()
    sub = df[df.method == m].reset_index(drop=True)
    rmsd = sub["rmsd"].to_numpy()
    s = sub["ranking_score"].to_numpy()
    strat = sub["novelty_stratum"].to_numpy().astype(int)
    out = {}
    for t in THRESHOLDS:
        y = (rmsd <= t).astype(int)
        # E2-style: global iid tau, per-stratum risk (pooled over repeats)
        acc_err = {}; acc_n = {}
        aurc_nat, aurc_comb = [], []
        for _ in range(n_repeats):
            perm = g.permutation(len(sub))
            a, b = int(0.4 * len(sub)), int(0.7 * len(sub))
            tr, cal, te = perm[:a], perm[a:b], perm[b:]
            tau = ltt_threshold(s[cal], y[cal], alpha=0.2, delta=DELTA)
            cc = conditional_coverage(s[te], y[te], strat[te], tau)
            for k, v in cc.items():
                if v["n_accept"]:
                    acc_err[k] = acc_err.get(k, 0) + v["selective_risk"] * v["n_accept"]
                    acc_n[k] = acc_n.get(k, 0) + v["n_accept"]
            comb = ScoreCombiner().fit(sub.iloc[tr], y[tr])
            aurc_nat.append(aurc(s[te], y[te]))
            aurc_comb.append(aurc(comb.predict(sub.iloc[te]), y[te]))
        strata_risk = {int(k): (acc_err[k] / acc_n[k]) for k in sorted(acc_n)}
        out[str(t)] = {
            "base_correct": float(y.mean()),
            "per_stratum_risk_global_tau": {k: round(v, 3) for k, v in strata_risk.items()},
            "aurc_native": float(np.mean(aurc_nat)),
            "aurc_combined": float(np.mean(aurc_comb)),
            "aurc_improve_pct": round(100 * (np.mean(aurc_nat) - np.mean(aurc_comb)) / np.mean(aurc_nat), 1),
        }
    return out


def feature_ablation(df, m="af3", n_repeats=120):
    g = rng()
    sub = df[df.method == m].reset_index(drop=True)
    y = sub["correct"].to_numpy()
    out = {}
    for name, feats in FEATURE_SETS.items():
        vals = []
        for _ in range(n_repeats):
            perm = g.permutation(len(sub))
            tr, te = perm[: len(sub) // 2], perm[len(sub) // 2:]
            comb = ScoreCombiner(features=feats).fit(sub.iloc[tr], y[tr])
            vals.append(aurc(comb.predict(sub.iloc[te]), y[te]))
        out[name] = {"aurc": float(np.mean(vals)), "n_features": len(feats)}
    return out


def main() -> None:
    df = load_delivered()
    res = {"threshold_sweep_af3": threshold_sweep(df), "feature_ablation_af3": feature_ablation(df)}
    save_json(res, RESDIR / "e5_ablations.json")

    print("=== RMSD-threshold sensitivity (AF3) ===")
    print(f"{'thresh':>7} {'base_ok':>8} {'S0':>6} {'S3':>6} {'S4':>6} {'AURC nat':>9} {'AURC comb':>10} {'improve':>8}")
    for t, v in res["threshold_sweep_af3"].items():
        ps = v["per_stratum_risk_global_tau"]
        print(f"{t:>7} {v['base_correct']:>8.3f} {ps.get(0,float('nan')):>6.3f} {ps.get(3,float('nan')):>6.3f} "
              f"{ps.get(4,float('nan')):>6.3f} {v['aurc_native']:>9.3f} {v['aurc_combined']:>10.3f} {v['aurc_improve_pct']:>7.1f}%")
    print("\n=== Feature ablation (AF3, cumulative AURC; lower better) ===")
    for name, v in res["feature_ablation_af3"].items():
        print(f"  {name:28} AURC {v['aurc']:.3f}  ({v['n_features']} feats)")


if __name__ == "__main__":
    main()
