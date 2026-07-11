"""E3b -- weighted conformal repair for the label-free covariate-shift setting.

Group-conditional (E3) needs labels in the novel-chemotype regime. Weighted
conformal does not: calibrate on the familiar (low-novelty) source, reweight the
calibration points by the estimated likelihood ratio over the novelty covariates,
and deploy on the novel (high-novelty) target. We compare, on the target:

  naive         threshold from source labels, no correction   (the E2 failure)
  weighted      source labels reweighted by novelty LR         (label-free repair)
  target-cal    threshold from held-out TARGET labels          (rigorous upper bound)

All three use the combined score (the paper's primary gate). Weighted coverage is
approximate (weight-estimation error); target-cal is the rigorous fallback.
"""

from __future__ import annotations

import numpy as np

from experiments._common import ALPHA, DELTA, RESDIR, load_delivered, methods_with_enough, rng, save_json
from foldgate.conformal import estimate_weights, ltt_threshold, weighted_threshold
from foldgate.scores import ScoreCombiner
from foldgate.selective import evaluate_gate

NOVELTY_FEATS = ["ligand_similarity", "pocket_similarity"]

# Two shift regimes. Moderate: certification is feasible, so the weighted repair
# is demonstrable. Extreme: the novel target is <55% correct at baseline, so no
# gate can certify a high-coverage 80%-correct set -- the honest limit.
REGIMES = {
    "moderate (S0,S1 -> S2)": ({0, 1}, {2}),
    "extreme (S0-S2 -> S3,S4)": ({0, 1, 2}, {3, 4}),
}


def run(source: set, target: set, n_repeats: int = 80) -> dict:
    df = load_delivered()
    methods = methods_with_enough(df)
    g = rng()
    out = {}

    for m in methods:
        sub = df[df.method == m].reset_index(drop=True)
        strat = sub["novelty_stratum"].to_numpy().astype(int)
        y = sub["correct"].to_numpy()
        src = np.where(np.isin(strat, list(source)))[0]
        tgt = np.where(np.isin(strat, list(target)))[0]
        if len(src) < 200 or len(tgt) < 100:
            continue

        acc = {k: {"risk": [], "cov": []} for k in ("naive", "weighted", "target_cal")}
        for _ in range(n_repeats):
            s_perm = g.permutation(src)
            tr, cal = s_perm[: len(src) // 2], s_perm[len(src) // 2:]
            t_perm = g.permutation(tgt)
            t_cal, t_test = t_perm[: len(tgt) // 2], t_perm[len(tgt) // 2:]

            comb = ScoreCombiner().fit(sub.iloc[tr], y[tr])
            sc_cal = comb.predict(sub.iloc[cal])
            sc_tcal = comb.predict(sub.iloc[t_cal])
            sc_ttest = comb.predict(sub.iloc[t_test])

            w = estimate_weights(sub.iloc[cal][NOVELTY_FEATS].to_numpy(),
                                 sub.iloc[t_test][NOVELTY_FEATS].to_numpy())

            tau_naive = ltt_threshold(sc_cal, y[cal], alpha=ALPHA, delta=DELTA)
            tau_weight = weighted_threshold(sc_cal, y[cal], w, alpha=ALPHA, delta=DELTA)
            tau_tcal = ltt_threshold(sc_tcal, y[t_cal], alpha=ALPHA, delta=DELTA)

            for name, tau in (("naive", tau_naive), ("weighted", tau_weight), ("target_cal", tau_tcal)):
                r = evaluate_gate(sc_ttest, y[t_test], tau)
                acc[name]["cov"].append(r["coverage"])
                if r["n_accept"]:
                    acc[name]["risk"].append(r["selective_risk"])

        out[m] = {
            "n_source": int(len(src)), "n_target": int(len(tgt)),
            **{name: {
                "risk_on_target": float(np.mean(v["risk"])) if v["risk"] else float("nan"),
                "coverage_on_target": float(np.mean(v["cov"])),
                "frac_risk_le_alpha": float(np.mean(np.array(v["risk"]) <= ALPHA)) if v["risk"] else float("nan"),
            } for name, v in acc.items()},
        }
    return out


def main() -> None:
    all_res = {}
    for label, (source, target) in REGIMES.items():
        res = run(source, target)
        all_res[label] = res
        print(f"\n=== E3b weighted covariate-shift repair | {label} | alpha={ALPHA}, delta={DELTA} ===")
        print(f"{'method':10} {'':>12} {'risk_on_tgt':>11} {'coverage':>9} {'P(risk<=a)':>11}")
        for m, r in res.items():
            for name in ("naive", "weighted", "target_cal"):
                v = r[name]
                print(f"{m if name=='naive' else '':10} {name:>12} {v['risk_on_target']:>11.3f} "
                      f"{v['coverage_on_target']:>9.3f} {v['frac_risk_le_alpha']:>11.3f}")
    save_json(all_res, RESDIR / "e3b_weighted_repair.json")
    print("\nRead: moderate regime -> weighted pulls target risk toward alpha without target "
          "labels. Extreme regime -> the novel target is too hard for any gate to certify high "
          "coverage; the layer's value there is abstaining, not the naive gate's false confidence.")


if __name__ == "__main__":
    main()
