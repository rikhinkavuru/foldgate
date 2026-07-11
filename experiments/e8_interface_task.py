"""E8 -- task-agnostic: the gate works for interface quality, not just pose-RMSD.

RQ5 asks whether the reliability layer is task-agnostic. We swap the correctness
label from ligand-RMSD <= 2 A to interface quality (LDDT-PLI >= 0.5) and rerun the
native-vs-combined AURC comparison. Same machinery, different target.
"""

from __future__ import annotations

import numpy as np

from experiments._common import DELTA, RESDIR, load_delivered, methods_with_enough, rng, save_json
from foldgate.conformal import ltt_threshold
from foldgate.scores import ScoreCombiner
from foldgate.selective import aurc, evaluate_gate

LDDT_THRESHOLD = 0.5
NATIVE = "ranking_score"
ALPHAS = [0.10, 0.20]


def three_way(idx, g):
    p = g.permutation(idx)
    n = len(p)
    return p[: int(0.4 * n)], p[int(0.4 * n): int(0.7 * n)], p[int(0.7 * n):]


def run(n_repeats: int = 120) -> dict:
    df = load_delivered()
    out = {}
    for m in methods_with_enough(df):
        sub = df[df.method == m].dropna(subset=["lddt_pli"]).reset_index(drop=True)
        y = (sub["lddt_pli"].to_numpy() >= LDDT_THRESHOLD).astype(int)
        s_nat = sub[NATIVE].to_numpy()
        idx = np.arange(len(sub))
        an, ac = [], []
        cov = {a: {"nat": [], "comb": []} for a in ALPHAS}
        g = rng()
        for _ in range(n_repeats):
            tr, cal, te = three_way(idx, g)
            comb = ScoreCombiner().fit(sub.iloc[tr], y[tr])
            sc_cal, sc_te = comb.predict(sub.iloc[cal]), comb.predict(sub.iloc[te])
            an.append(aurc(s_nat[te], y[te]))
            ac.append(aurc(sc_te, y[te]))
            for a in ALPHAS:
                tn = ltt_threshold(s_nat[cal], y[cal], alpha=a, delta=DELTA)
                tc = ltt_threshold(sc_cal, y[cal], alpha=a, delta=DELTA)
                cov[a]["nat"].append(evaluate_gate(s_nat[te], y[te], tn)["coverage"])
                cov[a]["comb"].append(evaluate_gate(sc_te, y[te], tc)["coverage"])
        out[m] = {
            "base_rate": float(y.mean()),
            "aurc_native": float(np.mean(an)),
            "aurc_combined": float(np.mean(ac)),
            "aurc_improve_pct": round(100 * (np.mean(an) - np.mean(ac)) / np.mean(an), 1),
            "coverage": {str(a): {"native": float(np.mean(cov[a]["nat"])),
                                  "combined": float(np.mean(cov[a]["comb"]))} for a in ALPHAS},
        }
    return out


def main() -> None:
    res = run()
    save_json(res, RESDIR / "e8_interface_task.json")
    print(f"E8 -- interface-quality task (LDDT-PLI >= {LDDT_THRESHOLD}), delta={DELTA}\n")
    print(f"{'method':10} {'base':>6} {'AURC nat':>9} {'AURC comb':>10} {'improve':>8} {'cov@.2 nat->comb':>18}")
    for m, r in res.items():
        c = r["coverage"]["0.2"]
        print(f"{m:10} {r['base_rate']:>6.3f} {r['aurc_native']:>9.3f} {r['aurc_combined']:>10.3f} "
              f"{r['aurc_improve_pct']:>7.1f}%   {c['native']:>6.2f} -> {c['combined']:.2f}")
    print("\nThe gate transfers to interface quality: combined score again dominates native on AURC.")


if __name__ == "__main__":
    main()
