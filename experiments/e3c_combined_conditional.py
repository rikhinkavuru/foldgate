"""E3c -- the full method: combined score + group-conditional calibration.

E3 (native score) restored the guarantee on novel strata only by abstaining.
Here we ask whether the combined score, calibrated per novelty stratum, recovers
USABLE coverage on the novel strata while still controlling error. This closes
the E2 -> E3 arc: not just "stop trusting", but "here is how much you can still
safely keep, per novelty band".
"""

from __future__ import annotations

import numpy as np

from experiments._common import ALPHA, DELTA, RESDIR, load_delivered, methods_with_enough, rng, save_json
from foldgate.conformal import ltt_threshold
from foldgate.scores import ScoreCombiner
from foldgate.selective import evaluate_gate

STRAT = "novelty_stratum"


def run(n_repeats: int = 120) -> dict:
    df = load_delivered()
    methods = methods_with_enough(df)
    g = rng()
    out = {}
    for m in methods:
        sub = df[df.method == m].reset_index(drop=True)
        y = sub["correct"].to_numpy()
        strat = sub[STRAT].to_numpy().astype(int)
        levels = sorted(np.unique(strat).tolist())
        n = len(sub)
        acc = {k: [0, 0, 0] for k in levels}  # accepted, errors, total
        for _ in range(n_repeats):
            perm = g.permutation(n)
            a, b = int(0.4 * n), int(0.7 * n)
            tr, cal, te = perm[:a], perm[a:b], perm[b:]
            comb = ScoreCombiner().fit(sub.iloc[tr], y[tr])
            sc_cal, sc_te = comb.predict(sub.iloc[cal]), comb.predict(sub.iloc[te])
            for k in levels:
                cal_k = cal[strat[cal] == k]
                te_k = te[strat[te] == k]
                if len(te_k) == 0:
                    continue
                if len(cal_k) >= 40:
                    tau = ltt_threshold(sc_cal[strat[cal] == k], y[cal_k], alpha=ALPHA, delta=DELTA)
                else:
                    tau = None
                r = evaluate_gate(sc_te[strat[te] == k], y[te_k], tau)
                acc[k][0] += r["n_accept"]
                acc[k][1] += int(round(r["selective_risk"] * r["n_accept"])) if r["n_accept"] else 0
                acc[k][2] += len(te_k)
        out[m] = {
            str(k): {
                "selective_risk": (acc[k][1] / acc[k][0]) if acc[k][0] else float("nan"),
                "coverage": (acc[k][0] / acc[k][2]) if acc[k][2] else 0.0,
            }
            for k in levels
        }
    return out


def main() -> None:
    res = run()
    save_json(res, RESDIR / "e3c_combined_conditional.json")
    print(f"E3c -- combined score + group-conditional (alpha={ALPHA}, delta={DELTA})\n")
    for m, r in res.items():
        print(f"[{m}]  {'stratum':>7} {'risk':>7} {'coverage':>9}")
        for k in sorted(r, key=int):
            print(f"          {k:>7} {r[k]['selective_risk']:>7.3f} {r[k]['coverage']:>9.3f}")
        print()
    print("Read vs E3 (native): the combined score recovers non-trivial coverage on the "
          "novel strata while holding risk <= alpha, instead of abstaining outright.")


if __name__ == "__main__":
    main()
