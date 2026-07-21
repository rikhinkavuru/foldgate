"""E45 -- within-stratum discrimination of the native score (reviewer R4.A5).

Is the gate anti-selective on novel strata? For each (model, stratum) we report the
accept-all error, and the AUROC of the native score computed WITHIN the stratum with a
bootstrap CI. AUROC near 0.5 would mean the score carries no pose-discriminative signal
there; AUROC well above 0.5 with a high base error means the break is a base-rate
problem, not a signal collapse. This is the single most consequential missing number:
it separates "the score is useless on novel targets" from "the score still ranks but
the target is too hard to certify alpha".

Output: results/e45_within_stratum_auroc.json
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from experiments._common import (
    CONF,
    RESDIR,
    load_delivered,
    methods_with_enough,
    rng,
    save_json,
)

AXES = {"ligand": "novelty_stratum", "pocket": "pocket_novelty_stratum"}
N_BOOT = 2000


def _auroc_ci(s, y, g, n_boot=N_BOOT):
    if len(np.unique(y)) < 2:
        return None
    a = float(roc_auc_score(y, s))
    n = len(s)
    bs = []
    for _ in range(n_boot):
        i = g.integers(0, n, n)
        if len(np.unique(y[i])) < 2:
            continue
        bs.append(roc_auc_score(y[i], s[i]))
    return {"auroc": round(a, 3),
            "ci90": [round(float(np.quantile(bs, 0.05)), 3), round(float(np.quantile(bs, 0.95)), 3)],
            "chance_in_ci": bool(np.quantile(bs, 0.05) <= 0.5 <= np.quantile(bs, 0.95))}


def run() -> dict:
    df = load_delivered()
    methods = methods_with_enough(df)
    g = rng()
    out = {"note": "within-stratum AUROC of native ranking_score; base_err = 1 - mean(correct).",
           "axes": {}}
    for axis, col in AXES.items():
        out["axes"][axis] = {}
        for m in methods:
            sub = df[df.method == m].dropna(subset=[CONF, col, "correct"])
            per = {}
            for k in sorted(int(x) for x in sub[col].unique()):
                gsub = sub[sub[col] == k]
                if len(gsub) < 20:
                    continue
                s = gsub[CONF].to_numpy()
                y = gsub["correct"].to_numpy().astype(int)
                per[f"S{k}"] = {"n": int(len(gsub)), "base_err": round(1 - float(y.mean()), 3),
                                **( _auroc_ci(s, y, g) or {"auroc": None})}
            out["axes"][axis][m] = per
    return out


def main() -> None:
    res = run()
    save_json(res, RESDIR / "e45_within_stratum_auroc.json")
    print("E45 -- within-stratum AUROC of native ranking_score (ligand axis)\n")
    print(f"{'model':>9} {'S':>3} {'n':>5} {'base_err':>8} {'AUROC':>6} {'chance?':>8}")
    for m, per in res["axes"]["ligand"].items():
        for k, r in per.items():
            ch = "CHANCE" if r.get("chance_in_ci") else ""
            print(f"{m:>9} {k:>3} {r['n']:>5} {r['base_err']:>8.3f} {str(r['auroc']):>6} {ch:>8}")


if __name__ == "__main__":
    main()
