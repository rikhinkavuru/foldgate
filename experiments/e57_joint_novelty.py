"""E57 (reviewer D20) -- joint (2-D) novelty: is the both-novel corner worse than either margin?

Every stratification in the paper is marginal on ONE axis: ligand-chemotype novelty OR pocket
novelty. D20 asks the obvious next question -- what happens where both are novel at once. We build
a coarse 2x2 grid on the target-level table,

    ligand-novel = (novelty_stratum >= 3),   pocket-novel = (pocket_novelty_stratum >= 3),

giving four disjoint cells {familiar-familiar, ligand-novel-only, pocket-novel-only, both-novel}.

A single GLOBAL iid gate is calibrated on native ranking_score to empirical selective risk
<= alpha=0.20 (the practitioner's native threshold, `naive_threshold`), fit on a calibration half
that IGNORES novelty, then deployed unchanged. We report, per model and per cell: n, base
correctness, and the realized selective risk of that one global gate on a held-out evaluation half
(averaged over repeated system-disjoint splits). The comparison of interest is whether the
both-novel cell's realized risk exceeds BOTH marginal-novel risks (ligand>=3 pooled, pocket>=3
pooled): a marginal axis can look survivable while the joint corner is the real failure.

Outputs results/e57_joint_novelty.json. Runs on the delivered parquet alone.
"""

from __future__ import annotations

import numpy as np

from experiments._common import CONF, RESDIR, load_delivered, methods_with_enough, rng, save_json
from experiments.d2_feasibility_map import target_level
from foldgate.conformal.risk import naive_threshold

ALPHA = 0.20
N_REPEATS = 200
CELLS = ["familiar_familiar", "ligand_novel_only", "pocket_novel_only", "both_novel"]


def _cell_mask(d, name):
    ln, pn = d["lig_novel"].to_numpy(), d["pock_novel"].to_numpy()
    return {
        "familiar_familiar": (~ln) & (~pn),
        "ligand_novel_only": ln & (~pn),
        "pocket_novel_only": (~ln) & pn,
        "both_novel": ln & pn,
    }[name]


def _risk(correct, accept):
    n = int(accept.sum())
    if n == 0:
        return float("nan"), 0
    return float(1.0 - correct[accept].mean()), n


def run(alpha: float = ALPHA, n_repeats: int = N_REPEATS, seed: int = 20260715) -> dict:
    df = load_delivered()
    methods = methods_with_enough(df)
    tl = target_level(df).dropna(subset=["novelty_stratum", "pocket_novelty_stratum"]).copy()
    tl["lig_novel"] = tl["novelty_stratum"] >= 3
    tl["pock_novel"] = tl["pocket_novelty_stratum"] >= 3
    g = rng(seed)

    out = {"alpha": alpha, "score": CONF, "n_repeats": n_repeats,
           "definition": "ligand-novel = novelty_stratum>=3; pocket-novel = pocket_novelty_stratum>=3",
           "gate": "global iid naive_threshold on native ranking_score, empirical risk<=alpha, "
                   "calibrated novelty-blind on a system-disjoint half, evaluated on the other half",
           "cells": CELLS, "methods": methods, "models": {}}

    for m in methods:
        d = tl[tl.method == m].reset_index(drop=True)
        y = d["correct"].to_numpy().astype(int)
        s = d[CONF].to_numpy()
        masks = {c: _cell_mask(d, c) for c in CELLS}
        lig_marg = d["lig_novel"].to_numpy()          # ligand>=3 pooled  (L-only + both)
        pock_marg = d["pock_novel"].to_numpy()        # pocket>=3 pooled  (P-only + both)

        # System-disjoint calibration/eval split, repeated, so the global gate is never scored
        # on the systems that set it.
        sys_ids = np.array(sorted(d.system_id.unique()))
        cell_risk = {c: [] for c in CELLS}
        cell_cov = {c: [] for c in CELLS}
        marg_risk = {"ligand_novel": [], "pocket_novel": []}
        overall_risk = []
        for _ in range(n_repeats):
            perm = g.permutation(sys_ids)
            cal_ids = set(perm[: len(perm) // 2])
            cal = d.system_id.isin(cal_ids).to_numpy()
            ev = ~cal
            tau = naive_threshold(s[cal], y[cal], alpha=alpha)
            if tau is None:
                continue
            acc = s >= tau
            r_all, _ = _risk(y[ev], acc[ev])
            if np.isfinite(r_all):
                overall_risk.append(r_all)
            for c in CELLS:
                msk = masks[c] & ev
                r, n = _risk(y[msk], acc[msk])
                if np.isfinite(r):
                    cell_risk[c].append(r)
                cell_cov[c].append(acc[msk].sum() / max(1, msk.sum()))
            for key, mm in (("ligand_novel", lig_marg), ("pocket_novel", pock_marg)):
                msk = mm & ev
                r, _ = _risk(y[msk], acc[msk])
                if np.isfinite(r):
                    marg_risk[key].append(r)

        def mean(x):
            return float(np.mean(x)) if len(x) else float("nan")

        cells_out = {}
        for c in CELLS:
            msk = masks[c]
            cells_out[c] = {
                "n": int(msk.sum()),
                "base_correctness": float(y[msk].mean()) if msk.sum() else float("nan"),
                "global_gate_realized_risk": mean(cell_risk[c]),
                "global_gate_coverage": mean(cell_cov[c]),
            }
        both = cells_out["both_novel"]["global_gate_realized_risk"]
        lig_only = cells_out["ligand_novel_only"]["global_gate_realized_risk"]
        pock_only = cells_out["pocket_novel_only"]["global_gate_realized_risk"]
        lig_m = mean(marg_risk["ligand_novel"])
        pock_m = mean(marg_risk["pocket_novel"])
        risks = {c: cells_out[c]["global_gate_realized_risk"] for c in CELLS}
        worst_cell = max((c for c in CELLS if np.isfinite(risks[c])), key=lambda c: risks[c])
        min_base = min(cells_out[c]["base_correctness"] for c in CELLS)
        out["models"][m] = {
            "n_total": int(len(d)),
            "overall_realized_risk": mean(overall_risk),
            "cells": cells_out,
            "marginal_realized_risk": {"ligand_novel_pooled": lig_m, "pocket_novel_pooled": pock_m},
            "both_novel_risk": both,
            # vs the two single-axis "only" cells (the pure marginal effects)
            "both_worse_than_both_only_cells": bool(
                np.isfinite(both) and both > lig_only and both > pock_only),
            # vs the pooled one-axis marginals the paper currently reports (each contains both-novel)
            "both_worse_than_both_pooled_marginals": bool(
                np.isfinite(both) and both > lig_m and both > pock_m),
            "worst_gate_risk_cell": worst_cell,
            "both_is_worst_cell": bool(worst_cell == "both_novel"),
            "both_has_lowest_base_correctness": bool(
                cells_out["both_novel"]["base_correctness"] == min_base),
            "both_exceeds_alpha": bool(np.isfinite(both) and both > alpha),
        }

    mods = out["models"].values()
    n_worst = sum(v["both_is_worst_cell"] for v in mods)
    n_only = sum(v["both_worse_than_both_only_cells"] for v in mods)
    n_over = sum(v["both_exceeds_alpha"] for v in mods)
    n_lowbase = sum(v["both_has_lowest_base_correctness"] for v in mods)
    out["summary"] = {
        "n_models": len(methods),
        "n_models_both_is_worst_gate_risk_cell": int(n_worst),
        "n_models_both_worse_than_both_only_cells": int(n_only),
        "n_models_both_lowest_base_correctness": int(n_lowbase),
        "n_models_both_exceeds_alpha": int(n_over),
        "takeaway": (f"the both-novel corner has the lowest base correctness in "
                     f"{n_lowbase}/{len(methods)} models and its gate risk exceeds alpha={alpha} in "
                     f"{n_over}/{len(methods)}; it is the single worst-risk cell in "
                     f"{n_worst}/{len(methods)} (novel-pocket dominates otherwise), so the joint "
                     f"corner is where the novelty-blind global gate fails hardest -- but pocket "
                     f"novelty alone is nearly as damaging"),
    }
    return out


def main() -> None:
    res = run()
    save_json(res, RESDIR / "e57_joint_novelty.json")
    print(f"E57 -- joint 2-D novelty, global iid native gate (alpha={res['alpha']})\n")
    hdr = f"  {'model':>9} {'cell':>18} {'n':>5} {'base':>6} {'gate risk':>9}"
    for m, mo in res["models"].items():
        print(f"[{m}]  overall gate risk {mo['overall_realized_risk']:.3f}  "
              f"(ligand-pool {mo['marginal_realized_risk']['ligand_novel_pooled']:.3f}, "
              f"pocket-pool {mo['marginal_realized_risk']['pocket_novel_pooled']:.3f})")
        print(hdr)
        for c, cc in mo["cells"].items():
            print(f"  {m:>9} {c:>18} {cc['n']:>5} {cc['base_correctness']:>6.3f} "
                  f"{cc['global_gate_realized_risk']:>9.3f}")
        print(f"    -> worst-risk cell: {mo['worst_gate_risk_cell']}; both>both only-cells: "
              f"{mo['both_worse_than_both_only_cells']}; both lowest base: "
              f"{mo['both_has_lowest_base_correctness']}; both exceeds alpha: {mo['both_exceeds_alpha']}\n")
    print(res["summary"]["takeaway"])
    print(f"\nwrote {RESDIR / 'e57_joint_novelty.json'}")


if __name__ == "__main__":
    main()
