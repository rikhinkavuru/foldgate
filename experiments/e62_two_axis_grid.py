"""E62 -- the two-axis (ligand x pocket) certified gate the paper recommends.

Sec. 6 states that a one-axis (ligand) certificate can under-cover the pocket-novel half
of a stratum by up to +0.32, and recommends the two-axis grid to remove this by
construction. This runs that grid so the recommendation is demonstrated, not just asserted.

For each model we cross-classify targets into a 2x2 familiar/novel grid on ligand novelty
(novelty_stratum <= 1 vs >= 2) and pocket novelty (pocket_novelty_stratum <= 1 vs >= 2),
calibrate a native-score threshold on the both-familiar cell, and report per cell the
CP-certified frontier (largest coverage whose Clopper-Pearson upper bound <= alpha under
the >=20-accept rule) and the resulting verdict, exactly as the one-axis cards do. The
point is the diagonal: the pocket-novel / ligand-familiar cell, which a ligand-only card
folds into a FEASIBLE stratum, is scored on its own here.

Output: results/e62_two_axis_grid.json  (per model, 2x2 coverage + verdict + risk)
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd

from experiments._common import CONF, DELTA, RESDIR
from experiments.d2_feasibility_map import COVERAGE_GRID, MIN_ACCEPT, _cell, _frontier

GOVERNED = ["af3", "boltz1", "boltz1x", "chai", "protenix"]
ALPHA = 0.20


def _top1(g):
    return g.sort_values(CONF, ascending=False).groupby(["system_id", "method"], as_index=False).first()


def _cell_verdict(loss, n_stratum):
    cells = {float(c): _cell(loss[:max(1, int(round(c * n_stratum)))], n_stratum, ALPHA, DELTA)
             for c in COVERAGE_GRID if int(round(c * n_stratum)) >= 1}
    cert = _frontier(cells, "certified")
    feas = _frontier(cells, "feasible")
    # min CP-lower over usable coverages decides infeasible vs underpowered
    usable = [v for v in cells.values() if v["n_accepted_targets"] >= MIN_ACCEPT]
    min_cp_low = min((v["R_Q_ci"][0] for v in usable), default=float("nan"))
    if cert > 0:
        verdict = "FEASIBLE"
    elif np.isfinite(min_cp_low) and min_cp_low > ALPHA:
        verdict = "ABSTAIN-infeasible"
    else:
        verdict = "ABSTAIN-underpowered"
    return {"cp_certified_coverage": cert, "point_frontier": feas, "verdict": verdict,
            "base_correct": round(float(1 - loss.mean()), 3)}


def main():
    d = pd.read_csv(RESDIR / "analysis_table.csv")
    d = d[d.method.isin(GOVERNED)].dropna(
        subset=[CONF, "correct", "novelty_stratum", "pocket_novelty_stratum"]).copy()
    # 3 bins per axis: familiar (S0-1), moderate (S2), novel (S3-4)
    def _bin(x):
        return np.where(x <= 1, 0, np.where(x == 2, 1, 2))
    d["lig_b"] = _bin(d["novelty_stratum"].astype(int))
    d["poc_b"] = _bin(d["pocket_novelty_stratum"].astype(int))
    names = {0: "fam", 1: "mod", 2: "novel"}

    out = {"alpha": ALPHA, "score": CONF, "grid": "3x3 familiar/moderate/novel on ligand and pocket",
           "cells_def": "bin: S0-1=familiar, S2=moderate, S3-4=novel, on each of ligand and pocket novelty",
           "per_model": {}}
    for m in GOVERNED:
        t = _top1(d[d.method == m])
        grid = {}
        for lig in (0, 1, 2):
            for poc in (0, 1, 2):
                g = t[(t.lig_b == lig) & (t.poc_b == poc)]
                key = f"lig_{names[lig]}__poc_{names[poc]}"
                if len(g) < MIN_ACCEPT:
                    grid[key] = {"n": int(len(g)), "verdict": "ABSTAIN-underpowered",
                                 "cp_certified_coverage": 0.0, "underpowered_small_n": True}
                    continue
                g = g.sort_values(CONF, ascending=False)
                loss = 1 - g["correct"].to_numpy()
                grid[key] = {"n": int(len(g)), **_cell_verdict(loss, len(g))}
        out["per_model"][m] = grid
    p = RESDIR / "e62_two_axis_grid.json"
    p.write_text(json.dumps(out, indent=1))
    print(f"wrote {p.name}")
    for m in GOVERNED:
        print(f"  {m}:")
        for k, v in out["per_model"][m].items():
            print(f"    {k:26s} n={v['n']:>4} {v['verdict']:22s} "
                  f"CP-cov={v.get('cp_certified_coverage',0):.2f} base={v.get('base_correct','-')}")


if __name__ == "__main__":
    main()
