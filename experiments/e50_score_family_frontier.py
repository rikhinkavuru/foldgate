"""E50 -- the feasibility frontier across the SCORE FAMILY (reviewers D1, A4).

The zero-frontier count is score-dependent: interface ipTM and ligand pLDDT are better
pose-triage signals than the global ranking_score the headline used, and they flip some
cells to feasible. So the honest impossibility is not a property of one score but of the
whole frozen score family: a cell is impossibility-robust iff NO score in the family
reaches alpha at any coverage. This computes every score on ONE protocol (d2's exact
feasibility map: fixed delivered pose = top-1 by ranking_score; source-split on S0;
coverage sweep; feasible iff realized risk < alpha with >= 20 accepted), so the counts
are apples-to-apples, and reports the per-score counts AND the intersection.

Scores: ranking_score (global), iface_iptm (interface), ligand_plddt_mean /
ligand_plddt_min (ligand-local, extracted from CIF B-factors in e46). PL-PAE is not in
the released dump (CIF + ranking only), noted as a gap.

Output: results/e50_score_family_frontier.json
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from experiments._common import DELTA, RESDIR, load_delivered, methods_with_enough, rng
from experiments.d2_feasibility_map import (
    AXES,
    COVERAGE_GRID,
    MIN_ACCEPT,
    SOURCE_STRATUM,
    _cell,
    target_level,
)
from experiments._common import save_json

SCORES = ["ranking_score", "iface_iptm", "ligand_plddt_mean", "ligand_plddt_min"]
ALPHAS = (0.20, 0.10)


def _merge_plddt(tl: pd.DataFrame) -> pd.DataFrame:
    p = pd.read_parquet("data/processed/ligand_local_plddt.parquet")
    return tl.merge(p[["system_id", "method", "ligand_plddt_mean", "ligand_plddt_min"]],
                    on=["system_id", "method"], how="left")


def _zero_frontier_cells(df, methods, score, alpha, delta, g_rng):
    """Return the set of (axis, model, stratum) non-reference cells with NO feasible
    operating point at any coverage (n>=20), on `score`, under d2's protocol."""
    zero = set()
    total_nonref = 0
    for axis, col in AXES.items():
        for m in methods:
            d = df[(df.method == m) & df[col].notna() & df[score].notna()].copy()
            if len(d) == 0:
                continue
            d["loss"] = (1 - d["correct"].astype(int)).astype(float)
            src_all = d[d[col] == SOURCE_STRATUM]
            sys_ids = np.array(sorted(src_all.system_id.unique()))
            g_rng.shuffle(sys_ids)
            cal_ids = set(sys_ids[: len(sys_ids) // 2])
            src_cal = src_all[src_all.system_id.isin(cal_ids)]
            src_eval = src_all[~src_all.system_id.isin(cal_ids)]
            if len(src_cal) < MIN_ACCEPT:
                continue
            frames = {int(gk): (src_eval if gk == SOURCE_STRATUM else d[d[col] == gk])
                      for gk in sorted(d[col].unique())}
            taus = {float(c): float(np.quantile(src_cal[score].to_numpy(), 1.0 - c))
                    for c in COVERAGE_GRID}
            for gk, dg in frames.items():
                if gk == SOURCE_STRATUM or len(dg) == 0:
                    continue
                total_nonref += 1
                feas_any = False
                sc = dg[score].to_numpy()
                loss = dg["loss"].to_numpy()
                nstr = len(dg)
                for c in COVERAGE_GRID:
                    acc = sc >= taus[float(c)]
                    if acc.sum() < MIN_ACCEPT:
                        continue
                    cell = _cell(loss[acc], nstr, alpha, delta)
                    if cell["feasible"]:
                        feas_any = True
                        break
                if not feas_any:
                    zero.add((axis, m, gk))
    return zero, total_nonref


def run() -> dict:
    df = _merge_plddt(target_level(load_delivered()))
    methods = methods_with_enough(load_delivered())
    out = {"protocol": "d2 feasibility map, fixed delivered pose (top-1 ranking_score), source-split S0",
           "pl_pae": "NOT AVAILABLE in released RNP dump (CIF + ranking_score only); needs regeneration",
           "per_alpha": {}}
    for alpha in ALPHAS:
        per_score = {}
        zero_sets = {}
        total = None
        for score in SCORES:
            z, tot = _zero_frontier_cells(df, methods, score, alpha, DELTA, rng())
            per_score[score] = len(z)
            zero_sets[score] = z
            total = tot
        # intersections
        global_family = zero_sets["ranking_score"] & zero_sets["iface_iptm"]
        full_family = global_family & zero_sets["ligand_plddt_mean"] & zero_sets["ligand_plddt_min"]
        out["per_alpha"][str(alpha)] = {
            "n_nonreference_cells": total,
            "zero_frontier_per_score": per_score,
            "intersection_global_scores": len(global_family),
            "intersection_full_family_incl_ligand_local": len(full_family),
        }
    return out


def main() -> None:
    res = run()
    save_json(res, RESDIR / "e50_score_family_frontier.json")
    print("E50 -- score-family feasibility frontier (one protocol, all scores)\n")
    for alpha, r in res["per_alpha"].items():
        print(f"alpha={alpha}  ({r['n_nonreference_cells']} non-reference cells)")
        for s, n in r["zero_frontier_per_score"].items():
            print(f"    {s:>20}: {n} zero-frontier")
        print(f"    intersection (global scores)      : {r['intersection_global_scores']}")
        print(f"    intersection (+ ligand-local)     : {r['intersection_full_family_incl_ligand_local']}")
    print("\nPL-PAE:", res["pl_pae"])


if __name__ == "__main__":
    main()
