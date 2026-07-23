"""E63 -- the CP-certified score-family infeasibility count.

The abstract's headline is "N of 40 cells admit no certifiable threshold on any score, so
abstention is the certified action." e50 gave the POINT-ESTIMATE intersection (13 cells
where no score's realized-risk frontier clears alpha). But "abstention is the certified
action" is a claim about a bound, not a point: it holds only where we can CERTIFY that no
score can reach alpha, i.e. where the Clopper-Pearson LOWER bound exceeds alpha at every
usable coverage, on all four scores at once. This computes that stricter number.

Per non-reference cell (2 axes x 5 governed models x strata S1..S4 = 40) and per score
(ranking_score, iface_iptm, ligand_plddt_mean, ligand_plddt_min): the cell is CP-infeasible
on that score iff, over every coverage with >= MIN_ACCEPT accepted targets, the CP lower
bound on the accepted-set error exceeds alpha. A cell counts toward the family intersection
iff it is CP-infeasible on ALL four scores. That count, not the point-estimate 13, is what
licenses "abstention is the certified action."

Output: results/e63_cp_family_intersection.json
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from experiments._common import CONF, DELTA, RESDIR
from experiments.d2_feasibility_map import AXES, COVERAGE_GRID, MIN_ACCEPT, SOURCE_STRATUM, target_level
from foldgate.selective.metrics import clopper_pearson

GOVERNED = ["af3", "boltz1", "boltz1x", "chai", "protenix"]
SCORES = ["ranking_score", "iface_iptm", "ligand_plddt_mean", "ligand_plddt_min"]
ALPHA = 0.20


def _merge_plddt(tl):
    p = pd.read_parquet(RESDIR.parent / "data" / "processed" / "ligand_local_plddt.parquet")
    return tl.merge(p[["system_id", "method", "ligand_plddt_mean", "ligand_plddt_min"]],
                    on=["system_id", "method"], how="left")


def _cp_infeasible(g, score, alpha, delta):
    """True iff the CP lower bound on accepted error > alpha at EVERY usable coverage."""
    d = g.dropna(subset=[score, "correct"]).sort_values(score, ascending=False)
    n = len(d)
    if n < MIN_ACCEPT:
        return None  # underpowered to even test
    loss = (1 - d["correct"].to_numpy())
    any_usable = False
    for c in COVERAGE_GRID:
        k = int(round(c * n))
        if k < MIN_ACCEPT:
            continue
        any_usable = True
        acc = loss[:k]
        lo, _ = clopper_pearson(int(acc.sum()), k, delta)
        if lo <= alpha:
            return False  # this coverage is not provably infeasible -> cell not CP-infeasible
    return True if any_usable else None


def main():
    df = target_level(pd.read_csv(RESDIR / "analysis_table.csv"))
    # analysis_table already carries ligand_plddt_mean/min and iface_iptm
    df = df[df.method.isin(GOVERNED)]

    per_score = {s: 0 for s in SCORES}
    total_cells = 0
    intersection = 0
    detail = []
    for axis_name, col in AXES.items():
        for m in GOVERNED:
            sub = df[df.method == m]
            strata = sorted(x for x in sub[col].dropna().unique() if x != SOURCE_STRATUM)
            for s in strata:
                g = sub[sub[col] == s]
                if len(g) < MIN_ACCEPT:
                    continue
                total_cells += 1
                flags = {}
                for score in SCORES:
                    flags[score] = _cp_infeasible(g, score, ALPHA, DELTA)
                    if flags[score] is True:
                        per_score[score] += 1
                # CP-infeasible on ALL scores that are testable, and at least one testable
                testable = [v for v in flags.values() if v is not None]
                all_infeasible = bool(testable) and all(v is True for v in testable)
                if all_infeasible:
                    intersection += 1
                detail.append({"axis": axis_name, "model": m, "stratum": int(s),
                               "n": int(len(g)), "flags": flags, "all_infeasible": all_infeasible})

    out = {
        "alpha": ALPHA, "delta": DELTA, "min_accept": MIN_ACCEPT,
        "definition": ("CP-infeasible on a score = CP lower bound on accepted error > alpha at "
                       "every coverage with >= min_accept; family intersection = CP-infeasible on "
                       "ALL four scores. This is the count that licenses 'abstention is certified'."),
        "n_nonreference_cells_tested": total_cells,
        "cp_infeasible_per_score": per_score,
        "cp_certified_family_intersection": intersection,
        "point_estimate_intersection_e50": 13,
        "detail": detail,
    }
    (RESDIR / "e63_cp_family_intersection.json").write_text(json.dumps(out, indent=1))
    print("wrote e63_cp_family_intersection.json")
    print(f"  cells tested: {total_cells}")
    print(f"  CP-infeasible per score: {per_score}")
    print(f"  CP-certified family intersection: {intersection} (point-estimate was 13)")


if __name__ == "__main__":
    main()
