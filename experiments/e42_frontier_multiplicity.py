"""E42 -- multiplicity correction on the feasibility-frontier grid (reviewer R3.7).

The frontier headline (21 zero-frontier cells at alpha=0.20, 14 CP-robust) uses
per-cell exact binomial lower bounds without a family-wise correction. Here we apply
Benjamini-Hochberg / Benjamini-Yekutieli FDR and a Holm FWER correction across the
40 non-reference cells, and report the corrected count of certifiably-infeasible cells.

Per cell we take the operating point that minimizes realized risk among coverages with
>= 20 accepted targets (the cell's best shot at feasibility) and test H0: R_Q <= alpha
with the exact binomial upper tail p = P(Bin(n, alpha) >= errors). A rejection means the
cell is certifiably infeasible (no threshold reaches alpha at any coverage). We then
FDR/FWER-correct across the family.

Output: results/e42_frontier_multiplicity.json
"""

from __future__ import annotations

import json

import numpy as np
from scipy.stats import binom

from experiments._common import RESDIR, bh, holm, save_json

AXIS = "ligand"  # report the ligand axis primary; pocket in the JSON too
MIN_ACC = 20


def _cell_pvalue(cells: dict, alpha: float):
    """Min-risk operating point (n>=20); return (feasible, p_infeasible, n, err, cov)."""
    best = None
    for _c, v in cells.items():
        if v.get("n_accepted_targets", 0) < MIN_ACC:
            continue
        r = v["R_Q"]
        if best is None or r < best[0]:
            n = v["n_accepted_targets"]
            err = int(round(r * n))
            best = (r, n, err, v["coverage"])
    if best is None:
        return None
    r, n, err, cov = best
    feasible = r <= alpha
    # H0: true risk <= alpha; reject (infeasible) if too many errors. Upper-tail exact binomial.
    p_infeasible = float(binom.sf(err - 1, n, alpha))  # P(Bin(n,alpha) >= err)
    return {"feasible": bool(feasible), "p_infeasible": p_infeasible,
            "n": n, "err": err, "min_R_Q": r, "coverage": cov}


def run() -> dict:
    d2 = json.load(open(RESDIR / "d2_feasibility_map.json"))
    out = {"min_accept": MIN_ACC, "axes": {}}
    for alpha_key in ("0.2", "0.1"):
        alpha = float(alpha_key)
        for axis in ("ligand", "pocket"):
            models = d2["axes"][axis]["models"]
            cells_meta, pvals = [], []
            zero_frontier = 0
            for model, md in models.items():
                strata = md["alpha"][alpha_key]["strata"]
                for k, sd in strata.items():
                    if k == "0":  # S0 reference, feasible by construction
                        continue
                    res = _cell_pvalue(sd.get("cells", {}), alpha)
                    if res is None:
                        continue
                    cells_meta.append({"model": model, "stratum": k, **res})
                    if not res["feasible"]:
                        zero_frontier += 1
                        pvals.append(res["p_infeasible"])
            pv = np.array(pvals)
            # BH-FDR and Holm-FWER at 0.10 over the zero-frontier tests
            bh_reject = int(bh(pv, q=0.10).sum()) if pv.size else 0
            holm_adj = holm(pv) if pv.size else np.array([])
            holm_reject = int((holm_adj <= 0.10).sum()) if pv.size else 0
            # Benjamini-Yekutieli: BH with the harmonic penalty c(m)=sum 1/i
            if pv.size:
                m = pv.size
                cm = np.sum(1.0 / np.arange(1, m + 1))
                by_reject = int(bh(pv, q=0.10 / cm).sum())
            else:
                by_reject = 0
            key = f"{axis}_alpha{alpha_key}"
            out["axes"][key] = {
                "n_nonreference_cells": len(cells_meta),
                "zero_frontier_uncorrected": zero_frontier,
                "certifiably_infeasible_BH_fdr10": bh_reject,
                "certifiably_infeasible_BY_fdr10": by_reject,
                "certifiably_infeasible_Holm_fwer10": holm_reject,
            }
    # combined-axis totals at alpha=0.20 (the headline family of 40 cells)
    lig = out["axes"]["ligand_alpha0.2"]
    poc = out["axes"]["pocket_alpha0.2"]
    out["headline_alpha0.20"] = {
        "n_nonreference_cells": lig["n_nonreference_cells"] + poc["n_nonreference_cells"],
        "zero_frontier_uncorrected": lig["zero_frontier_uncorrected"] + poc["zero_frontier_uncorrected"],
        "certifiably_infeasible_BH_fdr10": lig["certifiably_infeasible_BH_fdr10"] + poc["certifiably_infeasible_BH_fdr10"],
        "certifiably_infeasible_BY_fdr10": lig["certifiably_infeasible_BY_fdr10"] + poc["certifiably_infeasible_BY_fdr10"],
        "certifiably_infeasible_Holm_fwer10": lig["certifiably_infeasible_Holm_fwer10"] + poc["certifiably_infeasible_Holm_fwer10"],
    }
    return out


def main() -> None:
    res = run()
    save_json(res, RESDIR / "e42_frontier_multiplicity.json")
    h = res["headline_alpha0.20"]
    print("E42 -- frontier multiplicity (40 non-reference cells, alpha=0.20)")
    print(f"  zero-frontier uncorrected: {h['zero_frontier_uncorrected']}")
    print(f"  certifiably infeasible, BH-FDR 0.10:  {h['certifiably_infeasible_BH_fdr10']}")
    print(f"  certifiably infeasible, BY-FDR 0.10:  {h['certifiably_infeasible_BY_fdr10']}")
    print(f"  certifiably infeasible, Holm-FWER 0.10: {h['certifiably_infeasible_Holm_fwer10']}")
    for k, v in res["axes"].items():
        print(f"    {k}: zero={v['zero_frontier_uncorrected']} BH={v['certifiably_infeasible_BH_fdr10']} "
              f"BY={v['certifiably_infeasible_BY_fdr10']} Holm={v['certifiably_infeasible_Holm_fwer10']}")


if __name__ == "__main__":
    main()
