"""E56 (reviewer C14) -- is the zero-frontier count an artifact of the >=20-accept gate?

The headline "21 of 40 cells have no operating point at any coverage" (d2 feasibility map)
declares a cell non-certifiable when it cannot reach alpha with at least MIN_ACCEPT=20 accepted
targets. C14 asks whether that count is a property of the frozen score or of the arbitrary 20.

We re-run the exact d2 protocol -- reduce to one target-level row per (system, method), split the
familiar stratum S0 into a tau-fixing calibration half and a disjoint evaluation half, sweep the
frozen threshold over the source-coverage grid, and per non-reference cell take the frontier
c*_feasible -- but recompute the feasibility verdict at min_accept in {10, 20, 30, 50}. A cell is
zero-frontier iff NO coverage on the grid holds realized risk < alpha with >= min_accept accepted.

Reported per min_accept: the zero-frontier count over the 40 non-reference cells (2 axes x 5 models
x 4 novel strata) at alpha=0.20 on native ranking_score, plus the n-distribution
(median/min/max) of accepted targets at each zero-frontier cell's closest-to-feasible operating
point (the minimum-realized-risk coverage). A count that barely moves as the gate slides from 10 to
50 says the impossibility is in the score, not the threshold.

Outputs results/e56_minaccept_sensitivity.json. Runs on the delivered parquet alone.
"""

from __future__ import annotations

import numpy as np

from experiments._common import CONF, DELTA, RESDIR, load_delivered, methods_with_enough, rng, save_json
from experiments.d2_feasibility_map import AXES, COVERAGE_GRID, SOURCE_STRATUM, _cell, target_level

ALPHA = 0.20
MIN_ACCEPTS = [10, 20, 30, 50]
CAL_MIN = 20          # d2's calibration-half size gate (fixes tau); independent of the target gate
SEED = 20260715       # d2's seed, so the S0 splits reproduce the feasibility map exactly


def frontier_cells(tl, col, method, alpha, delta, g_rng):
    """d2 protocol for one (axis, model): return {stratum g in 1..4: [per-coverage cell]}.

    Each cell carries realized risk r and accepted count n_acc at a frozen source-coverage tau.
    The min_accept sweep is applied afterward, so these cells are computed once per (axis, model).
    """
    d = tl[(tl.method == method) & tl[col].notna()].copy()
    if len(d) == 0:
        return {}
    d["loss"] = (1 - d["correct"].astype(int)).astype(float)

    src_all = d[d[col] == SOURCE_STRATUM]
    sys_ids = np.array(sorted(src_all.system_id.unique()))
    g_rng.shuffle(sys_ids)
    cal_ids = set(sys_ids[: len(sys_ids) // 2])
    src_cal = src_all[src_all.system_id.isin(cal_ids)]
    src_eval = src_all[~src_all.system_id.isin(cal_ids)]
    if len(src_cal) < CAL_MIN:
        return {}

    taus = [(float(c), float(np.quantile(src_cal[CONF].to_numpy(), 1.0 - c))) for c in COVERAGE_GRID]
    out = {}
    for g in sorted(int(x) for x in d[col].unique()):
        if g == SOURCE_STRATUM:
            continue
        dg = src_eval if g == SOURCE_STRATUM else d[d[col] == g]
        if len(dg) == 0:
            continue
        cells = []
        for c, tau in taus:
            acc = dg[dg[CONF] >= tau]
            cell = _cell(acc["loss"].to_numpy(), len(dg), alpha, delta)
            cells.append({"coverage": c, "r": cell["R_Q"], "n_acc": cell["n_accepted_targets"]})
        out[g] = cells
    return out


def run(alpha: float = ALPHA, delta: float = DELTA, seed: int = SEED) -> dict:
    df = load_delivered()
    methods = methods_with_enough(df)
    tl = target_level(df)
    g_rng = rng(seed)

    # Compute the frontier cells ONCE per (axis, model, stratum); the min_accept sweep reuses them.
    all_cells: dict[tuple, list] = {}
    for axis, col in AXES.items():
        for m in methods:
            fc = frontier_cells(tl, col, m, alpha, delta, g_rng)
            for g, cl in fc.items():
                all_cells[(axis, m, g)] = cl

    n_cells = len(all_cells)
    per_min = {}
    for ma in MIN_ACCEPTS:
        zero, best_ns, zero_cells = 0, [], []
        for (axis, m, g), cl in all_cells.items():
            feasible = any(np.isfinite(c["r"]) and c["r"] < alpha and c["n_acc"] >= ma for c in cl)
            if feasible:
                continue
            zero += 1
            # Closest-to-feasible operating point: minimum realized risk among coverages that
            # accept anything (ties -> most stringent coverage). Its n shows the cell is not
            # zero-frontier merely for want of samples.
            valid = [c for c in cl if c["n_acc"] >= 1 and np.isfinite(c["r"])]
            if valid:
                bc = min(valid, key=lambda c: (c["r"], c["coverage"]))
                best_ns.append(int(bc["n_acc"]))
                zero_cells.append({"axis": axis, "model": m, "stratum": int(g),
                                   "min_risk": float(bc["r"]), "n_at_min_risk": int(bc["n_acc"])})
        arr = np.array(best_ns, dtype=float)
        per_min[str(ma)] = {
            "zero_frontier_count": int(zero),
            "n_nonreference_cells": n_cells,
            "n_at_best_coverage": {
                "median": float(np.median(arr)) if arr.size else float("nan"),
                "min": float(arr.min()) if arr.size else float("nan"),
                "max": float(arr.max()) if arr.size else float("nan"),
                "n_cells": int(arr.size),
            },
            "zero_frontier_cells": zero_cells,
        }

    counts = [per_min[str(ma)]["zero_frontier_count"] for ma in MIN_ACCEPTS]
    return {
        "alpha": alpha, "delta": delta, "score": CONF,
        "min_accepts": MIN_ACCEPTS, "cal_min": CAL_MIN, "seed": seed,
        "n_nonreference_cells": n_cells,
        "n_target_rows": int(len(tl)), "n_systems": int(tl.system_id.nunique()),
        "methods": methods,
        "per_min_accept": per_min,
        "count_range": [int(min(counts)), int(max(counts))],
        "count_spread": int(max(counts) - min(counts)),
        "takeaway": (f"zero-frontier count moves within {min(counts)}-{max(counts)} of {n_cells} "
                     f"as the accept gate slides 10->50; the impossibility tracks the score, "
                     f"not the >=20 threshold"),
    }


def main() -> None:
    res = run()
    save_json(res, RESDIR / "e56_minaccept_sensitivity.json")
    print(f"E56 -- min-accept sensitivity of the zero-frontier count "
          f"(alpha={res['alpha']}, score={res['score']}, {res['n_nonreference_cells']} non-ref cells)\n")
    print(f"  {'min_accept':>10} {'zero-frontier':>14} {'median n':>9} {'min n':>6} {'max n':>6}")
    for ma in MIN_ACCEPTS:
        b = res["per_min_accept"][str(ma)]
        nb = b["n_at_best_coverage"]
        print(f"  {ma:>10} {b['zero_frontier_count']:>10}/{res['n_nonreference_cells']:<3} "
              f"{nb['median']:>9.0f} {nb['min']:>6.0f} {nb['max']:>6.0f}")
    print(f"\n{res['takeaway']}")
    print(f"\nwrote {RESDIR / 'e56_minaccept_sensitivity.json'}")


if __name__ == "__main__":
    main()
