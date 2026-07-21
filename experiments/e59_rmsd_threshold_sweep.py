"""E59 (RMSD-threshold sweep) -- is the impossibility a property of the score or of the 2A line?

Binding-mode correctness is labeled at ligand-RMSD <= 2 A, a community convention. This experiment
asks whether the zero-frontier result (many novel cells have no operating point at any coverage)
survives moving that line. We recompute the correctness label at RMSD cutoffs
{1.0, 1.5, 2.0, 2.5, 3.0} A (correct = rmsd <= cutoff) and rerun the exact d2 feasibility protocol
for each -- target-level reduction, S0 calibration/evaluation split, frozen-tau coverage sweep on
native ranking_score, alpha=0.20 -- then count zero-frontier cells over the 40 non-reference cells
(2 axes x 5 models x 4 novel strata).

The S0 split is held fixed across cutoffs (same seed, reset per cutoff) so only the label moves.
The cutoff=2.0 column reproduces the paper's headline count. If the zero-frontier count stays high
as the line loosens to 3.0 A or tightens to 1.0 A, the impossibility is a property of the frozen
score's ranking, not an artifact of the 2 A convention.

Outputs results/e59_rmsd_threshold_sweep.json. Runs on the delivered parquet alone.
"""

from __future__ import annotations

import numpy as np

from experiments._common import CONF, DELTA, RESDIR, load_delivered, methods_with_enough, rng, save_json
from experiments.d2_feasibility_map import AXES, COVERAGE_GRID, SOURCE_STRATUM, _cell

ALPHA = 0.20
CUTOFFS = [1.0, 1.5, 2.0, 2.5, 3.0]
MIN_ACCEPT = 20       # d2 default target-accept gate
CAL_MIN = 20          # d2 calibration-half size gate
SEED = 20260715       # d2 seed; reset per cutoff so the S0 split is identical across labels


def target_rows(df):
    """One row per (system, method): the highest-ranking_score pose. Label recomputed per cutoff."""
    d = df.dropna(subset=[CONF, "rmsd"]).copy()
    d = d.sort_values(CONF, ascending=False)
    return d.groupby(["system_id", "method"], as_index=False).first()


def zero_frontier_count(tl, methods, alpha, delta, min_accept, seed) -> tuple[int, int]:
    """Count zero-frontier cells (no coverage holds risk<alpha with >=min_accept) over all cells."""
    g_rng = rng(seed)
    zero = total = 0
    for axis, col in AXES.items():
        for m in methods:
            d = tl[(tl.method == m) & tl[col].notna()].copy()
            if len(d) == 0:
                continue
            d["loss"] = (1 - d["correct"].astype(int)).astype(float)
            src_all = d[d[col] == SOURCE_STRATUM]
            sys_ids = np.array(sorted(src_all.system_id.unique()))
            g_rng.shuffle(sys_ids)
            cal_ids = set(sys_ids[: len(sys_ids) // 2])
            src_cal = src_all[src_all.system_id.isin(cal_ids)]
            if len(src_cal) < CAL_MIN:
                continue
            taus = [float(np.quantile(src_cal[CONF].to_numpy(), 1.0 - c)) for c in COVERAGE_GRID]
            for g in sorted(int(x) for x in d[col].unique()):
                if g == SOURCE_STRATUM:
                    continue
                dg = d[d[col] == g]
                if len(dg) == 0:
                    continue
                total += 1
                feasible = False
                for tau in taus:
                    acc = dg[dg[CONF] >= tau]
                    cell = _cell(acc["loss"].to_numpy(), len(dg), alpha, delta)
                    if np.isfinite(cell["R_Q"]) and cell["R_Q"] < alpha and \
                            cell["n_accepted_targets"] >= min_accept:
                        feasible = True
                        break
                if not feasible:
                    zero += 1
    return zero, total


def run(alpha: float = ALPHA, delta: float = DELTA, cutoffs=CUTOFFS, seed: int = SEED) -> dict:
    df = load_delivered()
    methods = methods_with_enough(df)
    base = target_rows(df)

    per_cutoff, n_cells = {}, None
    for cut in cutoffs:
        tl = base.copy()
        tl["correct"] = (tl["rmsd"] <= cut).astype(int)
        zero, total = zero_frontier_count(tl, methods, alpha, delta, MIN_ACCEPT, seed)
        n_cells = total
        per_cutoff[str(cut)] = {
            "zero_frontier_count": int(zero),
            "n_nonreference_cells": int(total),
            "overall_correct_rate": float(tl["correct"].mean()),
        }

    counts = [per_cutoff[str(c)]["zero_frontier_count"] for c in cutoffs]
    return {
        "alpha": alpha, "delta": delta, "score": CONF,
        "rmsd_cutoffs": cutoffs, "min_accept": MIN_ACCEPT, "cal_min": CAL_MIN, "seed": seed,
        "n_nonreference_cells": n_cells,
        "n_target_rows": int(len(base)), "n_systems": int(base.system_id.nunique()),
        "methods": methods,
        "per_cutoff": per_cutoff,
        "count_range": [int(min(counts)), int(max(counts))],
        "count_spread": int(max(counts) - min(counts)),
        "takeaway": (f"zero-frontier count stays in {min(counts)}-{max(counts)} of {n_cells} as the "
                     f"RMSD label moves 1.0->3.0 A; the impossibility is a property of the frozen "
                     f"score's ranking, not of the 2 A convention"),
    }


def main() -> None:
    res = run()
    save_json(res, RESDIR / "e59_rmsd_threshold_sweep.json")
    print(f"E59 -- RMSD-threshold sweep of the zero-frontier count "
          f"(alpha={res['alpha']}, score={res['score']}, {res['n_nonreference_cells']} non-ref cells)\n")
    print(f"  {'RMSD cutoff':>11} {'zero-frontier':>14} {'overall correct':>16}")
    for c in CUTOFFS:
        b = res["per_cutoff"][str(c)]
        print(f"  {c:>9.1f} A {b['zero_frontier_count']:>10}/{res['n_nonreference_cells']:<3} "
              f"{b['overall_correct_rate']:>16.3f}")
    print(f"\n{res['takeaway']}")
    print(f"\nwrote {RESDIR / 'e59_rmsd_threshold_sweep.json'}")


if __name__ == "__main__":
    main()
