"""E37 -- screening-table statistical-reporting fixes (reviewers R4.9, R4.10).

Two gaps in the E16 selective-screening table, fixed here without re-running screening:

  R4.9  EF DISCRETENESS. EF@1% on small libraries with few actives-in-top-1% is a
        coarse integer-valued grid, which is why several per-target-aggregated CIs
        pin the point estimate to an endpoint (e.g. ipTM 20.7 [18.1, 20.7]). For each
        target set we report n_targets, total actives, and per target the count of
        actives in the top 1% (k = round(0.01 * n_library);
        actives_in_top1 = round(EF * (n_act / n) * k)). We then report how many
        DISTINCT EF@1% values are realized across the target set (a small integer),
        which is the quantization that produces the endpoint CIs. BEDROC (alpha=80.5)
        is surfaced as a smoother secondary metric per method per target set.

  R4.10 PROPORTIONS WITH INTERVALS. The "targets: selective > random95" row
        (13% / 75% / 40%) is a count out of N. We recover the exact k/N from
        e16's frac_targets_selEF_beats_random95 and attach a Wilson 95% score
        interval. LIT-PCBA (N=5) cannot support any proportion claim and is flagged
        as a footnote rather than a column.

Reads results/e16_selective_screening.json, writes results/e37_screening_stats.json.
Torch-free, CPU-fast.
"""

from __future__ import annotations

import json
import math
from collections import Counter

from experiments._common import RESDIR, save_json

# Human-facing target-set names and their known active/target totals (sanity check).
DATASET_LABELS = {
    "dekois": "DEKOIS 2.0",
    "gpcr": "GPCR (recent)",
    "lipcba": "LIT-PCBA",
}
EXPECTED_TOTALS = {  # (n_targets, total_actives) from CLAUDE.md / task brief
    "dekois": (79, 3159),
    "gpcr": (16, 4673),
    "lipcba": (5, 529),
}

# EF metrics whose per-target discreteness we characterize. bedroc_* are the smoother
# secondary metrics carried alongside.
EF_METHODS = ["ef_dock", "ef_iptm", "ef_affinity", "ef_sel50"]
BEDROC_METHODS = ["bedroc_dock", "bedroc_affinity"]

# Rounding tolerance when counting DISTINCT realized EF values across a target set.
# EF grid spacing is n / (k * n_act); rounding to 3 decimals collapses float noise
# without merging genuinely distinct grid points.
EF_ROUND = 3


def wilson_interval(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion k/n at (default) 95% level.

    Returns (lo, hi). Defined for n >= 1; for k in {0, n} the interval is one-sided
    but still bounded away from the degenerate 0/1 that the Wald interval would give.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def discreteness_for_method(per_target: dict, method: str) -> dict:
    """Per-target actives-in-top-1% and the count of distinct realized EF values."""
    rows = []
    realized_ef = []
    for tid, rec in per_target.items():
        n = int(rec["n"])
        n_act = int(rec["n_act"])
        ef = rec.get(method)
        if ef is None or (isinstance(ef, float) and math.isnan(ef)):
            continue
        # For the ~50%-coverage selective metric the retained library is half-sized,
        # so its top-1% k is defined on the retained n; e16 does not store the
        # retained n per target, so for ef_sel50 we report on the full-library k as a
        # conservative (slightly finer) grid and flag it.
        k = max(1, round(0.01 * n))
        act_frac = n_act / n
        actives_in_top1 = round(ef * act_frac * k)
        # Grid spacing in EF units for one extra active in the top-k for THIS target.
        ef_step = 1.0 / (k * act_frac)  # = n / (k * n_act)
        # Max distinct EF values this target could take (0..min(k, n_act) actives).
        grid_size = min(k, n_act) + 1
        rows.append(
            {
                "target": tid,
                "n": n,
                "n_act": n_act,
                "k_top1": k,
                "ef": round(float(ef), 4),
                "actives_in_top1": actives_in_top1,
                "ef_step": round(ef_step, 4),
                "achievable_grid_size": grid_size,
            }
        )
        realized_ef.append(round(float(ef), EF_ROUND))

    realized_counter = Counter(realized_ef)
    ks = sorted(r["actives_in_top1"] for r in rows)
    med_k = ks[len(ks) // 2] if ks else None
    grids = sorted(r["achievable_grid_size"] for r in rows)
    med_grid = grids[len(grids) // 2] if grids else None
    return {
        "method": method,
        "n_targets_with_metric": len(rows),
        "n_distinct_realized_ef": len(realized_counter),
        "median_actives_in_top1": med_k,
        "min_actives_in_top1": ks[0] if ks else None,
        "max_actives_in_top1": ks[-1] if ks else None,
        "median_achievable_grid_size": med_grid,
        # The few EF values that soak up most of the mass -> endpoint CIs.
        "top5_realized_ef_by_frequency": [
            {"ef": ef, "count": c}
            for ef, c in realized_counter.most_common(5)
        ],
        "per_target": rows,
    }


def bedroc_summary(agg: dict) -> dict:
    out = {}
    for m in BEDROC_METHODS:
        if m in agg:
            a = agg[m]
            out[m] = {
                "median": round(float(a["median"]), 4),
                "ci90": [round(float(a["ci90"][0]), 4), round(float(a["ci90"][1]), 4)],
                "n": int(a["n"]),
            }
    return out


def main() -> None:
    src = RESDIR / "e16_selective_screening.json"
    e16 = json.loads(src.read_text())
    datasets = e16["datasets"]

    out: dict = {
        "_meta": {
            "source": str(src.relative_to(RESDIR.parent)),
            "reviewers": ["R4.9 (EF discreteness)", "R4.10 (proportions with intervals)"],
            "bedroc_alpha": 80.5,
            "wilson_level": 0.95,
        },
        "discreteness_R4_9": {},
        "beat_random_proportions_R4_10": {},
    }

    # --- R4.9 : EF discreteness + BEDROC secondary --------------------------------
    for name, ds in datasets.items():
        pt = ds["per_target"]
        n_targets = int(ds["n_targets"])
        total_actives = sum(int(r["n_act"]) for r in pt.values())
        exp = EXPECTED_TOTALS.get(name)
        methods = {m: discreteness_for_method(pt, m) for m in EF_METHODS if any(m in r for r in pt.values())}
        out["discreteness_R4_9"][name] = {
            "label": DATASET_LABELS.get(name, name),
            "n_targets": n_targets,
            "total_actives": total_actives,
            "total_actives_expected": exp[1] if exp else None,
            "total_actives_matches_expected": (exp is not None and total_actives == exp[1]),
            "bedroc_alpha80_5": bedroc_summary(ds["agg"]),
            "ef_methods": methods,
        }

    # --- R4.10 : selective > random95 proportion with Wilson interval -------------
    for name, ds in datasets.items():
        n_targets = int(ds["n_targets"])
        frac = float(ds["frac_targets_selEF_beats_random95"])
        k = round(frac * n_targets)
        lo, hi = wilson_interval(k, n_targets)
        rec = {
            "label": DATASET_LABELS.get(name, name),
            "k": k,
            "n": n_targets,
            "point_pct": round(100.0 * k / n_targets, 1),
            "wilson95_lo": round(lo, 4),
            "wilson95_hi": round(hi, 4),
            "wilson95_lo_pct": round(100.0 * lo, 1),
            "wilson95_hi_pct": round(100.0 * hi, 1),
            "underpowered": n_targets < 10,
        }
        if rec["underpowered"]:
            rec["recommendation"] = (
                "N<10: interval spans most of [0,1]; report as a footnote, not a table column."
            )
        out["beat_random_proportions_R4_10"][name] = rec

    dst = RESDIR / "e37_screening_stats.json"
    save_json(out, dst)

    # --- console takeaway ---------------------------------------------------------
    print(f"wrote {dst}")
    print("\n== R4.9 EF discreteness ==")
    for name, r in out["discreteness_R4_9"].items():
        iptm = r["ef_methods"].get("ef_iptm", {})
        print(
            f"  {r['label']:<14} n_targets={r['n_targets']:>3} actives={r['total_actives']:>4}"
            f" (expected {r['total_actives_expected']}, match={r['total_actives_matches_expected']})"
        )
        for m, md in r["ef_methods"].items():
            print(
                f"      {m:<12} distinct_EF={md['n_distinct_realized_ef']:>3}"
                f"  actives_in_top1 med={md['median_actives_in_top1']}"
                f" [{md['min_actives_in_top1']},{md['max_actives_in_top1']}]"
                f"  grid_size(med)={md['median_achievable_grid_size']}"
            )
        for bm, bv in r["bedroc_alpha80_5"].items():
            print(f"      {bm:<12} median={bv['median']} ci90={bv['ci90']}")

    print("\n== R4.10 selective > random95 (Wilson 95%) ==")
    for name, r in out["beat_random_proportions_R4_10"].items():
        flag = "  <-- FOOTNOTE (N<10)" if r["underpowered"] else ""
        print(
            f"  {r['label']:<14} {r['k']}/{r['n']} = {r['point_pct']}%"
            f"  Wilson95 [{r['wilson95_lo_pct']}%, {r['wilson95_hi_pct']}%]{flag}"
        )


if __name__ == "__main__":
    main()
