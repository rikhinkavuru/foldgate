"""E47 (reviewer A4): re-run the cross-model comparison on a COMMON score.

Reviewer A4: the five models' native "ranking_score" are DIFFERENT composite functions
(af3 ranking_score = 0.8 ipTM + 0.2 pTM + ...; boltz confidence_score; chai aggregate_score;
protenix its own), so a cross-model drift/frontier comparison keyed on ranking_score confounds
"model X is more overconfident" with "model X uses a different score function". This script
re-runs the two cross-model objects on interface ipTM (`iface_iptm`), which IS defined the same
way for all five models, as the PRIMARY cross-model comparison, and reports whether:

  1. reliability drift D_signed (E12 _drift, S0 reference, target-mass-weighted P(correct|score) gap)
     survives on the common score, and whether the cross-model DRIFT RANKING changes vs ranking_score;
  2. the ligand-axis zero-frontier count (D2-style: frozen "accept iff s>=tau" swept over source
     coverage, feasible iff realized risk<=alpha with >=MIN_ACCEPT accepted) changes on the common
     score, and whether any per-cell verdict flips.

Writes results/e47_common_score.json. Runs on the delivered parquet alone (no GPU).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from experiments._common import (  # noqa: E402
    DELTA,
    RESDIR,
    load_delivered,
    methods_with_enough,
    rng,
    save_json,
)

COMMON = "iface_iptm"          # the score defined the same way for all five models
NATIVE = "ranking_score"       # the per-model composite the paper quotes
AXES = {"ligand": "novelty_stratum", "pocket": "pocket_novelty_stratum"}
N_BINS = 5
N_BOOT = 2000
ALPHA = 0.20
MIN_ACCEPT = 20
SOURCE_STRATUM = 0
COVERAGE_GRID = np.round(np.arange(0.05, 1.0 + 1e-9, 0.05), 2)


# --------------------------------------------------------------------------------------
# Drift (mirror of experiments/e12_reliability_drift.py _drift/_edges), score-parameterized
# --------------------------------------------------------------------------------------
def _drift(conf_s, y_s, conf_t, y_t, edges):
    """Target-mass-weighted signed and absolute P(correct|score) gap, S0 -> Sk."""
    signed, absg, wts = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:], strict=False):
        ms = (conf_s >= lo) & (conf_s < hi)
        mt = (conf_t >= lo) & (conf_t < hi)
        if not ms.any() or not mt.any():
            continue
        ps, pt = float(y_s[ms].mean()), float(y_t[mt].mean())
        signed.append(ps - pt)
        absg.append(abs(ps - pt))
        wts.append(int(mt.sum()))
    if not wts:
        return float("nan"), float("nan")
    w = np.asarray(wts, float)
    return float(np.average(signed, weights=w)), float(np.average(absg, weights=w))


def _edges(conf_s, conf_t):
    e = np.quantile(np.concatenate([conf_s, conf_t]), np.linspace(0, 1, N_BINS + 1))
    e[0], e[-1] = -np.inf, np.inf
    return e


def drift_table(df, score, methods, g):
    """Per (model, axis, stratum) D_signed with a 90% bootstrap CI (shared per-axis resample)."""
    out = {m: {} for m in methods}
    for m in methods:
        sub = df[df.method == m]
        for axis, col in AXES.items():
            s = sub.dropna(subset=[score, col])
            conf = s[score].to_numpy()
            y = s["correct"].to_numpy().astype(int)
            strat = s[col].to_numpy().astype(int)
            levels = sorted(np.unique(strat).tolist())
            ref = levels[0]
            m_ref = strat == ref
            ks = [k for k in levels if k != ref and int((strat == k).sum()) >= 20]
            axis_out = {}
            if ks:
                edges_by_k = {k: _edges(conf[m_ref], conf[strat == k]) for k in ks}
                point = {k: _drift(conf[m_ref], y[m_ref], conf[strat == k], y[strat == k],
                                   edges_by_k[k]) for k in ks}
                n = len(conf)
                boot_s = {k: np.empty(N_BOOT) for k in ks}
                for b in range(N_BOOT):
                    bi = g.integers(0, n, n)
                    cb, yb, stb = conf[bi], y[bi], strat[bi]
                    ref_b = stb == ref
                    cref, yref = cb[ref_b], yb[ref_b]
                    for k in ks:
                        km = stb == k
                        ss, _ = _drift(cref, yref, cb[km], yb[km], edges_by_k[k])
                        boot_s[k][b] = ss
                for k in ks:
                    d_signed, d_abs = point[k]
                    bs = boot_s[k][np.isfinite(boot_s[k])]
                    lo = float(np.quantile(bs, 0.05)) if bs.size else float("nan")
                    hi = float(np.quantile(bs, 0.95)) if bs.size else float("nan")
                    axis_out[str(int(k))] = {
                        "D_signed": d_signed, "D_abs": d_abs,
                        "ci90": [lo, hi],
                        "n_stratum": int((strat == k).sum()), "n_ref": int(m_ref.sum()),
                    }
            out[m][axis] = axis_out
    return out


# --------------------------------------------------------------------------------------
# Frontier (D2-style simplified): frozen "accept iff s>=tau", swept over source coverage
# --------------------------------------------------------------------------------------
def target_level(df, score):
    """One delivered row per (system_id, method): the pose a caller acts on = highest `score`."""
    d = df.dropna(subset=[score, "correct"]).copy()
    d = d.sort_values(score, ascending=False)
    return d.groupby(["system_id", "method"], as_index=False).first()


def frontier_table(df, score, methods, seed=20260715):
    """Per (model, ligand stratum) c*_feasible on the frozen rule; zero-frontier = c*==0."""
    col = AXES["ligand"]
    tl = target_level(df, score)
    g_rng = rng(seed)
    models = {}
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
        src_eval = src_all[~src_all.system_id.isin(cal_ids)]
        if len(src_cal) < MIN_ACCEPT:
            continue
        frames = {int(g): (src_eval if g == SOURCE_STRATUM else d[d[col] == g])
                  for g in sorted(d[col].unique())}
        taus = {float(c): float(np.quantile(src_cal[score].to_numpy(), 1.0 - c))
                for c in COVERAGE_GRID}
        strata = {}
        for g, dg in frames.items():
            if len(dg) == 0:
                continue
            best_c = 0.0
            for c, tau in taus.items():
                acc = dg[dg[score] >= tau]
                loss = acc["loss"].to_numpy()
                n_acc = int(len(loss))
                if n_acc >= MIN_ACCEPT and float(loss.mean()) <= ALPHA:
                    best_c = max(best_c, c)
            strata[str(int(g))] = {
                "c_star_feasible": best_c,
                "zero_frontier": bool(best_c == 0.0),
                "n_targets_stratum": int(len(dg)),
            }
        models[m] = strata
    return models


def count_zero_frontier(models):
    """Zero-frontier count over the 20 non-reference ligand cells (5 models x S1..S4)."""
    n_zero = n_present = 0
    absent = []
    for m in sorted(models):
        for g in range(1, 5):
            cell = models[m].get(str(g))
            if cell is None:
                absent.append(f"{m}/S{g}")
                continue
            n_present += 1
            if cell["zero_frontier"]:
                n_zero += 1
    return {"n_zero_frontier": n_zero, "n_present": n_present, "absent_cells": absent}


def run():
    df = load_delivered()
    methods = methods_with_enough(df)
    g = rng()

    drift_common = drift_table(df, COMMON, methods, g)
    drift_native = drift_table(df, NATIVE, methods, g)

    fr_common = frontier_table(df, COMMON, methods)
    fr_native = frontier_table(df, NATIVE, methods)
    zf_common = count_zero_frontier(fr_common)
    zf_native = count_zero_frontier(fr_native)

    # per-cell frontier verdict flips on the ligand axis (non-ref cells)
    flips = []
    for m in sorted(set(fr_common) & set(fr_native)):
        for g_ in range(1, 5):
            c = fr_common[m].get(str(g_))
            n = fr_native[m].get(str(g_))
            if c is None or n is None:
                continue
            if c["zero_frontier"] != n["zero_frontier"]:
                flips.append({
                    "model": m, "stratum": g_,
                    "ranking_score": "zero" if n["zero_frontier"] else "feasible",
                    "iface_iptm": "zero" if c["zero_frontier"] else "feasible",
                    "c_star_native": n["c_star_feasible"], "c_star_common": c["c_star_feasible"],
                })

    # DRIFT RANKING: order models by S3 drift, on each axis, for each score
    def s3_rank(dt, axis):
        vals = {m: dt[m].get(axis, {}).get("3", {}).get("D_signed") for m in methods}
        vals = {m: v for m, v in vals.items() if v is not None and np.isfinite(v)}
        return [m for m, _ in sorted(vals.items(), key=lambda kv: -kv[1])], vals

    ranking = {}
    for axis in AXES:
        r_c, v_c = s3_rank(drift_common, axis)
        r_n, v_n = s3_rank(drift_native, axis)
        ranking[axis] = {
            "iface_iptm_order": r_c, "iface_iptm_S3": v_c,
            "ranking_score_order": r_n, "ranking_score_S3": v_n,
            "ranking_changed": bool(r_c != r_n),
        }

    return {
        "meta": {
            "common_score": COMMON, "native_score": NATIVE, "methods": methods,
            "alpha": ALPHA, "min_accept": MIN_ACCEPT, "n_boot": N_BOOT,
            "coverage_grid": COVERAGE_GRID.tolist(),
            "note": "iface_iptm is defined identically across all five models; ranking_score is a "
                    "per-model composite. Drift = E12 _drift (S0 ref, target-mass-weighted "
                    "P(correct|score) gap); frontier = D2-style frozen accept-iff-s>=tau sweep.",
        },
        "drift": {"iface_iptm": drift_common, "ranking_score": drift_native},
        "drift_ranking_S3": ranking,
        "frontier_ligand": {"iface_iptm": fr_common, "ranking_score": fr_native},
        "zero_frontier_ligand": {"iface_iptm": zf_common, "ranking_score": zf_native},
        "frontier_verdict_flips": flips,
    }


def _fmt(d):
    return "  --  " if d is None else f"{d['D_signed']:+.3f}[{d['ci90'][0]:+.2f},{d['ci90'][1]:+.2f}]"


def main():
    res = run()
    save_json(res, RESDIR / "e47_common_score.json")
    methods = res["meta"]["methods"]

    print("E47 (reviewer A4) -- cross-model comparison on the COMMON score iface_iptm\n")
    for axis in AXES:
        print(f"=== reliability drift D_signed on the {axis} axis (S1..S4) ===")
        print(f"  {'model':>9}  {'score':>13} " + " ".join(f"{'S'+str(k):>22}" for k in range(1, 5)))
        for score, key in [("iface_iptm", "iface_iptm"), ("ranking_score", "ranking_score")]:
            for m in methods:
                cells = res["drift"][key][m].get(axis, {})
                row = " ".join(f"{_fmt(cells.get(str(k))):>22}" for k in range(1, 5))
                print(f"  {m:>9}  {score:>13} {row}")
            print()
        rk = res["drift_ranking_S3"][axis]
        print(f"  S3 drift ranking (high->low):")
        print(f"     iface_iptm   : {rk['iface_iptm_order']}")
        print(f"     ranking_score: {rk['ranking_score_order']}")
        print(f"     ranking changed on {axis} axis: {rk['ranking_changed']}\n")

    print("=== ligand-axis zero-frontier count (of 20 = 5 models x S1..S4) ===")
    for key in ("iface_iptm", "ranking_score"):
        z = res["zero_frontier_ligand"][key]
        print(f"  {key:>13}: {z['n_zero_frontier']}/{z['n_present']} zero-frontier"
              + (f"  (absent: {z['absent_cells']})" if z["absent_cells"] else ""))
    flips = res["frontier_verdict_flips"]
    print(f"\n  frontier verdict flips (iptm vs ranking_score): {len(flips)}")
    for f in flips:
        print(f"     {f['model']}/S{f['stratum']}: ranking_score={f['ranking_score']} "
              f"-> iface_iptm={f['iface_iptm']} (c* {f['c_star_native']:.2f}->{f['c_star_common']:.2f})")
    print(f"\nwrote {RESDIR / 'e47_common_score.json'}")


if __name__ == "__main__":
    main()
