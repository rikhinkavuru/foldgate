"""D2 step 1: the per-stratum CERTIFICATION FEASIBILITY MAP on RNP (the Week-1 go/no-go gate).

The project theorem says a label-free certificate under-reports realized target selective risk by
the accept-region concept gap, so certifying a shifted stratum costs target labels. D2 asks the
dual question: where do those labels buy a certificate at all, and how many are needed?

The object is the certification margin of the deployed rule "accept iff s >= tau",

    m_g(tau) := alpha - R_Q,g(tau),     R_Q,g(tau) = E_Q[ L | s >= tau, stratum g ],

with the binding-mode error loss L = 1{ligand-RMSD > 2 A}. The sign of m_g is the whole map:

  * m_g > 0  -- FEASIBLE. The deployed rule really does hold alpha on this stratum, so target
    labels can certify it, and the label budget is governed by m_g (Target 1).
  * m_g <= 0 -- INFEASIBLE. The rule genuinely violates alpha here. No certificate, label-free or
    label-fed, can certify a false statement; the honest action is to abstain or drop coverage.
    This is the impossibility regime the project theorem predicts, seen from the label side.

WHY THIS IS A COVERAGE SWEEP AND NOT A SINGLE OPERATING POINT. Theorem 1(c) is stated *pinned at a
fixed coverage c* and deliberately does NOT claim that alpha is unreachable at lower coverage;
that stronger claim would need R_Q monotone in coverage, which is unproven and can fail in the
novel tail (THEOREM_RECONCILED scope note S1). Reporting one LTT-chosen tau would both hide that
open question and make the map an artifact of one calibrator: at the project default alpha = 0.20
the familiar stratum's own risk is already below alpha, so LTT certifies the *whole* set and the
"gate" degenerates to accept-everything, while for a model whose top-ranked poses contain two
early errors, fixed-sequence LTT breaks at its first hypothesis and returns no threshold at all.
Neither degeneracy is a fact about co-folding; both are facts about one calibration rule.

So the deliverable is a FRONTIER, not a point. We sweep the deployed rule over a grid of source
coverages, and per stratum report the largest coverage at which the rule still holds alpha:

    c*_g(alpha) := max { c : R_Q,g(tau(c)) <= alpha }.

c*_0 large and c*_g -> 0 on novel strata is the sharpest honest statement available: it says the
frozen score cannot reach alpha on that stratum *at any coverage*, which is strictly stronger than
the coverage-pinned theorem and is an empirical finding rather than a corollary. Where c*_g > 0 the
stratum is feasible and D2's label question is live; the LTT operating point is then reported as
one labelled point on this frontier, not as its definition.

Three conventions are load-bearing.

1. THE QUERY UNIT IS THE TARGET, NOT THE POSE. RNP ships several rows per (system, method) when a
   system has multiple ligand instances, and losses are strongly clustered by target (a hard pocket
   makes the models fail together). Pooling those rows as independent Bernoulli draws is
   anti-conservative and silently changes the estimand. We reduce to ONE delivered row per
   (system, method) -- the highest `ranking_score`, i.e. the pose a caller would actually act on --
   so draws are independent across systems, and every budget is reported in independent
   target-labels, per deployed model.

2. tau IS SET ON THE SOURCE AND DEPLOYED FROZEN. Each tau on the grid is a quantile of the FAMILIAR
   (source) stratum's scores, then applied unchanged to every target stratum. This is what a caller
   does: calibrate where you have labels, deploy where you do not. Refitting tau per stratum is
   Theorem 3's achievability construction and would erase the gap the map exists to show. The
   source stratum is split by system into a calibration half (which fixes tau) and a disjoint
   evaluation half (which reports its risk), so the source cell is not scored in-sample.

3. m_g IS MEASURED, NOT DECOMPOSED, FOR THE DECISION. The certification procedure observes R_Q,g
   only through queried labels; it never sees R_ref,g or the concept gap. The
   m_g = alpha - R_ref,g - Delta_bar_g split is reported alongside as an EXPLANATORY device for
   where the impossibility bites, not as a quantity any estimator exploits.

Outputs `results/d2_feasibility_map.json`. Runs on the delivered parquet alone (no coordinates,
no GPU).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from experiments._common import (  # noqa: E402
    CONF,
    DELTA,
    RESDIR,
    load_delivered,
    methods_with_enough,
    rng,
    save_json,
)
from foldgate.conformal.risk import ltt_threshold  # noqa: E402
from foldgate.conformal.shift_decomp import shift_decomposition  # noqa: E402
from foldgate.selective.metrics import clopper_pearson  # noqa: E402

# Novelty axes carried through the map. Both are shipped by RNP and pre-computed in
# `build_features`; stratum 0 = most similar to train, rising label = more novel, and the top
# stratum is the no-training-analog (NaN-similarity) extrapolation bin (features/novelty.py).
AXES = {
    "ligand": "novelty_stratum",
    "pocket": "pocket_novelty_stratum",
}
SOURCE_STRATUM = 0                                  # the familiar bin tau is calibrated on
COVERAGE_GRID = np.round(np.arange(0.05, 1.0 + 1e-9, 0.05), 2)   # source coverage knob
ALPHA_GRID = (0.10, 0.20)                           # 0.20 = project default; 0.10 = the biting regime
MIN_ACCEPT = 20                                     # below this a cell is declared non-certifiable
N_BOOT = 4000

# Candidate label-free PPI control covariates (Target 3). Each must be computable at deployment
# without a crystal structure. Sign is irrelevant (only rho^2 enters the variance reduction).
COVARIATES = [
    "ens_ranking_std",          # intra-model spread of ranking_score over diffusion samples
    "xmodel_iptm_std",          # cross-model spread of ipTM (scores-only, no coordinates)
    "intra_model_pose_std",     # intra-model pose spread (needs the coordinate tarball)
    "xmodel_pose_rmsd_median",  # cross-model pose disagreement (needs the coordinate tarball)
    CONF,                       # the deployed score itself, as a reference covariate
]


def target_level(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce to one delivered row per (system_id, method): the highest-`ranking_score` pose.

    This is the independence unit for every count and budget in this file (convention 1).
    """
    d = df.dropna(subset=[CONF, "correct"]).copy()
    d = d.sort_values(CONF, ascending=False)
    return d.groupby(["system_id", "method"], as_index=False).first()


def _risk_ci(loss: np.ndarray, delta: float = DELTA) -> tuple[float, float, float]:
    """Point estimate and exact Clopper-Pearson (1-delta) interval for a Bernoulli mean."""
    n = int(len(loss))
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    k = int(loss.sum())
    lo, hi = clopper_pearson(k, n, ci=1.0 - delta)
    return k / n, float(lo), float(hi)


def _rho(loss: np.ndarray, cov: np.ndarray) -> float:
    """corr(L, c) on the accepted subsample; NaN if the covariate is absent or constant."""
    ok = np.isfinite(cov) & np.isfinite(loss)
    if ok.sum() < 10:
        return float("nan")
    l, c = loss[ok], cov[ok]
    if l.std() == 0 or c.std() == 0:
        return float("nan")
    return float(np.corrcoef(l, c)[0, 1])


def _budget(sigma2: float, m: float, rho: float, delta: float = DELTA) -> float:
    """Target-1 label budget n_g ~ sigma^2 (1 - rho^2) log(1/delta) / m^2, or inf if infeasible.

    An order-of-magnitude planning number with the constant set to 1: it is the published PPI /
    CSA rate, not a finite-sample bound. The certification-vs-budget curves (d2_certify.py) are
    what actually establish achievability.
    """
    if not np.isfinite(m) or m <= 0 or not np.isfinite(sigma2):
        return float("inf")
    shrink = (1.0 - rho ** 2) if np.isfinite(rho) else 1.0
    return float(sigma2 * shrink * np.log(1.0 / delta) / (m ** 2))


def _cell(loss: np.ndarray, n_stratum: int, alpha: float, delta: float) -> dict:
    """One (stratum, tau) cell: realized risk, margin, and the two feasibility verdicts."""
    r, r_lo, r_hi = _risk_ci(loss, delta)
    n_acc = int(len(loss))
    sigma2 = float(r * (1 - r)) if np.isfinite(r) else float("nan")
    return {
        "n_accepted_targets": n_acc,
        "coverage": float(n_acc / n_stratum) if n_stratum else float("nan"),
        "R_Q": r, "R_Q_ci": [r_lo, r_hi], "sigma2": sigma2,
        "m": alpha - r if np.isfinite(r) else float("nan"),
        "m_ci": [alpha - r_hi, alpha - r_lo] if np.isfinite(r) else [float("nan")] * 2,
        # FEASIBLE: the rule truly holds alpha here (population claim, point estimate).
        "feasible": bool(np.isfinite(r) and r < alpha),
        # CERTIFIED: the labels in hand already prove it -- the exact CP upper bound clears alpha.
        # This is what a caller can actually stand behind, and it is what n_g must buy.
        "certified": bool(np.isfinite(r_hi) and r_hi <= alpha and n_acc >= MIN_ACCEPT),
    }


def _frontier(cells: dict, key: str) -> float:
    """Largest source coverage c at which `key` holds; 0.0 if it never holds."""
    ok = [c for c, cell in cells.items() if cell.get(key)]
    return float(max(ok)) if ok else 0.0


def run(alphas=ALPHA_GRID, delta: float = DELTA, seed: int = 20260715) -> dict:
    df = load_delivered()
    methods = methods_with_enough(df)
    tl = target_level(df)
    g_rng = rng(seed)

    out = {
        "alphas": list(alphas), "delta": delta, "score": CONF,
        "source_stratum": SOURCE_STRATUM,
        "coverage_grid": COVERAGE_GRID.tolist(),
        "min_accept": MIN_ACCEPT,
        "n_target_rows": int(len(tl)),
        "n_systems": int(tl.system_id.nunique()),
        "methods": methods,
        "axes": {},
    }

    for axis, col in AXES.items():
        axis_out = {"stratum_col": col, "models": {}}
        for m in methods:
            d = tl[(tl.method == m) & tl[col].notna()].copy()
            if len(d) == 0:
                continue
            d["loss"] = (1 - d["correct"].astype(int)).astype(float)

            # Source split by system: the calibration half fixes tau, the eval half reports risk
            # (convention 2), so the source cell is never scored on the points that set tau.
            src_all = d[d[col] == SOURCE_STRATUM]
            sys_ids = np.array(sorted(src_all.system_id.unique()))
            g_rng.shuffle(sys_ids)
            cal_ids = set(sys_ids[: len(sys_ids) // 2])
            src_cal = src_all[src_all.system_id.isin(cal_ids)]
            src_eval = src_all[~src_all.system_id.isin(cal_ids)]
            if len(src_cal) < MIN_ACCEPT:
                continue

            model_out = {
                "n_source_targets": int(len(src_all)),
                "n_source_cal": int(len(src_cal)),
                "n_source_eval": int(len(src_eval)),
                "alpha": {},
            }

            # Evaluation frames per stratum: the source uses its held-out half.
            frames = {int(g): (src_eval if g == SOURCE_STRATUM else d[d[col] == g])
                      for g in sorted(d[col].unique())}
            taus = {float(c): float(np.quantile(src_cal[CONF].to_numpy(), 1.0 - c))
                    for c in COVERAGE_GRID}

            for alpha in alphas:
                a_out = {"strata": {}}
                for g, dg in frames.items():
                    if len(dg) == 0:
                        continue
                    cells = {}
                    for c, tau in taus.items():
                        acc = dg[dg[CONF] >= tau]
                        cells[c] = _cell(acc["loss"].to_numpy(), len(dg), alpha, delta)
                        cells[c]["tau"] = tau
                    a_out["strata"][g] = {
                        "n_targets_stratum": int(len(dg)),
                        "cells": cells,
                        # The headline: the largest coverage at which the frozen score still
                        # holds alpha on this stratum. 0.0 = no operating point at ANY coverage.
                        "c_star_feasible": _frontier(cells, "feasible"),
                        "c_star_certified": _frontier(cells, "certified"),
                    }

                # The LTT operating point: one labelled point ON the frontier, for concreteness.
                tau_ltt = ltt_threshold(src_cal[CONF].to_numpy(), src_cal["correct"].to_numpy(),
                                        alpha=alpha, delta=delta)
                ltt = {"tau": tau_ltt}
                if tau_ltt is not None:
                    for g, dg in frames.items():
                        acc = dg[dg[CONF] >= tau_ltt]
                        cell = _cell(acc["loss"].to_numpy(), len(dg), alpha, delta)
                        loss = acc["loss"].to_numpy()
                        rhos = {cv: _rho(loss, acc[cv].to_numpy(dtype=float))
                                for cv in COVARIATES if cv in acc.columns}
                        cell["rho"] = rhos
                        finite = {k: v for k, v in rhos.items() if np.isfinite(v)}
                        best = max(finite, key=lambda k: abs(finite[k])) if finite else None
                        cell["rho_best_covariate"] = best
                        cell["rho_best"] = finite.get(best, float("nan")) if best else float("nan")
                        cell["n_labels_needed_label_only"] = _budget(
                            cell["sigma2"], cell["m"], float("nan"), delta)
                        cell["n_labels_needed_ppi"] = _budget(
                            cell["sigma2"], cell["m"], cell["rho_best"], delta)
                        cell["budget_reachable"] = bool(
                            cell["feasible"]
                            and np.isfinite(cell["n_labels_needed_label_only"])
                            and cell["n_labels_needed_label_only"] <= cell["n_accepted_targets"])

                        # Explanatory decomposition (convention 3), source vs this stratum.
                        if g != SOURCE_STRATUM:
                            dec = shift_decomposition(
                                src_cal[CONF].to_numpy(), src_cal["correct"].to_numpy().astype(int),
                                dg[CONF].to_numpy(), dg["correct"].to_numpy().astype(int),
                                tau=tau_ltt, n_bins=5, delta=delta, n_boot=N_BOOT, seed=0)
                            cell["explanatory"] = {
                                "gap_total": dec["gap_total"],
                                "gap_concept": dec["gap_concept"],
                                "gap_covariate": dec["gap_covariate"],
                                "gap_concept_ci": dec["ci"],
                                "concept_nonvacuous": dec["concept_nonvacuous"],
                            }
                        ltt[str(g)] = cell
                else:
                    ltt["note"] = ("fixed-sequence LTT rejected its first hypothesis (the smallest "
                                   "accept set) and returned no threshold; the frontier above is "
                                   "unaffected because it does not depend on this calibrator")
                a_out["ltt_operating_point"] = ltt
                model_out["alpha"][str(alpha)] = a_out
            axis_out["models"][m] = model_out
        out["axes"][axis] = axis_out

    # --- the go/no-go gate --------------------------------------------------------------
    gate = {}
    for alpha in alphas:
        feas = infeas = certd = 0
        for axis, ao in out["axes"].items():
            for m, mo in ao["models"].items():
                for g, s in mo["alpha"][str(alpha)]["strata"].items():
                    if s["c_star_feasible"] > 0:
                        feas += 1
                    else:
                        infeas += 1
                    if s["c_star_certified"] > 0:
                        certd += 1
        gate[str(alpha)] = {
            "n_strata_with_an_operating_point": feas,
            "n_strata_with_none_at_any_coverage": infeas,
            "n_strata_certified_from_labels_in_hand": certd,
            # The map is informative only if BOTH regimes exist: something to certify, and an
            # impossibility regime to show.
            "passes": bool(feas >= 2 and infeas >= 1 and certd >= 2),
        }
    out["gate"] = gate
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--delta", type=float, default=DELTA)
    args = ap.parse_args()
    res = run(delta=args.delta)
    save_json(res, RESDIR / "d2_feasibility_map.json")

    print(f"D2 feasibility map  (score={CONF}, delta={args.delta}, "
          f"source=S{SOURCE_STRATUM} calibration half)")
    print(f"target-level rows: {res['n_target_rows']} over {res['n_systems']} systems")
    print("c* = largest source coverage at which the frozen score still holds alpha "
          "(0.00 = no operating point at any coverage)\n")
    for axis, ao in res["axes"].items():
        for alpha in res["alphas"]:
            print(f"=== {axis}-novelty axis, alpha={alpha} ===")
            print(f"  {'model':>9} " + " ".join(f"{'S'+str(g):>13}" for g in range(5)))
            for m, mo in ao["models"].items():
                strata = mo["alpha"][str(alpha)]["strata"]
                cells = []
                for g in range(5):
                    s = strata.get(g)
                    cells.append("  --  " if s is None else
                                 f"{s['c_star_feasible']:.2f}/{s['c_star_certified']:.2f}"
                                 f"({s['n_targets_stratum']})")
                print(f"  {m:>9} " + " ".join(f"{c:>13}" for c in cells))
            g = res["gate"][str(alpha)]
            print(f"  gate: {g['n_strata_with_an_operating_point']} strata with an operating "
                  f"point / {g['n_strata_with_none_at_any_coverage']} with none at any coverage "
                  f"/ {g['n_strata_certified_from_labels_in_hand']} certified -> "
                  f"{'PASS' if g['passes'] else 'FAIL'}\n")
    print("(cells show c*_feasible / c*_certified (n_targets))")
    print(f"\nwrote {RESDIR / 'd2_feasibility_map.json'}")


if __name__ == "__main__":
    main()
