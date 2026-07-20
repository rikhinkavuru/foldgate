"""E27 -- PoseBusters joint label: strengthen the accept set and re-certify Y = correct AND pb_valid.

Reviewer R2.3 / addition III.8. The paper's label is Y = 1 iff ligand-RMSD <= 2 A.
PoseBusters practice certifies a pose only when it is BOTH close (RMSD <= 2 A) AND
physically valid (~30 RDKit checks). This script asks two questions per governed model:

  (2) Does the RMSD-only combined-score gate already deliver a physically valid accept
      set "for free"? We calibrate the gate on the RMSD label `correct`, then compare the
      PoseBusters-validity rate INSIDE the accepted set vs the rejected set.

  (3) What does it cost to certify the STRONGER joint label J = (correct AND pb_valid)
      directly? Same nested target-grouped LOTO protocol, but tau is calibrated on J.
      We report the pooled coverage, realized joint-error risk (1 - mean J among
      accepted), its Clopper-Pearson upper bound, the (1 - delta) Hoeffding-Bentkus
      certified upper bound, and folds_holding, then compare the coverage to the
      RMSD-only gate at the same alpha.

Protocol (per outer fold, GroupKFold on system_id; identical to E34):
  - test  = the held-out target fold.
  - within the training targets, a grouped 50/50 GroupShuffleSplit into a combiner-FIT
    subset and a calibration subset, disjoint at the target level.
  - fit ScoreCombiner (P(correct)) on the FIT subset; its scores on CAL/TEST are OOF.
  - RMSD gate:  tau_c = LTT(combined_cal, correct_cal, alpha, delta).
  - joint gate: tau_j = LTT(combined_cal, J_cal,       alpha, delta).
  The combined SCORE is identical for both gates; only the calibration label (hence tau)
  differs, so the coverage comparison is apples-to-apples.
Every target is entirely in fit, cal, or test; nothing leaks across roles.

NaN pb_valid is treated as physically invalid (pb=False, so J=0); the count of NaN rows
is reported per model.

Output: results/e27_pb_joint_label.json
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from experiments._common import (
    ALPHA,
    CONF,
    DELTA,
    RESDIR,
    load_delivered,
    methods_with_enough,
    save_json,
)
from foldgate.conformal import ltt_threshold
from foldgate.conformal.risk import hb_upper_bound
from foldgate.scores.combiner import DEFAULT_FEATURES, ScoreCombiner
from foldgate.selective.metrics import clopper_pearson

N_FOLDS = 5
CAL_FRAC = 0.5  # grouped fit/cal split of the training targets


def _pb_bool(series) -> np.ndarray:
    """PoseBusters validity as bool; NaN -> False (invalid)."""
    v = series.to_numpy(dtype=float)
    return (v == 1.0) & ~np.isnan(v)


def _rate(k: int, n: int) -> dict:
    if n == 0:
        return {"n": 0, "k": 0, "rate": None, "cp90": [None, None]}
    lo, hi = clopper_pearson(k, n, ci=0.90)
    return {"n": int(n), "k": int(k), "rate": round(k / n, 4),
            "cp90": [round(lo, 4), round(hi, 4)]}


def _model_result(df, m, alpha, delta) -> dict:
    raw = df[df.method == m]
    n_nan_pb = int(raw[CONF].notna().mul(raw["pb_valid"].isna()).sum()) if "pb_valid" in raw else 0
    # total NaN pb over the full model slice (before any CONF/system_id drop)
    total_nan_pb = int(raw["pb_valid"].isna().sum())

    sub = raw.dropna(subset=[CONF, "system_id"]).reset_index(drop=True)
    s = sub[CONF].to_numpy()
    y = sub["correct"].to_numpy().astype(int)
    pb = _pb_bool(sub["pb_valid"])
    J = (y == 1) & pb                       # joint label
    groups = sub["system_id"].to_numpy()
    n = len(sub)

    # ---- (1) base rates ----
    base = {
        "n": n,
        "n_nan_pb": total_nan_pb,
        "pb_valid_rate": round(float(pb.mean()), 4),
        "correct_rate": round(float(y.mean()), 4),
        "joint_rate": round(float(J.mean()), 4),
    }

    # ---- nested target-grouped LOTO, shared folds for both gates ----
    n_splits = min(N_FOLDS, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)

    # (2) RMSD-only gate: PB-validity inside accepted vs rejected (pooled)
    acc_pb_k = acc_pb_n = rej_pb_k = rej_pb_n = 0
    rmsd_cov_a = 0  # accepted count for RMSD gate (coverage comparison)

    # (3) joint-label gate
    j_a = j_err = 0                      # accepted, joint errors (accepted with J==0)
    j_holds = []
    j_folds = 0

    for train_idx, test_idx in gkf.split(s, y, groups):
        g_train = groups[train_idx]
        gss = GroupShuffleSplit(n_splits=1, test_size=CAL_FRAC, random_state=0)
        (fit_local, cal_local), = gss.split(train_idx, groups=g_train)
        fit_idx = train_idx[fit_local]
        cal_idx = train_idx[cal_local]

        comb = ScoreCombiner(features=DEFAULT_FEATURES).fit(sub.iloc[fit_idx], y[fit_idx])
        sc_cal = comb.predict(sub.iloc[cal_idx])
        sc_test = comb.predict(sub.iloc[test_idx])

        y_test = y[test_idx]
        pb_test = pb[test_idx]
        J_test = J[test_idx]

        # (2) RMSD gate calibrated on `correct`
        tau_c = ltt_threshold(sc_cal, y[cal_idx], alpha=alpha, delta=delta)
        if tau_c is not None:
            a_c = sc_test >= tau_c
            if a_c.sum() > 0:
                rmsd_cov_a += int(a_c.sum())
                acc_pb_k += int(pb_test[a_c].sum())
                acc_pb_n += int(a_c.sum())
                rej = ~a_c
                rej_pb_k += int(pb_test[rej].sum())
                rej_pb_n += int(rej.sum())

        # (3) joint gate calibrated on J
        tau_j = ltt_threshold(sc_cal, J[cal_idx], alpha=alpha, delta=delta)
        if tau_j is not None:
            a_j = sc_test >= tau_j
            na = int(a_j.sum())
            if na > 0:
                ne = int((~J_test[a_j]).sum())   # accepted poses failing the joint label
                j_a += na
                j_err += ne
                j_folds += 1
                j_holds.append(bool((ne / na) <= alpha))

    # ---- assemble (2) ----
    gate_rmsd = {
        "accepted": _rate(acc_pb_k, acc_pb_n),
        "rejected": _rate(rej_pb_k, rej_pb_n),
        "coverage": round(rmsd_cov_a / n, 4) if n else None,
    }

    # ---- assemble (3) ----
    if j_a == 0:
        gate_joint = {"coverage": 0.0, "n_accept": 0, "joint_error_risk": None,
                      "risk_cp90": [None, None], "certified_ub_hb": None,
                      "folds_holding": [0, j_folds], "vacuous": True}
    else:
        risk = j_err / j_a
        cp_lo, cp_hi = clopper_pearson(j_err, j_a, ci=0.90)
        ub = hb_upper_bound(risk, j_a, delta)
        gate_joint = {
            "coverage": round(j_a / n, 4),
            "n_accept": int(j_a),
            "joint_error_risk": round(risk, 4),
            "risk_cp90": [round(cp_lo, 4), round(cp_hi, 4)],
            "risk_cp_upper90": round(cp_hi, 4),
            "certified_ub_hb": round(ub, 4),
            "certified_pass": bool(ub <= alpha or risk <= alpha),
            "folds_holding": [int(sum(j_holds)), len(j_holds)],
            "vacuous": bool((j_a / n) < 0.05),
        }

    cov_cost = None
    if gate_rmsd["coverage"] is not None and gate_joint["coverage"] is not None:
        cov_cost = round(gate_rmsd["coverage"] - gate_joint["coverage"], 4)

    return {
        "base_rates": base,
        "gate_rmsd_label": gate_rmsd,       # (2) gate on correct; PB-validity by decision
        "gate_joint_label": gate_joint,     # (3) gate on J directly
        "coverage_cost_joint_vs_rmsd": cov_cost,
    }


def run() -> dict:
    df = load_delivered()
    methods = methods_with_enough(df)
    out = {
        "alpha": ALPHA,
        "delta": DELTA,
        "n_folds": N_FOLDS,
        "cal_frac": CAL_FRAC,
        "protocol": "nested target-grouped LOTO (GroupKFold outer, grouped 50/50 fit/cal inner); "
                    "combined score = ScoreCombiner P(correct); NaN pb_valid = invalid",
        "score": "combined",
        "per_model": {m: _model_result(df, m, ALPHA, DELTA) for m in methods},
    }
    return out


def main() -> None:
    res = run()
    save_json(res, RESDIR / "e27_pb_joint_label.json")
    a = res["alpha"]
    print(f"E27 -- PoseBusters joint label (alpha={a}, delta={res['delta']})\n")
    hdr = (f"{'model':>9} | {'n':>5} {'pb_rate':>7} {'joint':>6} | "
           f"{'acc_pb':>15} {'rej_pb':>15} | "
           f"{'J_cov':>6} {'RMSD_cov':>8} {'Jrisk':>6} {'HBub':>6} {'holds':>6}")
    print(hdr)
    print("-" * len(hdr))
    for m, r in res["per_model"].items():
        b = r["base_rates"]
        gr = r["gate_rmsd_label"]
        gj = r["gate_joint_label"]

        def pr(d):
            if d["rate"] is None:
                return "     -         "
            return f"{d['rate']:.3f}[{d['cp90'][0]:.2f},{d['cp90'][1]:.2f}]"

        jcov = gj["coverage"]
        rcov = gr["coverage"]
        jrisk = gj["joint_error_risk"]
        hb = gj["certified_ub_hb"]
        holds = gj["folds_holding"]
        print(f"{m:>9} | {b['n']:>5} {b['pb_valid_rate']:>7.3f} {b['joint_rate']:>6.3f} | "
              f"{pr(gr['accepted']):>15} {pr(gr['rejected']):>15} | "
              f"{jcov:>6.3f} {rcov:>8.3f} "
              f"{(jrisk if jrisk is not None else float('nan')):>6.3f} "
              f"{(hb if hb is not None else float('nan')):>6.3f} "
              f"{holds[0]}/{holds[1]:>3}")

    print("\nColumns: acc_pb/rej_pb = PoseBusters-validity rate [CP90] inside the RMSD-gate "
          "accepted / rejected set; J_cov = joint-label gate coverage; RMSD_cov = RMSD-only "
          "gate coverage; Jrisk = realized joint-error risk among accepted; HBub = HB certified "
          "upper bound; holds = folds with realized joint risk <= alpha.")


if __name__ == "__main__":
    main()
