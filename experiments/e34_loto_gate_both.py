"""E34 -- leakage-free (target-grouped) LOTO gate for BOTH native and combined scores.

Reviewer R3.1: the paper's headline coverage pair (71% combined vs 22% native at
alpha=0.20) comes from a pose-level random split (e4) that leaks 95% of test targets
into calibration; the only leakage-free number in the paper (52%) is the NATIVE gate
under LOTO (e13 calibrates its gate on ranking_score). So there is no leakage-free
COMBINED-gate coverage anywhere, and the headline pair is not matched.

This script fixes that: a fully target-grouped nested protocol that certifies BOTH
gates leakage-free, so the paper can headline a matched pair.

Protocol (per outer fold, GroupKFold on system_id):
  - test  = the held-out target fold.
  - within the training targets, a grouped 50/50 split (GroupShuffleSplit on
    system_id) into a combiner-FIT subset and a calibration subset, disjoint at the
    target level.
  - fit ScoreCombiner on the FIT subset; its scores on the CAL subset are therefore
    out-of-sample; calibrate tau (combined) by LTT on (combined_cal, y_cal).
  - native tau by LTT on (ranking_cal, y_cal).
  - evaluate both gates on the held-out test fold (combiner predicts test OOF).
Every target is entirely in fit, cal, or test; nothing leaks across roles.

We pool accepted/errors across folds and report, per model and per alpha in {0.20,0.10}:
  coverage, accepted n, realized risk, exact Clopper-Pearson upper bound on the risk,
  the (1-delta) Hoeffding-Bentkus certified upper bound, per-fold "holds" count, and
  the near-zero-coverage flag (Chai native). This gives R3.1's matched pair and
  R4.11's certified-bound + pass/fail column with an explicit accepted n.

Output: results/e34_loto_gate_both.json
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from experiments._common import (
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
ALPHAS = [0.20, 0.10]
CAL_FRAC = 0.5          # grouped fit/cal split of the training targets
MIN_COVERAGE_VACUOUS = 0.05


def _pool_gate(df, m, alpha, delta):
    """Nested target-grouped LOTO for native and combined gates at one alpha."""
    sub = df[df.method == m].dropna(subset=[CONF, "system_id"]).reset_index(drop=True)
    s = sub[CONF].to_numpy()
    y = sub["correct"].to_numpy().astype(int)
    groups = sub["system_id"].to_numpy()
    n = len(sub)
    n_splits = min(N_FOLDS, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)

    acc = {"native": {"a": 0, "e": 0, "holds": [], "folds": 0},
           "combined": {"a": 0, "e": 0, "holds": [], "folds": 0}}
    for train_idx, test_idx in gkf.split(s, y, groups):
        g_train = groups[train_idx]
        gss = GroupShuffleSplit(n_splits=1, test_size=CAL_FRAC, random_state=0)
        (fit_local, cal_local), = gss.split(train_idx, groups=g_train)
        fit_idx = train_idx[fit_local]
        cal_idx = train_idx[cal_local]

        comb = ScoreCombiner(features=DEFAULT_FEATURES).fit(sub.iloc[fit_idx], y[fit_idx])
        sc_cal = comb.predict(sub.iloc[cal_idx])
        sc_test = comb.predict(sub.iloc[test_idx])

        gates = {
            "native": (s[cal_idx], s[test_idx]),
            "combined": (sc_cal, sc_test),
        }
        for name, (score_cal, score_test) in gates.items():
            tau = ltt_threshold(score_cal, y[cal_idx], alpha=alpha, delta=delta)
            if tau is None:
                continue
            a = score_test >= tau
            na = int(a.sum())
            if na == 0:
                continue
            ne = int((1 - y[test_idx][a]).sum())
            acc[name]["a"] += na
            acc[name]["e"] += ne
            acc[name]["folds"] += 1
            acc[name]["holds"].append(bool((ne / na) <= alpha))

    out = {"n": n, "n_targets": int(len(np.unique(groups)))}
    for name in ("native", "combined"):
        a, e = acc[name]["a"], acc[name]["e"]
        if a == 0:
            out[name] = {"coverage": 0.0, "n_accept": 0, "realized_risk": None,
                         "risk_cp_upper90": None, "certified_ub_hb": None,
                         "certified": False, "folds_holding": [0, acc[name]["folds"]],
                         "vacuous": True}
            continue
        risk = e / a
        cp_lo, cp_hi = clopper_pearson(e, a, ci=0.90)
        ub = hb_upper_bound(risk, a, delta)
        cov = a / n
        out[name] = {
            "coverage": round(cov, 4),
            "n_accept": a,
            "realized_risk": round(risk, 4),
            "realized_risk_cp90": [round(cp_lo, 4), round(cp_hi, 4)],
            "risk_cp_upper90": round(cp_hi, 4),
            "certified_ub_hb": round(ub, 4),
            # A certificate requires the (1-delta) UPPER bound to sit at/below alpha, not
            # merely the point estimate. This is the paper's own standard (B4).
            "certified": bool(ub <= alpha),
            "folds_holding": [int(sum(acc[name]["holds"])), len(acc[name]["holds"])],
            "vacuous": bool(cov < MIN_COVERAGE_VACUOUS),
        }
    return out


def _certified_native_coverage(df, m, alpha, delta):
    """Largest leakage-free LOTO coverage at which the NATIVE gate's pooled HB upper bound
    is <= alpha (a properly certified native comparator, B4). Sweeps a shrinking coverage
    cap by testing progressively stricter fixed-sequence thresholds pooled across folds."""
    sub = df[df.method == m].dropna(subset=[CONF, "system_id"]).reset_index(drop=True)
    s = sub[CONF].to_numpy(); y = sub["correct"].to_numpy().astype(int)
    groups = sub["system_id"].to_numpy(); n = len(sub)
    n_splits = min(N_FOLDS, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)
    # For each candidate target coverage cap, pool accepts across folds using the per-fold
    # top-cap quantile threshold of the calibration scores, then check the pooled HB bound.
    best = {"coverage": 0.0, "n_accept": 0, "realized_risk": None, "certified_ub_hb": None}
    for cap in np.arange(0.05, 1.0 + 1e-9, 0.05):
        a = e = 0
        for tr, te in gkf.split(s, y, groups):
            tau = float(np.quantile(s[tr], 1.0 - cap))
            acc = s[te] >= tau
            na = int(acc.sum())
            if na:
                a += na; e += int((1 - y[te][acc]).sum())
        if a < 20:
            continue
        ub = hb_upper_bound(e / a, a, delta)
        if ub <= alpha and a / n > best["coverage"]:
            best = {"coverage": round(a / n, 4), "n_accept": a,
                    "realized_risk": round(e / a, 4), "certified_ub_hb": round(ub, 4)}
    return best


def run() -> dict:
    df = load_delivered()
    methods = methods_with_enough(df)
    out = {"protocol": "nested target-grouped LOTO (GroupKFold outer, grouped fit/cal inner)",
           "delta": DELTA, "cal_frac": CAL_FRAC, "per_alpha": {},
           "certified_native_coverage": {}}
    for alpha in ALPHAS:
        out["per_alpha"][str(alpha)] = {m: _pool_gate(df, m, alpha, DELTA) for m in methods}
    # Properly certified native comparator at alpha=0.20 (B4).
    out["certified_native_coverage"]["0.2"] = {
        m: _certified_native_coverage(df, m, 0.20, DELTA) for m in methods}
    return out


def main() -> None:
    res = run()
    save_json(res, RESDIR / "e34_loto_gate_both.json")
    print("E34 -- leakage-free nested-LOTO gate, native vs combined (matched pair)\n")
    for alpha, per in res["per_alpha"].items():
        print(f"alpha = {alpha}")
        for m, r in per.items():
            nv, cb = r["native"], r["combined"]
            def fmt(g):
                if g["n_accept"] == 0:
                    return "cov 0.00 (abstains)"
                return (f"cov {g['coverage']:.2f} (n={g['n_accept']}) risk {g['realized_risk']:.3f} "
                        f"CPub {g['risk_cp_upper90']:.3f} HBub {g['certified_ub_hb']:.3f} "
                        f"holds {g['folds_holding'][0]}/{g['folds_holding'][1]}")
            print(f"  [{m:>9}] native  : {fmt(nv)}")
            print(f"  [{' ':>9}] combined: {fmt(cb)}")
        print()


if __name__ == "__main__":
    main()
