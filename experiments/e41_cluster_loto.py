"""E41 -- leakage-free gate grouped on SEQUENCE CLUSTER, not system_id (reviewer R3.5).

The nested-LOTO headline (E34) groups on system_id, which does not stop a homologous
receptor from appearing in both calibration and test. Runs N' Poses ships a sequence
`cluster` label (1005 clusters). This re-runs the exact nested protocol grouping on
cluster instead, so no homolog straddles the split, and reports whether the 73% moves.

Output: results/e41_cluster_loto.json
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from experiments._common import (
    CONF,
    DELTA,
    RESDIR,
    ROOT,
    load_delivered,
    methods_with_enough,
    save_json,
)
from foldgate.conformal import ltt_threshold
from foldgate.conformal.risk import hb_upper_bound
from foldgate.scores.combiner import DEFAULT_FEATURES, ScoreCombiner
from foldgate.selective.metrics import clopper_pearson

N_FOLDS = 5
ALPHA = 0.20
CAL_FRAC = 0.5


def _merge_cluster(df: pd.DataFrame) -> pd.DataFrame:
    ann = pd.read_csv(ROOT / "data" / "raw" / "annotations.csv")[["system_id", "cluster"]]
    ann = ann.drop_duplicates("system_id")
    return df.merge(ann, on="system_id", how="left")


def _pool_gate_grouped(df, m, group_col, alpha, delta):
    sub = df[df.method == m].dropna(subset=[CONF, group_col]).reset_index(drop=True)
    s = sub[CONF].to_numpy()
    y = sub["correct"].to_numpy().astype(int)
    groups = sub[group_col].to_numpy()
    n = len(sub)
    n_splits = min(N_FOLDS, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)

    acc = {"native": {"a": 0, "e": 0, "holds": []},
           "combined": {"a": 0, "e": 0, "holds": []}}
    for train_idx, test_idx in gkf.split(s, y, groups):
        g_train = groups[train_idx]
        gss = GroupShuffleSplit(n_splits=1, test_size=CAL_FRAC, random_state=0)
        (fit_local, cal_local), = gss.split(train_idx, groups=g_train)
        fit_idx, cal_idx = train_idx[fit_local], train_idx[cal_local]
        comb = ScoreCombiner(features=DEFAULT_FEATURES).fit(sub.iloc[fit_idx], y[fit_idx])
        sc_cal, sc_test = comb.predict(sub.iloc[cal_idx]), comb.predict(sub.iloc[test_idx])
        for name, (score_cal, score_test) in {"native": (s[cal_idx], s[test_idx]),
                                              "combined": (sc_cal, sc_test)}.items():
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
            acc[name]["holds"].append(bool((ne / na) <= alpha))

    out = {"n": n, "n_groups": int(len(np.unique(groups)))}
    for name in ("native", "combined"):
        a, e = acc[name]["a"], acc[name]["e"]
        if a == 0:
            out[name] = {"coverage": 0.0, "n_accept": 0, "realized_risk": None,
                         "certified_ub_hb": None, "certified": False,
                         "folds_holding": [0, len(acc[name]["holds"])]}
            continue
        risk = e / a
        ub = hb_upper_bound(risk, a, delta)
        cp_lo, cp_hi = clopper_pearson(e, a, ci=0.90)
        out[name] = {
            "coverage": round(a / n, 4), "n_accept": a, "realized_risk": round(risk, 4),
            "risk_cp_upper90": round(cp_hi, 4), "certified_ub_hb": round(ub, 4),
            "certified": bool(ub <= alpha),
            "folds_holding": [int(sum(acc[name]["holds"])), len(acc[name]["holds"])],
        }
    return out


def run() -> dict:
    df = _merge_cluster(load_delivered())
    methods = methods_with_enough(df)
    out = {"alpha": ALPHA, "delta": DELTA,
           "grouping": "sequence cluster (RNP `cluster`, 1005 clusters)",
           "per_model": {}}
    for m in methods:
        out["per_model"][m] = {
            "cluster_grouped": _pool_gate_grouped(df, m, "cluster", ALPHA, DELTA),
            "system_grouped": _pool_gate_grouped(df, m, "system_id", ALPHA, DELTA),
        }
    return out


def main() -> None:
    res = run()
    save_json(res, RESDIR / "e41_cluster_loto.json")
    print("E41 -- cluster-grouped vs system-grouped nested LOTO (alpha=0.20)\n")
    print(f"{'model':>9} {'score':>8} | {'cluster cov/risk/HBub/folds':>34} | {'system cov/risk/HBub/folds':>34}")
    for m, r in res["per_model"].items():
        for score in ("native", "combined"):
            c, s = r["cluster_grouped"][score], r["system_grouped"][score]
            def f(x):
                if x["n_accept"] == 0:
                    return "abstains"
                return f"{x['coverage']:.2f}/{x['realized_risk']:.3f}/{x['certified_ub_hb']:.3f}/{x['folds_holding'][0]}-{x['folds_holding'][1]}"
            print(f"{m:>9} {score:>8} | {f(c):>34} | {f(s):>34}")


if __name__ == "__main__":
    main()
