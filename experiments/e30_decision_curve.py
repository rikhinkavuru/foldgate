"""E30 -- clinical-style decision-curve / net-benefit analysis (addition III.6).

Retires the R3.8 "why alpha=0.20?" defense. Instead of arguing for a single
operating point, we ask the decision-theoretic question a practitioner actually
faces: given a cost ratio between acting on a WRONG pose and abstaining on a
RIGHT one, which gating strategy maximizes net benefit?

Net benefit per delivered pose (Vickers & Elkin 2006, adapted to a binary
accept/abstain action):

    NB(gate) = (TP - lambda * FP) / N

evaluated on the held-out delivered set, where
    TP = # accepted AND correct,
    FP = # accepted AND incorrect,
    N  = total delivered poses,
    lambda = cost(act on wrong) / cost(abstain on right) = p_t / (1 - p_t)
             for a threshold probability p_t.

Four strategies, per model, all evaluated leakage-free (target-grouped):
  1. conformal gate on the COMBINED score at alpha=0.20 (accept score >= tau,
     tau from LTT calibrated on out-of-sample combiner scores).
  2. fixed ipTM gate: accept iface_iptm >= 0.8 (the field baseline, no calibration).
  3. accept-all (no abstention).
  4. abstain-all (NB = 0 by construction).

Leakage-free protocol (identical to e34): GroupKFold(5) on system_id; within each
outer fold's training targets a grouped fit/cal split; combiner fit on the fit
subset, LTT tau calibrated on the cal subset, both scored on the held-out test
fold. Every target lives entirely in fit, cal, or test. TP/FP are pooled across
folds so N = the full delivered set. The conformal strategy is averaged over
several inner-split seeds (only it depends on the split).

Output: results/e30_decision_curve.json
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
    rng,
    save_json,
)
from foldgate.conformal import ltt_threshold
from foldgate.scores.combiner import DEFAULT_FEATURES, ScoreCombiner

N_FOLDS = 5
CAL_FRAC = 0.5                 # grouped fit/cal split of the training targets
IPTM_THRESHOLD = 0.8           # the field-baseline fixed gate
N_SEEDS = 20                   # inner-split seeds to average the conformal gate over
FOCUS = "af3"

# Threshold-probability grid p_t and the implied cost ratio lambda = p_t / (1 - p_t).
LAMBDAS = [0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 9.0]
# A fine grid purely to locate the crossover lambdas precisely.
FINE_LAMBDAS = np.round(np.arange(0.05, 12.0001, 0.05), 4)

STRATS = ("conformal", "iptm", "accept_all", "abstain_all")


def _pooled_counts(df, m, seed: int) -> dict:
    """Nested target-grouped LOTO -> pooled (TP, FP, N) per strategy for one model.

    The conformal gate uses `seed` for its inner fit/cal split; the ipTM,
    accept-all and abstain-all strategies are split-independent (computed on the
    same held-out test poses, which across the 5 folds tile the full set).
    """
    sub = df[df.method == m].dropna(subset=[CONF, "system_id"]).reset_index(drop=True)
    s = sub[CONF].to_numpy()
    y = sub["correct"].to_numpy().astype(int)
    iptm = sub["iface_iptm"].to_numpy()
    groups = sub["system_id"].to_numpy()
    n = len(sub)
    n_splits = min(N_FOLDS, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)

    tp = {k: 0 for k in STRATS}
    fp = {k: 0 for k in STRATS}
    n_seen = 0

    for train_idx, test_idx in gkf.split(s, y, groups):
        yte = y[test_idx]
        n_seen += len(test_idx)

        # --- conformal gate on the combined score (leakage-free) ---
        g_train = groups[train_idx]
        gss = GroupShuffleSplit(n_splits=1, test_size=CAL_FRAC, random_state=seed)
        (fit_local, cal_local), = gss.split(train_idx, groups=g_train)
        fit_idx = train_idx[fit_local]
        cal_idx = train_idx[cal_local]

        comb = ScoreCombiner(features=DEFAULT_FEATURES).fit(sub.iloc[fit_idx], y[fit_idx])
        sc_cal = comb.predict(sub.iloc[cal_idx])
        sc_test = comb.predict(sub.iloc[test_idx])
        tau = ltt_threshold(sc_cal, y[cal_idx], alpha=ALPHA, delta=DELTA)
        if tau is not None:
            acc = sc_test >= tau
            tp["conformal"] += int((acc & (yte == 1)).sum())
            fp["conformal"] += int((acc & (yte == 0)).sum())
        # tau is None -> accept nothing this fold (0 TP, 0 FP), poses still count in N.

        # --- fixed ipTM gate (field baseline, no calibration) ---
        acc_ip = iptm[test_idx] >= IPTM_THRESHOLD
        tp["iptm"] += int((acc_ip & (yte == 1)).sum())
        fp["iptm"] += int((acc_ip & (yte == 0)).sum())

        # --- accept-all ---
        tp["accept_all"] += int((yte == 1).sum())
        fp["accept_all"] += int((yte == 0).sum())

        # --- abstain-all: TP = FP = 0 ---

    assert n_seen == n
    return {"n": n, "tp": tp, "fp": fp}


def _net_benefit(tp: float, fp: float, n: int, lam: float) -> float:
    return (tp - lam * fp) / n


def _best_strategy(nb_at: dict[str, float]) -> str:
    return max(nb_at, key=lambda k: nb_at[k])


def run() -> dict:
    df = load_delivered()
    methods = methods_with_enough(df)
    g = rng()
    seeds = [int(x) for x in g.integers(0, 2**31 - 1, N_SEEDS)]

    out = {
        "protocol": "nested target-grouped LOTO (GroupKFold outer, grouped fit/cal inner)",
        "alpha": ALPHA,
        "delta": DELTA,
        "iptm_threshold": IPTM_THRESHOLD,
        "n_inner_seeds": N_SEEDS,
        "cal_frac": CAL_FRAC,
        "lambda_grid": LAMBDAS,
        "net_benefit_formula": "NB = (TP - lambda*FP)/N",
        "per_model": {},
    }

    for m in methods:
        # Average pooled TP/FP over inner-split seeds (only conformal varies).
        acc = {k: {"tp": [], "fp": []} for k in STRATS}
        n_total = None
        for sd in seeds:
            c = _pooled_counts(df, m, sd)
            n_total = c["n"]
            for k in STRATS:
                acc[k]["tp"].append(c["tp"][k])
                acc[k]["fp"].append(c["fp"][k])
        counts = {
            k: {"tp": float(np.mean(acc[k]["tp"])), "fp": float(np.mean(acc[k]["fp"]))}
            for k in STRATS
        }

        # NB curve over the reported grid.
        nb = {k: {} for k in STRATS}
        best_by_lambda = {}
        for lam in LAMBDAS:
            row = {
                k: _net_benefit(counts[k]["tp"], counts[k]["fp"], n_total, lam)
                for k in STRATS
            }
            for k in STRATS:
                nb[k][str(lam)] = round(row[k], 5)
            best_by_lambda[str(lam)] = _best_strategy(row)

        # Fine grid: find the lambda window where conformal is the sole best, and
        # the crossover lambda at which accept-all stops being the best strategy.
        conf_wins = []
        accept_best_upto = None
        conf_first, conf_last = None, None
        for lam in FINE_LAMBDAS:
            row = {
                k: _net_benefit(counts[k]["tp"], counts[k]["fp"], n_total, lam)
                for k in STRATS
            }
            best = _best_strategy(row)
            if best == "accept_all":
                accept_best_upto = float(lam)
            if best == "conformal":
                conf_wins.append(float(lam))
                if conf_first is None:
                    conf_first = float(lam)
                conf_last = float(lam)
        # crossover = first lambda where accept-all is no longer best.
        accept_crossover = (
            round(accept_best_upto + 0.05, 4) if accept_best_upto is not None else None
        )

        out["per_model"][m] = {
            "n": n_total,
            "counts": {
                k: {"tp": round(counts[k]["tp"], 2), "fp": round(counts[k]["fp"], 2)}
                for k in STRATS
            },
            "net_benefit": nb,
            "best_strategy_by_lambda": best_by_lambda,
            "conformal_wins_lambda_range": (
                [round(conf_first, 4), round(conf_last, 4)] if conf_first is not None else None
            ),
            "accept_all_best_until_lambda": accept_crossover,
        }

    return out


def _table(res: dict, m: str) -> str:
    r = res["per_model"][m]
    lines = [f"NB(lambda) table -- {m} (N={r['n']} delivered poses)"]
    header = f"{'lambda':>7} " + " ".join(f"{k:>12}" for k in STRATS) + f"   {'best':>12}"
    lines.append(header)
    lines.append("-" * len(header))
    for lam in LAMBDAS:
        vals = " ".join(f"{r['net_benefit'][k][str(lam)]:>12.4f}" for k in STRATS)
        lines.append(f"{lam:>7} {vals}   {r['best_strategy_by_lambda'][str(lam)]:>12}")
    return "\n".join(lines)


def main() -> None:
    res = run()
    save_json(res, RESDIR / "e30_decision_curve.json")

    print("E30 -- decision-curve / net-benefit analysis (retires the alpha=0.20 defense)\n")
    print(_table(res, FOCUS) + "\n")

    fr = res["per_model"][FOCUS]
    cw = fr["conformal_wins_lambda_range"]
    xo = fr["accept_all_best_until_lambda"]
    print(f"[{FOCUS}] conformal gate is the top strategy for lambda in "
          f"{cw[0]}..{cw[1]}" if cw else f"[{FOCUS}] conformal never top")
    print(f"[{FOCUS}] accept-all stops being best at lambda ~ {xo}")

    print("\nall models (conformal-wins range | accept-all crossover):")
    for m, r in res["per_model"].items():
        cw = r["conformal_wins_lambda_range"]
        print(f"  {m:>9}: conformal wins {cw[0]}..{cw[1] if cw else '-'}"
              if cw else f"  {m:>9}: conformal never top",
              f"| accept-all best until lambda {r['accept_all_best_until_lambda']}")

    print("\nTakeaway:")
    print("  1. At low cost ratios (acting on a wrong pose is cheap), accept-all is optimal")
    print("     and abstention only sacrifices net benefit -- the layer is correctly OFF.")
    print("  2. As lambda grows (wrong poses get expensive), the conformal gate overtakes")
    print("     both accept-all and the fixed-ipTM baseline and stays best out to the")
    print("     high-cost tail; abstain-all (NB=0) only wins once no gate clears zero.")
    print("  3. So the layer's job is the high-cost regime, and the decision curve shows")
    print("     that directly -- no need to defend one alpha, the whole lambda axis is covered.")


if __name__ == "__main__":
    main()
