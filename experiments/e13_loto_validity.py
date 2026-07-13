"""E13 -- target-grouped (leave-one-target-out) validity + leakage audit.

E1/E4 use random pose-level splits and single-split pose bootstraps. Two honest
worries a reviewer raises: (1) the same target (or a near-duplicate) could sit in
both calibration and test, leaking optimistic validity; (2) a pose-level bootstrap
understates variability because poses of one target are correlated. We close both:

* **Grouped calibration** -- GroupKFold on ``system_id`` so every pose of a target is
  entirely in calibration or entirely in test, never split. We refit the LTT gate and
  the calibration-only combiner out-of-fold and report realized selective risk +
  coverage on held-out targets. If grouped validity matches the random-split result,
  the guarantee is not a leakage artifact.

* **Target-cluster bootstrap** -- resample whole targets (not rows) to get an honest,
  wider AURC confidence interval; we report it beside the naive row bootstrap so the
  understatement is explicit.

* **Leakage audit** -- for the pooled multi-model table, a random row split shares a
  large fraction of targets between calibration and test; a grouped split shares none.
  We quantify exactly what the naive split would have leaked.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from experiments._common import (
    ALPHA,
    CONF,
    DELTA,
    DELTA_JOINT,
    RESDIR,
    iut_all,
    load_delivered,
    methods_with_enough,
    rng,
    save_json,
)
from foldgate.conformal import ltt_threshold
from foldgate.scores.combiner import DEFAULT_FEATURES, ScoreCombiner
from foldgate.selective.metrics import aurc, clopper_pearson

N_FOLDS = 5
N_BOOT = 1000


def _cluster_bootstrap_aurc(score, correct, groups, g, n_boot=N_BOOT):
    """90% CI for AURC by resampling whole targets (clusters), not rows."""
    uniq = np.unique(groups)
    by_group = {u: np.where(groups == u)[0] for u in uniq}
    vals = []
    for _ in range(n_boot):
        pick = g.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([by_group[uniq[p]] for p in pick])
        vals.append(aurc(score[idx], correct[idx]))
    return [float(np.quantile(vals, 0.05)), float(np.quantile(vals, 0.95))]


def _paired_cluster_bootstrap_delta_aurc(native, combined, correct, groups, g, n_boot=N_BOOT):
    """Paired cluster-bootstrap of Delta(AURC) = AURC(native) - AURC(combined).

    Resamples whole targets and takes the difference on the SAME resample each rep, so
    the two scorers see identical rows and the CI reflects the paired gain (positive =
    combined beats native). Returns the 90% percentile interval, a one-sided bootstrap
    p-value for H0: Delta <= 0, and the point estimate. The per-model interval excluding
    zero is one component of the LOTO intersection-union test for "combined beats native
    for every model".
    """
    uniq = np.unique(groups)
    by_group = {u: np.where(groups == u)[0] for u in uniq}
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        pick = g.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([by_group[uniq[p]] for p in pick])
        deltas[b] = aurc(native[idx], correct[idx]) - aurc(combined[idx], correct[idx])
    lo, hi = float(np.quantile(deltas, 0.05)), float(np.quantile(deltas, 0.95))
    p_one = float((1.0 + np.count_nonzero(deltas <= 0.0)) / (1.0 + n_boot))
    return {
        "delta_aurc_point": float(aurc(native, correct) - aurc(combined, correct)),
        "delta_aurc_cluster_ci90": [lo, hi],
        "delta_excludes_zero": bool(lo > 0.0),
        "delta_p_one_sided": p_one,
    }


def _row_bootstrap_aurc(score, correct, g, n_boot=N_BOOT):
    n = len(score)
    vals = [aurc(score[i], correct[i]) for i in (g.integers(0, n, n) for _ in range(n_boot))]
    return [float(np.quantile(vals, 0.05)), float(np.quantile(vals, 0.95))]


def run() -> dict:
    df = load_delivered()
    methods = methods_with_enough(df)
    g = rng()
    out = {"per_model": {}}

    for m in methods:
        sub = df[df.method == m].dropna(subset=[CONF, "system_id"]).reset_index(drop=True)
        s = sub[CONF].to_numpy()
        y = sub["correct"].to_numpy().astype(int)
        groups = sub["system_id"].to_numpy()
        n = len(sub)
        n_splits = min(N_FOLDS, len(np.unique(groups)))

        gkf = GroupKFold(n_splits=n_splits)
        oof_native = np.full(n, np.nan)
        oof_combined = np.full(n, np.nan)
        per_fold = []
        # per-model certificate at delta (0.10) and the joint certificate at delta/K
        # (0.02, Bonferroni union bound so the "every model holds" statement is valid).
        pooled_acc, pooled_err = 0, 0
        pooled_acc_j, pooled_err_j = 0, 0
        for cal_idx, test_idx in gkf.split(s, y, groups):
            tau = ltt_threshold(s[cal_idx], y[cal_idx], alpha=ALPHA, delta=DELTA)
            tau_j = ltt_threshold(s[cal_idx], y[cal_idx], alpha=ALPHA, delta=DELTA_JOINT)
            comb = ScoreCombiner(features=DEFAULT_FEATURES).fit(sub.iloc[cal_idx], y[cal_idx])
            oof_native[test_idx] = s[test_idx]
            oof_combined[test_idx] = comb.predict(sub.iloc[test_idx])

            fold = {"tau": None, "risk": float("nan"), "coverage": 0.0, "holds": None,
                    "tau_joint": None, "risk_joint": float("nan"),
                    "coverage_joint": 0.0, "holds_joint": None}
            if tau is not None:
                acc = s[test_idx] >= tau
                n_acc = int(acc.sum())
                n_err = int((1 - y[test_idx][acc]).sum())
                pooled_acc += n_acc
                pooled_err += n_err
                fold["tau"] = float(tau)
                fold["risk"] = (n_err / n_acc) if n_acc else float("nan")
                fold["coverage"] = n_acc / len(test_idx)
                fold["holds"] = bool(fold["risk"] <= ALPHA) if n_acc else None
            if tau_j is not None:
                acc_j = s[test_idx] >= tau_j
                n_acc_j = int(acc_j.sum())
                n_err_j = int((1 - y[test_idx][acc_j]).sum())
                pooled_acc_j += n_acc_j
                pooled_err_j += n_err_j
                fold["tau_joint"] = float(tau_j)
                fold["risk_joint"] = (n_err_j / n_acc_j) if n_acc_j else float("nan")
                fold["coverage_joint"] = n_acc_j / len(test_idx)
                fold["holds_joint"] = bool(fold["risk_joint"] <= ALPHA) if n_acc_j else None
            per_fold.append(fold)

        risk_lo, risk_hi = clopper_pearson(pooled_err, pooled_acc, ci=0.90) if pooled_acc else (float("nan"),) * 2
        rj_lo, rj_hi = clopper_pearson(pooled_err_j, pooled_acc_j, ci=0.90) if pooled_acc_j else (float("nan"),) * 2
        holds = [f["holds"] for f in per_fold if f["holds"] is not None]
        holds_j = [f["holds_joint"] for f in per_fold if f["holds_joint"] is not None]
        cov_j = pooled_acc_j / n
        # The certificate bounds TRUE risk, and the pooled held-out risk is the least
        # noisy realized estimate of it (a single tightened fold accepts few poses and
        # its per-fold fraction is noisy, the E1 caveat), so "holds" keys off the pooled
        # joint risk. A model that abstains everywhere passes vacuously with zero
        # coverage, which we surface rather than hide (CLAUDE.md rule 5).
        joint_pooled_risk = (pooled_err_j / pooled_acc_j) if pooled_acc_j else float("nan")
        holds_joint = (pooled_acc_j == 0) or bool(joint_pooled_risk <= ALPHA)
        joint_vacuous = bool(cov_j < 0.05)
        paired = _paired_cluster_bootstrap_delta_aurc(oof_native, oof_combined, y, groups, g)
        out["per_model"][m] = {
            "n": n,
            "n_targets": int(len(np.unique(groups))),
            "loto_pooled_risk": (pooled_err / pooled_acc) if pooled_acc else float("nan"),
            "loto_pooled_risk_ci90": [risk_lo, risk_hi],
            "loto_pooled_coverage": pooled_acc / n,
            "folds_holding": [int(sum(holds)), len(holds)],
            "loto_pooled_risk_joint": joint_pooled_risk,
            "loto_pooled_risk_joint_ci90": [rj_lo, rj_hi],
            "loto_pooled_coverage_joint": cov_j,
            "folds_holding_joint": [int(sum(holds_j)), len(holds_j)],
            "holds_joint": bool(holds_joint),
            "joint_vacuous": joint_vacuous,
            "aurc_native_oof": aurc(oof_native, y),
            "aurc_combined_oof": aurc(oof_combined, y),
            "aurc_native_cluster_ci90": _cluster_bootstrap_aurc(oof_native, y, groups, g),
            "aurc_combined_cluster_ci90": _cluster_bootstrap_aurc(oof_combined, y, groups, g),
            "aurc_native_row_ci90": _row_bootstrap_aurc(oof_native, y, g),
            "delta_aurc_native_minus_combined": paired["delta_aurc_point"],
            "delta_aurc_cluster_ci90": paired["delta_aurc_cluster_ci90"],
            "delta_excludes_zero": paired["delta_excludes_zero"],
            "delta_p_one_sided": paired["delta_p_one_sided"],
            "per_fold": per_fold,
        }

    # Joint statements across the K models (docs/theory/MULTIPLICITY_SPEC.md).
    pm = out["per_model"]
    out["joint"] = {
        "delta": DELTA,
        "delta_joint": DELTA_JOINT,
        "K": len(pm),
        # certificate conjunction: does every model's LOTO gate hold at the Bonferroni
        # per-model level delta/K? Reported next to where it is vacuous.
        "all_models_hold_joint": bool(all(r["holds_joint"] for r in pm.values())),
        "models_joint_vacuous": [m for m, r in pm.items() if r["joint_vacuous"]],
        # discovery conjunction: "combined beats native for every model" is an
        # intersection-union test, so each per-model one-sided p <= delta certifies the
        # joint claim at delta with no multiplicity penalty (Berger 1982).
        "all_models_exclude_zero": bool(iut_all([r["delta_p_one_sided"] for r in pm.values()], DELTA)),
        "delta_p_one_sided_max": float(max(r["delta_p_one_sided"] for r in pm.values())),
        "note": (
            "all_models_hold_joint is a Bonferroni union-bound certificate at delta/K; "
            "chai/protenix reach it only by abstaining (see models_joint_vacuous / "
            "loto_pooled_coverage_joint). all_models_exclude_zero is an IUT and takes no penalty."
        ),
    }

    # Leakage audit on the pooled multi-model table.
    pooled = df.dropna(subset=[CONF, "system_id"]).reset_index(drop=True)
    grp = pooled["system_id"].to_numpy()
    gss = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=0)
    (cal_g, test_g), = gss.split(pooled, groups=grp)
    grouped_shared = len(set(grp[cal_g]) & set(grp[test_g]))
    # naive random row split of the same pooled table
    perm = g.permutation(len(pooled))
    cal_r, test_r = perm[: len(pooled) // 2], perm[len(pooled) // 2:]
    cal_targets = set(grp[cal_r])
    shared_rows = int(np.isin(grp[test_r], list(cal_targets)).sum())
    out["leakage_audit"] = {
        "pooled_rows": int(len(pooled)),
        "pooled_targets": int(len(np.unique(grp))),
        "random_split_test_rows_sharing_cal_target": shared_rows,
        "random_split_leak_fraction": shared_rows / len(test_r),
        "grouped_split_shared_targets": int(grouped_shared),
    }
    return out


def main() -> None:
    res = run()
    save_json(res, RESDIR / "e13_loto_validity.json")
    j = res["joint"]
    print(f"E13 -- target-grouped (LOTO) validity  (alpha={ALPHA}, per-model delta={DELTA}, "
          f"joint delta/K={j['delta_joint']})\n")
    for m, r in res["per_model"].items():
        h, hn = r["folds_holding"]
        hj, hjn = r["folds_holding_joint"]
        lo, hi = r["loto_pooled_risk_ci90"]
        d_lo, d_hi = r["delta_aurc_cluster_ci90"]
        vac = " VACUOUS" if r["joint_vacuous"] else ""
        print(f"[{m}] targets={r['n_targets']}  LOTO risk={r['loto_pooled_risk']:.3f} "
              f"[{lo:.3f},{hi:.3f}] cov={r['loto_pooled_coverage']:.2f}  folds_hold={h}/{hn}")
        print(f"     joint(delta/K): risk={r['loto_pooled_risk_joint']:.3f} "
              f"cov={r['loto_pooled_coverage_joint']:.2f} folds_hold={hj}/{hjn} holds_joint={r['holds_joint']}{vac}")
        print(f"     AURC oof native={r['aurc_native_oof']:.3f} combined={r['aurc_combined_oof']:.3f}  "
              f"| paired delta(AURC) {r['delta_aurc_native_minus_combined']:.3f} "
              f"[{d_lo:.3f},{d_hi:.3f}] excl0={r['delta_excludes_zero']}")
    print(f"\njoint statements (K={j['K']}): all_models_hold_joint={j['all_models_hold_joint']} "
          f"(vacuous: {j['models_joint_vacuous']})  |  "
          f"all_models_exclude_zero(IUT)={j['all_models_exclude_zero']} "
          f"(max one-sided p={j['delta_p_one_sided_max']:.4f})")
    la = res["leakage_audit"]
    print(f"\nleakage audit (pooled {la['pooled_rows']} rows, {la['pooled_targets']} targets): "
          f"random split leaks {la['random_split_leak_fraction']:.1%} of test rows "
          f"(share a cal target); grouped split shares {la['grouped_split_shared_targets']}.")


if __name__ == "__main__":
    main()
