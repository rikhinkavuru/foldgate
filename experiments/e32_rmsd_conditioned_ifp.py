"""E32 -- is the interaction-fingerprint (IFP) lift genuine, or an RMSD confound? (R2.8 / III.9)

E6b compares accepted-vs-rejected IFP recall by POOLING all accepted against all rejected
poses. But the gate accepts poses enriched for low ligand-RMSD, and low RMSD mechanically
buys correct crystal contacts, so part of that ~0.19 recall gap is the RMSD confound, not a
genuine gate lift on interactions. This script conditions the comparison two ways.

Gate protocol (leakage-free, target-grouped, matching e34):
  GroupKFold(5) on system_id (outer test folds). Within each fold's training targets, a
  grouped 50/50 fit/cal split (GroupShuffleSplit on system_id). Fit ScoreCombiner on the FIT
  targets, calibrate tau by LTT on the out-of-sample combined CAL scores vs `correct`
  (alpha=0.20, delta from _common), then label each held-out test pose accepted iff its
  combined score >= tau. Every pose gets exactly one out-of-fold accept/reject label.

Two conditioned analyses per model:
  1. WITHIN-CORRECT gap. Restrict to sub-2A (correct==1) poses only, then accepted-minus-
     rejected mean ifp_recall AMONG THE CORRECT POSES. If this shrinks toward zero vs the
     unconditioned e6b gap (AF3 ~0.19), that quantifies the confound; a surviving positive
     residual is a genuine, non-circular lift. 90% CI by row bootstrap.
  2. REGRESSION with gate covariate. OLS  ifp_recall ~ rmsd + accepted  over all evaluated
     poses (accepted = gate indicator 0/1). The `accepted` coefficient is the recovery lift
     attributable to the gate AFTER linearly controlling for RMSD; 90% CI + sign by row
     bootstrap (1000 reps). The plain rmsd coefficient is reported for context.

Output: results/e32_rmsd_conditioned_ifp.json
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from experiments._common import (
    ALPHA,
    DELTA,
    RESDIR,
    load_delivered,
    methods_with_enough,
    rng,
    save_json,
)
from foldgate.conformal import ltt_threshold
from foldgate.scores.combiner import DEFAULT_FEATURES, ScoreCombiner
from foldgate.selective.metrics import bootstrap_ci

N_FOLDS = 5
CAL_FRAC = 0.5          # grouped fit/cal split of the training targets
MIN_ROWS = 300
N_BOOT = 1000

# e6b unconditioned AF3 reference (accepted vs rejected pooled over ALL poses).
E6B_AF3_UNCONDITIONED_GAP = 0.190


def _oof_accept(sub, y, alpha, delta):
    """Out-of-fold gate indicator: accepted[i] in {0,1} for every row, -1 if its test
    fold produced no certifiable tau (those rows are dropped downstream)."""
    groups = sub["system_id"].to_numpy()
    n = len(sub)
    accepted = np.full(n, -1, dtype=int)
    n_splits = min(N_FOLDS, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)
    for train_idx, test_idx in gkf.split(np.arange(n), y, groups):
        g_train = groups[train_idx]
        gss = GroupShuffleSplit(n_splits=1, test_size=CAL_FRAC, random_state=0)
        (fit_local, cal_local), = gss.split(train_idx, groups=g_train)
        fit_idx, cal_idx = train_idx[fit_local], train_idx[cal_local]
        comb = ScoreCombiner(features=DEFAULT_FEATURES).fit(sub.iloc[fit_idx], y[fit_idx])
        tau = ltt_threshold(comb.predict(sub.iloc[cal_idx]), y[cal_idx], alpha=alpha, delta=delta)
        if tau is None:
            continue
        accepted[test_idx] = (comb.predict(sub.iloc[test_idx]) >= tau).astype(int)
    return accepted


def _gap(acc_mask, v):
    """mean(recall | accepted) - mean(recall | rejected)."""
    if not acc_mask.any() or not (~acc_mask).any():
        return float("nan")
    return float(np.nanmean(v[acc_mask]) - np.nanmean(v[~acc_mask]))


def _ols(y, rmsd, accepted):
    """OLS ifp_recall ~ 1 + rmsd + accepted; returns (rmsd_coef, accepted_coef)."""
    X = np.column_stack([np.ones_like(rmsd), rmsd, accepted.astype(float)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(beta[1]), float(beta[2])


def _boot_ols_accepted(y, rmsd, accepted, n_boot=N_BOOT, seed=0):
    """Row-bootstrap 90% CI + two-sided sign p-value for the `accepted` OLS coefficient."""
    g = np.random.default_rng(seed)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        idx = g.integers(0, n, n)
        try:
            _, ac = _ols(y[idx], rmsd[idx], accepted[idx])
        except np.linalg.LinAlgError:
            continue
        if np.isfinite(ac):
            vals.append(ac)
    if not vals:
        return (float("nan"), float("nan")), float("nan")
    vals = np.asarray(vals)
    lo, hi = float(np.quantile(vals, 0.05)), float(np.quantile(vals, 0.95))
    p_two_sided = float(min(1.0, 2.0 * min(np.mean(vals <= 0), np.mean(vals >= 0))))
    return (lo, hi), p_two_sided


def run() -> dict:
    df = load_delivered()
    if "ifp_recall" not in df.columns:
        return {"_status": "ifp features absent -- run `make pose-features`"}
    out = {
        "alpha": ALPHA,
        "delta": DELTA,
        "protocol": "target-grouped LOTO gate (GroupKFold outer, grouped fit/cal inner), combined score",
        "e6b_af3_unconditioned_gap_reference": E6B_AF3_UNCONDITIONED_GAP,
        "per_model": {},
    }
    _ = rng()  # pin global seed convention
    for m in methods_with_enough(df):
        sub = (
            df[df.method == m]
            .dropna(subset=["ifp_recall", "rmsd", "system_id"])
            .reset_index(drop=True)
        )
        if len(sub) < MIN_ROWS:
            continue
        y_correct = sub["correct"].to_numpy().astype(int)
        rec = sub["ifp_recall"].to_numpy(dtype=float)
        rmsd = sub["rmsd"].to_numpy(dtype=float)

        accepted = _oof_accept(sub, y_correct, ALPHA, DELTA)
        ev = accepted >= 0                       # evaluated (certifiable) rows only
        if ev.sum() < MIN_ROWS or accepted[ev].sum() == 0 or accepted[ev].sum() == ev.sum():
            out["per_model"][m] = {"n": int(len(sub)), "note": "no usable evaluated split"}
            continue

        acc_e = accepted[ev].astype(bool)
        rec_e, rmsd_e, corr_e = rec[ev], rmsd[ev], y_correct[ev].astype(bool)

        # ---- (0) unconditioned gap over ALL evaluated poses (recompute of the e6b gap) ----
        uncond_gap = _gap(acc_e, rec_e)
        uc_lo, uc_hi = bootstrap_ci(_gap, acc_e, rec_e, n_boot=N_BOOT)

        # ---- (1) within-correct gap: restrict to sub-2A poses ----
        acc_c = acc_e[corr_e]
        rec_c = rec_e[corr_e]
        if acc_c.any() and (~acc_c).any():
            wc_gap = _gap(acc_c, rec_c)
            wc_lo, wc_hi = bootstrap_ci(_gap, acc_c, rec_c, n_boot=N_BOOT)
        else:
            wc_gap, wc_lo, wc_hi = float("nan"), float("nan"), float("nan")

        # ---- (2) OLS ifp_recall ~ rmsd + accepted over all evaluated poses ----
        rmsd_coef, acc_coef = _ols(rec_e, rmsd_e, acc_e.astype(float))
        (ac_lo, ac_hi), ac_p = _boot_ols_accepted(rec_e, rmsd_e, acc_e.astype(float))

        out["per_model"][m] = {
            "n_total": int(len(sub)),
            "n_evaluated": int(ev.sum()),
            "n_correct_evaluated": int(corr_e.sum()),
            "coverage_evaluated": float(acc_e.mean()),
            "accepted_ifp_recall_all": float(np.nanmean(rec_e[acc_e])),
            "rejected_ifp_recall_all": float(np.nanmean(rec_e[~acc_e])),
            "unconditioned_gap": uncond_gap,
            "unconditioned_gap_ci90": [uc_lo, uc_hi],
            "within_correct": {
                "n_correct": int(corr_e.sum()),
                "n_correct_accepted": int(acc_c.sum()),
                "n_correct_rejected": int((~acc_c).sum()),
                "accepted_ifp_recall": float(np.nanmean(rec_c[acc_c])) if acc_c.any() else float("nan"),
                "rejected_ifp_recall": float(np.nanmean(rec_c[~acc_c])) if (~acc_c).any() else float("nan"),
                "gap": wc_gap,
                "gap_ci90": [wc_lo, wc_hi],
                "gap_excludes_zero": bool(np.isfinite(wc_lo) and wc_lo > 0),
                "shrinkage_vs_unconditioned": (
                    float(uncond_gap - wc_gap) if np.isfinite(uncond_gap) and np.isfinite(wc_gap) else float("nan")
                ),
            },
            "regression": {
                "model": "ifp_recall ~ 1 + rmsd + accepted",
                "rmsd_coef": rmsd_coef,
                "accepted_coef": acc_coef,
                "accepted_coef_ci90": [ac_lo, ac_hi],
                "accepted_coef_sign": int(np.sign(acc_coef)),
                "accepted_coef_p_two_sided": ac_p,
                "accepted_coef_excludes_zero": bool(np.isfinite(ac_lo) and (ac_lo > 0 or ac_hi < 0)),
            },
        }
    return out


def main() -> None:
    res = run()
    save_json(res, RESDIR / "e32_rmsd_conditioned_ifp.json")
    if "_status" in res:
        print(f"E32 -- {res['_status']}")
        return
    print(f"E32 -- IFP lift under RMSD conditioning (alpha={ALPHA}, delta={DELTA})\n")
    hdr = f"{'model':9} {'cov':>5} {'uncond':>8} {'within-corr gap (CI90)':>26} {'accepted coef (CI90)':>26}"
    print(hdr)
    for m, r in res["per_model"].items():
        if "regression" not in r:
            print(f"{m:9} {r.get('note','-')}")
            continue
        wc = r["within_correct"]
        rg = r["regression"]
        wc_ci = wc["gap_ci90"]
        ac_ci = rg["accepted_coef_ci90"]
        print(
            f"{m:9} {r['coverage_evaluated']:>5.2f} {r['unconditioned_gap']:>+8.3f} "
            f"{wc['gap']:>+9.3f} [{wc_ci[0]:+.3f},{wc_ci[1]:+.3f}]{'*' if wc['gap_excludes_zero'] else ' '} "
            f"{rg['accepted_coef']:>+9.3f} [{ac_ci[0]:+.3f},{ac_ci[1]:+.3f}]{'*' if rg['accepted_coef_excludes_zero'] else ' '}"
        )
    print(
        "\n* = CI excludes 0. within-corr gap conditions on the 2A label (removes the RMSD "
        "confound at the label boundary); the accepted coef conditions on continuous RMSD."
    )


if __name__ == "__main__":
    main()
