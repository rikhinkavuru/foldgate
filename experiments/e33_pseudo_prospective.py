"""E33 -- pseudo-prospective (time-split) deployment of the accept/abstain gate.

Addition III.4. Answers the "everything is retrospective" objection without a wet
lab. We freeze gate calibration on everything deposited BEFORE a cutoff date T and
evaluate on everything deposited on/after T, using NO stratum information at
calibration time. This mimics training a gate on the past and deploying it on the
future: the calibration set literally cannot see any post-T structure.

This is a TIME split, complementary to the structural-novelty story. RNP is entirely
post the 2021-era models' training cutoff (see E25), so the pre/post-T contrast here
ranks RECENCY among already-out-of-training structures, not in-training vs
out-of-training. It tests whether a gate calibrated on older depositions still
controls risk on newer ones -- temporal transfer of the certificate, not structural
generalization.

Protocol, per governed model:
  - Reduce to one delivered pose per (system_id, method): the top-ranking_score pose.
  - Split on structure deposition date at T (default: 60th percentile of release_date,
    chosen so every model clears ~200 post-T targets, the certification floor).
  - NATIVE gate: LTT-calibrate tau on pre-T (ranking_score, correct) at
    alpha=0.20, delta=0.10; deploy frozen on post-T.
  - COMBINED gate: grouped 50/50 split of the pre-T targets into a combiner-FIT subset
    and a disjoint calibration subset (both pre-T only). Fit ScoreCombiner on FIT,
    LTT-calibrate tau on the CAL subset's out-of-sample combined scores, deploy frozen
    on post-T combined scores.
  - Report, on post-T: coverage, accepted n, realized selective risk, exact
    Clopper-Pearson 90% interval, and the (1-delta) Hoeffding-Bentkus certified upper
    bound. Also report the pre-T EXPECTED risk of the accept set (what the gate thought
    it was buying) vs the realized post-T risk, per model, to show whether temporal
    deployment degrades control.

Interpretation: if realized post-T risk (or its CP upper bound) stays <= alpha, the
gate transfers prospectively in time; an overshoot is an honest limit.

Output: results/e33_pseudo_prospective.json
"""

from __future__ import annotations

import numpy as np
import pandas as pd

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
from foldgate.conformal.risk import hb_upper_bound
from foldgate.scores.combiner import DEFAULT_FEATURES, ScoreCombiner
from foldgate.selective.metrics import clopper_pearson

T_QUANTILE = 0.60        # cutoff = this percentile of release_date (~40% is "future")
MIN_POST_TARGETS = 200   # certification floor: post-T must clear this per model
CAL_FRAC = 0.5           # pre-T fit/calibration split for the combined gate
MIN_COVERAGE_VACUOUS = 0.05


def _one_pose_per_target(df):
    """Reduce to the top-ranking_score delivered pose per (system_id, method)."""
    d = df.dropna(subset=[CONF, "system_id", "release_dt"]).copy()
    idx = d.groupby(["method", "system_id"])[CONF].idxmax()
    return d.loc[idx].reset_index(drop=True)


def _expected_risk(score_cal, y_cal, tau):
    """Pre-T empirical risk of the accept set the gate was calibrated on."""
    acc = score_cal >= tau
    na = int(acc.sum())
    if na == 0:
        return None, 0
    return round(float((1 - y_cal[acc]).mean()), 4), na


def _deploy_metrics(score_post, y_post, tau, alpha, delta, expected_risk, expected_n):
    """Frozen-gate metrics on the post-T (future) set."""
    n = len(score_post)
    if tau is None:
        return {"tau": None, "coverage": 0.0, "n_accept": 0, "n_post": n,
                "realized_risk": None, "vacuous": True,
                "note": "LTT certified nothing on pre-T calibration"}
    acc = score_post >= tau
    na = int(acc.sum())
    if na == 0:
        return {"tau": round(float(tau), 4), "coverage": 0.0, "n_accept": 0,
                "n_post": n, "realized_risk": None, "vacuous": True,
                "note": "frozen tau accepts nothing on post-T"}
    e = int((1 - y_post[acc]).sum())
    risk = e / na
    cp_lo, cp_hi = clopper_pearson(e, na, ci=0.90)
    ub = hb_upper_bound(risk, na, delta)
    cov = na / n
    return {
        "tau": round(float(tau), 4),
        "coverage": round(cov, 4),
        "n_accept": na,
        "n_post": n,
        "realized_risk": round(risk, 4),
        "realized_risk_cp90": [round(cp_lo, 4), round(cp_hi, 4)],
        "risk_cp_upper90": round(cp_hi, 4),
        "certified_ub_hb": round(ub, 4),
        # transfers in time if realized risk (or its exact CP upper bound) stays <= alpha
        "risk_controlled": bool(risk <= alpha),
        "cp_upper_controlled": bool(cp_hi <= alpha),
        "expected_risk_preT": expected_risk,
        "expected_accept_n_preT": expected_n,
        "risk_inflation_post_minus_pre": (
            None if expected_risk is None else round(risk - expected_risk, 4)
        ),
        "vacuous": bool(cov < MIN_COVERAGE_VACUOUS),
    }


def _per_model(sub, T, alpha, delta, g):
    pre = sub[sub.release_dt < T]
    post = sub[sub.release_dt >= T]
    out = {
        "n_pre": int(len(pre)),
        "n_post": int(len(post)),
        "base_correct_pre": round(float(pre["correct"].mean()), 4) if len(pre) else None,
        "base_correct_post": round(float(post["correct"].mean()), 4) if len(post) else None,
    }
    if len(pre) < 30 or len(post) < 30:
        out["skipped"] = "insufficient pre or post data"
        return out

    y_pre = pre["correct"].to_numpy().astype(int)
    y_post = post["correct"].to_numpy().astype(int)

    # ---- NATIVE gate: calibrate on pre-T ranking_score, deploy frozen on post-T ----
    s_pre = pre[CONF].to_numpy()
    s_post = post[CONF].to_numpy()
    tau_nat = ltt_threshold(s_pre, y_pre, alpha=alpha, delta=delta)
    exp_r, exp_n = (None, 0) if tau_nat is None else _expected_risk(s_pre, y_pre, tau_nat)
    out["native"] = _deploy_metrics(s_post, y_post, tau_nat, alpha, delta, exp_r, exp_n)

    # ---- COMBINED gate: grouped fit/cal split of pre-T, deploy frozen on post-T ----
    # After the one-pose reduction each pre-T row is a distinct target, so a plain
    # random split is already target-disjoint (no system_id leaks across fit/cal).
    n_pre = len(pre)
    perm = g.permutation(n_pre)
    n_cal = int(round(CAL_FRAC * n_pre))
    cal_local = perm[:n_cal]
    fit_local = perm[n_cal:]
    pre_fit = pre.iloc[fit_local]
    pre_cal = pre.iloc[cal_local]

    comb = ScoreCombiner(features=DEFAULT_FEATURES).fit(pre_fit, y_pre[fit_local])
    sc_cal = comb.predict(pre_cal)
    sc_post = comb.predict(post)
    y_cal = y_pre[cal_local]

    tau_cmb = ltt_threshold(sc_cal, y_cal, alpha=alpha, delta=delta)
    exp_rc, exp_nc = (None, 0) if tau_cmb is None else _expected_risk(sc_cal, y_cal, tau_cmb)
    out["combined"] = _deploy_metrics(sc_post, y_post, tau_cmb, alpha, delta, exp_rc, exp_nc)
    out["combined"]["n_fit_preT"] = int(len(pre_fit))
    out["combined"]["n_cal_preT"] = int(len(pre_cal))
    return out


def run() -> dict:
    df = load_delivered()
    df["release_dt"] = pd.to_datetime(df["release_date"], errors="coerce")
    methods = methods_with_enough(df)
    red = _one_pose_per_target(df[df.method.isin(methods)])

    T = red["release_dt"].quantile(T_QUANTILE)
    # Guard the certification floor: if any model's post-T set is too thin, step T back.
    for q in (T_QUANTILE, 0.55, 0.50, 0.45, 0.40):
        Tq = red["release_dt"].quantile(q)
        post_counts = red[red.release_dt >= Tq].groupby("method").size()
        if post_counts.reindex(methods).fillna(0).min() >= MIN_POST_TARGETS:
            T, T_QUANTILE_USED = Tq, q
            break
    else:
        T_QUANTILE_USED = T_QUANTILE  # keep default even if floor unmet; flagged below

    g = rng()
    span = {
        "release_date_min": str(red["release_dt"].min().date()),
        "release_date_max": str(red["release_dt"].max().date()),
    }

    per_model = {}
    for m in methods:
        sub = red[red.method == m]
        per_model[m] = _per_model(sub, T, ALPHA, DELTA, g)

    counts = {
        m: {"pre": per_model[m]["n_pre"], "post": per_model[m]["n_post"]}
        for m in methods
    }
    floor_met = all(per_model[m]["n_post"] >= MIN_POST_TARGETS for m in methods)

    out = {
        "config": {
            "alpha": ALPHA,
            "delta": DELTA,
            "T_quantile": round(float(T_QUANTILE_USED), 3),
            "T_cutoff": str(T.date()),
            "min_post_targets_floor": MIN_POST_TARGETS,
            "cal_frac_preT": CAL_FRAC,
            "one_pose_per_target": "top ranking_score per (system_id, method)",
            "release_date_span": span,
            "post_floor_met": bool(floor_met),
        },
        "pre_post_counts": counts,
        "per_model": per_model,
    }
    out["_summary"] = _takeaway(out)
    return out


def _takeaway(out) -> dict:
    T = out["config"]["T_cutoff"]
    lines = []
    lines.append(
        f"Time split at T={T} ({int(round(out['config']['T_quantile']*100))}th pct of "
        f"release_date): calibrate on pre-T depositions, deploy frozen on post-T, no "
        f"stratum info used. Post-T clears the {out['config']['min_post_targets_floor']}"
        f"-target floor for every model: {out['config']['post_floor_met']}."
    )
    nat_ok = sum(
        1 for m, r in out["per_model"].items()
        if r.get("native", {}).get("risk_controlled") or r.get("native", {}).get("cp_upper_controlled")
    )
    cmb_ok = sum(
        1 for m, r in out["per_model"].items()
        if r.get("combined", {}).get("risk_controlled") or r.get("combined", {}).get("cp_upper_controlled")
    )
    nmods = len(out["per_model"])
    lines.append(
        f"Native gate holds post-T risk <= alpha={out['config']['alpha']} (realized or CP-upper) "
        f"for {nat_ok}/{nmods} models; combined gate for {cmb_ok}/{nmods}. Where it holds, the "
        f"certificate calibrated on the past transfers forward in time."
    )
    infl = [
        r["combined"]["risk_inflation_post_minus_pre"]
        for r in out["per_model"].values()
        if r.get("combined", {}).get("risk_inflation_post_minus_pre") is not None
    ]
    if infl:
        lines.append(
            f"Realized-minus-expected risk (combined) spans {min(infl):+.3f}..{max(infl):+.3f} across "
            f"models: small gaps mean temporal deployment does not silently break control; a large "
            f"positive gap on any model is an honest recency-transfer limit."
        )
    return {"takeaway": lines}


def main() -> None:
    res = run()
    save_json(res, RESDIR / "e33_pseudo_prospective.json")
    c = res["config"]
    print("E33 -- pseudo-prospective (time-split) gate deployment")
    print(f"T = {c['T_cutoff']} ({c['T_quantile']} quantile of release_date), "
          f"span {c['release_date_span']['release_date_min']}..{c['release_date_span']['release_date_max']}")
    print(f"post-floor ({c['min_post_targets_floor']}/model) met: {c['post_floor_met']}\n")
    print(f"{'model':>9} | {'pre':>5} {'post':>5} | "
          f"{'gate':>8}  cov   n_acc  risk   CP90-up  HBub   exp(pre)  transfer")
    for m, r in res["per_model"].items():
        pre, post = r["n_pre"], r["n_post"]
        for gate in ("native", "combined"):
            g = r.get(gate)
            if not g:
                continue
            if g.get("n_accept", 0) == 0:
                print(f"{m:>9} | {pre:>5} {post:>5} | {gate:>8}  abstains (tau {g.get('tau')})")
                continue
            transfer = "yes" if (g["risk_controlled"] or g["cp_upper_controlled"]) else "NO"
            print(f"{m:>9} | {pre:>5} {post:>5} | {gate:>8}  "
                  f"{g['coverage']:.2f}  {g['n_accept']:>5}  {g['realized_risk']:.3f}  "
                  f"{g['risk_cp_upper90']:.3f}   {g['certified_ub_hb']:.3f}  "
                  f"{str(g['expected_risk_preT']):>7}   {transfer}")
    print()
    for line in res["_summary"]["takeaway"]:
        print("- " + line)


if __name__ == "__main__":
    main()
