"""E22 -- distributionally-robust and simultaneous certificates for the accept/abstain gate.

Two extensions of the label-free worst-subpopulation certificate (E17):

* a chi-square robustness RADIUS rho*: the accepted selective risk stays <= alpha for every test
  law within chi-square divergence rho* of the calibration law (Cauchois et al., Robust
  Validation, JASA 2024; the exact two-point Pearson worst-case for the binary label). It is the
  same certificate as the CVaR mass m*, reparameterized by rho = (1 - m*) / m*, and reported at
  each coverage so the robustness-vs-coverage tradeoff is explicit.
* a SIMULTANEOUS certificate across the five models via a Bonferroni delta / K correction (Snell
  et al., Quantile Risk Control, Thm 4.6), so the statement "every model's accepted risk is
  controlled" holds jointly with probability 1 - delta.

As with E17, tightening the gate (lower coverage) buys a larger margin; at the aggressive LTT
operating point the margin is near zero, which we report rather than hide.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupKFold

from experiments._common import ALPHA, CONF, DELTA, RESDIR, load_delivered, methods_with_enough, save_json
from foldgate.conformal import (
    ltt_threshold,
    robustness_radius,
    simultaneous_certificate,
    worst_subpopulation_certificate,
)
from foldgate.scores.combiner import DEFAULT_FEATURES, ScoreCombiner

COVERAGES = [0.1, 0.2, 0.3, 0.4, 0.5]


def _oof_combined(sub, y):
    groups = sub["system_id"].to_numpy()
    oof = np.full(len(sub), np.nan)
    for cal, test in GroupKFold(n_splits=min(5, len(np.unique(groups)))).split(sub, y, groups):
        oof[test] = ScoreCombiner(features=DEFAULT_FEATURES).fit(sub.iloc[cal], y[cal]).predict(sub.iloc[test])
    return oof


def run() -> dict:
    df = load_delivered()
    methods = methods_with_enough(df)
    out = {"alpha": ALPHA, "delta": DELTA, "per_model": {}}
    ltt_counts = []
    for m in methods:
        sub = df[df.method == m].dropna(subset=[CONF, "system_id"]).reset_index(drop=True)
        y = sub["correct"].to_numpy().astype(int)
        score = _oof_combined(sub, y)
        order = np.argsort(-score)
        y_sorted = y[order]
        n = len(y)

        curve = []
        for c in COVERAGES:
            k = max(int(round(c * n)), 1)
            errors = int((1 - y_sorted[:k]).sum())
            wsc = worst_subpopulation_certificate(errors, k, ALPHA, DELTA)
            rr = robustness_radius(errors, k, ALPHA, DELTA)
            curve.append({"coverage": c, "m_star": wsc["m_star"], "rho_star": rr["rho_star"],
                          "certified": wsc["certified"], "r_ucb": wsc["r_ucb"]})

        # LTT operating point (the deployed gate) for the simultaneous certificate
        tau = ltt_threshold(score, y, alpha=ALPHA, delta=DELTA)
        if tau is not None:
            acc = score >= tau
            ltt_counts.append((int((1 - y[acc]).sum()), int(acc.sum())))
        rho3 = next(p["rho_star"] for p in curve if abs(p["coverage"] - 0.3) < 1e-9)
        out["per_model"][m] = {"n": n, "radius_curve": curve, "rho_star_at_cov_0.3": rho3}

    out["simultaneous_at_ltt"] = simultaneous_certificate(ltt_counts, ALPHA, DELTA)
    # a conservative simultaneous point: top-20%-by-combined per model
    cons = []
    for m in methods:
        sub = df[df.method == m].dropna(subset=[CONF, "system_id"]).reset_index(drop=True)
        y = sub["correct"].to_numpy().astype(int)
        score = _oof_combined(sub, y)
        k = max(int(round(0.2 * len(y))), 1)
        top = np.argsort(-score)[:k]
        cons.append((int((1 - y[top]).sum()), k))
    out["simultaneous_at_cov_0.2"] = simultaneous_certificate(cons, ALPHA, DELTA)
    return out


def main() -> None:
    res = run()
    save_json(res, RESDIR / "e22_robust_certificates.json")
    print(f"E22 -- DRO radius + simultaneous certificates  (alpha={ALPHA}, delta={DELTA})\n")
    for m, r in res["per_model"].items():
        pts = ", ".join(f"cov{p['coverage']}:rho*{p['rho_star']:.3f}(m*{p['m_star']:.2f})" for p in r["radius_curve"])
        print(f"[{m}] {pts}")
    s = res["simultaneous_at_cov_0.2"]
    print(f"\nsimultaneous @ top-20% (delta/K={s['delta_per_model']:.3f}): joint m*={s['joint_m_star']:.3f} "
          f"joint rho*={s['joint_rho_star']:.4f} all_certified={s['all_certified']}")
    sl = res["simultaneous_at_ltt"]
    print(f"simultaneous @ LTT gate: joint m*={sl['joint_m_star']:.3f} rho*={sl['joint_rho_star']:.4f} "
          f"all_certified={sl['all_certified']} (aggressive gate -> near-zero margin, honest)")


if __name__ == "__main__":
    main()
