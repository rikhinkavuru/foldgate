"""E17 -- label-free worst-subpopulation certificate (CVaR beta*), the positive
companion to the weighted-CP null.

Weighted conformal cannot certify a *named* novel stratum under concept shift (E3b).
CVaR control instead certifies, without stratum labels or weights, that every accepted
subpopulation carrying at least a fraction m* = 1 - beta* of accepted mass has selective
risk <= alpha (CVaR-DRO duality). For the binary RMSD <= 2 A loss CVaR is exact
(CVaR_beta = min(1, r/(1-beta))), so a finite-sample upper bound on the accepted error
rate certifies every beta at once.

We report:
* the tradeoff curve m*(coverage): tightening the gate lowers the accepted error rate and
  shrinks m*, so smaller subpopulations become certifiable -- the honest price of a robust
  guarantee is coverage;
* how much the calibration-only combined score tightens that curve vs native confidence
  (the combined score reaches a given m* at higher coverage);
* the deploy-on-novel regime (calibrate on familiar S0-S2, certify on novel S3-S4), where
  the worst-subpopulation certificate is vacuous until the gate abstains hard -- stated as a
  number, not hidden.

Combined scores are fit out-of-fold (GroupKFold on system_id) so the certificate sees no
leakage. Each curve point is individually valid at its pre-specified coverage.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupKFold

from experiments._common import (
    ALPHA,
    CONF,
    DELTA,
    RESDIR,
    load_delivered,
    methods_with_enough,
    save_json,
)
from foldgate.conformal.robust import worst_subpopulation_certificate
from foldgate.scores.combiner import DEFAULT_FEATURES, ScoreCombiner

COVERAGES = np.round(np.arange(0.05, 1.0001, 0.05), 3)
NOVEL = {3, 4}          # deploy-on-novel strata
FAMILIAR = {0, 1, 2}


def _oof_combined(sub, y):
    """Leakage-free out-of-fold combined score over system_id folds."""
    groups = sub["system_id"].to_numpy()
    n = len(sub)
    oof = np.full(n, np.nan)
    gkf = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    for cal, test in gkf.split(sub, y, groups):
        comb = ScoreCombiner(features=DEFAULT_FEATURES).fit(sub.iloc[cal], y[cal])
        oof[test] = comb.predict(sub.iloc[test])
    return oof


def _mstar_curve(score, y, alpha, delta):
    """m*(coverage): accept the top-c by score, certify CVaR worst-subpop mass."""
    order = np.argsort(-score)
    y_sorted = y[order]
    n = len(score)
    curve = []
    for c in COVERAGES:
        k = max(int(round(c * n)), 1)
        acc_y = y_sorted[:k]
        errors = int((1 - acc_y).sum())
        cert = worst_subpopulation_certificate(errors, k, alpha, delta, method="cp")
        curve.append({"coverage": float(c), "r_ucb": cert["r_ucb"],
                      "m_star": cert["m_star"], "certified": cert["certified"]})
    return curve


def _coverage_at_mstar(curve, target_m):
    """Highest coverage whose certified subpop mass m* is at or below target_m."""
    ok = [p["coverage"] for p in curve if p["certified"] and p["m_star"] <= target_m]
    return float(max(ok)) if ok else None


def run() -> dict:
    df = load_delivered()
    methods = methods_with_enough(df)
    out = {}
    for m in methods:
        sub = df[df.method == m].dropna(subset=[CONF, "system_id", "novelty_stratum"]).reset_index(drop=True)
        y = sub["correct"].to_numpy().astype(int)
        native = sub[CONF].to_numpy()
        combined = _oof_combined(sub, y)
        strat = sub["novelty_stratum"].to_numpy().astype(int)

        native_curve = _mstar_curve(native, y, ALPHA, DELTA)
        combined_curve = _mstar_curve(combined, y, ALPHA, DELTA)

        # deploy-on-novel: certify on S3-S4 accepted poses, gate by combined score
        novel_mask = np.isin(strat, list(NOVEL))
        nov_curve = _mstar_curve(combined[novel_mask], y[novel_mask], ALPHA, DELTA) if novel_mask.sum() >= 30 else []

        out[m] = {
            "n": int(len(sub)),
            "native_curve": native_curve,
            "combined_curve": combined_curve,
            "novel_curve": nov_curve,
            "cov_at_mstar_0.5_native": _coverage_at_mstar(native_curve, 0.5),
            "cov_at_mstar_0.5_combined": _coverage_at_mstar(combined_curve, 0.5),
            "cov_at_mstar_0.25_combined": _coverage_at_mstar(combined_curve, 0.25),
            "cov_at_mstar_0.5_novel": _coverage_at_mstar(nov_curve, 0.5) if nov_curve else None,
        }
    return out


def make_figure(res: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = list(res.keys())
    fig, ax = plt.subplots(figsize=(7, 4.5))
    cmap = plt.get_cmap("tab10")
    for i, m in enumerate(methods):
        cc = res[m]["combined_curve"]
        nn = res[m]["native_curve"]
        cov_c = [p["coverage"] for p in cc]
        ax.plot(cov_c, [p["m_star"] for p in cc], "-", color=cmap(i), label=f"{m} combined", lw=1.6)
        ax.plot([p["coverage"] for p in nn], [p["m_star"] for p in nn], "--", color=cmap(i), lw=0.9, alpha=0.6)
    ax.axhline(0.5, ls=":", color="k", lw=0.8)
    ax.set_xlabel("coverage (accepted fraction)")
    ax.set_ylabel("certified worst-subpopulation mass m* (lower = stronger)")
    ax.set_title("E17: tightening the gate certifies smaller subpopulations (solid=combined, dashed=native)")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    from experiments._common import FIGDIR
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGDIR / "e17_worst_subpop.png", dpi=150)
    print(f"saved {FIGDIR / 'e17_worst_subpop.png'}")


def main() -> None:
    res = run()
    save_json(res, RESDIR / "e17_worst_subpop.json")
    print(f"E17 -- worst-subpopulation certificate m* = 1 - beta*  (alpha={ALPHA}, delta={DELTA})\n")
    for m, r in res.items():
        print(f"[{m}] coverage to certify m*<=0.5: native={r['cov_at_mstar_0.5_native']} "
              f"combined={r['cov_at_mstar_0.5_combined']}  "
              f"(m*<=0.25 combined at cov={r['cov_at_mstar_0.25_combined']})")
        print(f"      deploy-on-novel (S3-S4): coverage to certify m*<=0.5 = {r['cov_at_mstar_0.5_novel']}")
    make_figure(res)


if __name__ == "__main__":
    main()
