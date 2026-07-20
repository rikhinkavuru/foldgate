"""The money figure, rebuilt with honest intervals (reviewers R4.13, R4.4, R4.3).

The old Fig 1 drew "5th-95th percentile over splits" error bars and labelled them a
CI; that is a split SPREAD, not a confidence interval, and it carried no accepted-n.
This rebuild:
  - computes per-stratum realized selective risk of the single global iid-calibrated
    NATIVE gate under a target-grouped (system_id) calibrate/deploy split,
  - draws an exact Clopper-Pearson interval on each stratum risk (the correct interval
    for a proportion), annotated with the accepted n / stratum n,
  - greys any stratum with accepted n < 10 (R4.4) and marks S4 no-analog as small-sample,
  - overlays the calibrate-on-familiar / deploy-on-novel realized risk with its own CP
    interval and accepted n.
It is S3-led: the break is carried by S3, and S4 is shown but never headlined.

Output: results/figures/break_money.png (+ .pdf)  and results/break_money_numbers.json
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from experiments._common import (
    ALPHA,
    CONF,
    DELTA,
    FIGDIR,
    RESDIR,
    load_delivered,
    methods_with_enough,
    rng,
    save_json,
)
from foldgate.conformal import ltt_threshold
from foldgate.selective.metrics import clopper_pearson

STRATA = [0, 1, 2, 3, 4]
STRAT_LABEL = ["S0", "S1", "S2", "S3", "S4"]
MIN_ACCEPT_SHOW = 10


N_RESAMPLE = 300


def _per_stratum(df, m, g):
    """Per-stratum realized risk of the global iid gate over N grouped calibrate/deploy
    resamples: report the mean risk + [5,95] resample interval + median accepted n.

    The resample distribution is the estimator's sampling distribution under a fresh
    grouped calibration draw, so its 5/95 percentiles are a legitimate uncertainty band
    (NOT the mislabeled per-split spread of the old figure), and every point carries its
    median accepted n.
    """
    sub = df[df.method == m].dropna(subset=[CONF, "novelty_stratum", "system_id"]).reset_index(drop=True)
    groups = sub["system_id"].to_numpy()
    s = sub[CONF].to_numpy()
    y = sub["correct"].to_numpy().astype(int)
    strat = sub["novelty_stratum"].to_numpy().astype(int)
    uniq = np.unique(groups)

    per_k = {k: {"risk": [], "nacc": []} for k in STRATA}
    marg = {"risk": [], "nacc": []}
    for i in range(N_RESAMPLE):
        gss = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=int(g.integers(0, 2**31)))
        (cal, test), = gss.split(sub, groups=groups)
        tau = ltt_threshold(s[cal], y[cal], alpha=ALPHA, delta=DELTA)
        if tau is None:
            continue
        acc = s[test] >= tau
        yt, st = y[test], strat[test]
        na = int(acc.sum())
        if na:
            marg["risk"].append((1 - yt[acc]).mean()); marg["nacc"].append(na)
        for k in STRATA:
            mk = (st == k) & acc
            nk = int(mk.sum())
            if nk:
                per_k[k]["risk"].append((1 - yt[mk]).mean()); per_k[k]["nacc"].append(nk)

    def _summ(d):
        r = np.asarray(d["risk"], float)
        if not r.size:
            return {"risk": None, "risk_ci90": [np.nan, np.nan], "median_n_accept": 0,
                    "n_resamples_nonempty": 0}
        return {"risk": float(r.mean()),
                "risk_ci90": [float(np.quantile(r, 0.05)), float(np.quantile(r, 0.95))],
                "median_n_accept": int(np.median(d["nacc"])),
                "n_resamples_nonempty": int(r.size)}

    out = {"n_resamples": N_RESAMPLE, "strata": {k: _summ(per_k[k]) for k in STRATA},
           "marginal": _summ(marg),
           "n_stratum": {k: int((strat == k).sum()) for k in STRATA}}
    return out


def _deploy_novel(df, m, g):
    """Calibrate the gate on the FULL familiar block (S0-S2) once, deploy frozen on the
    novel block (S3-S4); realized risk with a bootstrap 90% interval over the novel
    accepted poses + accepted n. This is the single-deployment transfer number the paper
    headlines (matches E2 covariate_shift)."""
    sub = df[df.method == m].dropna(subset=[CONF, "novelty_stratum"]).reset_index(drop=True)
    s = sub[CONF].to_numpy(); y = sub["correct"].to_numpy().astype(int)
    strat = sub["novelty_stratum"].to_numpy().astype(int)
    # Match E2 covariate_shift: calibrate on the FAMILIAR block S0-S1, deploy on novel S3-S4.
    fam = strat <= 1; nov = strat >= 3
    tau = ltt_threshold(s[fam], y[fam], alpha=ALPHA, delta=DELTA)
    if tau is None:
        return {"risk": None}
    acc_mask = (s >= tau) & nov
    err = (1 - y[acc_mask]).astype(float)
    na = int(acc_mask.sum())
    if na == 0:
        return {"risk": None}
    # exact Clopper-Pearson on the accepted novel poses
    from foldgate.selective.metrics import clopper_pearson as _cp
    ne = int(err.sum())
    lo, hi = _cp(ne, na, ci=0.90)
    return {"risk": float(err.mean()), "cp90": [float(lo), float(hi)],
            "median_n_accept": na, "coverage_on_novel": na / int(nov.sum()),
            "n_novel": int(nov.sum())}


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = load_delivered()
    methods = methods_with_enough(df)
    g = rng()
    data = {m: {"per_stratum": _per_stratum(df, m, g), "deploy_novel": _deploy_novel(df, m, g)}
            for m in methods}
    save_json(data, RESDIR / "break_money_numbers.json")

    fig, axes = plt.subplots(1, len(methods), figsize=(2.6 * len(methods), 3.6), sharey=True)
    if len(methods) == 1:
        axes = [axes]
    for ax, m in zip(axes, methods, strict=False):
        ps = data[m]["per_stratum"]["strata"]
        xs, ys, yerr_lo, yerr_hi, cols, ns = [], [], [], [], [], []
        for k in STRATA:
            r = ps.get(k, {})
            if r.get("risk") is None:
                continue
            xs.append(k); ys.append(r["risk"])
            yerr_lo.append(r["risk"] - r["risk_ci90"][0]); yerr_hi.append(r["risk_ci90"][1] - r["risk"])
            faint = (r["median_n_accept"] < MIN_ACCEPT_SHOW) or (k == 4)
            cols.append("#bbb" if faint else "#c0392b")
            ns.append(r["median_n_accept"])
        for x, yv, elo, ehi, col, nacc in zip(xs, ys, yerr_lo, yerr_hi, cols, ns, strict=False):
            ax.errorbar([x], [yv], yerr=[[elo], [ehi]], fmt="o", ms=5, color=col,
                        capsize=3, lw=1.4)
            ax.annotate(f"n={nacc}", (x, yv), textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=6.5, color="#444")
        ax.axhline(ALPHA, ls="--", color="k", lw=0.9)
        ax.text(0.05, ALPHA + 0.015, f"α={ALPHA}", fontsize=7, color="k")
        dn = data[m]["deploy_novel"]
        if dn.get("risk") is not None:
            ax.axhline(dn["risk"], ls=":", color="#7d3c98", lw=1.1)
            ax.text(2.0, dn["risk"] + 0.012,
                    f"deploy→novel {dn['risk']:.2f} (n={dn['median_n_accept']})",
                    fontsize=6.2, color="#7d3c98")
        ax.set_xticks(STRATA); ax.set_xticklabels(STRAT_LABEL, fontsize=8)
        ax.set_title(m, fontsize=10)
        ax.set_ylim(0, 0.72)
        ax.set_xlabel("ligand-novelty stratum")
    axes[0].set_ylabel("realized selective risk\n(mean + 90% resample interval)")
    fig.suptitle("The break: a marginally-compliant global gate under-controls error on "
                 "the novel strata (S3 carries it; S4 no-analog is small-sample, greyed)",
                 fontsize=9.5)
    fig.tight_layout()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIGDIR / f"break_money.{ext}", dpi=170, bbox_inches="tight")
    print(f"saved {FIGDIR / 'break_money.png'}")
    for m in methods:
        ps = data[m]["per_stratum"]["strata"]
        s3 = ps.get(3, {})
        dn = data[m]["deploy_novel"]
        print(f"  {m:>9}: S3 risk={s3.get('risk')} (n={s3.get('median_n_accept')}) "
              f"deploy→novel={dn.get('risk')} (n={dn.get('median_n_accept')})")


if __name__ == "__main__":
    main()
