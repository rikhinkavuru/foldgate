"""E2 -- the exchangeability break (the money figure).

Two complementary failures of a native, iid-calibrated selective-risk gate:

  (A) Conditional-coverage failure: a global RCPS threshold calibrated on a
      random split keeps MARGINAL risk <= alpha, but the risk is grossly uneven
      across novelty strata -- low on familiar ligands, far above alpha on novel
      ones. Marginal validity hides per-group under-control.

  (B) Covariate-shift failure: calibrate the threshold on low-novelty systems
      (what a developer near the training distribution would do) and deploy on
      high-novelty systems, and even the MARGINAL guarantee is violated.

Both motivate the shift-robust repair in E3.
"""

from __future__ import annotations

import numpy as np

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
from foldgate.conformal import rcps_threshold
from foldgate.selective import evaluate_gate

STRAT = "novelty_stratum"
LOW_STRATA = {0, 1}
HIGH_STRATA_MIN = 3   # strata >= this are "high novelty"


def conditional_break(sub, s, y, strat, g, n_repeats=300):
    """Per-stratum realized risk under a single global iid-calibrated tau."""
    n = len(sub)
    levels = sorted(np.unique(strat).tolist())
    acc_err = {k: 0 for k in levels}
    acc_n = {k: 0 for k in levels}
    tot_n = {k: int((strat == k).sum()) for k in levels}
    per_repeat_risk = {k: [] for k in levels}
    per_repeat_cov = {k: [] for k in levels}
    marg_risks = []

    for _ in range(n_repeats):
        perm = g.permutation(n)
        cal, test = perm[: n // 2], perm[n // 2:]
        tau = rcps_threshold(s[cal], y[cal], alpha=ALPHA, delta=DELTA)
        if tau is None:
            continue
        acc = s[test] >= tau
        marg = evaluate_gate(s[test], y[test], tau)
        if marg["n_accept"]:
            marg_risks.append(marg["selective_risk"])
        for k in levels:
            in_k = strat[test] == k
            n_test_k = int(in_k.sum())
            if not n_test_k:
                continue
            mk = acc & in_k
            nk = int(mk.sum())
            per_repeat_cov[k].append(nk / n_test_k)
            if nk:
                ek = int((1 - y[test][mk]).sum())
                acc_err[k] += ek
                acc_n[k] += nk
                per_repeat_risk[k].append(ek / nk)

    out = {}
    for k in levels:
        risk = acc_err[k] / acc_n[k] if acc_n[k] else float("nan")
        pr = np.array(per_repeat_risk[k], dtype=float)
        out[k] = {
            "n_stratum": tot_n[k],
            "pooled_selective_risk": float(risk),
            "risk_p05": float(np.nanpercentile(pr, 5)) if len(pr) else float("nan"),
            "risk_p95": float(np.nanpercentile(pr, 95)) if len(pr) else float("nan"),
            "mean_coverage": float(np.mean(per_repeat_cov[k])) if per_repeat_cov[k] else 0.0,
        }
    return out, float(np.nanmean(marg_risks))


def covariate_shift_break(sub, s, y, strat, g, n_repeats=300):
    """Calibrate on low-novelty, deploy on high-novelty; check the marginal guarantee."""
    low = np.where(np.isin(strat, list(LOW_STRATA)))[0]
    high = np.where(strat >= HIGH_STRATA_MIN)[0]
    if len(low) < 100 or len(high) < 50:
        return None
    risks, covs, held = [], [], []
    for _ in range(n_repeats):
        cal = g.choice(low, size=len(low) // 2, replace=False)
        tau = rcps_threshold(s[cal], y[cal], alpha=ALPHA, delta=DELTA)
        res = evaluate_gate(s[high], y[high], tau)
        if res["n_accept"] == 0:
            continue
        risks.append(res["selective_risk"])
        covs.append(res["coverage"])
        held.append(res["selective_risk"] <= ALPHA)
    if not risks:
        return {"n_low": int(len(low)), "n_high": int(len(high)),
                "mean_realized_risk_on_high": float("nan"),
                "mean_coverage_on_high": 0.0, "frac_splits_risk_le_alpha": float("nan"),
                "target_guarantee": 1 - DELTA}
    return {
        "n_low": int(len(low)), "n_high": int(len(high)),
        "mean_realized_risk_on_high": float(np.mean(risks)),
        "mean_coverage_on_high": float(np.mean(covs)),
        "frac_splits_risk_le_alpha": float(np.mean(held)),
        "target_guarantee": 1 - DELTA,
    }


def run(n_repeats: int = 300) -> dict:
    df = load_delivered()
    methods = methods_with_enough(df)
    g = rng()
    out = {}
    for m in methods:
        sub = df[df.method == m].dropna(subset=[CONF, STRAT]).reset_index(drop=True)
        s = sub[CONF].to_numpy()
        y = sub["correct"].to_numpy()
        strat = sub[STRAT].to_numpy().astype(int)
        cond, marg = conditional_break(sub, s, y, strat, g, n_repeats)
        shift = covariate_shift_break(sub, s, y, strat, g, n_repeats)
        out[m] = {"marginal_risk": marg, "conditional": cond, "covariate_shift": shift}
    return out


def make_figure(res: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = list(res.keys())
    fig, ax = plt.subplots(figsize=(8, 5))
    for m in methods:
        cond = res[m]["conditional"]
        ks = sorted(cond.keys())
        risks = [cond[k]["pooled_selective_risk"] for k in ks]
        ax.plot(ks, risks, marker="o", label=m)
    ax.axhline(ALPHA, ls="--", color="k", lw=1)
    ax.text(0, ALPHA + 0.01, f"target alpha = {ALPHA}", fontsize=9)
    ax.set_xlabel("ligand-novelty stratum  (0 = familiar,  top = no training analog)")
    ax.set_ylabel("realized selective risk among accepted")
    ax.set_title("E2: a global iid-calibrated gate under-controls error on novel ligands")
    ax.legend(fontsize=8)
    fig.tight_layout()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGDIR / "e2_exchangeability_break.png", dpi=150)
    print(f"saved {FIGDIR / 'e2_exchangeability_break.png'}")


def main() -> None:
    res = run()
    save_json(res, RESDIR / "e2_exchangeability_break.json")
    print(f"E2 -- exchangeability break  (alpha={ALPHA}, delta={DELTA})\n")
    for m, r in res.items():
        cond = r["conditional"]
        ks = sorted(cond.keys())
        print(f"[{m}]  marginal risk under global tau = {r['marginal_risk']:.3f}  (target <= {ALPHA})")
        print(f"    {'stratum':>7} {'n':>6} {'sel_risk':>9} {'[p05':>7} {'p95]':>7}")
        for k in ks:
            c = cond[k]
            print(f"    {k:>7} {c['n_stratum']:>6} {c['pooled_selective_risk']:>9.3f} "
                  f"{c['risk_p05']:>7.3f} {c['risk_p95']:>7.3f}")
        sh = r["covariate_shift"]
        if sh:
            print(f"    covariate-shift (cal=low-novelty -> deploy=high-novelty): "
                  f"risk_on_high={sh['mean_realized_risk_on_high']:.3f} (target<={ALPHA}), "
                  f"P(risk<=a)={sh['frac_splits_risk_le_alpha']:.2f} vs target {1-DELTA}")
        print()
    make_figure(res)


if __name__ == "__main__":
    main()
