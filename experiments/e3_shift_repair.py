"""E3 -- shift-robust repair via group-conditional (Mondrian) conformal.

E2 showed a single global threshold under-controls error on novel strata. The
fix that needs only stratum labels (which RNP ships): calibrate a SEPARATE LTT
threshold per novelty stratum. By construction each stratum then satisfies the
error guarantee -- at an honest coverage cost on the novel strata (you abstain
more where the model is unreliable, which is the whole point).

We report, per stratum, realized selective risk + coverage for the global gate
(E2) vs the group-conditional gate (E3), pooled over repeated splits.
"""

from __future__ import annotations

import numpy as np

from experiments._common import ALPHA, CONF, DELTA, FIGDIR, RESDIR, load_delivered, methods_with_enough, rng, save_json
from foldgate.conformal import ltt_threshold
from foldgate.selective import evaluate_gate

STRAT = "novelty_stratum"


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
        levels = sorted(np.unique(strat).tolist())
        n = len(sub)

        # accumulators: pooled accepted-count / accepted-errors / stratum test count
        acc = {mode: {k: [0, 0, 0] for k in levels} for mode in ("global", "group")}
        for _ in range(n_repeats):
            perm = g.permutation(n)
            cal, test = perm[: n // 2], perm[n // 2:]
            tau_global = ltt_threshold(s[cal], y[cal], alpha=ALPHA, delta=DELTA)
            tau_group = {}
            for k in levels:
                ck = cal[strat[cal] == k]
                tau_group[k] = ltt_threshold(s[ck], y[ck], alpha=ALPHA, delta=DELTA) if len(ck) >= 40 else None

            for k in levels:
                tk = test[strat[test] == k]
                if len(tk) == 0:
                    continue
                sk, yk = s[tk], y[tk]
                for mode, tau in (("global", tau_global), ("group", tau_group[k])):
                    if tau is None:
                        acc[mode][k][2] += len(tk)
                        continue
                    a = sk >= tau
                    acc[mode][k][0] += int(a.sum())
                    acc[mode][k][1] += int((1 - yk[a]).sum())
                    acc[mode][k][2] += len(tk)

        strata_out = {}
        for k in levels:
            row = {}
            for mode in ("global", "group"):
                n_acc, n_err, n_tot = acc[mode][k]
                row[mode] = {
                    "selective_risk": (n_err / n_acc) if n_acc else float("nan"),
                    "coverage": (n_acc / n_tot) if n_tot else 0.0,
                }
            strata_out[k] = row
        out[m] = {"n": n, "strata": strata_out}
    return out


def make_figure(res: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m = "af3" if "af3" in res else next(iter(res))
    strata = res[m]["strata"]
    ks = sorted(strata.keys())
    glob = [strata[k]["global"]["selective_risk"] for k in ks]
    grp = [strata[k]["group"]["selective_risk"] for k in ks]
    grp_cov = [strata[k]["group"]["coverage"] for k in ks]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(ks))
    ax.bar(x - 0.2, glob, 0.4, label="global gate (E2)", color="#c44")
    ax.bar(x + 0.2, grp, 0.4, label="group-conditional gate (E3)", color="#48c")
    ax.axhline(ALPHA, ls="--", color="k", lw=1)
    ax.text(0, ALPHA + 0.01, f"target alpha = {ALPHA}", fontsize=9)
    for xi, cov in zip(x + 0.2, grp_cov):
        ax.text(xi, 0.005, f"cov\n{cov:.2f}", ha="center", va="bottom", fontsize=7, color="#246")
    ax.set_xticks(x)
    ax.set_xticklabels([f"S{k}" for k in ks])
    ax.set_xlabel("ligand-novelty stratum (S0 familiar -> top no analog)")
    ax.set_ylabel("realized selective risk among accepted")
    ax.set_title(f"E3 ({m}): group-conditional calibration restores error control on novel ligands")
    ax.legend(fontsize=8)
    fig.tight_layout()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGDIR / "e3_shift_repair.png", dpi=150)
    print(f"saved {FIGDIR / 'e3_shift_repair.png'}")


def main() -> None:
    res = run()
    save_json(res, RESDIR / "e3_shift_repair.json")
    print(f"E3 -- group-conditional repair  (alpha={ALPHA}, delta={DELTA})\n")
    for m, r in res.items():
        print(f"[{m}]  {'stratum':>7} | {'global risk':>11} {'cov':>5} | {'group risk':>10} {'cov':>5}")
        for k in sorted(r["strata"]):
            gl = r["strata"][k]["global"]; gr = r["strata"][k]["group"]
            print(f"          {k:>7} | {gl['selective_risk']:>11.3f} {gl['coverage']:>5.2f} "
                  f"| {gr['selective_risk']:>10.3f} {gr['coverage']:>5.2f}")
        print()
    make_figure(res)


if __name__ == "__main__":
    main()
