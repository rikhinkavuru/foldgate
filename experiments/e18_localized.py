"""E18 -- randomly-localized conformal keyed continuously on novelty.

E2 showed a single global threshold under-controls error on novel strata; E3
repaired it with a per-stratum (Mondrian) threshold. Mondrian holds one tau per
stratum and inherits the arbitrary stratum edges. This experiment slides the
threshold smoothly along the continuous novelty axis (``ligand_similarity``, max
Tanimoto to the training set) with randomly-localized conformal (Hore & Barber,
arXiv:2310.07850): draw a random reference point in a kernel around the query
novelty, reweight the calibration poses by that kernel, and grow the accept set
while the locally-weighted error stays <= alpha.

We compare, per novelty stratum, realized selective risk + coverage for three
gates over repeated splits:

  global      one LTT threshold for all poses (the E2 baseline)
  group       one LTT threshold per stratum (the E3 Mondrian baseline)
  localized   the randomly-localized threshold tau(novelty)

Because RLCP is a marginal-coverage statement, the localized gate's selective
risk is validated empirically here, not by a new theorem (the marginal-coverage
guarantee itself is checked in ``foldgate.conformal.localized._synthetic_...``).
We also report the effective kernel sample size per stratum, so the reader sees
where the kernel goes thin and the certificate is vacuous. The no-analog stratum
(S4) has no Tanimoto coordinate at all, so the localizer is undefined there and
the gate abstains by construction -- the honest outcome, not a failure to hide.
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
from foldgate.conformal import ltt_threshold
from foldgate.conformal.localized import (
    _kernel_weights,
    default_bandwidth,
    kish_ess,
    localized_threshold,
)

STRAT = "novelty_stratum"
LOCALIZER = "ligand_similarity"   # continuous novelty; higher = more familiar
KERNEL = "gaussian"
BW_SCALE = 1.0                    # Silverman bandwidth on the similarity axis
GRID = 48                        # novelty grid nodes for the smooth threshold
MIN_EFF = 40.0                   # min local Kish ESS to stand on a kernel
MIN_ACCEPT_EFF = 20.0            # min effective accepted mass to certify an accept set
MODES = ("global", "group", "localized")


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
        nov = sub[LOCALIZER].to_numpy()            # NaN on the no-analog stratum
        levels = sorted(np.unique(strat).tolist())
        n = len(sub)

        # [n_accept, n_error, n_test] per mode; localized also tracks uncertifiable + ESS
        acc = {mode: {k: [0, 0, 0] for k in levels} for mode in MODES}
        loc_uncert = {k: 0 for k in levels}
        loc_ess = {k: [0.0, 0] for k in levels}     # [sum_ess, count]
        tau_curve = np.zeros(GRID)                   # mean localized tau over repeats (for figure)
        tau_curve_grid = None
        tau_curve_n = np.zeros(GRID)

        for _ in range(n_repeats):
            perm = g.permutation(n)
            cal, test = perm[: n // 2], perm[n // 2:]

            # -- global + group (Mondrian), exactly as E2/E3 --
            tau_global = ltt_threshold(s[cal], y[cal], alpha=ALPHA, delta=DELTA)
            tau_group = {}
            for k in levels:
                ck = cal[strat[cal] == k]
                tau_group[k] = (
                    ltt_threshold(s[ck], y[ck], alpha=ALPHA, delta=DELTA)
                    if len(ck) >= 40 else None
                )

            # -- localized: build a smooth threshold on a novelty grid from calibration --
            cs = nov[cal]
            valid = np.isfinite(cs)
            grid = tau_grid = ess_grid = None
            if valid.sum() >= 2 * MIN_EFF:
                h = default_bandwidth(cs[valid], scale=BW_SCALE)
                grid = np.linspace(np.nanmin(cs[valid]), np.nanmax(cs[valid]), GRID)
                tau_grid = localized_threshold(
                    s[cal], y[cal], cs, grid, alpha=ALPHA, h=h, kernel=KERNEL,
                    generator=g, min_eff=MIN_EFF, min_accept_eff=MIN_ACCEPT_EFF,
                )
                ess_grid = np.array([
                    kish_ess(_kernel_weights(x, cs[valid], h, KERNEL)) for x in grid
                ])
                if tau_curve_grid is None:
                    tau_curve_grid = grid
                finite = np.isfinite(tau_grid)
                tau_curve[finite] += tau_grid[finite]
                tau_curve_n[finite] += 1

            for k in levels:
                tk = test[strat[test] == k]
                if len(tk) == 0:
                    continue
                sk, yk, xk = s[tk], y[tk], nov[tk]

                for mode in ("global", "group"):
                    tau = tau_global if mode == "global" else tau_group[k]
                    if tau is None:
                        acc[mode][k][2] += len(tk)
                        continue
                    a = sk >= tau
                    acc[mode][k][0] += int(a.sum())
                    acc[mode][k][1] += int((1 - yk[a]).sum())
                    acc[mode][k][2] += len(tk)

                # localized: map each test pose to the nearest grid node, accept iff s >= tau(x)
                acc["localized"][k][2] += len(tk)
                if grid is None:
                    loc_uncert[k] += len(tk)          # no localizer available this split
                    continue
                node = np.clip(
                    np.rint((xk - grid[0]) / (grid[-1] - grid[0]) * (GRID - 1)), 0, GRID - 1
                )
                undefined = ~np.isfinite(xk)
                node_i = np.where(undefined, 0, node).astype(int)
                tau_loc = np.where(undefined, np.nan, tau_grid[node_i])
                abstain = ~np.isfinite(tau_loc)       # undefined coord or uncertifiable node
                loc_uncert[k] += int(abstain.sum())
                a = (sk >= tau_loc) & ~abstain
                acc["localized"][k][0] += int(a.sum())
                acc["localized"][k][1] += int((1 - yk[a]).sum())
                # kernel ESS seen by the certifiable test poses in this stratum
                certifiable = ~undefined
                if certifiable.any():
                    loc_ess[k][0] += float(ess_grid[node_i[certifiable]].sum())
                    loc_ess[k][1] += int(certifiable.sum())

        strata_out = {}
        for k in levels:
            row = {}
            for mode in MODES:
                n_acc, n_err, n_tot = acc[mode][k]
                row[mode] = {
                    "selective_risk": (n_err / n_acc) if n_acc else float("nan"),
                    "coverage": (n_acc / n_tot) if n_tot else 0.0,
                }
            row["localized"]["abstain_rate"] = (
                loc_uncert[k] / acc["localized"][k][2] if acc["localized"][k][2] else float("nan")
            )
            row["localized"]["kernel_ess"] = (
                loc_ess[k][0] / loc_ess[k][1] if loc_ess[k][1] else float("nan")
            )
            row["mean_similarity"] = float(np.nanmean(nov[strat == k])) if np.isfinite(
                nov[strat == k]).any() else float("nan")
            strata_out[k] = row

        curve = None
        if tau_curve_grid is not None:
            with np.errstate(invalid="ignore"):
                mean_tau = np.where(tau_curve_n > 0, tau_curve / np.maximum(tau_curve_n, 1), np.nan)
            curve = {"similarity": tau_curve_grid.tolist(), "tau": mean_tau.tolist(),
                     "defined_frac": (tau_curve_n / n_repeats).tolist()}
        out[m] = {"n": n, "bandwidth": float(default_bandwidth(nov[np.isfinite(nov)], scale=BW_SCALE)),
                  "strata": strata_out, "localized_curve": curve}
    return out


def make_figure(res: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m = "af3" if "af3" in res else next(iter(res))
    strata = res[m]["strata"]
    ks = sorted(strata.keys())

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # -- left: per-stratum selective risk for the three gates --
    x = np.arange(len(ks))
    width = 0.26
    colors = {"global": "#c44", "group": "#48c", "localized": "#2a2"}
    labels = {"global": "global (E2)", "group": "group-conditional (E3)",
              "localized": "localized (E18)"}
    for i, mode in enumerate(MODES):
        vals = [strata[k][mode]["selective_risk"] for k in ks]
        ax.bar(x + (i - 1) * width, vals, width, label=labels[mode], color=colors[mode])
    ax.axhline(ALPHA, ls="--", color="k", lw=1)
    ax.text(-0.4, ALPHA + 0.01, f"target alpha = {ALPHA}", fontsize=9)
    for i, mode in enumerate(MODES):
        for xi, k in zip(x + (i - 1) * width, ks, strict=False):
            cov = strata[k][mode]["coverage"]
            ax.text(xi, 0.004, f"{cov:.2f}", ha="center", va="bottom", fontsize=6, rotation=90,
                    color="white")
    ax.set_xticks(x)
    ax.set_xticklabels([f"S{k}" for k in ks])
    ax.set_xlabel("ligand-novelty stratum (S0 familiar -> S4 no analog)")
    ax.set_ylabel("realized selective risk among accepted")
    ax.set_title(f"E18 ({m}): selective risk per stratum (coverage labelled in bars)")
    ax.legend(fontsize=8, loc="upper left")

    # -- right: the smoothly-varying localized threshold along novelty --
    curve = res[m]["localized_curve"]
    if curve is not None:
        sim = np.array(curve["similarity"])
        tau = np.array(curve["tau"], dtype=float)
        ax2.plot(sim, tau, "-o", ms=3, color="#2a2", label="localized tau(novelty)")
        for k in ks:
            ms = strata[k]["mean_similarity"]
            if np.isfinite(ms):
                ax2.axvline(ms, color="#999", ls=":", lw=0.8)
                ax2.text(ms, ax2.get_ylim()[1], f"S{k}", fontsize=7, ha="center", va="top",
                         color="#555")
        ax2.set_xlabel("ligand_similarity (higher = more familiar)")
        ax2.set_ylabel("accept threshold tau (ranking_score)")
        ax2.set_title("smooth threshold: stricter on novel ligands")
        ax2.invert_xaxis()   # novel (low similarity) on the right, matching the stratum order
        ax2.legend(fontsize=8)

    fig.tight_layout()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGDIR / "e18_localized.png", dpi=150)
    print(f"saved {FIGDIR / 'e18_localized.png'}")


def main() -> None:
    res = run()
    save_json(res, RESDIR / "e18_localized.json")
    print(f"E18 -- randomly-localized conformal  (alpha={ALPHA}, kernel={KERNEL})\n")
    for m, r in res.items():
        print(f"[{m}]  n={r['n']}  bandwidth={r['bandwidth']:.3f}")
        print(f"   {'S':>2} | {'glob risk':>9} {'cov':>4} | {'grp risk':>8} {'cov':>4} "
              f"| {'loc risk':>8} {'cov':>4} {'abst':>4} {'kESS':>5}")
        for k in sorted(r["strata"]):
            gl = r["strata"][k]["global"]
            gr = r["strata"][k]["group"]
            lo = r["strata"][k]["localized"]
            print(f"   S{k:>1} | {gl['selective_risk']:>9.3f} {gl['coverage']:>4.2f} "
                  f"| {gr['selective_risk']:>8.3f} {gr['coverage']:>4.2f} "
                  f"| {lo['selective_risk']:>8.3f} {lo['coverage']:>4.2f} "
                  f"{lo['abstain_rate']:>4.2f} {lo['kernel_ess']:>5.0f}")
        print()
    make_figure(res)


if __name__ == "__main__":
    main()
