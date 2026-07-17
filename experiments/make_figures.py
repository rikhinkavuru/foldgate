"""Consolidated multi-model figures for the paper, built from results/*.json."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments._common import ALPHA, FIGDIR, RESDIR


def _load(name):
    p = RESDIR / name
    return json.loads(p.read_text()) if p.exists() else None


def fig_e2_all_models():
    e2 = _load("e2_exchangeability_break.json")
    if not e2:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for m, r in e2.items():
        cond = r.get("conditional", {})
        ks = sorted(cond, key=int)
        ax.plot([int(k) for k in ks], [cond[k]["pooled_selective_risk"] for k in ks], marker="o", label=m)
    ax.axhline(ALPHA, ls="--", color="k", lw=1)
    ax.text(0, ALPHA + 0.01, f"alpha = {ALPHA}", fontsize=9)
    ax.set_xlabel("ligand-novelty stratum (0 familiar -> top no analog)")
    ax.set_ylabel("realized selective risk (global gate)")
    ax.set_title("E2: exchangeability break across all models")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGDIR / "e2_all_models.png", dpi=150)
    plt.close(fig)


def fig_e4_aurc_bars():
    e4 = _load("e4_selective_utility.json")
    if not e4:
        return
    models = [m for m in e4 if not m.startswith("_")]  # skip _joint aggregate
    x = np.arange(len(models))
    nat = [e4[m]["aurc_native"] for m in models]
    comb = [e4[m]["aurc_combined"] for m in models]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - 0.2, nat, 0.4, label="native confidence", color="#c44")
    ax.bar(x + 0.2, comb, 0.4, label="combined score", color="#48c")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15)
    ax.set_ylabel("AURC (lower = better)")
    ax.set_title("E4: combined reliability score vs native confidence (AURC)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGDIR / "e4_aurc_all_models.png", dpi=150)
    plt.close(fig)


def fig_e7_axes():
    e7 = _load("e7_shift_axes.json")
    if not e7:
        return
    axes_names = ["ligand_novelty", "pocket_novelty", "temporal"]
    fig, axs = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, an in zip(axs, axes_names, strict=False):
        for m, axd in e7.items():
            if an in axd:
                ks = sorted(axd[an], key=int)
                ax.plot([int(k) for k in ks], [axd[an][k]["risk"] for k in ks], marker="o", label=m)
        ax.axhline(ALPHA, ls="--", color="k", lw=1)
        ax.set_title(an.replace("_", " "))
        ax.set_xlabel("stratum (novel ->)")
    axs[0].set_ylabel("realized selective risk")
    axs[-1].legend(fontsize=7)
    fig.suptitle("E7: the break is structural/chemical novelty, not recency")
    fig.tight_layout()
    fig.savefig(FIGDIR / "e7_shift_axes.png", dpi=150)
    plt.close(fig)


# Okabe-Ito colorblind-safe palette + display names, shared across paper figures.
_MODEL_ORDER = ["af3", "boltz1", "boltz1x", "chai", "protenix"]
_MODEL_NAME = {"af3": "AF3", "boltz1": "Boltz-1", "boltz1x": "Boltz-1x",
               "chai": "Chai-1", "protenix": "Protenix"}
_MODEL_COLOR = {"af3": "#0072B2", "boltz1": "#E69F00", "boltz1x": "#009E73",
                "chai": "#CC79A7", "protenix": "#D55E00"}
# Below this accepted coverage a per-stratum "risk" is measured on a handful of
# poses; the group-conditional gate has effectively abstained (folded) there, so we
# mark abstention rather than plotting a noise-dominated risk point.
_ABSTAIN_COV = 0.02


def fig_e3_repair():
    """The repair: group-conditional calibration holds realized risk <= alpha where
    the global gate over-shot, at an honest (novelty-growing) coverage cost.

    Panel (a): AF3 mechanism -- global gate (native score, breaks on S3/S4) vs the
    full method (combined score + group-conditional), with per-stratum coverage
    annotated. Panel (b): the repaired gate across all five models, marker size
    proportional to retained coverage; where coverage vanishes the gate abstains.
    """
    e3 = _load("e3_shift_repair.json")          # global gate, native score (the break)
    e3c = _load("e3c_combined_conditional.json")  # group-conditional, full method (repair)
    if not e3 or not e3c:
        return
    strata = ["0", "1", "2", "3", "4"]
    xlabels = ["S0\nfamiliar", "S1", "S2", "S3", "S4\nno analog"]
    x = np.arange(len(strata))

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(11, 4.3))

    # -- Panel (a): AF3, global gate vs full-method repair --------------------
    g = e3["af3"]["strata"]
    glob_risk = [g[k]["global"]["selective_risk"] for k in strata]
    rep = e3c["af3"]
    rep_risk = [rep[k]["selective_risk"] for k in strata]
    rep_cov = [rep[k]["coverage"] for k in strata]
    axa.bar(x - 0.20, glob_risk, 0.38, color="#c44", label="global gate (native)")
    rep_plot = [r if (c >= _ABSTAIN_COV and np.isfinite(r)) else 0.0
                for r, c in zip(rep_risk, rep_cov, strict=True)]
    axa.bar(x + 0.20, rep_plot, 0.38, color=_MODEL_COLOR["af3"],
            label="group-conditional (full method)")
    for xi, (r, c) in enumerate(zip(rep_risk, rep_cov, strict=True)):
        if c >= _ABSTAIN_COV and np.isfinite(r):
            axa.annotate(f"{int(round(100 * c))}%", (xi + 0.20, r), textcoords="offset points",
                         xytext=(0, 3), ha="center", va="bottom", fontsize=8,
                         color=_MODEL_COLOR["af3"])
        else:
            axa.annotate("abstains", (xi + 0.20, 0.006), rotation=90, ha="center",
                         va="bottom", fontsize=8, color="0.35")
    axa.axhline(ALPHA, ls="--", color="k", lw=1)
    axa.text(4.35, ALPHA + 0.006, r"$\alpha=0.20$", ha="right", va="bottom", fontsize=9)
    axa.set_xticks(x)
    axa.set_xticklabels(xlabels, fontsize=8.5)
    axa.set_ylabel("realized selective risk\n(error among accepted)")
    axa.set_ylim(0, 0.46)
    axa.set_title("(a) AF3: the repair restores control", fontsize=10)
    axa.legend(fontsize=8, loc="upper left", frameon=False)
    axa.text(1.55, 0.435, "% = retained coverage", fontsize=8, color=_MODEL_COLOR["af3"])

    # -- Panel (b): five-model generality of the repaired gate -----------------
    smax = 320.0  # marker area at full coverage
    for m in _MODEL_ORDER:
        d = e3c[m]
        xs, ys, ss = [], [], []
        ax_lo, ax_ab = [], []
        for xi, k in enumerate(strata):
            r, c = d[k]["selective_risk"], d[k]["coverage"]
            xj = xi + (_MODEL_ORDER.index(m) - 2) * 0.09  # jitter models apart
            if c >= _ABSTAIN_COV and np.isfinite(r):
                xs.append(xj); ys.append(r); ss.append(30 + smax * c)
            else:
                ax_lo.append(xj); ax_ab.append(0.012)
        axb.scatter(xs, ys, s=ss, color=_MODEL_COLOR[m], alpha=0.85,
                    edgecolors="white", linewidths=0.6, label=_MODEL_NAME[m], zorder=3)
        axb.scatter(ax_lo, ax_ab, s=26, facecolors="none", edgecolors=_MODEL_COLOR[m],
                    linewidths=0.9, marker="v", zorder=2)
    axb.axhline(ALPHA, ls="--", color="k", lw=1)
    axb.text(4.4, ALPHA + 0.004, r"$\alpha=0.20$", ha="right", va="bottom", fontsize=9)
    axb.set_xticks(x)
    axb.set_xticklabels(xlabels, fontsize=8.5)
    axb.set_ylim(0, 0.30)
    axb.set_xlim(-0.5, 4.6)
    axb.set_ylabel("realized selective risk")
    axb.set_title("(b) held below $\\alpha$ across five models", fontsize=10)
    axb.legend(fontsize=7.5, loc="upper left", frameon=False, ncol=2, handletextpad=0.2)
    axb.text(4.45, 0.115, "marker area $\\propto$ retained coverage\n$\\triangledown$ = abstains",
             fontsize=7.5, color="0.4", ha="right", va="top")

    fig.tight_layout()
    fig.savefig(FIGDIR / "e3_repair.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig_e2_all_models()
    fig_e4_aurc_bars()
    fig_e7_axes()
    fig_e3_repair()
    print("wrote:", ", ".join(p.name for p in sorted(FIGDIR.glob("*.png"))))


if __name__ == "__main__":
    main()
