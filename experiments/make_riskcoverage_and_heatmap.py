"""Two audit-requested figures, both from committed data.

1. results/figures/risk_coverage.png -- the operating-characteristic figure a
   selective-prediction paper is expected to lead with: selective risk vs coverage for the
   native ranking-score gate, marginal and per ligand-novelty stratum, with the alpha line.
   The familiar strata drop below alpha at high coverage; the novel strata never do at
   usable coverage, which is the break as an operating curve.

2. results/figures/certificate_heatmap.png -- replaces the unreadable 25-card figure with a
   model x stratum verdict matrix (FEASIBLE / ABSTAIN-underpowered / ABSTAIN-infeasible),
   coverage annotated in each cell, hatching added so the three verdicts survive grayscale.

Grayscale-safe: verdict encoded by both color and hatch; the S4 column carries a distinct
hatch. Reads in print and in black-and-white.
"""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIGS = ROOT / "results" / "figures"
ALPHA = 0.20
GREEN, AMBER, RED, GREY, DARK = "#1a7f4b", "#b8860b", "#c1272d", "#8a8a8a", "#1a1a1a"
STRAT_COLORS = ["#2c6fbb", "#5aa0d6", "#8a8a8a", "#c1272d", "#7a1a1a"]

plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"]})


def risk_coverage():
    d = pd.read_csv(ROOT / "results" / "analysis_table.csv")
    d = d[d.method == "af3"].dropna(subset=["ranking_score", "correct", "novelty_stratum"])
    d["novelty_stratum"] = d["novelty_stratum"].astype(int)
    cov = np.linspace(0.05, 1.0, 40)

    def curve(g):
        g = g.sort_values("ranking_score", ascending=False)
        loss = 1 - g["correct"].to_numpy()
        n = len(g)
        return [loss[:max(1, int(round(c * n)))].mean() for c in cov]

    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.plot(cov, curve(d), color=DARK, lw=2.0, label="marginal", zorder=5)
    for s in range(4):
        g = d[d.novelty_stratum == s]
        if len(g) < 20:
            continue
        ax.plot(cov, curve(g), color=STRAT_COLORS[s], lw=1.3,
                label=f"$S_{s}$" + (" (familiar)" if s == 0 else " (novel)" if s == 3 else ""))
    ax.axhline(ALPHA, ls=(0, (4, 3)), lw=1.0, color=DARK)
    ax.text(1.0, ALPHA + 0.008, r"target $\alpha=0.20$", ha="right", va="bottom",
            fontsize=7, color=DARK)
    ax.set_xlabel("coverage (accepted fraction)", fontsize=8.5)
    ax.set_ylabel("selective risk (accepted-set error)", fontsize=8.5)
    ax.set_xlim(0.05, 1.0); ax.set_ylim(0, 0.6)
    ax.tick_params(labelsize=7.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(fontsize=6.8, frameon=False, loc="upper left", ncol=1, handlelength=1.4)
    ax.set_title("AF3 native-score risk-coverage curve", fontsize=9, color=DARK)
    fig.tight_layout(pad=0.4)
    fig.savefig(FIGS / "risk_coverage.png", dpi=600)
    plt.close(fig)
    print("wrote risk_coverage.png")


def heatmap():
    c = json.loads((ROOT / "results" / "certificate_cards.json").read_text())["cards"]
    models = ["af3", "boltz1", "boltz1x", "chai", "protenix"]
    mlabels = ["AF3", "Boltz-1", "Boltz-1x", "Chai-1", "Protenix"]
    verdict_style = {  # color, hatch
        "FEASIBLE": (GREEN, ""),
        "ABSTAIN-underpowered": (AMBER, "///"),
        "ABSTAIN-infeasible": (RED, "xxx"),
    }
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    for i, m in enumerate(models):
        for s in range(5):
            card = c[m][str(s)]
            v = card["verdict"]
            col, hatch = verdict_style.get(v, (GREY, ""))
            ax.add_patch(plt.Rectangle((s, i), 1, 1, facecolor=col, edgecolor="white",
                                       lw=1.2, hatch=hatch, alpha=0.9))
            cov = card.get("retained_coverage")
            acc = card.get("accepted_n_targets")
            if v == "FEASIBLE" and isinstance(cov, (int, float)):
                txt = f"{cov:.2f}\nn={acc}"
            elif v == "ABSTAIN-underpowered":
                txt = f"labels\nn={acc}" if acc else "labels"
            else:
                txt = "abandon"
            ax.text(s + 0.5, i + 0.5, txt, ha="center", va="center", fontsize=5.2,
                    color="white", weight="bold", linespacing=1.0)
    ax.set_xticks(np.arange(5) + 0.5)
    ax.set_xticklabels([f"$S_{s}$" for s in range(5)], fontsize=8)
    ax.set_yticks(np.arange(5) + 0.5)
    ax.set_yticklabels(mlabels, fontsize=8)
    ax.set_xlim(0, 5); ax.set_ylim(0, 5)
    ax.invert_yaxis()
    ax.set_xlabel("ligand novelty  (familiar $\\rightarrow$ novel)", fontsize=8.5)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    # legend
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=col, hatch=h, edgecolor="white")
               for (col, h) in verdict_style.values()]
    ax.legend(handles, ["FEASIBLE (coverage)", "ABSTAIN: collect labels", "ABSTAIN: infeasible"],
              fontsize=6.2, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=1)
    ax.set_title("Certificate verdict by model and novelty stratum (native score)",
                 fontsize=8, color=DARK)
    fig.tight_layout(pad=0.4)
    fig.savefig(FIGS / "certificate_heatmap.png", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print("wrote certificate_heatmap.png")


if __name__ == "__main__":
    risk_coverage()
    heatmap()
