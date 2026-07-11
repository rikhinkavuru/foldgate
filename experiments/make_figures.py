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
    fig.tight_layout(); fig.savefig(FIGDIR / "e2_all_models.png", dpi=150); plt.close(fig)


def fig_e4_aurc_bars():
    e4 = _load("e4_selective_utility.json")
    if not e4:
        return
    models = list(e4)
    x = np.arange(len(models))
    nat = [e4[m]["aurc_native"] for m in models]
    comb = [e4[m]["aurc_combined"] for m in models]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - 0.2, nat, 0.4, label="native confidence", color="#c44")
    ax.bar(x + 0.2, comb, 0.4, label="combined score", color="#48c")
    ax.set_xticks(x); ax.set_xticklabels(models, rotation=15)
    ax.set_ylabel("AURC (lower = better)")
    ax.set_title("E4: combined reliability score vs native confidence (AURC)")
    ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(FIGDIR / "e4_aurc_all_models.png", dpi=150); plt.close(fig)


def fig_e7_axes():
    e7 = _load("e7_shift_axes.json")
    if not e7:
        return
    axes_names = ["ligand_novelty", "pocket_novelty", "temporal"]
    fig, axs = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, an in zip(axs, axes_names):
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
    fig.tight_layout(); fig.savefig(FIGDIR / "e7_shift_axes.png", dpi=150); plt.close(fig)


def main():
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig_e2_all_models(); fig_e4_aurc_bars(); fig_e7_axes()
    print("wrote:", ", ".join(p.name for p in sorted(FIGDIR.glob("*.png"))))


if __name__ == "__main__":
    main()
