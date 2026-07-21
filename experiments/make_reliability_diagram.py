"""Reliability diagram (reviewer J): score vs empirical accuracy, per novelty stratum.

The most basic diagnostic behind the reliability-drift number D(nu): bin the frozen
score and plot empirical P(correct) per bin, one curve per ligand-novelty stratum. If
the curves separate, the same reported score means different accuracies on familiar vs
novel targets -- the score-conditional reliability drift, made visible. This is the
picture D(nu) summarizes.

Output: results/figures/reliability_diagram.png (+ .pdf)
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments._common import CONF, FIGDIR, load_delivered

MODELS = ["af3", "chai", "protenix"]
N_BINS = 8
STRATA = [0, 1, 2, 3]  # skip S4 (small-sample); S0..S3
COLORS = {0: "#2c7fb8", 1: "#7fcdbb", 2: "#fdae61", 3: "#d7191c"}
LABELS = {0: "$S_0$ familiar", 1: "$S_1$", 2: "$S_2$", 3: "$S_3$ novel"}


def _curve(s, y, edges):
    xs, ys, ns = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:], strict=False):
        m = (s >= lo) & (s < hi)
        if m.sum() < 10:
            continue
        xs.append(0.5 * (lo + hi))
        ys.append(float(y[m].mean()))
        ns.append(int(m.sum()))
    return np.array(xs), np.array(ys), np.array(ns)


def main() -> None:
    df = load_delivered()
    fig, axes = plt.subplots(1, len(MODELS), figsize=(3.4 * len(MODELS), 3.4), sharey=True)
    if len(MODELS) == 1:
        axes = [axes]
    for ax, m in zip(axes, MODELS, strict=False):
        sub = df[df.method == m].dropna(subset=[CONF, "correct", "novelty_stratum"])
        s_all = sub[CONF].to_numpy()
        # shared score bins (quantiles of the whole model) so strata are comparable
        edges = np.quantile(s_all, np.linspace(0, 1, N_BINS + 1))
        edges[0], edges[-1] = -np.inf, np.inf
        for k in STRATA:
            g = sub[sub.novelty_stratum == k]
            if len(g) < 40:
                continue
            xs, ys, ns = _curve(g[CONF].to_numpy(), g["correct"].to_numpy().astype(int), edges)
            if not len(xs):
                continue
            ax.plot(xs, ys, "-o", ms=3.5, color=COLORS[k], label=LABELS[k], lw=1.5)
        ax.axhline(0.8, ls=":", color="0.6", lw=0.8)
        ax.set_title(m, fontsize=10)
        ax.set_xlabel("ranking score (shared quantile bins)")
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("empirical P(correct)")
    axes[-1].legend(fontsize=7.5, title="ligand novelty", loc="lower right")
    fig.suptitle("Reliability diagram: at the same reported score, novel-stratum poses are less "
                 "often correct (the score-conditional drift D(ν) made visible)", fontsize=9.2)
    fig.tight_layout()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIGDIR / f"reliability_diagram.{ext}", dpi=170, bbox_inches="tight")
    print(f"saved {FIGDIR/'reliability_diagram.png'}")


if __name__ == "__main__":
    main()
