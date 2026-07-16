"""Figures for D1 (the label-free danger floor) and D2 (the certification feasibility map).

D1's execution plan calls the floor-vs-realized-risk plot per novelty stratum "the single figure
that is the minimal viable result", and D2's calls the labels-to-certify curve its MVR. Both were
reported as numbers first; this draws them.

Panel A (D1). Certified floor against realized ensemble risk per ligand-novelty stratum, AF3's
accept region at 50% coverage. The floor is label-free; the realized risk is validation truth a
deployed caller never sees. The point is that the floor tracks novelty and stays under the truth,
and that it goes vacuous on S4.

Panel B (D1). The consensus blind spot, which is why the floor cannot audit the concept gap: the
consensus rate falls with novelty while the risk INSIDE consensus rises, so the error the floor is
blind to (their product) grows even as the consensus mass shrinks.

Panel C (D2). The feasibility frontier c*_g: the largest source coverage at which the deployed rule
still holds alpha. Zero means no operating point anywhere on the swept grid.

Panel D (D2). Labels-to-certify by certifier on the feasible cells, showing the exact binomial
dominating the variance-adaptive and Hoeffding-type bounds.

Reads the result JSONs only; no recomputation. Writes results/figures/d1_d2_summary.png.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RES = ROOT / "results"
FIG = RES / "figures" / "d1_d2_summary.png"
STRATA = [0, 1, 2, 3, 4]
COV = "0.5"
DEPLOYED = "af3"


def main() -> None:
    d1 = json.loads((RES / "d1_floor.json").read_text())
    d2 = json.loads((RES / "d2_feasibility_map.json").read_text())
    cert = json.loads((RES / "d2_certify.json").read_text())

    fig, ax = plt.subplots(2, 2, figsize=(11, 8))

    # --- A: floor vs realized risk -----------------------------------------------------------
    cells = d1["axes"]["ligand"]["deployed"][DEPLOYED]["strata"]
    floor, real, ns = [], [], []
    for g in STRATA:
        c = cells[str(g)]["cells"][COV]
        floor.append(c["floor_packing"])
        real.append(c["realized_R_bar"])
        ns.append(c["n"])
    x = np.arange(len(STRATA))
    a = ax[0, 0]
    a.bar(x - 0.2, real, 0.4, label=r"realized $\bar R$ (needs labels)", color="#c44e52")
    a.bar(x + 0.2, floor, 0.4, label="certified floor (label-free)", color="#4c72b0")
    for i, n in enumerate(ns):
        a.text(i, max(real[i], floor[i]) + 0.015, f"n={n}", ha="center", fontsize=8, color="#555")
    a.set_xticks(x)
    a.set_xticklabels([f"S{g}" for g in STRATA])
    a.set_xlabel("ligand-novelty stratum (S4 = no training analog)")
    a.set_ylabel("ensemble risk")
    a.set_title(f"A. D1: the floor tracks novelty and never crosses the truth\n"
                f"({DEPLOYED}, accept region at 50% coverage, K=5)", fontsize=10)
    a.legend(fontsize=8, loc="upper left")
    a.set_ylim(0, 0.55)
    a.annotate("vacuous:\nfloor = 0", xy=(4.2, 0.02), fontsize=8, color="#c44e52")

    # --- B: the consensus blind spot ----------------------------------------------------------
    cons, r_cons, hidden = [], [], []
    for g in STRATA:
        c = cells[str(g)]["cells"][COV]
        cons.append(c["rate_consensus"])
        rc = c["R_bar_on_consensus"]
        r_cons.append(rc)
        hidden.append(c["rate_consensus"] * rc if np.isfinite(rc) else np.nan)
    b = ax[0, 1]
    b.plot(x, cons, "o-", label="consensus rate (floor reads 0 here)", color="#4c72b0")
    b.plot(x, r_cons, "s-", label="risk INSIDE consensus", color="#c44e52")
    b.plot(x, hidden, "^--", label="error the floor cannot see (product)", color="#000000")
    b.set_xticks(x)
    b.set_xticklabels([f"S{g}" for g in STRATA])
    b.set_xlabel("ligand-novelty stratum")
    b.set_ylabel("rate")
    # The trend is read through S3. S4 is the no-analog bin at n=11 here and bounces, which is the
    # same thinness the text reports; the title must not claim a monotone shrink the plot denies.
    b.axvspan(3.5, 4.5, color="#eeeeee", zorder=0)
    b.text(4.0, 0.93, "S4: n=11,\nunresolved", ha="center", fontsize=7, color="#777")
    b.set_title("B. D1: why the escape is narrow. Through S3 the consensus mass\n"
                "shrinks while the error hidden inside it grows", fontsize=10)
    b.legend(fontsize=8, loc="center left")
    b.set_ylim(0, 1.0)

    # --- C: the D2 feasibility frontier -------------------------------------------------------
    c_ax = ax[1, 0]
    models = d2["methods"]
    width = 0.15
    for i, m in enumerate(models):
        s = d2["axes"]["ligand"]["models"][m]["alpha"]["0.2"]["strata"]
        vals = [s.get(str(g), {}).get("c_star_feasible", np.nan) for g in STRATA]
        c_ax.bar(np.arange(len(STRATA)) + (i - 2) * width, vals, width, label=m)
    c_ax.set_xticks(np.arange(len(STRATA)))
    c_ax.set_xticklabels([f"S{g}" for g in STRATA])
    c_ax.set_xlabel("ligand-novelty stratum")
    c_ax.set_ylabel(r"$c^\star$: largest coverage holding $\alpha$")
    c_ax.set_title(r"C. D2: the feasibility frontier collapses ($\alpha=0.20$)."
                   "\n" r"$c^\star=0$: no operating point anywhere on the grid", fontsize=10)
    c_ax.legend(fontsize=7, ncol=2)
    c_ax.set_ylim(0, 1.05)

    # --- D: labels to certify ------------------------------------------------------------------
    d_ax = ax[1, 1]
    order = ["hoeffding", "hb", "wsr", "binom"]
    nice = {"hoeffding": "Hoeffding\n(pure)", "hb": "Hoeffding-\nBentkus",
            "wsr": "WSR betting\n(variance-adaptive)", "binom": "exact\nbinomial"}
    med = cert["summary"]["median_labels_to_certify_where_it_fires"]
    ncert = cert["summary"]["n_cells_certified"]
    tot = cert["summary"]["n_feasible_cells_with_curves"]
    vals = [med[k] for k in order]
    cols = ["#999999", "#8172b2", "#dd8452", "#55a868"]
    bars = d_ax.bar(range(len(order)), vals, color=cols)
    for i, k in enumerate(order):
        d_ax.text(i, vals[i] + 2, f"{vals[i]:.0f}\n({ncert[k]}/{tot} cells)",
                  ha="center", fontsize=8)
    d_ax.set_xticks(range(len(order)))
    d_ax.set_xticklabels([nice[k] for k in order], fontsize=8)
    d_ax.set_ylabel("median independent target-labels to certify")
    d_ax.set_title("D. D2: the exact binomial dominates. Variance-adaptivity\n"
                   "beats pure Hoeffding but not a binomial term", fontsize=10)
    d_ax.set_ylim(0, max(vals) * 1.3)
    bars[3].set_edgecolor("black")
    bars[3].set_linewidth(1.5)

    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=160)
    print(f"wrote {FIG}")


if __name__ == "__main__":
    main()
