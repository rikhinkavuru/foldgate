"""Figures for the two new additions: label-cost curve (Fig 4) and decision curve (Fig 5).

Reads results/e28_label_cost_curve.json and results/e30_decision_curve.json (no
recomputation). Deterministic.
"""

from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments._common import FIGDIR, RESDIR


def label_cost_figure():
    d = json.load(open(RESDIR / "e28_label_cost_curve.json"))
    af = d["curves_combined"]["af3"]
    fig, ax = plt.subplots(figsize=(4.7, 3.5))
    colors = {"S0": "#22aa77", "S1": "#3388cc", "S2": "#ee8800", "S3": "#cc3333"}
    for sk in ["S0", "S1", "S2", "S3"]:
        pts = af.get(sk, [])
        xs = [p["n_g"] for p in pts]
        ys = [p["median_cov"] for p in pts]
        lo = [p["median_cov"] - p["cov_lo"] for p in pts]
        hi = [p["cov_hi"] - p["median_cov"] for p in pts]
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="-o", ms=4, color=colors[sk],
                    label=sk, lw=1.5, capsize=2, elinewidth=0.8, alpha=0.9)
    ax.axhline(0.2, ls=":", color="0.5", lw=0.8)
    ax.text(6, 0.225, "usable (0.2)", fontsize=7, color="0.4")
    ax.set_xscale("log")
    ax.set_xlabel(r"in-stratum target labels $n_g$")
    ax.set_ylabel("certified coverage (median over 200 draws)")
    ax.set_title("Label-cost curve (AF3, combined score):\ncertified coverage bought per stratum by $n_g$ labels",
                 fontsize=9.5)
    ax.legend(title="ligand stratum", fontsize=8, loc="center left")
    ax.set_ylim(-0.03, 1.05)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIGDIR / f"label_cost.{ext}", dpi=170, bbox_inches="tight")
    print(f"saved {FIGDIR/'label_cost.png'}")


def decision_curve_figure():
    d = json.load(open(RESDIR / "e30_decision_curve.json"))
    af = d["per_model"]["af3"]
    nb = af["net_benefit"]
    lam = sorted(float(k) for k in nb["conformal"].keys())
    labels = {"conformal": ("#cc3333", "-", "conformal gate"),
              "iptm": ("#3388cc", "--", "fixed ipTM $\\geq$ 0.8"),
              "accept_all": ("#888888", ":", "accept-all"),
              "abstain_all": ("#000000", "-.", "abstain-all")}
    fig, ax = plt.subplots(figsize=(4.7, 3.5))
    for name, (col, ls, lab) in labels.items():
        ys = [nb[name][f"{x:g}"] if f"{x:g}" in nb[name] else nb[name].get(str(x)) for x in lam]
        ys = [nb[name].get(k) for k in sorted(nb[name].keys(), key=float)]
        xs = sorted(float(k) for k in nb[name].keys())
        ax.plot(xs, ys, ls, color=col, lw=1.7, label=lab)
    lo, hi = af["conformal_wins_lambda_range"]
    ax.axvspan(lo, hi, color="#cc3333", alpha=0.08)
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel(r"cost ratio $\lambda$ = cost(act on wrong) / cost(abstain on right)")
    ax.set_ylabel("net benefit per delivered pose")
    ax.set_title(f"Decision curve (AF3): the gate wins for "
                 f"$\\lambda\\in[{lo:g},{hi:g}]$,\ndominating a fixed-ipTM threshold everywhere",
                 fontsize=9.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIGDIR / f"decision_curve.{ext}", dpi=170, bbox_inches="tight")
    print(f"saved {FIGDIR/'decision_curve.png'}")


if __name__ == "__main__":
    label_cost_figure()
    decision_curve_figure()
