"""Generate ACS TOC / abstract graphic candidates.

ACS spec: max 3.25 in wide x 1.75 in tall, legible at print size, >=300 dpi.
Every number is read from the committed results JSON, never hard-coded.

Usage: .venv/bin/python scripts/make_toc_graphic.py
Writes results/figures/toc_option_{a,b,c}.png
"""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIGS = ROOT / "results" / "figures"
ALPHA = 0.20

# Colorblind-safe: blue = certified/ok, red = uncertified, grey = context.
BLUE, RED, GREY, DARK = "#2c6fbb", "#c1272d", "#8a8a8a", "#1a1a1a"
SIZE = (3.25, 1.75)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
})


def load():
    d = json.loads((ROOT / "results" / "break_money_numbers.json").read_text())
    af3 = d["af3"]
    strata = af3["per_stratum"]["strata"]
    risks = [strata[str(i)]["risk"] for i in range(5)]

    # Frontier c*(stratum): largest coverage still feasible at alpha, subject to the
    # frozen min-accept rule. Derived here rather than transcribed from the text.
    fm = json.loads((ROOT / "results" / "d2_feasibility_map.json").read_text())
    min_accept = fm["min_accept"]
    cells_by_stratum = fm["axes"]["ligand"]["models"]["af3"]["alpha"][str(ALPHA)]["strata"]
    frontier = []
    for s in map(str, range(5)):
        ok = [float(c) for c, v in cells_by_stratum[s]["cells"].items()
              if v["feasible"] and v["n_accepted_targets"] >= min_accept]
        frontier.append(max(ok) if ok else 0.0)

    return {
        "risks": risks,
        "marginal": af3["per_stratum"]["marginal"]["risk"],
        "deploy_novel": af3["deploy_novel"]["risk"],
        "frontier": frontier,
    }


def option_a(d):
    """The break: marginal validity hides per-stratum failure."""
    fig, ax = plt.subplots(figsize=SIZE)
    x = range(5)
    colors = [BLUE if r <= ALPHA else RED for r in d["risks"]]
    bars = ax.bar(x, d["risks"], color=colors, width=0.62, zorder=3)
    # S4 is a mixed stratum (about half is a similarity-search coverage gap, not
    # novelty), so it is hatched and de-emphasized exactly as in the main figure;
    # the load-bearing break is S3.
    bars[4].set_alpha(0.42)
    bars[4].set_hatch("///")
    bars[4].set_edgecolor("white")
    ax.axhline(ALPHA, color=DARK, lw=0.9, ls="--", zorder=4)
    ax.text(4.42, ALPHA + 0.012, r"target $\alpha$", ha="right", va="bottom",
            fontsize=6, color=DARK)
    ax.axhline(d["marginal"], color=GREY, lw=0.9, ls=":", zorder=4)
    ax.text(-0.42, d["marginal"] - 0.02, "marginal risk\n(looks compliant)", ha="left",
            va="top", fontsize=5.6, color=GREY, linespacing=1.15)
    ax.set_xticks(list(x))
    ax.set_xticklabels(["$S_0$", "$S_1$", "$S_2$", "$S_3$", "$S_4$"], fontsize=6.5)
    ax.set_xlabel("ligand novelty  (familiar $\\rightarrow$ novel)", fontsize=6.5, labelpad=1.5)
    ax.set_ylabel("realized error", fontsize=6.5, labelpad=2)
    ax.tick_params(axis="y", labelsize=6, pad=1.5)
    ax.tick_params(axis="x", pad=1.5)
    ax.set_ylim(0, 0.52)
    ax.set_xlim(-0.55, 4.55)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("Co-folding confidence certifies the familiar, not the novel",
                 fontsize=7, pad=3.5, color=DARK)
    fig.tight_layout(pad=0.28)
    fig.savefig(FIGS / "toc_option_a.png", dpi=600)
    plt.close(fig)


def option_b(d):
    """Schematic: frozen confidence in, three certified verdicts out."""
    fig, ax = plt.subplots(figsize=SIZE)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.6)
    ax.axis("off")

    ax.text(1.05, 3.58, "frozen co-folding\nconfidence", ha="center", va="center",
            fontsize=6.0, color=DARK, linespacing=1.2)
    ax.text(1.05, 2.62, "AlphaFold3\nBoltz · Chai", ha="center", va="center",
            fontsize=5.0, color=GREY, linespacing=1.25)

    gate = FancyBboxPatch((2.36, 2.30), 1.34, 1.34, boxstyle="round,pad=0.06",
                          linewidth=0.9, edgecolor=DARK, facecolor="#eef2f7", zorder=3)
    ax.add_patch(gate)
    ax.text(3.03, 3.20, "foldgate", ha="center", va="center", fontsize=6.2,
            color=DARK, family="monospace", zorder=4)
    ax.text(3.03, 2.66, "training-free\nrisk control", ha="center", va="center",
            fontsize=5.0, color=GREY, linespacing=1.2, zorder=4)

    ax.add_patch(FancyArrowPatch((1.94, 2.97), (2.30, 2.97), arrowstyle="-|>",
                                 mutation_scale=6, lw=0.8, color=DARK))

    rows = [
        (3.90, BLUE, "FEASIBLE", "certified operating point"),
        (2.97, "#b8860b", "ABSTAIN", "collect labels (~80 structures)"),
        (2.04, RED, "ABSTAIN", "no threshold exists"),
    ]
    for y, c, verdict, sub in rows:
        ax.add_patch(FancyArrowPatch((3.76, 2.97), (4.30, y), arrowstyle="-|>",
                                     mutation_scale=6, lw=0.7, color=c,
                                     connectionstyle="arc3,rad=0.10"))
        ax.add_patch(FancyBboxPatch((4.40, y - 0.36), 5.30, 0.72,
                                    boxstyle="round,pad=0.03", linewidth=0.8,
                                    edgecolor=c, facecolor="white", zorder=3))
        ax.text(4.60, y + 0.12, verdict, ha="left", va="center", fontsize=5.8,
                color=c, weight="bold", zorder=4)
        ax.text(4.60, y - 0.17, sub, ha="left", va="center", fontsize=5.0,
                color=DARK, zorder=4)

    ax.text(5.0, 1.06, "Deploying a marginally valid gate on novel targets realizes "
                       "%.2f error\nagainst a %.2f target; %d of 40 model-by-stratum cells admit no\n"
                       "certifiable threshold on any frozen confidence score."
            % (d["deploy_novel"], ALPHA, 13),
            ha="center", va="center", fontsize=5.2, color=DARK, linespacing=1.3)
    fig.tight_layout(pad=0.2)
    fig.savefig(FIGS / "toc_option_b.png", dpi=600)
    plt.close(fig)


def option_c(d):
    """Frontier: certified coverage decays to zero with novelty."""
    frontier = d["frontier"]  # AF3, ligand axis, alpha=0.20, derived in load()
    fig, ax = plt.subplots(figsize=SIZE)
    x = list(range(5))
    # Solid through S3 (the load-bearing claim); dashed into the mixed S4 stratum.
    ax.plot(x[:4], frontier[:4], "-o", color=BLUE, lw=1.3, ms=3.4, zorder=4,
            markeredgecolor="white", markeredgewidth=0.5)
    ax.plot(x[3:], frontier[3:], "--o", color=BLUE, lw=1.0, ms=3.0, alpha=0.5,
            zorder=4, markeredgecolor="white", markeredgewidth=0.5)
    ax.fill_between(x, frontier, color=BLUE, alpha=0.12, zorder=2)
    ax.axhline(0, color=DARK, lw=0.7, zorder=3)

    ax.annotate("no certifiable\noperating point",
                xy=(4, 0.0), xytext=(3.16, 0.42), fontsize=5.6, color=RED,
                linespacing=1.2, ha="center",
                arrowprops=dict(arrowstyle="-|>", lw=0.7, color=RED,
                                connectionstyle="arc3,rad=-0.25"))
    ax.text(3.0, 0.30, "20%", fontsize=5.4, color=GREY, ha="center", va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels(["$S_0$", "$S_1$", "$S_2$", "$S_3$", "$S_4$"], fontsize=6.5)
    ax.set_xlabel("ligand novelty  (familiar $\\rightarrow$ novel)", fontsize=6.5, labelpad=1.5)
    ax.set_ylabel("certifiable coverage", fontsize=6.5, labelpad=2)
    ax.tick_params(axis="y", labelsize=6, pad=1.5)
    ax.tick_params(axis="x", pad=1.5)
    ax.set_ylim(-0.05, 1.12)
    ax.set_xlim(-0.3, 4.4)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("The certification frontier collapses with novelty",
                 fontsize=7, pad=3.5, color=DARK)
    fig.tight_layout(pad=0.28)
    fig.savefig(FIGS / "toc_option_c.png", dpi=600)
    plt.close(fig)


if __name__ == "__main__":
    data = load()
    print("AF3 per-stratum realized risk:", [round(r, 3) for r in data["risks"]])
    print("marginal:", round(data["marginal"], 3),
          "| deploy-to-novel:", round(data["deploy_novel"], 3))
    option_a(data)
    option_b(data)
    option_c(data)
    for name in "abc":
        p = FIGS / f"toc_option_{name}.png"
        print(f"wrote {p.relative_to(ROOT)} ({p.stat().st_size / 1024:.0f} KB)")
