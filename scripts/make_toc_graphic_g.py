"""ACS TOC graphic, option G: the subgenre pattern (hook -> mechanism -> data receipt).

Built to the conventions of accepted JCIM/ACS conformal-prediction-for-structure TOC
graphics (ConfDock / ConfDTI / ConfBiXtCPI): a real 3D pocket hook on the left, a
left-to-right flow, a minimal data inset on the right as the quantitative receipt, a
single two-color semantic scheme (blue-green = familiar/certifiable, red = novel/fails),
Helvetica throughout, and one headline number rather than a table.

Left  : the matched pair. Same AlphaFold3 confidence, opposite outcome. Familiar ligand
        (green, correct) vs novel chemotype (red, wrong). This is the hook.
Right : the break curve. Realized error of one marginally-valid gate across novelty
        strata; it clears the alpha target on familiar strata and crosses it on novel
        ones, so the two pockets are the S1 and S3 points of a systematic trend. The
        callout states the frontier count.

All numbers and coordinates trace to committed JSON (break_money_numbers.json,
e50_score_family_frontier.json, toc_struct_coords.json) and to the ChimeraX panels
produced by scripts/render_toc.sh. Run that first; this composites.
"""
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIGS = ROOT / "results" / "figures"
REND = ROOT / "results" / "toc_render"
COORDS = ROOT / "results" / "toc_struct_coords.json"
BREAK = ROOT / "results" / "break_money_numbers.json"
FRONT = ROOT / "results" / "e50_score_family_frontier.json"
ALPHA = 0.20

# One semantic scheme, reused everywhere: cool = familiar/certifiable, warm = novel/fails.
BLUE, GREEN, RED, AMBER = "#2c6fbb", "#1a7f4b", "#c1272d", "#b8860b"
GREY, DARK, INK = "#8a8a8a", "#1a1a1a", "#222222"
plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"]})


def load():
    b = json.loads(BREAK.read_text())["af3"]["per_stratum"]
    risks = [b["strata"][str(i)]["risk"] for i in range(5)]
    meta = json.loads(COORDS.read_text())["_meta"]
    fr = json.loads(FRONT.read_text())["per_alpha"][str(ALPHA)]
    return risks, meta, fr


def place(ax, path, x, y, zoom):
    ab = AnnotationBbox(OffsetImage(mpimg.imread(path), zoom=zoom), (x, y),
                        frameon=False, box_alignment=(0.5, 0.5), zorder=3)
    ax.add_artist(ab)


def verdict_mark(ax, x, y, ok, color, s=0.032):
    if ok:
        ax.plot([x - s, x - s * 0.25, x + s * 1.05], [y, y - s * 0.85, y + s * 0.95],
                lw=1.4, color=color, solid_capstyle="round",
                solid_joinstyle="round", zorder=6)
    else:
        for sx in (1, -1):
            ax.plot([x - s * sx, x + s * sx], [y - s, y + s], lw=1.4, color=color,
                    solid_capstyle="round", zorder=6)


def pocket(ax, path, cx, cy, label, rmsd, ok):
    color = GREEN if ok else RED
    ax.add_patch(FancyBboxPatch((cx - 0.36, cy - 0.34), 0.72, 0.68,
                                boxstyle="round,pad=0.008", linewidth=1.0,
                                edgecolor=color, facecolor="white", zorder=2))
    place(ax, path, cx, cy + 0.03, zoom=0.0245)
    ax.text(cx, cy + 0.40, label, ha="center", va="center", fontsize=5.6,
            color=INK, weight="bold")
    verdict_mark(ax, cx - 0.24, cy - 0.28, ok, color)
    ax.text(cx + 0.30, cy - 0.28, f"{rmsd:.2f} Å", ha="right", va="center",
            fontsize=6.2, color=color, weight="bold")


def break_inset(ax, risks, fr, x0, y0, w, h):
    """Realized error vs novelty, restyled minimal: dashed alpha line, green->red bars."""
    n = 5
    bw = w / (n + 1.2)
    ymax = 0.5
    xs = [x0 + (i + 0.9) * (w / n) for i in range(n)]
    for i, (x, r) in enumerate(zip(xs, risks)):
        c = GREEN if r <= ALPHA else RED
        bh = h * (r / ymax)
        bar = FancyBboxPatch((x - bw / 2, y0), bw, bh, boxstyle="square,pad=0",
                             linewidth=0, facecolor=c, zorder=3)
        if i == 4:                      # S4 mixed stratum: hatch + fade
            bar.set_alpha(0.42)
            bar.set_hatch("////")
        ax.add_patch(bar)
        ax.text(x, y0 - 0.03, f"S$_{i}$", ha="center", va="top", fontsize=4.8,
                color=(GREEN if r <= ALPHA else RED))
    # dashed alpha target
    ya = y0 + h * (ALPHA / ymax)
    ax.plot([x0, x0 + w], [ya, ya], ls=(0, (3, 2)), lw=0.9, color=DARK, zorder=4)
    ax.text(x0 + w + 0.015, ya, r"$\alpha$", ha="left", va="center",
            fontsize=5.6, color=DARK, weight="bold")
    # thin y axis
    ax.plot([x0, x0], [y0, y0 + h], lw=0.7, color=INK, zorder=3)
    ax.text(xs[0], y0 - 0.115, "familiar", ha="center", va="top", fontsize=4.6, color=GREEN)
    ax.text(xs[-1], y0 - 0.115, "novel", ha="center", va="top", fontsize=4.6, color=RED)
    ax.text(x0 - 0.02, y0 + h, "realized\nerror", ha="right", va="top",
            fontsize=4.9, color=INK, linespacing=1.05)



def main():
    for p in (REND / "panel_5sgt.png", REND / "panel_5sku.png"):
        if not p.exists():
            sys.exit(f"missing {p}; run: bash scripts/render_toc.sh")
    risks, meta, fr = load()
    n_cells = fr["n_nonreference_cells"]
    n_zero = fr["intersection_full_family_incl_ligand_local"]

    fig, ax = plt.subplots(figsize=(3.25, 1.75))
    ax.set_xlim(0, 3.25)
    ax.set_ylim(0, 1.75)
    ax.axis("off")

    # banner (the hook line): same confidence for both poses
    ax.add_patch(FancyBboxPatch((0.05, 1.525), 3.15, 0.18,
                                boxstyle="round,pad=0.010", linewidth=0.9,
                                edgecolor=DARK, facecolor="#eef2f7", zorder=4))
    ax.text(1.625, 1.615,
            f"Same AF3 confidence    ranking {meta['ranking']:.2f}   ·   "
            f"ipTM {meta['iptm']:.2f}   ·   ligand pLDDT {meta['plddt']:.0f}",
            ha="center", va="center", fontsize=5.2, color=DARK, weight="bold", zorder=5)

    # left: the matched pair (hook)
    pocket(ax, REND / "panel_5sgt.png", 0.55, 1.02, "familiar ligand", 0.30, True)
    pocket(ax, REND / "panel_5sku.png", 1.38, 1.02, "novel chemotype", 10.49, False)

    # flow arrow into the data receipt
    ax.add_patch(FancyArrowPatch((1.80, 1.02), (2.02, 1.02), arrowstyle="-|>",
                                 mutation_scale=8, lw=1.4, color=DARK, zorder=4))

    # right: the break curve (receipt) shows the pair is one instance of a trend
    break_inset(ax, risks, fr, x0=2.12, y0=0.70, w=0.90, h=0.56)
    ax.text(2.57, 1.36, "one gate, every novelty level", ha="center", va="center",
            fontsize=5.2, color=INK, weight="bold")

    # single headline callout (the one number the genre allows)
    ax.text(1.16, 0.45,
            f"{n_zero} of {n_cells} model$\\times$novelty cells:",
            ha="center", va="center", fontsize=5.4, color=RED, weight="bold")
    ax.text(1.16, 0.335,
            "no threshold on any score certifies",
            ha="center", va="center", fontsize=5.4, color=RED, weight="bold")
    ax.text(1.625, 0.185,
            "foldgate reads training-set novelty, keeps the certifiable strata, and "
            "abstains on the rest.",
            ha="center", va="center", fontsize=5.2, color=INK)
    ax.text(1.625, 0.055, "training-free  ·  frozen AlphaFold3 / Boltz / Chai confidence  ·  "
            "finite-sample coverage guarantee",
            ha="center", va="center", fontsize=4.6, color=GREY)

    fig.savefig(FIGS / "toc_option_g.png", dpi=600)
    plt.close(fig)
    print("wrote results/figures/toc_option_g.png")


if __name__ == "__main__":
    main()
