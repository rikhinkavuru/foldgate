"""ACS TOC graphic, option H: the three-act composition, real structures + real data.

Reproduces the strong SciDraw layout (PROBLEM -> BREAK -> REPAIR) in the reproducible
pipeline, so every pixel is data-traceable and the two hero pockets are the real
validated 5sgt / 5sku ChimeraX renders rather than AI-invented proteins.

PROBLEM  (left)   : the matched pair. Same AF3 confidence, opposite outcome.
BREAK    (center) : realized error rising across novelty strata past the alpha target,
                    with the accept/abstain gate as the mechanism.
REPAIR   (right)  : the certificate card. Frontier count (13/40), the group-conditional
                    recovery (52% of the moderate-novel stratum, combined score, per the
                    paper's coverage-map table), and the FEASIBLE/ABSTAIN legend.

Numbers trace to committed JSON: confidence + RMSD from toc_struct_coords.json, the
per-stratum error from break_money_numbers.json, the frontier count from
e50_score_family_frontier.json. The 52% recovery is the combined-score group-conditional
S1 coverage stated in the manuscript (Sec. repair / coverage-map table). Pockets are the
ChimeraX panels from scripts/render_toc.sh; run that first.
"""
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, RegularPolygon
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIGS = ROOT / "results" / "figures"
REND = ROOT / "results" / "toc_render"
COORDS = ROOT / "results" / "toc_struct_coords.json"
BREAK = ROOT / "results" / "break_money_numbers.json"
FRONT = ROOT / "results" / "e50_score_family_frontier.json"
ALPHA = 0.20
KEEP_PCT = 52          # combined-score group-conditional S1 coverage (paper coverage-map)

GREEN, RED, GREY, DARK, INK = "#1a7f4b", "#c1272d", "#8a8a8a", "#1a1a1a", "#222222"
CARD = "#fbfbfb"
plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"]})


def load():
    risks = [json.loads(BREAK.read_text())["af3"]["per_stratum"]["strata"][str(i)]["risk"]
             for i in range(5)]
    meta = json.loads(COORDS.read_text())["_meta"]
    fr = json.loads(FRONT.read_text())["per_alpha"][str(ALPHA)]
    return risks, meta, fr


def img(ax, path, x, y, zoom):
    ax.add_artist(AnnotationBbox(OffsetImage(mpimg.imread(path), zoom=zoom), (x, y),
                                 frameon=False, box_alignment=(0.5, 0.5), zorder=3))


def check(ax, x, y, color, s=0.028):
    ax.plot([x - s, x - s * 0.25, x + s * 1.05], [y, y - s * 0.85, y + s * 0.95],
            lw=1.3, color=color, solid_capstyle="round", solid_joinstyle="round", zorder=6)


def cross(ax, x, y, color, s=0.026):
    for sx in (1, -1):
        ax.plot([x - s * sx, x + s * sx], [y - s, y + s], lw=1.3, color=color,
                solid_capstyle="round", zorder=6)


def act_title(ax, x, t):
    ax.text(x, 1.40, t, ha="center", va="center", fontsize=7.2, color=DARK, weight="bold")


def pocket(ax, path, cx, cy, label, rmsd, ok, w=0.40):
    color = GREEN if ok else RED
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - w / 2), w, w,
                                boxstyle="round,pad=0.006", linewidth=1.0,
                                edgecolor=color, facecolor="white", zorder=2))
    img(ax, path, cx, cy, zoom=0.0142)
    ax.text(cx, cy + w / 2 + 0.055, label, ha="center", va="center", fontsize=4.7,
            color=INK, weight="bold")
    # verdict + RMSD on one line BELOW the box, so the mark never overlaps the number
    yb = cy - w / 2 - 0.075
    (check if ok else cross)(ax, cx - 0.13, yb, color)
    ax.text(cx + 0.02, yb, f"{rmsd:.2f} Å", ha="left", va="center",
            fontsize=5.2, color=color, weight="bold")


def break_chart(ax, risks, x0, y0, w, h):
    n, ymax = 5, 0.5
    bw = w / (n + 1.4)
    xs = [x0 + (i + 0.9) * (w / n) for i in range(n)]
    for i, (x, r) in enumerate(zip(xs, risks)):
        c = GREEN if r <= ALPHA else RED
        bar = FancyBboxPatch((x - bw / 2, y0), bw, h * (r / ymax), boxstyle="square,pad=0",
                             linewidth=0, facecolor=c, zorder=3)
        if i == 4:
            bar.set_alpha(0.42); bar.set_hatch("////")
        ax.add_patch(bar)
        ax.text(x, y0 - 0.035, f"S$_{i}$", ha="center", va="top", fontsize=4.6,
                color=(GREEN if r <= ALPHA else RED))
    ya = y0 + h * (ALPHA / ymax)
    ax.plot([x0, x0 + w], [ya, ya], ls=(0, (3, 2)), lw=0.9, color=DARK, zorder=4)
    ax.text(x0 + w + 0.012, ya, r"$\alpha$", ha="left", va="center", fontsize=5.4,
            color=DARK, weight="bold")
    ax.plot([x0, x0], [y0, y0 + h], lw=0.7, color=INK, zorder=3)
    ax.text(x0 - 0.045, y0 + h / 2, "realized error", ha="center", va="center",
            fontsize=4.4, color=INK, rotation=90)
    ax.text(xs[0], y0 - 0.115, "familiar", ha="center", va="top", fontsize=4.3, color=GREEN)
    ax.text(xs[-1], y0 - 0.115, "novel", ha="center", va="top", fontsize=4.3, color=RED)


def gate(ax, cx, cy):
    """Accept/abstain split: a node forking to a green check and a red stop octagon."""
    ax.add_patch(Circle((cx, cy), 0.013, facecolor=DARK, edgecolor="none", zorder=5))
    ax.add_patch(FancyArrowPatch((cx, cy), (cx - 0.17, cy - 0.15), arrowstyle="-|>",
                                 mutation_scale=6, lw=1.1, color=GREEN,
                                 connectionstyle="arc3,rad=0.15", zorder=4))
    ax.add_patch(FancyArrowPatch((cx, cy), (cx + 0.17, cy - 0.15), arrowstyle="-|>",
                                 mutation_scale=6, lw=1.1, color=RED,
                                 connectionstyle="arc3,rad=-0.15", zorder=4))
    # ACCEPT: bare check (a ringed check reads as a no-entry sign at this size)
    check(ax, cx - 0.20, cy - 0.205, GREEN, s=0.03)
    # ABSTAIN: red stop octagon
    ax.add_patch(RegularPolygon((cx + 0.20, cy - 0.205), numVertices=8, radius=0.042,
                                orientation=0.393, facecolor=RED, edgecolor=RED, zorder=5))
    ax.text(cx - 0.20, cy - 0.29, "ACCEPT", ha="center", va="top", fontsize=4.6,
            color=GREEN, weight="bold")
    ax.text(cx + 0.20, cy - 0.29, "ABSTAIN", ha="center", va="top", fontsize=4.6,
            color=RED, weight="bold")


def certificate(ax, x0, y0, w, h, n_zero, n_cells):
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0.012",
                                linewidth=1.0, edgecolor=GREY, facecolor=CARD, zorder=2))
    cx = x0 + w / 2
    ax.text(cx, y0 + h - 0.095, f"{n_zero} of {n_cells} cells:", ha="center", va="center",
            fontsize=5.6, color=RED, weight="bold", zorder=4)
    ax.text(cx, y0 + h - 0.195, "no threshold certifies", ha="center", va="center",
            fontsize=5.6, color=RED, weight="bold", zorder=4)
    ax.plot([x0 + 0.06, x0 + w - 0.06], [y0 + h - 0.265, y0 + h - 0.265],
            lw=0.5, color="#dddddd", zorder=3)
    ax.text(cx, y0 + h - 0.345, "group-conditional gate", ha="center", va="center",
            fontsize=5.0, color=GREEN, weight="bold", zorder=4)
    ax.text(cx, y0 + h - 0.435, f"keeps {KEEP_PCT}%, abstains on the rest",
            ha="center", va="center", fontsize=4.6, color=GREEN, zorder=4)
    # legend chips, sized to their text
    cw = 0.355
    for i, (lab, sub, col) in enumerate([("FEASIBLE", "certified", GREEN),
                                         ("ABSTAIN", "abstain", RED)]):
        ly = y0 + 0.185 - i * 0.115
        ax.add_patch(FancyBboxPatch((x0 + 0.10, ly - 0.032), cw, 0.064,
                                    boxstyle="round,pad=0.003", linewidth=0,
                                    facecolor=col, zorder=4))
        ax.text(x0 + 0.10 + cw / 2, ly, lab, ha="center", va="center", fontsize=4.0,
                color="white", weight="bold", zorder=5)
        ax.text(x0 + 0.10 + cw + 0.04, ly, sub, ha="left", va="center", fontsize=4.6,
                color=INK, zorder=5)


def main():
    for p in (REND / "panel_5sgt.png", REND / "panel_5sku.png"):
        if not p.exists():
            sys.exit(f"missing {p}; run: bash scripts/render_toc.sh")
    risks, meta, fr = load()

    fig, ax = plt.subplots(figsize=(3.25, 1.75))
    ax.set_xlim(0, 3.25); ax.set_ylim(0, 1.75); ax.axis("off")

    # banner
    ax.add_patch(FancyBboxPatch((0.05, 1.55), 3.15, 0.165, boxstyle="round,pad=0.008",
                                linewidth=0.9, edgecolor="#cccccc", facecolor="#f4f4f5", zorder=4))
    ax.text(1.625, 1.632,
            f"Same AF3 confidence:  ranking {meta['ranking']:.2f}   ·   "
            f"ipTM {meta['iptm']:.2f}   ·   pLDDT {meta['plddt']:.0f}",
            ha="center", va="center", fontsize=5.4, color=DARK, weight="bold", zorder=5)

    act_title(ax, 0.53, "PROBLEM")
    act_title(ax, 1.68, "BREAK")
    act_title(ax, 2.74, "REPAIR")

    # ACT 1 -- problem
    pocket(ax, REND / "panel_5sgt.png", 0.30, 0.88, "familiar", 0.30, True)
    pocket(ax, REND / "panel_5sku.png", 0.78, 0.88, "novel", 10.49, False)
    ax.add_patch(FancyArrowPatch((1.03, 0.88), (1.17, 0.88), arrowstyle="-|>",
                                 mutation_scale=7, lw=1.5, color=DARK, zorder=4))

    # ACT 2 -- break
    break_chart(ax, risks, x0=1.34, y0=0.86, w=0.62, h=0.40)
    gate(ax, 1.66, 0.62)
    ax.add_patch(FancyArrowPatch((2.10, 0.88), (2.24, 0.88), arrowstyle="-|>",
                                 mutation_scale=7, lw=1.5, color=DARK, zorder=4))

    # ACT 3 -- repair
    certificate(ax, 2.30, 0.40, 0.88, 0.86,
                fr["intersection_full_family_incl_ligand_local"], fr["n_nonreference_cells"])

    # footer
    ax.plot([0.15, 3.10], [0.16, 0.16], lw=0.5, color="#e2e2e2", zorder=1)
    ax.text(1.625, 0.085,
            "training-free   ·   frozen AlphaFold3 / Boltz / Chai confidence   ·   "
            "finite-sample coverage guarantee",
            ha="center", va="center", fontsize=4.6, color=GREY)

    fig.savefig(FIGS / "toc_option_h.png", dpi=600)
    plt.close(fig)
    print("wrote results/figures/toc_option_h.png")


if __name__ == "__main__":
    main()
