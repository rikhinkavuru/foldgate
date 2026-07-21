"""ACS TOC graphic, option F: composite the ChimeraX-rendered pocket panels.

Reads the two transparent panels produced by scripts/toc_chimerax.cxc and lays them
into the ACS layout with the shared-confidence banner, per-panel verdict, and caption.
The confidence values, PDB ids, and RMSDs are read from the committed coordinate JSON,
so the numbers stay bound to the validated data. The 3D panels are real superposed
structures rendered by ChimeraX; nothing here is drawn by hand.

Run scripts/render_toc.sh, which produces the panels then calls this. If the panels
are missing this exits with instructions rather than a stack trace.
"""
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIGS = ROOT / "results" / "figures"
REND = ROOT / "results" / "toc_render"
COORDS = ROOT / "results" / "toc_struct_coords.json"

GREEN, RED, GREY, DARK = "#1a7f4b", "#c1272d", "#8a8a8a", "#1a1a1a"
plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["Times New Roman", "DejaVu Serif"]})


def place(ax, img_path, x, y, zoom):
    img = mpimg.imread(img_path)
    ab = AnnotationBbox(OffsetImage(img, zoom=zoom), (x, y),
                        frameon=False, box_alignment=(0.5, 0.5))
    ax.add_artist(ab)


def mark(ax, x, y, ok, color, s=0.05):
    if ok:
        ax.plot([x - s, x - s * 0.25, x + s * 1.05], [y, y - s * 0.85, y + s * 0.95],
                lw=1.5, color=color, solid_capstyle="round",
                solid_joinstyle="round", zorder=8)
    else:
        for sx in (1, -1):
            ax.plot([x - s * sx, x + s * sx], [y - s, y + s], lw=1.5, color=color,
                    solid_capstyle="round", zorder=8)


def verdict(ax, x0, pdbid, rmsd, ok):
    color = GREEN if ok else RED
    ax.text(x0 - 0.66, 0.32, pdbid, ha="left", va="center", fontsize=5.2,
            color=GREY, family="monospace")
    mark(ax, x0 - 0.10, 0.325, ok, color)
    ax.text(x0 + 0.68, 0.32, f"{rmsd:.2f} Å", ha="right", va="center",
            fontsize=7.2, color=color, weight="bold")


def main():
    panels = [REND / "panel_5sgt.png", REND / "panel_5sku.png"]
    missing = [p for p in panels if not p.exists()]
    if missing:
        sys.exit("Rendered panels not found: " + ", ".join(str(p) for p in missing)
                 + "\nRun: bash scripts/render_toc.sh")

    d = json.loads(COORDS.read_text())
    meta = d["_meta"]

    fig, ax = plt.subplots(figsize=(3.25, 1.75))
    ax.set_xlim(0, 3.25)
    ax.set_ylim(0, 1.75)
    ax.axis("off")

    ax.add_patch(FancyBboxPatch((0.07, 1.495), 3.11, 0.21,
                                boxstyle="round,pad=0.015", linewidth=0.9,
                                edgecolor=DARK, facecolor="#eef2f7", zorder=3))
    ax.text(1.625, 1.600,
            f"Same AlphaFold3 confidence:  ranking {meta['ranking']:.2f}   "
            f"ipTM {meta['iptm']:.2f}   ligand pLDDT {meta['plddt']:.0f}",
            ha="center", va="center", fontsize=5.5, color=DARK, zorder=4)

    # Zoom is tuned to the 1400 px panels; adjust if the render size changes.
    place(ax, panels[0], 0.85, 0.90, zoom=0.083)
    place(ax, panels[1], 2.40, 0.90, zoom=0.083)

    ax.text(0.85, 1.37, "familiar ligand", ha="center", va="center",
            fontsize=6.3, color=DARK, weight="bold")
    ax.text(2.40, 1.37, "novel chemotype", ha="center", va="center",
            fontsize=6.3, color=DARK, weight="bold")
    ax.plot([1.625, 1.625], [0.24, 1.44], lw=0.6, color=GREY, alpha=0.5, zorder=1)

    verdict(ax, 0.85, "5sgt", d["5sgt"]["shipped_rmsd"], True)
    verdict(ax, 2.40, "5sku", d["5sku"]["shipped_rmsd"], False)

    ax.text(1.625, 0.135, "Confidence cannot see novelty. foldgate gates on novelty",
            ha="center", va="center", fontsize=5.7, color=DARK)
    ax.text(1.625, 0.048, "instead, and abstains where no threshold can certify.",
            ha="center", va="center", fontsize=5.7, color=DARK)

    fig.savefig(FIGS / "toc_option_f.png", dpi=600)
    plt.close(fig)
    print("wrote results/figures/toc_option_f.png")


if __name__ == "__main__":
    main()
