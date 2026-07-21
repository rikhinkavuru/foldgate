"""ACS TOC graphic, option D: the matched-pair before/after.

One idea: AlphaFold3 reports the SAME confidence, across every score in the frozen
family, for a pose that is right and a pose that is badly wrong. What separates them
is novelty, which the confidence cannot see. So the gate must key on novelty.

The pair is real and selected by an auditable rule (see pick_pair): among AF3 poses,
find a familiar/correct pose and a novel-chemotype/incorrect pose whose ranking score,
interface ipTM, and mean ligand pLDDT all agree. 2,552 such pairs exist, so the example
is representative rather than cherry-picked; we report that count on the graphic.

The confidence values, PDB ids, RMSDs, and pair count are all real, read from the
committed analysis table. The ligand cartoons are schematic: they are not the deposited
geometry, and the drawn displacement is exaggerated for legibility at 3.25 inches rather
than drawn to scale. Any caption accompanying this graphic must say so.

Usage: .venv/bin/python scripts/make_toc_graphic_d.py
"""
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, FancyBboxPatch, Wedge

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIGS = ROOT / "results" / "figures"

BLUE, RED, GREY, DARK = "#2c6fbb", "#c1272d", "#8a8a8a", "#1a1a1a"
GREEN = "#1a7f4b"
POCKET = "#d9dee5"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
})


def pick_pair():
    """Matched pair: same frozen confidence, opposite outcome, different novelty."""
    df = pd.read_csv(ROOT / "results" / "analysis_table.csv")
    d = df[df.method == "af3"].dropna(
        subset=["ranking_score", "rmsd", "iface_iptm", "ligand_plddt_mean"])
    fam = d[(d.novelty_stratum <= 1) & (d.correct == 1) & (d.rmsd < 1.5)]
    nov = d[(d.novelty_stratum == 3) & (d.correct == 0) & (d.rmsd > 4.0)]

    rows = []
    for _, n in nov.iterrows():
        m = fam[((fam.ranking_score - n.ranking_score).abs() < 0.015)
                & ((fam.iface_iptm - n.iface_iptm).abs() < 0.03)
                & ((fam.ligand_plddt_mean - n.ligand_plddt_mean).abs() < 4.0)]
        for _, f in m.iterrows():
            gap = (abs(f.ranking_score - n.ranking_score)
                   + abs(f.iface_iptm - n.iface_iptm)
                   + abs(f.ligand_plddt_mean - n.ligand_plddt_mean) / 100)
            rows.append((gap, f, n))
    rows.sort(key=lambda r: r[0])
    _, f, n = rows[0]
    return f, n, len(rows)


def molecule(ax, cx, cy, color, lw=0.9, scale=1.0, alpha=1.0, dashed=False):
    """Small schematic ligand: fused ring plus a short tail, legible at 3 mm."""
    ls = (0, (1.5, 1.1)) if dashed else "-"
    r = 0.105 * scale
    for dx in (-r * 0.86, r * 0.86):
        th = np.linspace(0, 2 * np.pi, 7)
        ax.plot(cx + dx + r * np.cos(th), cy + r * np.sin(th), ls=ls, lw=lw,
                color=color, alpha=alpha, solid_joinstyle="round", zorder=6)
    ax.plot([cx + r * 1.9, cx + r * 2.8], [cy + r * 0.5, cy - r * 0.2],
            ls=ls, lw=lw, color=color, alpha=alpha, zorder=6)
    ax.add_patch(Circle((cx + r * 2.8, cy - r * 0.2), r * 0.28, facecolor=color,
                        edgecolor="none", alpha=alpha, zorder=6))


def verdict_mark(ax, x, y, ok, color, s=0.052):
    """Draw the check / cross, since Times lacks those glyphs."""
    if ok:
        ax.plot([x - s, x - s * 0.25, x + s * 1.05],
                [y, y - s * 0.85, y + s * 0.95],
                lw=1.5, color=color, solid_capstyle="round",
                solid_joinstyle="round", zorder=7)
    else:
        for sx in (1, -1):
            ax.plot([x - s * sx, x + s * sx], [y - s, y + s], lw=1.5, color=color,
                    solid_capstyle="round", zorder=7)


def panel(ax, x0, title, pdbid, rmsd, ok, offset):
    """One before/after panel: pocket, crystal pose, predicted pose, verdict."""
    color = GREEN if ok else RED
    cy = 0.95
    # Pocket: an open concave cradle the ligand sits in.
    ax.add_patch(Wedge((x0, cy - 0.34), 0.60, 20, 160, width=0.15,
                       facecolor=POCKET, edgecolor="none", zorder=2))
    ax.text(x0, 1.32, title, ha="center", va="center", fontsize=6.3,
            color=DARK, weight="bold")

    molecule(ax, x0 - 0.10, cy, GREY, lw=0.75, dashed=True, alpha=0.95)  # crystal
    molecule(ax, x0 - 0.10 + offset, cy + offset * 0.22, color, lw=1.15)  # predicted

    ax.text(x0 - 0.64, 0.50, pdbid, ha="left", va="center",
            fontsize=5.3, color=GREY, family="monospace")
    # Mark sits well clear of the number: 10.49 must never read as 0.49.
    verdict_mark(ax, x0 - 0.08, 0.505, ok, color)
    ax.text(x0 + 0.66, 0.50, f"{rmsd:.2f} Å", ha="right", va="center",
            fontsize=7.4, color=color, weight="bold")


def main():
    f, n, n_pairs = pick_pair()
    print(f"familiar {f.entry_pdb_id}: rank={f.ranking_score:.3f} iptm={f.iface_iptm:.2f} "
          f"plddt={f.ligand_plddt_mean:.0f} rmsd={f.rmsd:.2f}")
    print(f"novel    {n.entry_pdb_id}: rank={n.ranking_score:.3f} iptm={n.iface_iptm:.2f} "
          f"plddt={n.ligand_plddt_mean:.0f} rmsd={n.rmsd:.2f}")
    print(f"matched pairs available: {n_pairs}")

    fig, ax = plt.subplots(figsize=(3.25, 1.75))
    ax.set_xlim(0, 3.25)
    ax.set_ylim(0, 1.75)
    ax.axis("off")

    # Shared confidence banner: the whole point is that this is ONE readout.
    ax.add_patch(FancyBboxPatch((0.07, 1.495), 3.11, 0.21,
                                boxstyle="round,pad=0.015", linewidth=0.9,
                                edgecolor=DARK, facecolor="#eef2f7", zorder=3))
    ax.text(1.625, 1.600,
            f"Same AlphaFold3 confidence:  ranking {n.ranking_score:.2f}   "
            f"ipTM {n.iface_iptm:.2f}   ligand pLDDT {n.ligand_plddt_mean:.0f}",
            ha="center", va="center", fontsize=5.5, color=DARK, zorder=4)

    panel(ax, 0.84, "familiar ligand", f.entry_pdb_id, f.rmsd, True, 0.035)
    panel(ax, 2.41, "novel chemotype", n.entry_pdb_id, n.rmsd, False, 0.34)

    ax.plot([1.625, 1.625], [0.40, 1.42], lw=0.6, color=GREY, alpha=0.5, zorder=1)

    ax.text(1.625, 0.245,
            "Confidence cannot see novelty. foldgate gates on novelty instead,",
            ha="center", va="center", fontsize=5.9, color=DARK)
    ax.text(1.625, 0.145,
            "and abstains where no threshold can certify.",
            ha="center", va="center", fontsize=5.9, color=DARK)
    ax.text(1.625, 0.045, f"{n_pairs:,} such matched pairs in the benchmark",
            ha="center", va="center", fontsize=4.8, color=GREY)

    fig.savefig(FIGS / "toc_option_d.png", dpi=600)
    plt.close(fig)
    p = FIGS / "toc_option_d.png"
    print(f"wrote {p.relative_to(ROOT)} ({p.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
