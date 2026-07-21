"""ACS TOC graphic, option E: real structures, matched confidence, opposite outcome.

Same idea as option D, drawn from deposited and predicted coordinates instead of
cartoons. AlphaFold3 reports the same confidence across its whole frozen score family
for both systems; one predicted pose sits on the crystal pose, the other is expelled
from the site. The only difference between them is ligand novelty.

Coordinates come from scripts/prep_toc_structures.py, which superposes each AF3
prediction onto its crystal frame on pocket residues and refuses to emit anything
unless the recomputed ligand RMSD reproduces the shipped BiSyRMSD label. Both
receptors are single-chain, so the protomer trap of the integrity appendix cannot
apply. Nothing here is drawn by hand.

Usage: .venv/bin/python scripts/make_toc_graphic_e.py
"""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyBboxPatch

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIGS = ROOT / "results" / "figures"
COORDS = ROOT / "results" / "toc_struct_coords.json"

GREEN, RED, GREY, DARK = "#1a7f4b", "#c1272d", "#8a8a8a", "#1a1a1a"
TRACE = "#9fb3c8"
BOND_MAX = 1.85     # heavy-atom bond cutoff, angstrom
CONTEXT_R = 12.0    # how much backbone trace to draw around the site
SURFACE_R = 12.0    # radius of the soft pocket-envelope impression
SURF = "#9fb3c8"

plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["Times New Roman", "DejaVu Serif"]})


def frame(lig_c, lig_p, ca):
    """Orthonormal view basis: put the crystal-to-predicted displacement in-plane."""
    c0 = lig_c.mean(0)
    d = lig_p.mean(0) - c0
    e1 = d / np.linalg.norm(d) if np.linalg.norm(d) > 1.0 else None
    if e1 is None:                       # co-located poses: use the ligand's long axis
        v = lig_c - c0
        e1 = np.linalg.svd(v - v.mean(0), full_matrices=False)[2][0]
    near = ca[np.linalg.norm(ca - c0, axis=1) < CONTEXT_R]
    spread = near - near.mean(0) if len(near) > 3 else lig_c - c0
    # second axis: the widest receptor direction orthogonal to e1
    proj = spread - np.outer(spread @ e1, e1)
    e2 = np.linalg.svd(proj - proj.mean(0), full_matrices=False)[2][0]
    e2 = e2 - (e2 @ e1) * e1
    e2 /= np.linalg.norm(e2)
    e3 = np.cross(e1, e2)
    return c0, np.vstack([e1, e2, e3])


def project(X, c0, B):
    Y = (X - c0) @ B.T
    return Y[:, :2], Y[:, 2]


def smooth(run, factor=6):
    """Catmull-Rom-ish resampling so the backbone reads as a ribbon, not a zigzag."""
    if len(run) < 4:
        return run
    t = np.arange(len(run))
    tt = np.linspace(0, len(run) - 1, (len(run) - 1) * factor + 1)
    return np.column_stack([np.interp(tt, t, run[:, k]) for k in range(run.shape[1])])


def draw_surface(ax, ca, c0, B, x0, y0, s):
    """Soft blob behind the trace so the site reads as an enclosed pocket.

    Overlapping translucent discs at CA positions approximate a molecular envelope.
    It is an impression of bulk, not a computed solvent-accessible surface, and the
    caption must not describe it as one.
    """
    dist = np.linalg.norm(ca - c0, axis=1)
    keep = dist < SURFACE_R
    if keep.sum() == 0:
        return
    xy, z = project(ca[keep], c0, B)
    zr = (z - z.min()) / max(np.ptp(z), 1e-6)
    fade = np.clip(1.0 - (dist[keep] - 4.0) / (SURFACE_R - 4.0), 0.12, 1.0)
    for size, alpha in ((104.0, 0.016), (76.0, 0.018), (54.0, 0.020),
                        (36.0, 0.022), (22.0, 0.024)):
        ax.scatter(x0 + xy[:, 0] * s, y0 + xy[:, 1] * s,
                   s=size * s / 0.040, c=SURF,
                   alpha=None, linewidths=0, zorder=1,
                   edgecolors="none",
                   facecolors=[(0.58, 0.68, 0.79, alpha * f * (0.45 + 0.55 * r))
                               for f, r in zip(fade, zr)])


def draw_trace(ax, ca, c0, B, x0, y0, s):
    """CA trace near the site, drawn as contiguous smoothed runs and depth-shaded."""
    keep = np.linalg.norm(ca - c0, axis=1) < CONTEXT_R
    xy, z = project(ca, c0, B)
    pts = np.column_stack([xy, z])

    # Split into contiguous in-view runs, so we draw strands rather than confetti.
    runs, cur = [], []
    for i in range(len(ca)):
        broken = i > 0 and np.linalg.norm(ca[i] - ca[i - 1]) > 4.5
        if keep[i] and not broken:
            cur.append(pts[i])
        else:
            if len(cur) >= 4:
                runs.append(np.array(cur))
            cur = [pts[i]] if keep[i] else []
    if len(cur) >= 4:
        runs.append(np.array(cur))

    segs, depths = [], []
    for run in runs:
        sm = smooth(run)
        for i in range(len(sm) - 1):
            segs.append([(x0 + sm[i, 0] * s, y0 + sm[i, 1] * s),
                         (x0 + sm[i + 1, 0] * s, y0 + sm[i + 1, 1] * s)])
            depths.append((sm[i, 2] + sm[i + 1, 2]) / 2)
    if not segs:
        return
    depths = np.array(depths)
    rank = (depths - depths.min()) / max(np.ptp(depths), 1e-6)
    lc = LineCollection(segs, colors=[TRACE] * len(segs),
                        linewidths=1.3 + 1.3 * rank, alpha=0.30 + 0.45 * rank,
                        capstyle="round", joinstyle="round", zorder=2)
    ax.add_collection(lc)


def draw_ligand(ax, L, c0, B, x0, y0, s, color, lw=1.05, dashed=False, alpha=1.0):
    xy, z = project(L, c0, B)
    order = np.argsort(z)
    d = np.linalg.norm(L[:, None, :] - L[None, :, :], axis=2)
    ls = (0, (1.5, 1.1)) if dashed else "-"
    for i in range(len(L)):
        for j in range(i + 1, len(L)):
            if d[i, j] <= BOND_MAX:
                ax.plot([x0 + xy[i, 0] * s, x0 + xy[j, 0] * s],
                        [y0 + xy[i, 1] * s, y0 + xy[j, 1] * s],
                        ls=ls, lw=lw, color=color, alpha=alpha,
                        solid_capstyle="round", zorder=6)
    ax.scatter(x0 + xy[order, 0] * s, y0 + xy[order, 1] * s,
               s=2.1, c=color, alpha=alpha, linewidths=0, zorder=7)


def panel(ax, data, x0, title, pdbid, ok, scale):
    color = GREEN if ok else RED
    ca = np.array(data["receptor_ca"])
    lig_c = np.array(data["ligand_crystal"])
    lig_p = np.array(data["ligand_pred"])
    c0, B = frame(lig_c, lig_p, ca)
    y0 = 0.92

    draw_surface(ax, ca, c0, B, x0, y0, scale)
    draw_trace(ax, ca, c0, B, x0, y0, scale)
    # Crystal pose sits underneath and slightly heavier, so its grey rim stays
    # visible even where the predicted pose lands on top of it.
    draw_ligand(ax, lig_c, c0, B, x0, y0, scale, GREY, lw=2.0, alpha=0.55)
    draw_ligand(ax, lig_p, c0, B, x0, y0, scale, color, lw=1.0)

    ax.text(x0, 1.37, title, ha="center", va="center", fontsize=6.3,
            color=DARK, weight="bold")
    ax.text(x0 - 0.66, 0.335, pdbid, ha="left", va="center", fontsize=5.2,
            color=GREY, family="monospace")
    mark(ax, x0 - 0.10, 0.340, ok, color)
    ax.text(x0 + 0.68, 0.335, f"{data['shipped_rmsd']:.2f} Å", ha="right",
            va="center", fontsize=7.2, color=color, weight="bold")


def mark(ax, x, y, ok, color, s=0.05):
    if ok:
        ax.plot([x - s, x - s * 0.25, x + s * 1.05], [y, y - s * 0.85, y + s * 0.95],
                lw=1.5, color=color, solid_capstyle="round",
                solid_joinstyle="round", zorder=8)
    else:
        for sx in (1, -1):
            ax.plot([x - s * sx, x + s * sx], [y - s, y + s], lw=1.5, color=color,
                    solid_capstyle="round", zorder=8)


def main():
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

    panel(ax, d["5sgt"], 0.84, "familiar ligand", "5sgt", True, meta["scale"])
    panel(ax, d["5sku"], 2.41, "novel chemotype", "5sku", False, meta["scale"])
    ax.plot([1.625, 1.625], [0.26, 1.44], lw=0.6, color=GREY, alpha=0.5, zorder=1)

    ax.text(1.625, 0.135, "Confidence cannot see novelty. foldgate gates on novelty",
            ha="center", va="center", fontsize=5.7, color=DARK)
    ax.text(1.625, 0.048, "instead, and abstains where no threshold can certify.",
            ha="center", va="center", fontsize=5.7, color=DARK)

    fig.savefig(FIGS / "toc_option_e.png", dpi=600)
    plt.close(fig)
    print(f"wrote results/figures/toc_option_e.png")


if __name__ == "__main__":
    main()
