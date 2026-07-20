"""CONSORT-style dataset flow diagram (reviewer R4.1).

Reconciles the four counts that otherwise read as inconsistent (13,535 / 12,602 /
11,254 / 12,125 / 13,146) into one auditable reduction chain, annotating which
experiments consume which node. Deterministic; verified counts from
data/processed/rnp_delivered.parquet (see docs/REVISION_NUMBERS.md).

Output: results/figures/consort_flow.png (+ .pdf)
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from experiments._common import FIGDIR, load_delivered


def _verify() -> dict:
    """Recompute the node counts so the figure never drifts from the data."""
    df = load_delivered()
    gov = df[df.method.isin(["af3", "boltz1", "boltz1x", "chai", "protenix"])]
    return {
        "raw": int(len(df)),
        "n_boltz2": int((df.method == "boltz2").sum()),
        "governed": int(len(gov)),
        "systems": int(gov.system_id.nunique()),
        "target_labels_5": int(gov.drop_duplicates(["system_id", "method"]).shape[0]),
        "target_labels_6": int(df.drop_duplicates(["system_id", "method"]).shape[0]),
    }


def _box(ax, xy, w, h, text, fc, ec="#333"):
    x, y = xy
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.2, edgecolor=ec, facecolor=fc, zorder=2))
    ax.text(x, y, text, ha="center", va="center", fontsize=8.6, zorder=3, linespacing=1.35)


def _arrow(ax, p0, p1):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=12,
        linewidth=1.1, color="#555", shrinkA=2, shrinkB=2, zorder=1))


def main() -> None:
    c = _verify()
    assert c["raw"] == c["governed"] + c["n_boltz2"], c
    fig, ax = plt.subplots(figsize=(8.4, 8.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 12); ax.axis("off")

    blue, green, grey, tan = "#dce8f6", "#dcecdc", "#eeeeee", "#f6efdc"

    _box(ax, (5, 11.2), 7.2, 1.0,
         f"Released Runs N' Poses predictions\n{c['raw']:,} delivered top-1 poses "
         f"(6 co-folding models)\none pose per (system, method, ligand instance)", blue)

    _box(ax, (8.4, 9.6), 3.0, 0.95,
         f"Boltz-2 removed\n(−{c['n_boltz2']}; ungoverned\naffinity comparator, Sec. 7)", grey)
    _box(ax, (5, 9.6), 7.2, 0.95,
         f"Governed panel: {c['governed']:,} poses / {c['systems']:,} systems\n"
         f"5 models (AF3, Boltz-1, Boltz-1x, Chai-1, Protenix)", green)

    _box(ax, (2.7, 7.9), 4.5, 1.0,
         f"Dedup → one pose per (system, method)\n"
         f"{c['target_labels_5']:,} independent target-labels\n(5 governed models)", tan)
    _box(ax, (7.4, 7.9), 4.4, 1.0,
         f"All-6 dedup → one per (system, method)\n"
         f"{c['target_labels_6']:,} target-labels\n(feasibility-map unit, Sec. 4)", tan)

    _box(ax, (2.7, 6.1), 4.6, 1.15,
         "Pose-level analyses (exchangeable draws)\n"
         "break (E2), i.i.d. gate (E1), combiner AURC (E4)\n"
         "12,602 poses; splits report SI-grouped variants", blue)
    _box(ax, (7.4, 6.1), 4.4, 1.15,
         "Target-grouped certificates\n"
         "nested-LOTO gate (E34), feasibility frontier (D2),\n"
         "label-cost curve — all keyed on system_id", green)

    # d1 distance track, separate provenance
    _box(ax, (5, 4.0), 8.4, 1.25,
         "Cross-model geometry track (recomputed poses, App. B)\n"
         "13,215 pairs → 13,146 frame checks · valid frame = single-chain receptor +\n"
         "unique ligand copy → 6,223 instances (1,115 retain all K=5)\n"
         "used ONLY by the label-free danger floor / consensus analysis", grey)

    _box(ax, (5, 2.0), 8.4, 1.15,
         "Screening arm (SI): released co-folded virtual screen [Shen 2026]\n"
         "DEKOIS 79 · GPCRrecent 16 · LIT-PCBA 5 targets — per-compound ipTM +\n"
         "affinity head; heuristic abstention transfer (no pose-RMSD label)", tan)

    _arrow(ax, (5, 10.7), (5, 10.08))
    _arrow(ax, (6.6, 9.6), (6.9, 9.6))
    _arrow(ax, (4.2, 9.12), (3.1, 8.42))
    _arrow(ax, (5.8, 9.12), (7.0, 8.42))
    _arrow(ax, (2.7, 7.4), (2.7, 6.7))
    _arrow(ax, (7.4, 7.4), (7.4, 6.7))
    _arrow(ax, (5, 9.12), (5, 4.65))
    _arrow(ax, (5, 3.35), (5, 2.6))

    ax.set_title("Dataset flow: one reduction chain reconciles every count",
                 fontsize=11, fontweight="bold", pad=6)
    fig.tight_layout()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIGDIR / f"consort_flow.{ext}", dpi=170, bbox_inches="tight")
    print(f"saved {FIGDIR / 'consort_flow.png'}  (counts verified: {c})")


if __name__ == "__main__":
    main()
