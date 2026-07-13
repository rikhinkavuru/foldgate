"""E16 -- selective virtual screening: does abstaining on unreliable co-folded poses
change enrichment, and does the reliability story matter under chemical-novelty shift?

This is the downstream decision number the RNP study could not produce. It reuses a
released co-folded screen (Shen et al., Chem Sci 2026, Zenodo 10.5281/zenodo.17568813):
per-compound Boltz-2 co-folding confidence (ipTM, affinity probability) plus active/decoy
labels for DEKOIS2.0 (79 targets), LIT-PCBA (5), and GPCRrecent (16 post-2022 novel
targets). No GPU, no crystal coordinates.

Three questions, each honest about what it shows:
1. Baseline. How well does a co-folding confidence rank actives vs a docking baseline (Gnina
   CNNscore)? EF@1% and BEDROC, aggregated over targets with a target-level bootstrap CI.
2. Selective screening. Rank by the Boltz-2 binder-affinity probability, then ABSTAIN on
   structurally unreliable poses (low ipTM), and measure EF on the retained library against a
   random-abstention control. The gate is a heuristic transfer, not a certified one: the
   foldgate guarantee is calibrated on pose correctness, which a decoy has no analogue of, so
   we report the enrichment lift without a coverage guarantee and pre-register the null.
3. Shift. Using the shipped active-to-training similarity tables, show enrichment degrades as
   actives become dissimilar to the training set, and GPCRrecent (novel targets) enriches worse
   than DEKOIS. This is the screening analogue of the E2 coverage collapse and is why a
   reliability layer is worth having.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

from experiments._common import RESDIR, rng, save_json
from foldgate.selective.enrichment import (
    active_retention_curve,
    bedroc,
    enrichment_factor,
    random_abstention_ef,
    roc_auc,
    selective_enrichment_curve,
)

SCREEN_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "external", "screening")
DATASETS = {"dekois": "dekois_scores", "lipcba": "lipcba_scores", "gpcr": "gpcr_scores"}
FRAC = 0.01


def _load_target(dset_dir: str, target: str) -> pd.DataFrame | None:
    """Per-compound Boltz-2 confidence + Gnina docking baseline + label, merged on compound id."""
    tdir = os.path.join(dset_dir, target)
    bpath = os.path.join(tdir, "boltz_scores.csv")
    if not os.path.exists(bpath):
        return None
    b = pd.read_csv(bpath)
    need = {"lid", "label", "iptm", "affinity_probability_binary"}
    if not need.issubset(b.columns):
        return None
    b = b[["lid", "label", "iptm", "affinity_probability_binary"]].dropna()
    b = b.groupby("lid", as_index=False).agg(
        label=("label", "max"),
        iptm=("iptm", "max"),
        affinity_probability_binary=("affinity_probability_binary", "max"),
    )
    gpath = os.path.join(tdir, "gnina_scores.csv")
    if os.path.exists(gpath):
        gd = pd.read_csv(gpath)
        idc = "ID" if "ID" in gd.columns else ("lid" if "lid" in gd.columns else None)
        if idc and "cnnscore" in gd.columns:
            gd = gd[[idc, "cnnscore"]].rename(columns={idc: "lid"}).groupby("lid", as_index=False).max()
            b = b.merge(gd, on="lid", how="left")
    if "cnnscore" not in b.columns:
        b["cnnscore"] = np.nan
    if b["label"].sum() < 3 or (b["label"] == 0).sum() < 20:
        return None
    return b


def _target_metrics(df: pd.DataFrame) -> dict:
    y = df["label"].to_numpy().astype(int)
    aff = df["affinity_probability_binary"].to_numpy()
    iptm = df["iptm"].to_numpy()
    dock = df["cnnscore"].to_numpy()
    row = {
        "n": len(df), "n_act": int(y.sum()),
        "ef_affinity": enrichment_factor(aff, y, FRAC),
        "ef_iptm": enrichment_factor(iptm, y, FRAC),
        "ef_dock": enrichment_factor(dock, y, FRAC) if np.isfinite(dock).any() else float("nan"),
        "bedroc_affinity": bedroc(aff, y),
        "bedroc_dock": bedroc(dock, y) if np.isfinite(dock).any() else float("nan"),
        "auc_affinity": roc_auc(aff, y),
    }
    # selective: rank by affinity, abstain by ipTM (pose reliability); EF at 50% coverage
    sel = selective_enrichment_curve(aff, iptm, y, coverages=(1.0, 0.75, 0.5), frac=FRAC)
    row["ef_full"] = sel[0]["ef_at_frac"]
    row["ef_sel50"] = sel[2]["ef_at_frac"]
    rmean, rlo, rhi = random_abstention_ef(aff, y, coverage=0.5, frac=FRAC)
    row["ef_rand50_mean"], row["ef_rand50_lo"], row["ef_rand50_hi"] = rmean, rlo, rhi
    ret = active_retention_curve(iptm, y, coverages=(0.5,))[0]
    row["active_retained_50"] = ret["active_retained"]
    row["decoy_retained_50"] = ret["decoy_retained"]
    return row


def _boot_median(vals, g, n_boot=2000):
    v = np.asarray([x for x in vals if np.isfinite(x)], dtype=float)
    if len(v) == 0:
        return {"median": float("nan"), "ci90": [float("nan"), float("nan")], "n": 0}
    meds = [np.median(v[g.integers(0, len(v), len(v))]) for _ in range(n_boot)]
    return {"median": float(np.median(v)),
            "ci90": [float(np.quantile(meds, 0.05)), float(np.quantile(meds, 0.95))], "n": int(len(v))}


def _shift_table(stat_dir: str, fname: str) -> list[dict]:
    """Mean Boltz-2 EF0.01 as a function of active-to-training similarity cutoff."""
    p = os.path.join(stat_dir, fname)
    if not os.path.exists(p):
        return []
    d = pd.read_csv(p)
    d = d[d["method"].astype(str).str.contains("Boltz", case=False, na=False)]
    if d.empty or "sim" not in d.columns or "EF0.01" not in d.columns:
        return []
    out = []
    for sim, grp in d.groupby("sim"):
        ef = grp["EF0.01"].replace([np.inf, -np.inf], np.nan).dropna()
        out.append({"sim_cutoff": float(sim), "mean_ef01": float(ef.mean()), "n_targets": int(len(ef))})
    return sorted(out, key=lambda r: r["sim_cutoff"])


def run() -> dict:
    g = rng()
    out = {"datasets": {}, "shift": {}}
    for name, sub in DATASETS.items():
        dset_dir = os.path.join(SCREEN_DIR, sub)
        if not os.path.isdir(dset_dir):
            continue
        targets = sorted(os.path.basename(t) for t in glob.glob(os.path.join(dset_dir, "*")) if os.path.isdir(t))
        rows = []
        for t in targets:
            df = _load_target(dset_dir, t)
            if df is not None:
                rows.append((t, _target_metrics(df)))
        if not rows:
            continue
        keys = ["ef_affinity", "ef_iptm", "ef_dock", "bedroc_affinity", "bedroc_dock", "auc_affinity",
                "ef_full", "ef_sel50", "ef_rand50_mean", "active_retained_50", "decoy_retained_50"]
        agg = {k: _boot_median([r[1][k] for r in rows], g) for k in keys}
        # how often does ipTM-gated selective EF beat the random-abstention 95th pct?
        beats = [1.0 if (np.isfinite(r[1]["ef_sel50"]) and r[1]["ef_sel50"] > r[1]["ef_rand50_hi"]) else 0.0
                 for r in rows]
        out["datasets"][name] = {
            "n_targets": len(rows),
            "agg": agg,
            "frac_targets_selEF_beats_random95": float(np.mean(beats)),
            "per_target": {t: m for t, m in rows},
        }
        stat_dir = os.path.join(SCREEN_DIR, sub.replace("_scores", "_stat"))
        out["shift"][name] = {
            "molecular_sim": _shift_table(stat_dir, "molecular_sim.csv"),
            "scaffold_sim": _shift_table(stat_dir, "scaffold_sim.csv"),
        }
    return out


def main() -> None:
    res = run()
    save_json(res, RESDIR / "e16_selective_screening.json")
    print("E16 -- selective virtual screening (Boltz-2 co-folding; Zenodo 17568813)\n")
    for name, r in res["datasets"].items():
        a = r["agg"]
        print(f"[{name}]  targets={r['n_targets']}")
        print(f"   EF@1% median: affinity={a['ef_affinity']['median']:.2f} "
              f"ipTM={a['ef_iptm']['median']:.2f} docking(Gnina)={a['ef_dock']['median']:.2f}")
        print(f"   BEDROC median: affinity={a['bedroc_affinity']['median']:.3f} "
              f"docking={a['bedroc_dock']['median']:.3f}"
              f"  AUROC affinity={a['auc_affinity']['median']:.3f}")
        print(f"   selective (rank=affinity, abstain=ipTM): EF full={a['ef_full']['median']:.2f} "
              f"-> EF@50%cov={a['ef_sel50']['median']:.2f} vs random={a['ef_rand50_mean']['median']:.2f}; "
              f"beats-random in {r['frac_targets_selEF_beats_random95']:.0%} of targets")
        print(f"   retention@50%: actives={a['active_retained_50']['median']:.2f} "
              f"decoys={a['decoy_retained_50']['median']:.2f}")
        ms = res["shift"][name]["molecular_sim"]
        if ms:
            trend = ", ".join(f"sim{r_['sim_cutoff']:.1f}:EF{r_['mean_ef01']:.1f}" for r_ in ms)
            print(f"   shift (mol-sim -> mean EF0.01): {trend}")
        print()


if __name__ == "__main__":
    main()
