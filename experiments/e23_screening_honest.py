"""E23 -- honest screening ledger: separate the guaranteed object from the sold decision.

Two related audits, both recomputed from the RAW screen CSVs under
``data/external/screening/{dekois,lipcba,gpcr}_scores/<target>/`` (Shen et al., Chem Sci 2026,
Zenodo 10.5281/zenodo.17568813). No GPU, no crystal coordinates.

J2 (decision != guarantee). The conformal reliability layer is calibrated on pose correctness
(ligand-RMSD <= 2A). It therefore governs only the POSE-CONFIDENCE signals. We split every
screening signal into three ledgers and recompute enrichment for each, so the paper cannot sell
an enrichment number the guarantee does not cover:
  (A) guaranteed_layer  -- pose-confidence signals the conformal layer is calibrated on:
      Boltz-2 ipTM, Boltz-2 confidence_score (its ranking field), Protenix ranking_score and
      ipTM, and AF3 ranking_score where the screen ships it (DEKOIS only).
  (B) not_guaranteed.affinity_head -- Boltz-2 affinity_probability_binary. This is the binder
      head, NOT covered by the pose guarantee, and it is what carries the headline enrichment.
  (C) not_guaranteed.docking -- Gnina CNNscore and Glide, the external baselines to beat.
For each dataset we report EF@1%, BEDROC(alpha=80.5) and AUROC, one row per compound (best pose
by max over poses/seeds), top-k = round(0.01 * N), with target-level bootstrap 90% CIs. The
headline is the pose-confidence number (the guaranteed object) beside docking; the affinity head
is reported separately and labelled as outside the guarantee.

W3 (honest shift). Construction 2 of the shift curve: disjoint ec_sim bins of the actives held
against all decoys (ec_sim = ECFP Tanimoto of the active to the training set, shipped per active
in actives_similarity.csv). We report it for DEKOIS and GPCR, for both the pose-confidence signal
and the affinity head, with the bin edges stated explicitly. GPCR is the clean monotone
"novelty hurts" case. LIT-PCBA ships no actives_similarity.csv, so Construction 2 is not
computable there; for LIT-PCBA we inherit the molecular_sim / scaffold_sim curves from E16 and
note that both novelty axes agree that the most novel actives enrich worst.
"""

from __future__ import annotations

import glob
import json
import os
import subprocess

import numpy as np
import pandas as pd

from experiments._common import RESDIR, rng, save_json
from foldgate.selective.enrichment import bedroc, enrichment_factor, roc_auc

SCREEN_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "external", "screening")
DATASETS = {"dekois": "dekois_scores", "lipcba": "lipcba_scores", "gpcr": "gpcr_scores"}
FRAC = 0.01
BEDROC_ALPHA = 80.5
N_BOOT = 2000
SHIFT_BINS = [(-0.01, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.01)]  # disjoint ec_sim bins (lo, hi]

# Pose-confidence signals the conformal layer is calibrated on (label: guaranteed).
# (signal name -> (score file, column, higher_is_better))
POSE_SIGNALS = {
    "boltz_iptm": ("boltz_scores.csv", "iptm", True),
    "boltz_confidence_score": ("boltz_scores.csv", "confidence_score", True),
    "protenix_ranking_score": ("pix_scores.csv", "ranking_score", True),
    "protenix_iptm": ("pix_scores.csv", "iptm", True),
    "af3_ranking_score": ("af3_scores.csv", "ranking_score", True),  # DEKOIS only
}
# Affinity head -- NOT covered by the pose guarantee.
AFFINITY_SIGNAL = ("boltz_scores.csv", "affinity_probability_binary", True)
# Docking baselines. Glide differs per dataset; Glide energy is lower-is-better (we negate).
GNINA_SIGNAL = ("gnina_scores.csv", "cnnscore", True)
GLIDE_SIGNAL = {
    "dekois": ("af3_glide_scores.csv", "glide_min_SP_top1", False),
    "gpcr": ("glide_scores.csv", "score", False),
    "lipcba": ("glide_scores.csv", "score", False),
}

# Headline pose-confidence signal (the guaranteed object) and headline docking signal.
HEADLINE_POSE = "boltz_iptm"
HEADLINE_DOCK = "gnina_cnnscore"


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=os.path.dirname(__file__)
        ).decode().strip()
    except Exception:
        return "unknown"


def _load_signal(tdir: str, fname: str, col: str, higher_better: bool) -> pd.DataFrame | None:
    """One row per compound (best pose by max over poses/seeds) for a single score column.

    Returns columns ``lid, label, _val`` where ``_val`` is oriented so higher == more active
    (Glide-style energies are negated). Duplicate id/label columns (Protenix ships a second
    ``label`` header) are ignored; the score column is coerced to numeric and NaNs dropped.
    """
    path = os.path.join(tdir, fname)
    if not os.path.exists(path):
        return None
    d = pd.read_csv(path)
    idc = "lid" if "lid" in d.columns else ("ID" if "ID" in d.columns else None)
    if idc is None or "label" not in d.columns or col not in d.columns:
        return None
    d = d.loc[:, [idc, "label", col]].copy()
    d.columns = ["lid", "label", "_raw"]  # positional -> drops any duplicate label header
    d["_raw"] = pd.to_numeric(d["_raw"], errors="coerce")
    d = d.dropna(subset=["_raw"])
    d["_val"] = d["_raw"] if higher_better else -d["_raw"]
    return d.groupby("lid", as_index=False).agg(label=("label", "max"), _val=("_val", "max"))


def _metrics(g: pd.DataFrame | None) -> dict | None:
    """EF@1%, BEDROC and AUROC for one target, or None if the target is too thin."""
    if g is None:
        return None
    y = g["label"].to_numpy().astype(int)
    if y.sum() < 3 or (y == 0).sum() < 20:
        return None
    v = g["_val"].to_numpy()
    return {
        "ef": enrichment_factor(v, y, FRAC),
        "bedroc": bedroc(v, y, BEDROC_ALPHA),
        "auroc": roc_auc(v, y),
        "n": int(len(y)),
        "n_act": int(y.sum()),
    }


def _boot_median(vals, g, n_boot: int = N_BOOT) -> dict:
    """Target-level bootstrap of the median with a 90% CI."""
    v = np.asarray([x for x in vals if np.isfinite(x)], dtype=float)
    if len(v) == 0:
        return {"median": float("nan"), "ci90": [float("nan"), float("nan")], "n": 0}
    meds = [np.median(v[g.integers(0, len(v), len(v))]) for _ in range(n_boot)]
    return {
        "median": float(np.median(v)),
        "ci90": [float(np.quantile(meds, 0.05)), float(np.quantile(meds, 0.95))],
        "n": int(len(v)),
    }


def _agg_signal(dset_dir: str, targets: list[str], fname: str, col: str,
                higher_better: bool, g) -> dict:
    """Aggregate EF / BEDROC / AUROC across a dataset's targets for one signal."""
    rows = []
    for t in targets:
        m = _metrics(_load_signal(os.path.join(dset_dir, t), fname, col, higher_better))
        if m is not None:
            rows.append(m)
    if not rows:
        return {"not_computable": True,
                "reason": f"no usable '{col}' in '{fname}' for any target (file absent or thin)",
                "n_targets_scored": 0}
    return {
        "ef": _boot_median([r["ef"] for r in rows], g),
        "bedroc": _boot_median([r["bedroc"] for r in rows], g),
        "auroc": _boot_median([r["auroc"] for r in rows], g),
        "n_targets_scored": len(rows),
    }


def _shift_curve(dset_dir: str, targets: list[str], fname: str, col: str,
                 higher_better: bool, g) -> dict:
    """Construction 2: EF@1% within disjoint ec_sim bins of the actives, all decoys held fixed.

    Each active is assigned to a bin by its ec_sim (Tanimoto-to-train). Within a bin we keep that
    bin's actives plus every decoy, recompute EF@1% (top-k = round(0.01 * N_bin)), and aggregate
    the per-target EFs. Returns one entry per bin with the target-level bootstrap median CI.
    """
    acc = {f"({lo},{hi}]": [] for lo, hi in SHIFT_BINS}
    for t in targets:
        tdir = os.path.join(dset_dir, t)
        simp = os.path.join(tdir, "actives_similarity.csv")
        if not os.path.exists(simp):
            continue
        sim = pd.read_csv(simp)
        sidc = "ID" if "ID" in sim.columns else ("lid" if "lid" in sim.columns else None)
        if sidc is None or "ec_sim" not in sim.columns:
            continue
        d = _load_signal(tdir, fname, col, higher_better)
        if d is None:
            continue
        y = d["label"].to_numpy().astype(int)
        v = d["_val"].to_numpy()
        simmap = dict(zip(sim[sidc].astype(str), sim["ec_sim"], strict=False))
        asim = d["lid"].astype(str).map(simmap)
        for lo, hi in SHIFT_BINS:
            in_bin = (d["label"] == 1) & (asim > lo) & (asim <= hi)
            if in_bin.sum() < 2:
                continue
            mask = ((d["label"] == 0) | in_bin.to_numpy())
            acc[f"({lo},{hi}]"].append(enrichment_factor(v[mask], y[mask], FRAC))
    return {b: _boot_median(vs, g) for b, vs in acc.items()}


def _inherited_lipcba_shift() -> dict:
    """LIT-PCBA has no actives_similarity.csv, so echo the E16 molecular/scaffold curves."""
    p = RESDIR / "e16_selective_screening.json"
    if not p.exists():
        return {"not_computable": True, "reason": "results/e16_selective_screening.json absent"}
    e16 = json.loads(p.read_text())
    sh = e16.get("shift", {}).get("lipcba", {})
    return {
        "source": "inherited from results/e16_selective_screening.json (Boltz-2, cumulative "
                  "<=-threshold on active-to-train similarity; mean EF0.01 per cutoff)",
        "molecular_sim": sh.get("molecular_sim", []),
        "scaffold_sim": sh.get("scaffold_sim", []),
    }


def run() -> dict:
    g = rng()
    out = {
        "meta": {
            "git_sha": git_sha(),
            "seed": 20260710,
            "frac": FRAC,
            "bedroc_alpha": BEDROC_ALPHA,
            "n_boot": N_BOOT,
            "topk_rule": "round(0.01 * N) per target",
            "per_compound": "best pose by max over poses/seeds",
            "ci": "target-level bootstrap 90% CI (p05, p95)",
            "shift_bin_edges_ec_sim": [list(b) for b in SHIFT_BINS],
            "source": "Shen et al., Chem Sci 2026, Zenodo 10.5281/zenodo.17568813",
        },
        "guaranteed_layer": {
            "description": "Pose-confidence signals the conformal reliability layer is calibrated "
                           "on (pose-RMSD <= 2A correctness). These are the fields the guarantee "
                           "governs.",
            "by_dataset": {},
        },
        "not_guaranteed": {
            "description": "Signals the pose conformal guarantee does NOT cover.",
            "affinity_head": {
                "description": "Boltz-2 affinity_probability_binary (binder head). Carries the "
                               "headline enrichment but is OUTSIDE the pose guarantee.",
                "by_dataset": {},
            },
            "docking": {
                "description": "External docking baselines to beat (not a co-folding guarantee "
                               "object).",
                "by_dataset": {},
            },
        },
        "headline": {
            "description": "Guaranteed pose-confidence EF@1% (Boltz-2 ipTM) vs docking "
                           "(Gnina CNNscore), with the affinity head shown separately and "
                           "flagged as not covered by the pose guarantee.",
            "by_dataset": {},
        },
        "shift": {
            "construction": "Construction 2 -- disjoint ec_sim bins of actives, all decoys held "
                            "fixed; EF@1% recomputed per bin. Bin edges (lo, hi] on ec_sim: "
                            f"{[list(b) for b in SHIFT_BINS]}.",
            "by_dataset": {},
        },
    }

    for name, sub in DATASETS.items():
        dset_dir = os.path.join(SCREEN_DIR, sub)
        if not os.path.isdir(dset_dir):
            continue
        targets = sorted(os.path.basename(t) for t in glob.glob(os.path.join(dset_dir, "*"))
                         if os.path.isdir(t))

        # (A) guaranteed pose-confidence signals
        pose = {}
        for sig, (fname, col, hb) in POSE_SIGNALS.items():
            pose[sig] = _agg_signal(dset_dir, targets, fname, col, hb, g)
        out["guaranteed_layer"]["by_dataset"][name] = {"n_targets": len(targets), "signals": pose}

        # (B) affinity head
        af_fname, af_col, af_hb = AFFINITY_SIGNAL
        aff = _agg_signal(dset_dir, targets, af_fname, af_col, af_hb, g)
        out["not_guaranteed"]["affinity_head"]["by_dataset"][name] = aff

        # (C) docking baselines
        gn_fname, gn_col, gn_hb = GNINA_SIGNAL
        gl_fname, gl_col, gl_hb = GLIDE_SIGNAL[name]
        out["not_guaranteed"]["docking"]["by_dataset"][name] = {
            "gnina_cnnscore": _agg_signal(dset_dir, targets, gn_fname, gn_col, gn_hb, g),
            "glide": _agg_signal(dset_dir, targets, gl_fname, gl_col, gl_hb, g),
        }

        # headline row
        pose_ef = pose[HEADLINE_POSE].get("ef")
        dock_ef = out["not_guaranteed"]["docking"]["by_dataset"][name]["gnina_cnnscore"].get("ef")
        aff_ef = aff.get("ef")
        out["headline"]["by_dataset"][name] = {
            "pose_iptm_ef_GUARANTEED": pose_ef,
            "docking_gnina_ef": dock_ef,
            "affinity_head_ef_NOT_GUARANTEED": aff_ef,
        }

        # (W3) shift
        if name in ("dekois", "gpcr"):
            out["shift"]["by_dataset"][name] = {
                "pose_boltz_iptm": _shift_curve(dset_dir, targets, "boltz_scores.csv", "iptm",
                                                True, g),
                "affinity_head": _shift_curve(dset_dir, targets, "boltz_scores.csv",
                                              "affinity_probability_binary", True, g),
            }
        else:  # lipcba
            out["shift"]["by_dataset"][name] = {
                "construction2": {"not_computable": True,
                                  "reason": "no actives_similarity.csv (no per-active ec_sim) for "
                                            "LIT-PCBA; Construction 2 is empty"},
                "inherited_construction1": _inherited_lipcba_shift(),
            }
    return out


def _fmt(m) -> str:
    if not isinstance(m, dict) or "median" not in m:
        return "n/a"
    lo, hi = m["ci90"]
    return f"{m['median']:6.2f} [{lo:.2f},{hi:.2f}]"


def main() -> None:
    res = run()
    save_json(res, RESDIR / "e23_screening_honest.json")

    print("E23 -- honest screening ledger (guaranteed pose-confidence vs sold decision)")
    print(f"git {res['meta']['git_sha'][:10]}  EF@1% (top-k=round(0.01N)), BEDROC(a=80.5), AUROC; "
          "target-level bootstrap 90% CI\n")

    print("HEADLINE  EF@1%   pose-ipTM (GUARANTEED) | docking Gnina | affinity head (NOT guaranteed)")
    for ds, r in res["headline"]["by_dataset"].items():
        print(f"  {ds:7s}  {_fmt(r['pose_iptm_ef_GUARANTEED'])} | {_fmt(r['docking_gnina_ef'])} "
              f"| {_fmt(r['affinity_head_ef_NOT_GUARANTEED'])}")

    print("\nGUARANTEED LAYER -- pose-confidence signals (EF@1% median [ci90])")
    for ds, r in res["guaranteed_layer"]["by_dataset"].items():
        print(f"  [{ds}] n_targets={r['n_targets']}")
        for sig, m in r["signals"].items():
            if m.get("not_computable"):
                print(f"     {sig:24s} NOT COMPUTABLE ({m['reason']})")
            else:
                print(f"     {sig:24s} EF {_fmt(m['ef'])}  BEDROC {m['bedroc']['median']:.3f}  "
                      f"AUROC {m['auroc']['median']:.3f}  (n={m['n_targets_scored']})")

    print("\nNOT GUARANTEED -- affinity head + docking (EF@1% median [ci90])")
    for ds in res["headline"]["by_dataset"]:
        aff = res["not_guaranteed"]["affinity_head"]["by_dataset"][ds]
        dock = res["not_guaranteed"]["docking"]["by_dataset"][ds]
        gl = dock["glide"]
        gl_s = _fmt(gl["ef"]) if not gl.get("not_computable") else "n/a"
        print(f"  [{ds}] affinity {_fmt(aff['ef'])} | gnina {_fmt(dock['gnina_cnnscore']['ef'])} "
              f"| glide {gl_s}")

    print("\nW3 SHIFT -- Construction 2, disjoint ec_sim bins (lo,hi], median EF@1% [ci90], n_targets")
    for ds, r in res["shift"]["by_dataset"].items():
        if ds == "lipcba":
            inh = r["inherited_construction1"]
            print(f"  [{ds}] Construction 2 NOT COMPUTABLE (no actives_similarity.csv).")
            if not inh.get("not_computable"):
                ms = ", ".join(f"{x['sim_cutoff']:.1f}:{x['mean_ef01']:.1f}" for x in inh["molecular_sim"])
                sc = ", ".join(f"{x['sim_cutoff']:.1f}:{x['mean_ef01']:.1f}" for x in inh["scaffold_sim"])
                print(f"        inherited molecular_sim (cutoff:meanEF): {ms}")
                print(f"        inherited scaffold_sim  (cutoff:meanEF): {sc}")
                print("        both axes agree: most novel actives enrich worst.")
            continue
        print(f"  [{ds}]")
        for ranker in ("pose_boltz_iptm", "affinity_head"):
            trend = ", ".join(f"{b}:{m['median']:.1f}(n{m['n']})" for b, m in r[ranker].items())
            print(f"     {ranker:16s} {trend}")


if __name__ == "__main__":
    main()
