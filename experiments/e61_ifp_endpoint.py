"""E61 -- an interaction-fingerprint correctness endpoint (reviewer H-P Peng).

Peng's point: a single geometric cutoff (ligand-RMSD <= 2 A) overcounts failures for
flexible ligands, where a long solvent-exposed tail pushes whole-ligand RMSD past 2 A even
when the core pose and the key protein-ligand interactions are reproduced. He suggests a
ProLIF-style interaction fingerprint (recall / precision / F1) as a pose-correctness label
that is robust to that tail motion.

We already ship typed IFP recovery as a downstream quality check (e51). Here we take the
sharper step Peng proposes and treat interaction recovery as an alternative CORRECTNESS
LABEL, then ask the two questions that matter:

  1. Does it change which poses count as correct, and is the change concentrated on flexible
     ligands (his mechanism)?
  2. Does the novel-stratum coverage break survive the new label, or was it an artifact of
     the RMSD cutoff? We re-run the exact break and feasibility-frontier machinery
     (d2_feasibility_map) with the IFP label swapped for the RMSD label.

IFP F1 is derived from the per-pose untyped IFP Jaccard already in analysis_table
(F1 = 2J / (1 + J)); recall is analysis_table.ifp_recall. Everything is CPU-only from the
committed per-pose table. DockQ v2 (Bianchi & Elofsson, Bioinformatics 2024, btae586) is the
standardized pocket-aligned symmetry-aware RMSD metric class our BiSyRMSD label already
belongs to; it standardizes the geometry but is still whole-ligand RMSD, so it does not
address the flexible-tail concern that this interaction endpoint does.

Output: results/e61_ifp_endpoint.json
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from experiments._common import CONF, DELTA, RESDIR, save_json
from experiments.d2_feasibility_map import COVERAGE_GRID, MIN_ACCEPT, _cell, _frontier

ATABLE = RESDIR / "analysis_table.csv"
TYPED = ROOT_TYPED = __import__("pathlib").Path(__file__).resolve().parents[1] / \
    "data" / "processed" / "e51_typed_recovery.parquet"
GOVERNED = ["af3", "boltz1", "boltz1x", "chai", "protenix"]
F1_GRID = (0.4, 0.5, 0.6, 0.7, 0.8)
ALPHA = 0.20
N_STRATA = 5          # S0..S4
KEY_TYPES = ["hbond", "salt_bridge", "pi_stacking"]   # binding-determinant interactions


def _key_recall(row) -> float:
    """Count-weighted recall over the crystal's polar/directional interactions.

    Peng's "key protein-ligand interactions": excludes hydrophobic contacts, which are
    numerous and easy to recover partially, so an all-contact fingerprint is lenient.
    """
    num = den = 0.0
    for ty in KEY_TYPES:
        n = row[f"n_true_{ty}"]
        rec = row[f"{ty}_recall"]
        if n > 0 and not pd.isna(rec):
            num += rec * n
            den += n
    return num / den if den > 0 else np.nan


def load() -> pd.DataFrame:
    d = pd.read_csv(ATABLE)
    d = d[d["method"].isin(GOVERNED)].copy()
    d = d.dropna(subset=["rmsd", "ifp_jaccard", "novelty_stratum", CONF])
    d["ifp_f1"] = 2 * d["ifp_jaccard"] / (1 + d["ifp_jaccard"])       # untyped, all-contact
    d["rmsd_correct"] = (d["rmsd"] <= 2.0).astype(int)
    d["novelty_stratum"] = d["novelty_stratum"].astype(int)
    # merge the typed (key-interaction) recovery, computed in e51
    if TYPED.exists():
        t = pd.read_parquet(TYPED)
        t["key_recall"] = t.apply(_key_recall, axis=1)
        d = d.merge(t[["system_id", "method", "key_recall"]],
                    on=["system_id", "method"], how="left")
    else:
        d["key_recall"] = np.nan
    return d


def disagreement_by_flexibility(d: pd.DataFrame, f1_thr: float = 0.5) -> dict:
    """Where do RMSD and IFP disagree, and is it concentrated on flexible ligands?"""
    x = d.dropna(subset=["ligand_num_rot_bonds"]).copy()
    x["ifp_correct"] = (x["ifp_f1"] >= f1_thr).astype(int)
    # rotatable-bond bins: rigid / moderate / flexible (Peng's long-tail regime)
    x["flex"] = pd.cut(x["ligand_num_rot_bonds"], [-1, 4, 7, 1e9],
                       labels=["rigid(<=4)", "moderate(5-7)", "flexible(>=8)"])
    out = {}
    for lab, g in x.groupby("flex", observed=True):
        rmsd_wrong_ifp_ok = int(((g.rmsd_correct == 0) & (g.ifp_correct == 1)).sum())
        rmsd_ok_ifp_wrong = int(((g.rmsd_correct == 1) & (g.ifp_correct == 0)).sum())
        out[str(lab)] = {
            "n": int(len(g)),
            "rmsd_correct_rate": round(float(g.rmsd_correct.mean()), 3),
            "ifp_correct_rate": round(float(g.ifp_correct.mean()), 3),
            "rmsd_wrong_but_ifp_ok": rmsd_wrong_ifp_ok,          # Peng's flexible-tail cases
            "rmsd_ok_but_ifp_wrong": rmsd_ok_ifp_wrong,
            "median_rot_bonds": float(g.ligand_num_rot_bonds.median()),
            "median_rmsd_of_disagreement": round(float(
                g.loc[(g.rmsd_correct == 0) & (g.ifp_correct == 1), "rmsd"].median()), 2)
                if rmsd_wrong_ifp_ok else None,
        }
    return out


def per_stratum_correctness(d: pd.DataFrame) -> dict:
    """Base correctness per novelty stratum, RMSD vs IFP-F1 across thresholds."""
    out = {}
    for meth, g in d.groupby("method"):
        rec = {"RMSD<=2": {int(s): round(float(v), 3) for s, v in
                           g.groupby("novelty_stratum")["rmsd_correct"].mean().items()}}
        for thr in F1_GRID:
            lab = (g["ifp_f1"] >= thr).astype(int)
            rec[f"IFP-F1>={thr}"] = {int(s): round(float(lab[g.novelty_stratum == s].mean()), 3)
                                     for s in sorted(g.novelty_stratum.unique())}
        out[meth] = rec
    return out


def _top1(g: pd.DataFrame, label: str) -> pd.DataFrame:
    """One delivered pose per (system, method): top-1 by ranking_score, carrying `label`."""
    return (g.sort_values(CONF, ascending=False)
            .groupby(["system_id", "method"], as_index=False).first())


def break_under_label(d: pd.DataFrame, label: str, delta: float = DELTA) -> dict:
    """Calibrate a marginally-valid gate on familiar strata, deploy on novel; realized error."""
    out = {}
    for meth, g in d.groupby("method"):
        t = _top1(g, label)
        fam = t[t.novelty_stratum.isin([0, 1])]
        nov = t[t.novelty_stratum.isin([2, 3])]
        if len(fam) < MIN_ACCEPT or len(nov) < MIN_ACCEPT:
            out[meth] = {"underpowered": True}
            continue
        fam = fam.sort_values(CONF, ascending=False)
        fam_loss = 1 - fam[label].to_numpy()
        tau = next((float(fam[CONF].iloc[k - 1]) for k in range(len(fam), 0, -1)
                    if fam_loss[:k].mean() <= ALPHA), None)
        if tau is None:
            out[meth] = {"no_familiar_gate": True}
            continue
        acc = nov[nov[CONF] >= tau]
        out[meth] = {
            "novel_realized_error": round(float((1 - acc[label]).mean()), 3) if len(acc) else None,
            "novel_coverage": round(float(len(acc) / len(nov)), 3),
            "target_alpha": ALPHA,
        }
    return out


def frontier_under_label(d: pd.DataFrame, label: str, delta: float = DELTA) -> dict:
    """Per (method, stratum) frontier c* on `label`; count zero-frontier non-reference cells."""
    zero, total, detail = 0, 0, {}
    for meth, g in d.groupby("method"):
        t = _top1(g, label)
        per = {}
        for s in range(N_STRATA):
            gs = t[t.novelty_stratum == s].sort_values(CONF, ascending=False)
            n = len(gs)
            if n < MIN_ACCEPT:
                per[str(s)] = {"n": int(n), "frontier": None, "underpowered": True}
                continue
            loss = (1 - gs[label].to_numpy())
            cells = {float(c): _cell(loss[:max(1, int(round(c * n)))], n, ALPHA, delta)
                     for c in COVERAGE_GRID if int(round(c * n)) >= 1}
            fr = _frontier(cells, "feasible")
            per[str(s)] = {"n": int(n), "frontier": fr,
                           "base_correct": round(float(gs[label].mean()), 3)}
            if s != 0:                       # S0 is the reference stratum
                total += 1
                if fr == 0.0:
                    zero += 1
        detail[meth] = per
    return {"zero_frontier_cells": zero, "nonreference_cells": total, "per_method": detail}


def main() -> None:
    d = load()
    # three endpoint families: strict geometry, lenient all-contact IFP, and key interactions
    d["ifp_correct_05"] = (d["ifp_f1"] >= 0.5).astype(int)
    d["ifp_correct_07"] = (d["ifp_f1"] >= 0.7).astype(int)
    dk = d.dropna(subset=["key_recall"]).copy()
    dk["key_correct_05"] = (dk["key_recall"] >= 0.5).astype(int)
    dk["key_correct_07"] = (dk["key_recall"] >= 0.7).astype(int)

    labels = {
        "rmsd_2A": (d, "rmsd_correct"),
        "ifp_all_contact_f1_0.5": (d, "ifp_correct_05"),
        "ifp_all_contact_f1_0.7": (d, "ifp_correct_07"),
        "key_interaction_recall_0.5": (dk, "key_correct_05"),
        "key_interaction_recall_0.7": (dk, "key_correct_07"),
    }
    result = {
        "n_poses": int(len(d)),
        "n_poses_with_key_interactions": int(len(dk)),
        "note": ("Three correctness endpoints. rmsd_2A: strict whole-ligand geometry (the primary "
                 "BiSyRMSD label; DockQ v2 / btae586 is the same metric class and does not change "
                 "the flexible-tail behavior). ifp_all_contact_f1: untyped IFP F1 = 2J/(1+J), "
                 "lenient because hydrophobic contacts dominate. key_interaction_recall: "
                 "count-weighted recall of H-bond / salt-bridge / pi-stacking, the binding "
                 "determinants Peng means by 'key interactions'. The novel-stratum break survives "
                 "the geometry and key-interaction endpoints; it relaxes only under the lenient "
                 "all-contact fingerprint, which is the honest scope of the impossibility."),
        "disagreement_by_flexibility": disagreement_by_flexibility(d, 0.5),
        "per_stratum_correctness": per_stratum_correctness(d),
        "break": {k: break_under_label(df, lab) for k, (df, lab) in labels.items()},
        "frontier": {k: frontier_under_label(df, lab) for k, (df, lab) in labels.items()},
    }
    out = RESDIR / "e61_ifp_endpoint.json"
    save_json(result, out)
    print(f"wrote {out.name}  (n={len(d)}, key-interaction subset n={len(dk)})")
    for k in labels:
        fr = result["frontier"][k]
        b = result["break"][k].get("af3", {})
        print(f"  {k:28s}: zero-frontier {fr['zero_frontier_cells']}/{fr['nonreference_cells']} "
              f"| AF3 deploy-novel error {b.get('novel_realized_error')}")


if __name__ == "__main__":
    main()
