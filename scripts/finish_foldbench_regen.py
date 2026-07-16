#!/usr/bin/env python3
"""Post-batch: join the regenerated Protenix scores with the FoldBench novelty flag,
write the analysis table, and validate the regeneration against FoldBench's own top-1.

Input: the scorer output (pdb_id, target_ccd, seed, sample, iptm_iface, ranking_score,
lrmsd, status). Output: data/external/foldbench/foldbench_protenix_regen.csv with
is_unseen_protein + correct, plus a printed validation comparing our delivered-pose
(top-1 by ranking_score) success rate against FoldBench's own delivered-pose success rate
computed from their released table (apples-to-apples: both top-1 by ranking_score).
"""
import sys
from pathlib import Path

import pandas as pd

FB = Path("data/external/foldbench")
CONF_CSV = FB / "foldbench_protein_ligand_confidence_rmsd.csv"
NOV_CSV = FB / "foldbench_protein_ligand_rmsd_lddtlp.csv"
OUT = FB / "foldbench_protenix_regen.csv"


def main(scored_csv: str) -> None:
    df = pd.read_csv(scored_csv)
    df = df[df["status"] == "ok"].copy()
    df["correct"] = (df["lrmsd"] <= 2.0).astype(int)

    nov = pd.read_csv(NOV_CSV)
    pdb_unseen = nov.groupby("pdb_id")["is_unseen_protein"].first()
    df["is_unseen_protein"] = df["pdb_id"].map(pdb_unseen)
    df = df.dropna(subset=["is_unseen_protein", "iptm_iface", "lrmsd"]).copy()
    df["is_unseen_protein"] = df["is_unseen_protein"].astype(bool)

    df.to_csv(OUT, index=False)

    # --- validation: our top-1 vs FoldBench's own top-1 (both by ranking_score) ---
    fb = pd.read_csv(CONF_CSV)
    fbp = fb[fb["model"].str.contains("Protenix", case=False)].copy()
    fbp_top = (fbp.sort_values("ranking_score", ascending=False)
               .drop_duplicates("pdb_id"))
    fbp_top["correct"] = (fbp_top["lrmsd"] <= 2.0).astype(int)

    common = set(df["pdb_id"]) & set(fbp_top["pdb_id"])
    fb_rate = fbp_top[fbp_top.pdb_id.isin(common)]["correct"].mean()
    our_rate = df[df.pdb_id.isin(common)]["correct"].mean()

    print(f"scored (ok) targets: {len(df)}")
    print(f"  seen:   {int((~df.is_unseen_protein).sum())}")
    print(f"  unseen: {int(df.is_unseen_protein.sum())}\n")
    print("VALIDATION -- delivered-pose (top-1 by ranking_score) success rate, lrmsd<=2A")
    print(f"  common targets: {len(common)}")
    print(f"  FoldBench released Protenix top-1: {fb_rate:.3f}")
    print(f"  our regenerated Protenix top-1:    {our_rate:.3f}")
    print(f"  delta: {our_rate - fb_rate:+.3f}  "
          f"({'consistent' if abs(our_rate - fb_rate) < 0.08 else 'CHECK -- large gap'})")
    print(f"\n  (FoldBench published best-of-25 leaderboard is 0.507; top-1 is stricter.)")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "regen_scores.csv")
