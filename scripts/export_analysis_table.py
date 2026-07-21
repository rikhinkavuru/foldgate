"""Export the per-(model, system) analysis table (reviewer E / H).

One row per delivered pose with everything needed to re-derive every tabular result in
the paper without the structure stream: the scores, the label, both similarities and
strata, sequence cluster, crystal resolution, and the extracted ligand-local pLDDT.
This is the single highest-value reproducibility artifact.

Output: results/analysis_table.csv  (+ a small data dictionary printed)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    d = pd.read_parquet(ROOT / "data/processed/rnp_delivered.parquet")
    keep = ["system_id", "ligand_instance_chain", "method", "ranking_score", "iface_iptm",
            "correct", "rmsd", "lddt_pli", "pb_valid", "ligand_similarity", "novelty_stratum",
            "pocket_similarity", "pocket_novelty_stratum", "release_date", "temporal_stratum",
            "ligand_molecular_weight", "ligand_num_heavy_atoms", "ligand_num_rot_bonds",
            "ifp_recall", "ifp_jaccard"]
    tab = d[[c for c in keep if c in d.columns]].copy()

    # sequence cluster + PDB id + crystal resolution
    ann = pd.read_csv(ROOT / "data/raw/annotations.csv")
    acols = [c for c in ["system_id", "cluster", "entry_pdb_id", "morgan_tanimoto",
                         "sucos_shape_pocket_qcov", "protein_seqsim_max", "target_release_date"]
             if c in ann.columns]
    tab = tab.merge(ann[acols].drop_duplicates("system_id"), on="system_id", how="left")

    res = ROOT / "data/processed/pdb_resolution.csv"
    if res.exists():
        r = pd.read_csv(res)
        idc = "entry_pdb_id" if "entry_pdb_id" in r.columns else r.columns[0]
        rescol = next((c for c in r.columns if "resol" in c.lower()), None)
        if rescol and "entry_pdb_id" in tab.columns:
            tab = tab.merge(r[[idc, rescol]].rename(columns={idc: "entry_pdb_id", rescol: "resolution"}),
                            on="entry_pdb_id", how="left")

    plddt = ROOT / "data/processed/ligand_local_plddt.parquet"
    if plddt.exists():
        p = pd.read_parquet(plddt)
        tab = tab.merge(p[["system_id", "method", "ligand_plddt_mean", "ligand_plddt_min"]],
                        on=["system_id", "method"], how="left")

    out = ROOT / "results/analysis_table.csv"
    tab.to_csv(out, index=False)
    print(f"wrote {out}  ({len(tab)} rows, {tab.shape[1]} cols)")
    print("columns:", ", ".join(tab.columns))
    print("\nData dictionary: one row per delivered pose (top-1 by ranking_score per system/model).")
    print("  correct = 1[BiSyRMSD <= 2A]; novelty_stratum/pocket_novelty_stratum in 0..4 (4 = no-computable-analog);")
    print("  ligand_similarity/pocket_similarity = similarity to nearest pre-cutoff PDB analog;")
    print("  ligand_plddt_* = per-ligand-atom predicted confidence; resolution = crystal resolution (A).")


if __name__ == "__main__":
    main()
