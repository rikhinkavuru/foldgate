"""Build the processed delivered-pose table from released RNP predictions.

Output: data/processed/rnp_delivered.parquet -- one row per (system, ligand,
method) top-1 pose, with native confidence, novelty score/strata, and the
RMSD<=2A correctness label. This is the input to E1-E4.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from foldgate.features import (  # noqa: E402
    add_cross_model_agreement,
    add_novelty,
    attach_pose_features,
    temporal_stratum,
)
from foldgate.io import load_rnp  # noqa: E402

METHODS = ["af3", "boltz2", "chai", "protenix", "boltz1", "boltz1x"]
CONF_COLS = ["ranking_score", "iface_iptm"]
POSE_COLS = ["ens_ranking_std", "ens_ranking_mean", "ens_ranking_range",
             "ens_iptm_std", "ens_n_samples", "pb_valid"]
PHYS_COLS = ["ligand_molecular_weight", "ligand_num_rot_bonds", "ligand_num_heavy_atoms"]
XMODEL_COLS = ["xmodel_iptm_mean", "xmodel_iptm_std", "xmodel_n_models"]
OUT = Path("data/processed/rnp_delivered.parquet")


def main() -> None:
    data = load_rnp(methods=METHODS, proper_only=True, keep_samples=True)
    df = attach_pose_features(data.delivered, data.samples, "data/raw", METHODS)
    df = add_cross_model_agreement(df)
    df = add_novelty(df)
    if "release_date" in df.columns:
        df["temporal_stratum"] = temporal_stratum(df, "release_date")

    keep = [
        "system_id", "ligand_instance_chain", "method", "seed", "sample",
        *CONF_COLS, *POSE_COLS, *PHYS_COLS, *XMODEL_COLS, "rmsd", "lddt_pli", "correct",
        "ligand_similarity", "ligand_novelty", "novelty_stratum",
        "pocket_similarity", "pocket_novelty_stratum",
        "release_date", "temporal_stratum",
        "target_release_date", "num_training_systems_with_similar_ccds",
    ]
    df = df[[c for c in keep if c in df.columns]].copy()

    # W1: left-join cross-model + intra-model pose-agreement features if they have been built
    # (experiments/build_pose_features.py, from the structure tarball). Optional: absent columns
    # are simply skipped by the combiner, so the pipeline stays green without them.
    pose_path = OUT.parent / "rnp_pose_features.parquet"
    if pose_path.exists():
        import pandas as pd
        pose = pd.read_parquet(pose_path)
        df = df.merge(pose, on=["system_id", "method"], how="left")
        print(f"joined pose features from {pose_path.name} "
              f"({df['intra_model_pose_std'].notna().mean():.0%} coverage)")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)

    print(f"wrote {OUT}  ({len(df)} rows)")
    print("\nper-method delivered poses + base correctness rate:")
    summ = df.groupby("method").agg(
        n=("correct", "size"),
        base_correct=("correct", "mean"),
        median_novelty=("ligand_novelty", "median"),
    )
    print(summ.round(3).to_string())

    print("\nnovelty stratum sizes (af3) [0 = least novel; top = no training analog]:")
    af3 = df[df.method == "af3"]
    g = af3.groupby("novelty_stratum").agg(
        n=("correct", "size"),
        correct_rate=("correct", "mean"),
        median_similarity=("ligand_similarity", "median"),
    )
    print(g.round(3).to_string())
    print(f"\ndropped (no annotation join): {data.dropped}")


if __name__ == "__main__":
    main()
