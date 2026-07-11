"""Load the FoldBench protein-ligand table as a second dataset.

FoldBench (BEAM-Labs, Nat. Commun. 2025) ships a per-pose table with a native
confidence (`ranking_score`) and a ligand-RMSD label (`lrmsd`) for five models,
extracted from the paper's Source Data. It does NOT ship ipTM/PoseBusters or
per-pose training-similarity, so it supports the validity + risk-coverage
transfer (E1/E4-style), not the novelty break (which needs similarity metadata).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

KEY = ["pdb_id", "native_chain_id_1", "native_chain_id_2", "model"]
TARGET = ["pdb_id", "native_chain_id_1", "native_chain_id_2"]
CSV = "foldbench_protein_ligand_confidence_rmsd.csv"
RMSD_THRESHOLD_A = 2.0


def load_foldbench(raw_dir: str | Path = "data/external/foldbench") -> pd.DataFrame:
    """Return delivered-pose records (top-1 by ranking_score per target/model)."""
    df = pd.read_csv(Path(raw_dir) / CSV)
    df = df.dropna(subset=["ranking_score", "lrmsd"])

    # intra-model ensemble spread across seeds/samples
    ens = df.groupby(KEY)["ranking_score"].agg(ens_ranking_std="std").reset_index()

    top = df.sort_values("ranking_score", ascending=False).drop_duplicates(KEY).reset_index(drop=True)
    top = top.merge(ens, on=KEY, how="left")
    top["correct"] = (top["lrmsd"] <= RMSD_THRESHOLD_A).astype(int)
    top["rmsd"] = top["lrmsd"]

    # ranking_score scales differ by model -> rank-normalise within model, then
    # cross-model agreement = leave-one-out mean of others' percentile rank.
    top["rank_pct"] = top.groupby("model")["ranking_score"].rank(pct=True)
    g = top.groupby(TARGET)["rank_pct"]
    gsum, gcnt = g.transform("sum"), g.transform("count")
    n_other = gcnt - 1
    with np.errstate(invalid="ignore"):
        top["xmodel_rank_mean"] = ((gsum - top["rank_pct"]) / n_other).where(n_other > 0)
    top["xmodel_n_models"] = n_other
    return top
