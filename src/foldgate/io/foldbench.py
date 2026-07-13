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
NOVELTY_CSV = "foldbench_protein_ligand_rmsd_lddtlp.csv"
GROUP_KEY = ["pdb_id", "ligand", "model"]
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


def load_foldbench_novelty(raw_dir: str | Path = "data/external/foldbench") -> pd.DataFrame:
    """Top-1 FoldBench poses carrying the is_unseen_protein novelty flag.

    The public confidence table ships ranking_score and ligand-RMSD but no
    per-pose training similarity. The companion rmsd_lddtlp table carries an
    ``is_unseen_protein`` flag (a low-homology protein unseen relative to the
    training cutoff) that is constant per pdb_id, so we map it onto each pose by
    pdb_id. The delivered pose is the top-1 by ranking_score per
    (pdb_id, ligand, model), matching the RNP delivered convention.
    ``correct`` = 1 iff ligand-RMSD <= 2 A.

    This flag is the one novelty axis FoldBench exposes, so it is the axis on
    which a frozen RNP-calibrated gate can be checked for the E2 coverage break.
    """
    raw = Path(raw_dir)
    df = pd.read_csv(raw / CSV).dropna(subset=["ranking_score", "lrmsd"])

    nov = pd.read_csv(raw / NOVELTY_CSV)
    pdb_unseen = nov.groupby("pdb_id")["is_unseen_protein"].first()
    df["is_unseen_protein"] = df["pdb_id"].map(pdb_unseen).astype("boolean")

    top = (
        df.sort_values("ranking_score", ascending=False)
        .drop_duplicates(GROUP_KEY)
        .reset_index(drop=True)
    )
    top["correct"] = (top["lrmsd"] <= RMSD_THRESHOLD_A).astype(int)
    top["rmsd"] = top["lrmsd"]
    return top
