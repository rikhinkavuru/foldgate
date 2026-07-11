"""Load released Runs N' Poses (RNP) predictions into unified records.

RNP ships one CSV per method under ``predictions/`` (keyed on ``target`` =
PLINDER system_id, ``ligand_instance_chain``, ``seed``, ``sample``) plus
``annotations.csv`` with pre-computed training-set similarity metadata. We do
NOT need the multi-GB structure/MSA tarballs for the tabular reliability layer.

The decision-relevant unit is the *delivered pose*: the top-1 sample per
(system, ligand) ranked by the model's own ``ranking_score`` -- what a
practitioner actually uses. Label: binding-mode correct iff BiSyRMSD <= 2 A.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# Methods shipped in RNP predictions/. Keys map to CSV basenames.
RNP_METHODS = {
    "af3": "af3",
    "af3_no_template": "af3_no_template",
    "boltz1": "boltz",
    "boltz1x": "boltz1x",
    "boltz2": "boltz2",
    "chai": "chai",
    "protenix": "protenix",
    "rfaa": "rfaa",
}

KEY = ["system_id", "ligand_instance_chain"]
RMSD_THRESHOLD_A = 2.0


@dataclass
class RNPData:
    """Delivered-pose table (one row per system/ligand/method) + raw samples."""

    delivered: pd.DataFrame           # top-1 by ranking_score, joined w/ novelty + label
    samples: pd.DataFrame | None = None  # all seeds/samples, for ensemble features
    dropped: dict = field(default_factory=dict)


def _interface_iptm_col(df: pd.DataFrame) -> str | None:
    """Prefer the rmsd-chain-mapped average protein-ligand chain-pair ipTM."""
    for c in (
        "prot_lig_chain_iptm_average_rmsd",
        "prot_lig_chain_iptm_average_lddt_pli",
        "lig_prot_chain_iptm_average_rmsd",
        "lig_prot_chain_iptm_average_lddt_pli",
    ):
        if c in df.columns:
            return c
    return None


def _load_one_method(path: Path, method: str, proper_only: bool) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df = df.rename(columns={"target": "system_id"})
    # ligand_is_proper can be mixed-typed; coerce robustly.
    if proper_only and "ligand_is_proper" in df.columns:
        proper = df["ligand_is_proper"].astype(str).str.lower().isin({"true", "1", "1.0"})
        df = df[proper].copy()
    if "ranking_score" not in df.columns or "rmsd" not in df.columns:
        raise ValueError(f"{path.name} missing ranking_score/rmsd")
    iptm = _interface_iptm_col(df)
    df["iface_iptm"] = df[iptm] if iptm else np.nan
    df["method"] = method
    return df


def load_rnp(
    raw_dir: str | Path = "data/raw",
    methods: list[str] | None = None,
    proper_only: bool = True,
    keep_samples: bool = False,
) -> RNPData:
    """Return delivered-pose records for the requested methods.

    Parameters
    ----------
    methods : subset of RNP_METHODS keys; default = the three headline models.
    proper_only : drop ions/artifacts (RNP guidance: analyse proper ligands).
    keep_samples : also return the full multi-sample frame (for ensemble features).
    """
    raw = Path(raw_dir)
    pred_dir = raw / "predictions" / "predictions"
    methods = methods or ["af3", "boltz2", "chai"]

    ann = pd.read_csv(raw / "annotations.csv", low_memory=False)
    # Similarity metadata is per (system_id, ligand_instance_chain). Keep the
    # columns the reliability layer keys on; the rest stay available downstream.
    ann_key = ann.drop_duplicates(KEY).set_index(KEY)

    delivered_parts, sample_parts, dropped = [], [], {}
    for m in methods:
        base = RNP_METHODS[m]
        raw_df = _load_one_method(pred_dir / f"{base}.csv", m, proper_only)
        if keep_samples:
            sample_parts.append(raw_df)
        # Top-1 delivered pose per (system, ligand) by model-native ranking_score.
        top1 = (
            raw_df.sort_values("ranking_score", ascending=False)
            .drop_duplicates(KEY)
            .reset_index(drop=True)
        )
        n_before = len(top1)
        top1 = top1.join(ann_key, on=KEY, how="inner", rsuffix="_ann")
        dropped[m] = {"no_annotation": n_before - len(top1), "n": len(top1)}
        delivered_parts.append(top1)

    delivered = pd.concat(delivered_parts, ignore_index=True)
    # A pose with no BiSyRMSD cannot be labelled; drop it rather than silently
    # scoring NaN as incorrect (which would inflate the base error rate).
    delivered = delivered.dropna(subset=["rmsd"]).reset_index(drop=True)
    delivered["correct"] = (delivered["rmsd"] <= RMSD_THRESHOLD_A).astype(int)

    samples = pd.concat(sample_parts, ignore_index=True) if keep_samples else None
    return RNPData(delivered=delivered, samples=samples, dropped=dropped)
