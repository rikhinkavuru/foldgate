"""Derived per-pose features for the combined nonconformity score.

Two label-free signals beyond native confidence:
  - intra-model ensemble disagreement across the 5 diffusion samples per seed
    (spread of ranking_score / interface ipTM): a pose the model is internally
    unsure about is riskier.
  - PoseBusters physical validity (PB-valid = all ~30 RDKit checks pass): a
    physically implausible pose is unlikely to be a correct binding mode.

Both are available in the released RNP tarballs; neither uses the RMSD label.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

KEY = ["system_id", "ligand_instance_chain", "method"]

# posebusters_results/ basenames per method key (Boltz-2 has no PB file shipped).
PB_FILES = {
    "af3": "af3",
    "af3_no_template": "af3_no_templ",
    "boltz1": "boltz",
    "boltz1x": "boltz1x",
    "chai": "chai",
    "protenix": "protenix",
    "rfaa": "rfaa",
}

# Non-check columns in posebusters CSVs (identifiers / load flags handled separately).
_PB_ID_COLS = {"molecule", "position", "system_id", "seed", "sample", "ligand_chain", "method"}


def ensemble_stats(samples: pd.DataFrame) -> pd.DataFrame:
    """Per (system, ligand, method) spread of native confidence across samples."""
    g = samples.groupby(KEY, dropna=False)
    agg = g.agg(
        ens_ranking_std=("ranking_score", "std"),
        ens_ranking_mean=("ranking_score", "mean"),
        ens_ranking_range=("ranking_score", lambda x: x.max() - x.min()),
        ens_iptm_std=("iface_iptm", "std"),
        ens_n_samples=("ranking_score", "size"),
    ).reset_index()
    return agg


def load_posebusters(raw_dir: str | Path, methods: list[str]) -> pd.DataFrame:
    """Return PB-valid flag per (system, seed, sample, method).

    PB-valid = every boolean physical-plausibility check passes (NaN check = fail).
    """
    pb_dir = Path(raw_dir) / "posebusters" / "posebusters_results"
    parts = []
    for m in methods:
        base = PB_FILES.get(m)
        path = pb_dir / f"{base}.csv" if base else None
        if not path or not path.exists():
            continue
        df = pd.read_csv(path, low_memory=False)
        check_cols = [c for c in df.columns if c not in _PB_ID_COLS and df[c].dtype == bool]
        if not check_cols:
            continue
        df["pb_valid"] = df[check_cols].fillna(False).all(axis=1).astype(float)
        df["method"] = m
        # PB is per-ligand; keep ligand_chain so multi-ligand systems don't fan out.
        df = df.rename(columns={"ligand_chain": "ligand_instance_chain"})
        parts.append(df[["system_id", "seed", "sample", "method", "ligand_instance_chain", "pb_valid"]])
    if not parts:
        cols = ["system_id", "seed", "sample", "method", "ligand_instance_chain", "pb_valid"]
        return pd.DataFrame(columns=cols)
    return pd.concat(parts, ignore_index=True)


def attach_pose_features(
    delivered: pd.DataFrame, samples: pd.DataFrame, raw_dir: str | Path, methods: list[str]
) -> pd.DataFrame:
    """Merge ensemble spread + PB validity onto the delivered-pose table."""
    out = delivered.merge(ensemble_stats(samples), on=KEY, how="left")
    pb = load_posebusters(raw_dir, methods)
    n_before = len(out)
    if len(pb):
        pb = pb.drop_duplicates(["system_id", "seed", "sample", "method", "ligand_instance_chain"])
        out = out.merge(pb, on=["system_id", "seed", "sample", "method", "ligand_instance_chain"], how="left")
    else:
        out["pb_valid"] = float("nan")
    assert len(out) == n_before, f"PB merge changed row count {n_before} -> {len(out)}"
    return out
