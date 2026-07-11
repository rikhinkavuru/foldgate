"""Training-set novelty features -- the covariate-shift variable.

Novelty both stratifies (group-conditional conformal) and, later, weights
(weighted conformal). RNP ships similarity-to-nearest-training-system on a
0-100 scale; crucially, a NaN means *no similar training system was found* =
maximally novel (the extrapolation regime), so NaN is a signal, not missing
data, and gets its own extreme stratum.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Similarity columns in RNP annotations (0-100). Higher = closer to training.
LIGAND_SIM = "morgan_tanimoto"          # ECFP4 Tanimoto to nearest train ligand
POCKET_SIM = "sucos_shape_pocket_qcov"  # pocket shape x coverage to nearest train
PROTEIN_SIM = "protein_seqsim_max"      # max protein sequence identity to train


def similarity(df: pd.DataFrame, col: str = LIGAND_SIM) -> pd.Series:
    """Return similarity-to-training in [0, 1]; NaN preserved as no-analog.

    RNP stores these on a 0-100 scale; any value above 1 means the column is on
    that scale, so we normalise. A genuine [0,1] column (max <= 1) is left as-is.
    """
    s = pd.to_numeric(df[col], errors="coerce")
    nn = s.dropna()
    if len(nn) and nn.max() > 1.5:
        s = s / 100.0
    return s.clip(0.0, 1.0)


def novelty(df: pd.DataFrame, col: str = LIGAND_SIM) -> pd.Series:
    """novelty = 1 - similarity; NaN (no training analog) -> 1.0 (max novel)."""
    return (1.0 - similarity(df, col)).fillna(1.0)


def make_strata(
    df: pd.DataFrame,
    col: str = LIGAND_SIM,
    n_bins: int = 4,
    no_analog_stratum: bool = True,
) -> pd.Series:
    """Bin by similarity into ordered novelty strata (0 = least novel).

    Points with no training analog (NaN similarity) form their own top
    stratum, the sharpest test of extrapolation. Remaining points are split
    into ``n_bins`` quantile bins of similarity.
    """
    sim = similarity(df, col)
    strata = pd.Series(np.nan, index=df.index, dtype="float")

    has = sim.notna()
    if has.any():
        # qcut on similarity; label so that higher label = more novel.
        q = pd.qcut(sim[has], q=n_bins, labels=False, duplicates="drop")
        n_levels = int(np.nanmax(q)) + 1
        strata.loc[has] = (n_levels - 1) - q  # invert: low sim -> high novelty

    if no_analog_stratum:
        top = (strata.max() + 1) if has.any() else 0
        strata.loc[~has] = top
    else:
        strata = strata.fillna(strata.max())

    return strata.astype(int)


def temporal_stratum(df: pd.DataFrame, date_col: str = "release_date", n_bins: int = 4) -> pd.Series:
    """Bin systems by release date into ordered temporal strata (0 = earliest).

    Later release = further past the training cutoff = a temporal-shift axis
    complementary to chemical/pocket novelty.
    """
    dates = pd.to_datetime(df[date_col], errors="coerce")
    ordinal = dates.astype("int64").where(dates.notna())  # ns since epoch; NaT -> NaN
    strata = pd.Series(np.nan, index=df.index, dtype="float")
    has = ordinal.notna()
    if has.any():
        strata.loc[has] = pd.qcut(ordinal[has], q=n_bins, labels=False, duplicates="drop")
    return strata.fillna(strata.median()).astype(int)


def add_novelty(df: pd.DataFrame, ligand_col: str = LIGAND_SIM,
                pocket_col: str = POCKET_SIM, n_bins: int = 4) -> pd.DataFrame:
    """Attach novelty score + strata columns in place-safe fashion."""
    out = df.copy()
    out["ligand_similarity"] = similarity(out, ligand_col)
    out["ligand_novelty"] = novelty(out, ligand_col)
    out["novelty_stratum"] = make_strata(out, ligand_col, n_bins=n_bins)
    if pocket_col in out.columns:
        out["pocket_similarity"] = similarity(out, pocket_col)
        out["pocket_novelty_stratum"] = make_strata(out, pocket_col, n_bins=n_bins)
    return out
