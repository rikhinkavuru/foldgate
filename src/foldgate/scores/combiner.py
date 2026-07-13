"""Combined nonconformity score: a calibration-only combiner over native
confidence + physical validity + ensemble spread + ligand difficulty.

The reliability layer stays training-free w.r.t. the co-folding model, but the
gate itself is a thin model fit ONLY on calibration data (never the test fold),
mapping cheap per-prediction features to P(correct). Higher = more likely
correct, so it plugs directly into the same threshold machinery as a raw
confidence. Training-set novelty is deliberately excluded here -- novelty enters
through conformal calibration, not the score, keeping the two roles separate.

HistGradientBoosting handles NaN natively (e.g. Boltz-2 has no PoseBusters file,
single-sample runs have no ensemble spread), so missing features need no imputation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

# Primary combined score: released-tabular-data-only (the cheap, structure-free reliability
# layer). Consistent across all experiments.
DEFAULT_FEATURES = [
    "ranking_score",        # native global confidence
    "iface_iptm",           # interface chain-pair ipTM
    "ens_ranking_std",      # intra-model ensemble disagreement
    "ens_ranking_range",
    "ens_iptm_std",
    "pb_valid",             # PoseBusters physical validity
    "xmodel_iptm_mean",     # cross-model confidence agreement (others' mean ipTM)
    "xmodel_iptm_std",      # cross-model disagreement
    "xmodel_n_models",
    "ligand_molecular_weight",
    "ligand_num_rot_bonds",
    "ligand_num_heavy_atoms",
]

# W1 structure-based upgrade: pose-agreement features from the model's diffusion samples +
# cross-model structures (build_pose_features.py). Opt-in (DEFAULT + POSE_FEATURES); the
# combiner skips any that are absent, so the tabular default stays valid without structures.
POSE_FEATURES = [
    "intra_model_pose_std",       # spread of the model's own samples' ligand placement
    "intra_model_pose_median",
    "pose_consensus_frac",        # fraction of samples in the delivered binding mode
    "xmodel_pose_rmsd_median",    # cross-model structural agreement (others' delivered poses)
    "xmodel_pose_rmsd_min",
    "pose_consensus_cluster_size",
]


class ScoreCombiner:
    """Fit on calibration only; predict P(correct) as a confidence score."""

    def __init__(self, features: list[str] | None = None, random_state: int = 0):
        self.features = features or DEFAULT_FEATURES
        self.random_state = random_state
        self.model: HistGradientBoostingClassifier | None = None
        self.used_: list[str] = []

    def _matrix(self, df: pd.DataFrame) -> np.ndarray:
        cols = [c for c in self.features if c in df.columns]
        self.used_ = cols
        X = df[cols].apply(pd.to_numeric, errors="coerce")
        return X.to_numpy(dtype=float)

    def fit(self, df: pd.DataFrame, y: np.ndarray) -> ScoreCombiner:
        X = self._matrix(df)
        self.model = HistGradientBoostingClassifier(
            max_depth=3, max_iter=200, learning_rate=0.05,
            l2_regularization=1.0, random_state=self.random_state,
        )
        self.model.fit(X, np.asarray(y, dtype=int))
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("fit() first")
        X = df[self.used_].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        return self.model.predict_proba(X)[:, 1]
