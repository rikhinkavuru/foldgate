"""foldgate.features -- novelty (the covariate-shift variable) + derived signals.

Novelty both stratifies (group-conditional conformal) and weights (weighted
conformal). Ensemble/cross-model agreement and PoseBusters validity land here.
"""

from .agreement import add_cross_model_agreement
from .novelty import (
    LIGAND_SIM,
    POCKET_SIM,
    PROTEIN_SIM,
    add_novelty,
    make_strata,
    novelty,
    similarity,
    temporal_stratum,
)
from .pose import attach_pose_features, ensemble_stats, load_posebusters

__all__ = [
    "add_novelty",
    "novelty",
    "similarity",
    "make_strata",
    "temporal_stratum",
    "LIGAND_SIM",
    "POCKET_SIM",
    "PROTEIN_SIM",
    "attach_pose_features",
    "ensemble_stats",
    "load_posebusters",
    "add_cross_model_agreement",
]
