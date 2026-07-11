"""Cross-model agreement features.

When several co-folding models are run on the same target, whether they AGREE is
a signal orthogonal to any one model's self-confidence. We use interface
chain-pair ipTM (comparable across models on a 0-1 scale, unlike each model's own
ranking score) and, per delivered (system, ligand), summarise the OTHER models'
confidence: a leave-one-out mean (are the others confident too?), the spread
(do they disagree?), and how many models delivered this target.

This needs only the released confidences, no structures. Pose-level agreement
(pairwise ligand-RMSD across models) would need the coordinate tarball and is
left out here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

KEY = ["system_id", "ligand_instance_chain"]
CONF = "iface_iptm"


def add_cross_model_agreement(delivered: pd.DataFrame) -> pd.DataFrame:
    """Attach leave-one-out cross-model confidence-agreement features.

    Operates on the multi-method delivered table (one row per system/ligand/model).
    """
    out = delivered.copy()
    c = pd.to_numeric(out[CONF], errors="coerce")
    grp = c.groupby([out[k] for k in KEY])
    gsum = grp.transform("sum")
    gcnt = grp.transform("count")
    gsumsq = (c * c).groupby([out[k] for k in KEY]).transform("sum")

    n_other = gcnt - 1
    with np.errstate(invalid="ignore", divide="ignore"):
        loo_mean = (gsum - c) / n_other
        # variance of the other models' confidence (leave-one-out)
        loo_var = (gsumsq - c * c) / n_other - loo_mean**2
    out["xmodel_iptm_mean"] = loo_mean.where(n_other > 0)
    out["xmodel_iptm_std"] = np.sqrt(loo_var.clip(lower=0)).where(n_other > 0)
    out["xmodel_n_models"] = n_other
    return out
