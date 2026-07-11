"""foldgate.scores -- nonconformity scores from confidence + derived signals.

`native_score` uses a single raw confidence (baseline). `ScoreCombiner` fits a
calibration-only model over native confidence + PoseBusters validity + ensemble
spread + ligand difficulty, returning P(correct) as the confidence. Both are
model-agnostic and plug into the same threshold machinery.
"""

from .combiner import DEFAULT_FEATURES, ScoreCombiner

__all__ = ["ScoreCombiner", "DEFAULT_FEATURES"]
