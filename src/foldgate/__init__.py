"""foldgate — a model-agnostic reliability layer for protein-ligand co-folding.

foldgate converts native co-folding confidence (AF3 / Boltz / Chai) into
risk-controlled accept/abstain decisions with conformal coverage guarantees,
and makes those guarantees robust to the novel-pocket / novel-chemotype
distribution shift central to drug discovery.

Pipeline (see PLAN.md for the science):

    io       parse AF3 / Boltz / Chai outputs -> unified Prediction records
    features novelty (Tanimoto, pocket similarity, temporal), ensemble /
             cross-model agreement, physical validity (PoseBusters)
    scores   turn confidence + derived signals into a nonconformity score
    conformal split / weighted / Mondrian(group-conditional) / RCPS / LTT
    selective accept/abstain gate, risk-coverage curves, AURC
    eval     coverage-vs-novelty, conditional coverage, downstream utility

The public API is intentionally thin: wrap a frozen model's outputs, calibrate
on a held-out set, then gate new predictions. Nothing here retrains a model.
"""

__version__ = "0.1.0"

__all__ = [
    "io",
    "features",
    "scores",
    "conformal",
    "selective",
    "eval",
]
