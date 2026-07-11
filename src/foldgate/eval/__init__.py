"""foldgate.eval — experiment drivers and figures (E1-E6 in PLAN.md).

Turns calibrated predictions into the paper's evidence. Each function maps to a
figure so results are reproducible from released prediction sets.

Planned public surface:

    e1_iid_validity(...)        native-confidence conformal hits nominal coverage
    e2_exchangeability_break(.) coverage vs novelty stratum -> the collapse
    e3_shift_repair(...)        weighted + group-conditional restore coverage
    e4_selective_utility(...)   risk-coverage + AURC, gate vs native threshold
    e5_generality(...)          repeat across AF3/Boltz/Chai and tasks
    e6_downstream(...)          screening enrichment / FEP starting-structure lift
    labels                      ligand-RMSD<=2A (primary), continuous RMSD,
                                DockQ, binder-vs-decoy (screening arm)

Report across RMSD thresholds, not just 2A (PLAN.md §8).
"""
