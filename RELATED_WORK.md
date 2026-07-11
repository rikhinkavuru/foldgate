# Related Work & Novelty Defense

The literature splits into four clusters. None occupies our cell: a **distribution-free, risk-controlled accept/abstain decision procedure for co-folding pose correctness that is model-agnostic, training-free, and shift-robust** (weighted / group-conditional conformal keyed on training-set structural/chemical similarity). Each cluster below ends with the delta.

## Cluster 1 — Confidence-correlates-with-accuracy / enrichment studies

These show co-folding confidence tracks pose accuracy or enriches actives, then stop. No coverage guarantee, no formal abstention rule.

- **Mac1 prospective evaluation** (Kim, Correy, Hall … Shoichet, Fraser; eLife reviewed preprint 110475 / bioRxiv 2025.12.25.696505). AF3/Boltz-2/Chai-1 each reproduce >50% of Mac1 poses to <2 Å; AF3 & Chai-1 confidence weakly-but-significantly tracks potency; Boltz-2 iPTM gives AUC 0.73. Notably, AF3 ligand-pose confidence did **not** separate true ligands from high-scoring false positives as well as docking scores or Boltz-2 affinity in prospective screens — direct evidence that raw confidence is an imperfect trust signal, motivating a calibrated layer.
- **AF3 for Structure-guided Ligand Discovery / COValid** (bioRxiv 2025.12.04.692352). Introduces COValid (covalent virtual-screening enrichment benchmark); an AF3 confidence metric beats physics-based methods at enriching covalent ligands. Enrichment/ranking, no guarantee. (Note: COValid's "valid" is unrelated to conformal validity/coverage.)
- **Discovery of Covalent Ligands with AlphaFold3** (JACS 2025, doi:10.1021/jacs.5c22222). Prospective hit discovery prioritized by AF3 confidence. Discovery result, not a decision procedure.

**Delta:** we deliver a finite-sample, distribution-free accept/abstain guarantee on pose correctness and evaluate it as selective prediction (risk-coverage, AURC), not a correlation.

## Cluster 2 — Conformal prediction under shift (for scalars)

- **CoDrug** (Nguyen et al., NeurIPS 2023, arXiv:2310.12033). Conformal prediction for **scalar drug-property** prediction under covariate shift, with an energy-based model + KDE for likelihood-ratio weights; reduces the coverage gap up to ~35% on scaffold/fingerprint splits. **The single closest prior art.**

**Delta (pre-empt the reviewer):** CoDrug conformalizes a *scalar property* with KDE density-ratio weights and an *asymptotic* guarantee. We conformalize a *structured binary pose-correctness label* (≤2 Å) and key the shift weighting on *training-set structural/chemical similarity* — the axis co-folding accuracy actually degrades along. First to apply selective prediction / conformal to co-folding pose reliability.

## Cluster 3 — Novelty-degradation benchmarks (our motivation, not a solution)

These quantify that accuracy collapses on novel pockets/ligands. We reuse them as the shift axis instead of re-deriving the degradation.

- **Runs N' Poses** — "Have protein-ligand co-folding methods moved beyond memorisation?" (Škrinjar, Eberhardt, Tauriello, Schwede, Durairaj; bioRxiv 2025.02.03.636309 → Nature Struct. Mol. Biol. s41594-026-01797-5). ~2,073 post-cutoff systems; AF3/Protenix/Chai-1/Boltz largely memorise, accuracy near-linearly worse with distance to training (down to ~20% in the sparsest bin).
- **FoldBench** (BEAM-Labs; bioRxiv 2025.05.22.655600 → Nature Comms s41467-025-67127-3). Low-homology all-atom benchmark; ligand-docking success drops sharply as ligand Tanimoto-to-train falls; "unseen ligand" (<0.50 Tanimoto) subset. Concludes models "rely substantially on memorized binding modes."
- **Do co-folding models learn the physics?** (Nature Comms s41467-025-63947-5). Argues predictions come from features largely independent of true physics/pose — confidence is not physics-grounded.
- **Covalent co-folding benchmark** (Acta Pharmacologica Sinica s41401-025-01721-5). Performance "markedly declines for novel pocket-ligand pairs."

**Delta:** they report the drop; we turn novelty into a calibrated weight that provably repairs coverage and yields an operational abstention rule.

## Cluster 4 — Recall/quality-improvement methods (complementary, not competing)

These make weak predictions better; they do not decide whether to trust a prediction. Our layer sits on top of any of them.

- **SiteAF3** (PNAS 2025, doi:10.1073/pnas.2521048122). Conditional-diffusion fine-tune of AF3 that improves the bottom-20% lowest-ranking-score predictions.
- **AF-ClaSeq** (builds on AF-Cluster). Sequence purification to sample alternative conformational states at high confidence — a recall-of-states method.

**Delta:** those improve or diversify predictions; we decide whether to trust one. Model-agnostic + training-free → our layer composes on top, turning "why not just improve the model?" into a strength.

## Reliability-adjacent to cite and distinguish

- **On the Reliability of Boltz-2** (arXiv:2603.05532, 2026). Evaluation: Boltz-2 affinity did not correlate with predicted pose (r=−0.03) though iPTM tracked outcomes. Evaluation, not a decision procedure.
- **PoseBusters** (Chem. Sci. 2024, doi:10.1039/D3SC04185A). Physical-validity benchmark; AF3 ~80.5% within-2 Å unconstrained, ~93.2% with a specified pocket. Supplies our PB-valid feature/label; provides no guarantee itself.

## Scoop risks (re-sweep near submission)

- Residue-level **conformal risk control for protein structure** with pLDDT abstention (arXiv:2509.20345) — abstains per-residue, not per protein-ligand complex on a decision-relevant pose label.
- Generic ranked-abstention theory — **Confidence Gate Theorem** (arXiv:2603.09947).
- Any follow-up that bolts a calibration/abstention layer onto RNP or FoldBench — the most likely genuine competitor.

Do a targeted final sweep for "conformal" + "co-folding / pose / binding mode" before submitting.

## The defensible novelty cell

model-agnostic **×** training-free **×** guaranteed accept/abstain **×** shift-robust (weighted + group-conditional on training similarity) **×** pose-correctness label. No found paper occupies it.
