# Know When to Fold — Project Plan

**Working title:** Know When to Fold: Distribution-Shift-Aware Selective Prediction for Protein–Ligand Co-Folding

This is the canonical research plan and north-star document. `CLAUDE.md` holds the working conventions and grounded facts; this file holds the scientific argument. Keep both in sync when scope changes.

---

## 1. One-sentence contribution

A calibrated, model-agnostic reliability layer that converts co-folding confidence into risk-controlled accept/abstain decisions with coverage guarantees; demonstrates that standard conformal guarantees collapse under the novel-pocket / novel-chemotype shift central to drug discovery; and restores them with shift-robust conformal keyed on training similarity.

## 2. Precise novelty delta (related-work defense)

- **vs. Mac1 / COValid / AF3-ligand-discovery papers:** they show confidence correlates with accuracy/potency and stop there. We deliver a *guaranteed decision procedure* and evaluate it as *selective prediction*, not correlation.
- **vs. CoDrug and weighted-conformal-under-shift:** that is conformal for *scalar molecular properties*. We are first to apply selective prediction / conformal to *co-folding pose reliability* (a structured 3D output), and to use *training-set structural/chemical similarity* as the covariate-shift variable.
- **vs. RNP / FoldBench:** they report that accuracy drops with novelty. We turn novelty into a stratifier that *repairs per-stratum coverage* with group-conditional conformal (the operative guarantee) and yields an operational abstention rule; weighted conformal is the label-free complement, valid only where reliability drift is small (it abstains under the concept shift novel pockets induce).
- **vs. SiteAF3 / AF-ClaSeq:** those *improve* low-confidence predictions. We *decide whether to trust any prediction* — complementary, not overlapping.

## 3. Research questions and hypotheses

| RQ | Question | Hypothesis |
|----|----------|-----------|
| RQ1 | Can native co-folding confidences (ipTM, PAE, PDE, ligand-pLDDT) be conformalized into valid coverage on i.i.d. test data? | H1: yes, roughly. |
| RQ2 | Does that coverage hold under novelty-stratified test sets? | H2: no — systematic under-coverage on high-novelty strata. |
| RQ3 | Do shift-robust conformal methods (weighted / group-conditional) restore guarantees using training-similarity? | H3: yes, closing most of the gap. |
| RQ4 | Does the abstention rule beat native-confidence thresholding on risk–coverage / AURC, and improve a downstream decision (screening enrichment or FEP starting-structure quality)? | H4: yes. |
| RQ5 | Is the layer model-agnostic (AF3/Boltz/Chai) and task-agnostic (pose / interface / affinity-ranking)? | H5: yes, with model-specific calibration. |

## 4. Formal setup

- **Prediction target Y:** binding-mode correctness — binary "ligand-RMSD ≤ 2 Å" (primary); secondary continuous pose-RMSD, interface DockQ, and binder-vs-decoy for the screening arm.
- **Nonconformity score / features X:** native confidences (ipTM, PAE, PDE, per-ligand pLDDT, Boltz-2 affinity confidence) + derived signals: (a) intra-model ensemble disagreement across diffusion samples, (b) cross-model agreement (AF3 vs Boltz vs Chai), (c) physical-validity flags (PoseBusters pass, strain), (d) training-novelty (ligand Tanimoto-to-nearest-train, pocket sequence/structure similarity, temporal post-cutoff flag).
- **Selective prediction:** a gate `g(X) ∈ {accept, abstain}`; report *risk* (error among accepted) vs *coverage* (fraction accepted), the risk–coverage curve, and AURC.
- **Conformal:** split conformal for the i.i.d. baseline guarantee; then weighted conformal (likelihood-ratio weights from a novelty/density model) and Mondrian / group-conditional conformal (calibrate per novelty stratum) for shift-robust coverage; optionally Learn-then-Test / risk-controlling prediction sets to control the RMSD-threshold risk directly.
- **Shift variable:** training similarity is both the *stratifier* (for conditional coverage) and the *weight source* (for weighted conformal) — the conceptual center.

## 5. Data (reuse-first, low GPU)

- **Released predictions to consume:** Runs N' Poses (multi-model predictions + training-similarity metadata), FoldBench (multi-task, novelty-graded), the Mac1 557-ligand prospective set (post-cutoff, single-target depth). PoseBusters V2 for physical-validity features.
- **Generate only what's missing:** a modest held-out slice for cross-model agreement / diffusion-sample ensembles if not in the releases — the only real GPU cost, and it's bounded.
- **Splits that respect shift:** calibration/test partitions along three shift axes — ligand-novelty, pocket-novelty, temporal — plus a novelty-stratified test grid so RQ2 is directly measurable.

## 6. Experiments (figure-by-figure)

- **E1 — Baseline validity (i.i.d.):** native-confidence conformal achieves nominal coverage on random splits. Establishes the method isn't broken.
- **E2 — The exchangeability break (money figure):** coverage vs novelty stratum shows systematic under-coverage on novel pockets/chemotypes. The negative result that motivates everything.
- **E3 — The fix:** weighted + group-conditional conformal restores coverage; report residual gap and where it persists.
- **E4 — Selective-prediction utility:** risk–coverage curves and AURC for (native-confidence gate) vs (conformal reliability layer); quote operating points ("at 50% coverage, retained poses 95% correct"). Include conditional coverage by stratum, not just marginal.
- **E5 — Generality:** repeat across AF3/Boltz/Chai and across pose/interface/affinity-rank; show it's a property of the approach.
- **E6 — Downstream payoff:** does abstaining on unreliable poses improve virtual-screening enrichment (Mac1 screens or a DUD-E-style set) or FEP starting-structure quality vs. using all predictions? This lifts it from "nice statistics" to "changes practice."
- **Ablations:** feature importance (novelty vs ensemble disagreement vs native confidence); marginal vs conditional coverage; calibration-set-size sensitivity for rare strata.

## 7. Falsifiable success criteria

E2 shows a clear, quantified coverage collapse on high-novelty strata; E3 closes most of it; E4 shows the conformal gate dominates native-confidence thresholding on AURC; E6 shows a measurable downstream lift. **If E2 shows no collapse** (native conformal already robust), that is a publishable positive surprise about co-folding calibration — the project is designed to be worth running either way.

## 8. Risks, limitations, mitigations

- **Estimating the shift/weights is hard** (weighted conformal's Achilles heel) → report sensitivity to the density model; fall back to group-conditional conformal, which needs only stratum labels, not weights.
- **Label-definition sensitivity** (2 Å is a convention) → report across thresholds and use continuous RMSD risk.
- **Small calibration sets in rare strata** → pool strata hierarchically; report where guarantees become vacuous rather than hiding it.
- **"Is the guarantee practically useful?"** → E6 answers directly; without it, reviewers will ask.
- **Honest scope limit:** guarantees are over the chosen correctness definition and the sampled shift axes, not a universal "trustworthiness."

## 9. Anticipated reviewer objections → responses

- *"Confidence-tracks-accuracy is known."* → Correct, and cited; our contribution is the guaranteed decision rule and the exchangeability-break analysis, not the correlation.
- *"Conformal under shift exists (CoDrug)."* → For scalar properties; we're first for structured pose reliability and first to key the shift on training similarity for co-folding.
- *"Why not retrain a better confidence head?"* → Model-agnostic, training-free, works on frozen released models, and provides guarantees a learned head does not.

## 10. Deliverables and venue mapping

- **Open-source reliability layer** (pip-installable, wraps AF3/Boltz/Chai outputs) → the artifact anchoring a *Journal of Cheminformatics* or *Digital Discovery* paper.
- **Method + theory-flavored framing** → MLSB @ NeurIPS (early Sept, non-archival) and MoML (Sept, non-archival); reach for a main-conference selective-prediction framing if E2/E3 are clean.
- **Full applied paper with the downstream arm** → *PLOS Computational Biology* (reach) or the fast *Digital Discovery* / *J. Cheminformatics* route for a Dec–Jan acceptance.
- **bioRxiv + arXiv day one.**

## 11. Why this is the best version

Not a benchmark (crowded) or a correlation study (being scooped) — a method with a guarantee, a sharp negative result at its heart, a model-agnostic released tool, and a downstream payoff, on a compute profile (reuse + CPU-heavy conformal) the available assets suit. The exchangeability break is a clean, quotable finding that gets cited beyond the immediate subfield.
