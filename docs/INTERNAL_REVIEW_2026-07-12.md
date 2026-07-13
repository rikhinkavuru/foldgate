# Internal review — foldgate paper (independent 4-lens panel, 2026-07-12)

A 3-reviewer + chair panel (conformal-statistician, structural-bio domain expert, harshest
Reviewer-2; a 4th "applied" reviewer returned a null stub and was ignored) read the paper plus
PLAN/RELATED_WORK/RESULTS/PROGRESS and the conformal source, and **verified the headline numbers
against the results JSON**. This is the honest read, kept for the author.

## Consensus rating
- Non-archival workshop (MoML / MLSB): **strong accept**, ~6.5–7/10.
- Archival journal (Digital Discovery / J. Cheminf) **as-is**: **borderline / reject-pending-additions**, ~4.5–5.5.
- Top-tier ML/stats (NeurIPS/ICLR/AISTATS/JMLR) on methods novelty: **not close**, ~2–3.

## Novelty
Domain-application novelty, **not** methods novelty. Every conformal ingredient is off-the-shelf
and applied as published (LTT/RCPS, Mondrian, Snell QRC-CVaR, Hore-Barber RLCP, Cauchois chi-square
DRO, Tibshirani weighted CP, WSR). No new estimator, bound, or theorem. Genuinely new-and-real:
(1) first application of conformal selective prediction to co-folding pose correctness keyed on
training-set similarity, in a scoop cell still unoccupied as of the 2026-07 sweep; (2) the scoped,
CI-bounded concept-vs-covariate decomposition that proves which repair is admissible. "Reliability
drift" is weaker — a rebranding of known ipTM/ranking-score miscalibration under novelty.

## Rigor
Adequate-to-rigorous with exemplary internal honesty, but a headline that oversells the internals.
Strong: math correct where checked (CVaR monotone lift, exact two-point Pearson chi-square worst
case, WSR/Ville, fixed-sequence LTT); the **95.6% target-leakage audit** caught and corrected
(targets recur across models → naive pooled split leaks; switched to GroupKFold/LOTO, AURC gain
survives out-of-fold); synthetic per-certifier validity + bootstrap CIs excluding zero; vacuous
certificates reported as numbers.

## The real weaknesses (verified in the JSON — fix before an archival submission)
1. **FUNDAMENTAL — guarantee/decision disconnect.** The flagship screening number (GPCR EF@1%
   **26.6 vs docking 2.0**) is the **Boltz-2 affinity-probability head**, which the conformal layer
   never touches. The **pose-confidence** the guarantee is actually built on yields EF **9.3**, and
   the screening ranking carries **no coverage guarantee at all**. The rigorous machinery and the
   impressive number never meet on the same result. Fix: demote/drop the affinity-head EF as the
   headline; report the ipTM-based selective number honestly.
2. **FUNDAMENTAL — certificates vacuous exactly on the novel regime the paper is about.** On
   deploy-on-novel: group-conditional abstains on S3/S4; CVaR m* never reaches 0.5; DRO at the
   actual LTT gate is `all_certified=false`, m*=1.0, rho*=0.0 for every model (e17/e22). The touted
   "m*=0.52, rho*=0.10 across all 5 models" holds only at an arbitrary fixed 20%-coverage point, not
   the deployed operating point. So "we repair coverage three ways" is not supported on truly novel
   pockets; the only working response there is **abstain** (which restates base pose-correctness
   ~53% on S4 and needs no conformal machinery). Not fully fixable without a stronger base model or
   a prospective labeled novel set — partly a property of the co-folding models, not the layer.
3. **FIXABLE-BUT-BLOCKING — essentially single-dataset.** Every guaranteed-conformal claim is
   RNP-only. FoldBench transfer is feature-limited, under-controls for Protenix, and the learned
   combiner does not generalize (AURC −23% to +9%), so the 25–40% AURC headline may be RNP-specific.
   Fix: a second real dataset with matched novelty features (this is the FoldBench Protenix GPU regen).
4. **FIXABLE — screening baseline fairness + inert gate.** The docking baseline (Gnina EF ~2) is
   inherited un-validated from the source screen (Chem Sci D5SC06481C), so "near random" may be a
   weak protocol as much as difficulty. The pre-registered gate is **inert** at its threshold
   (`ef_reg_gate == ef_full`), so the selective claim rides a post-hoc 50%-coverage median-ipTM cut
   (+~0.5 EF over full ranking). Fix: expert-prepped docking, scaffold-split actives, decoy-quality check.
5. **FIXABLE, honesty-critical — selective reporting.** The "enrichment collapses under novelty"
   story holds on GPCR/DEKOIS but **reverses on LIT-PCBA**; only the favorable direction is
   foregrounded (and PROGRESS concedes the similarity-axis semantics were never verified). "No
   covariate reweighting can help" overstates a proposition proven only for reweighting through the
   scalar score. Fix: tighten prose to match the scoped code.
6. **For a stats venue only — zero new methodology + uncontrolled multiplicity** across the 5 models
   for the main claims. Decisive only if targeting AISTATS/NeurIPS/JMLR.

## Genuinely valuable contributions (rank)
1. The concept-vs-covariate decomposition (AF3 S3: 0.309 of the 0.328 gap is concept, CI excludes
   zero) — a useful negative result telling a practitioner which repair is even possible.
2. Statistical hygiene as a contribution to how the subfield should evaluate such tools (the leakage
   audit, LOTO, exact-binomial LTT, honest vacuity reporting).
3. The per-stratum error quantification on accepted poses (marginal ~0.18 while S3/S4 = 0.37–0.43),
   cleanly split along ligand/pocket axes and near-zero on temporal.
4. The out-of-fold within-RNP AURC gain (25–40%) + non-circular downstream signal (accepted poses
   recover 0.90 vs 0.72 correct crystal contacts).
5. A released, reuse-first, CPU-only, torch-free pip package — engineering/reproducibility above norm.

## Bottom line
A genuinely honest, unusually well-audited domain-application paper with one real intellectual
payoff (the concept-shift decomposition) and one exemplary rigor move (the leakage catch). Not a
methods contribution and doesn't pretend to be. Two fundamental problems — the guarantee is vacuous
on the novel regime it is about, and the flagship decision is decoupled from the guaranteed layer —
are wounds to the impact story, not the execution. **Ship now as a workshop paper; foreground the
per-stratum error and the concept-shift floor, and stop leading with the affinity-head EF.** For an
archival journal, not ready until (1) a second real dataset replicates the break and the AURC gain,
and (2) the reported decision and the guarantee meet on the same object.
