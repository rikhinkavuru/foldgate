# Journal / STS / Nature Upgrade Plan

*Scope: the ARCHIVAL paper only (`paper/foldgate_journal.tex`, Digital Discovery / Regeneron STS,
Nature-tier ambition). NOT the MoML short paper, which is finalized and submit-ready. Author: Rikhin
Kavuru. Written 2026-07-17, grounded against the repo state on disk (every "current state" line below
was verified, not assumed). Companion to `PLAN.md`, `RESULTS.md`, and `docs/theory/D1_D2_EXECUTION_PLAN.md`.*

This is the build sheet for the six improvements that raise the archival paper from "strong method
paper" toward finalist / high-impact grade, plus the one thing that is the real ceiling: independent
validation. Each item states what exists today, the exact change, the effort, the gate, and the
honest risk. Nothing here is required for MoML.

---

## 0. The dataset reality (read first, it reframes everything)

**You have two datasets, not one, and one model that is not a dataset.**

- Datasets: **Runs N' Poses (RNP)** and **FoldBench**.
- Models (six): AF3, Boltz-1, Boltz-1x, Boltz-2, Chai-1, Protenix. Protenix is a model, it appears
  in both datasets, it is not itself a dataset.

RNP is the full validation: five governed models, released predictions, released BiSyRMSD labels,
one calibration/test split. FoldBench (experiment E15b) is a **genuine but partial** second dataset:

- Protenix-only. The other FoldBench models ship `ranking_score` only, so the RNP gate's feature
  (interface-ipTM) is unavailable for them; they were not regenerated
  (`experiments/e15b_foldbench_iptm_transfer.py:22`).
- The interface-ipTM feature was **regenerated locally** (Protenix v0.5.5, 5 seeds x 5 samples,
  ColabFold MSAs), not taken from FoldBench's public release, and the ligand-RMSD labels were
  **self-scored** against the deposited assemblies.
- Regen top-1 success is **0.40 vs FoldBench's released 0.57**, so the regeneration is not a faithful
  reproduction of FoldBench Protenix (feature and label are self-consistent within the one run, which
  is why the transfer is still valid, but it is not an independent benchmark).

So the honest position is "1.5 datasets": one full (RNP), one partial single-model regenerated
(FoldBench). This is the single biggest limitation and the thing that most limits a Nature-tier claim.
Items 1 and 7 below both attack it.

The result on that partial dataset is real and worth keeping: the frozen RNP interface-ipTM gate
transfers with risk-coverage AURC **0.380** against the matched `ranking_score` control **0.454**, and
the same frozen threshold accepts **5%** of FoldBench against **26%** at home (the coverage collapse,
reproduced cross-dataset). Source: `results/e15b_foldbench_iptm_transfer.json`.

---

## 1. Archive the FoldBench regeneration to a Zenodo dataset DOI (reproducibility)

**Priority: do first. Effort: half a day, no GPU. Closes an audit finding.**

- *Current state.* The regen data is on disk (`data/external/foldbench/`, ~21 MB:
  `foldbench_protenix_regen.csv`, `regen_scores.csv`, `foldbench_protein_ligand_rmsd_lddtlp.csv`,
  `foldbench_protein_ligand_confidence_rmsd.csv`) but is **gitignored and unarchived**, so a referee
  cannot reproduce the 0.380-vs-0.454 numbers. The paper currently says the predictions are "available
  on request." The existing `.zenodo.json` is an `upload_type: software` record for the code; this is a
  separate **dataset** record.
- *The change.* Create a Zenodo dataset record (see the step-by-step in the handoff message / the
  README you write alongside it): upload the four CSVs plus a `README_regen.md` stating Protenix
  v0.5.5, 5x5 sampling, ColabFold MSAs, self-scored vs deposited assemblies, and the 0.40-vs-0.57
  success gap. Reserve the DOI, add `isDerivedFrom` -> FoldBench DOI and `isSupplementTo` -> the arXiv
  preprint. Then replace the "available on request" hedge in the paper's Data-and-code-availability
  statement with the DOI.
- *Gate.* None; this is pure hygiene. *Risk.* Confirm FoldBench's license permits redistribution of a
  derived-predictions table (FoldBench is MIT, so yes; cite it).

---

## 2. Add the repair figure (complete the visual narrative)

**Priority: cheap win, do now. Effort: ~2 hours, no GPU.**

- *Current state.* The break figure already sits in the paper (`results/figures/e2_all_models.png`,
  Fig. 1). The repair result exists as data and a rendered figure but is NOT in the paper:
  `results/figures/e3_shift_repair.png` and `results/e3_shift_repair.json` / `e3c_combined_conditional.json`.
- *The change.* Add `e3_shift_repair.png` (or re-cut it for print) as a second figure in
  `sec:repair`, showing group-conditional calibration restoring per-stratum realized risk under alpha
  where the global gate over-shot. Reference it from the repair prose. If the current PNG is busy,
  re-cut a two-panel version (global gate risk-by-stratum vs group-conditional risk-by-stratum) from
  `e3_shift_repair.json` in matplotlib; the `experiments/d1_d2_figures.py` scaffolding is a template.
- *Gate.* The figure must show risk under alpha post-repair on the strata the global gate violated
  (S3 especially). *Risk.* Low. Keep it honest about the coverage cost (repair abstains more).

---

## 3. Lead the utility with operating points, not AURC (reframing)

**Priority: cheap, high-persuasion. Effort: ~2 hours, no GPU, no new experiment.**

- *Current state.* `sec:screen` leads with AURC reductions (28-41%). The operating-point numbers
  already exist in `results/e4_selective_utility.json` and in the paper's own Table 1 (coverage at
  certified error, e.g. AF3 0.22 -> 0.71 native -> combined at alpha=0.2). A practitioner reads
  "at 50% coverage I keep 95% correct poses" far better than "AURC down a third."
- *The change.* Reorder the utility paragraph to lead with the quotable operating point ("at X%
  coverage the retained poses are Y% correct, against Z% for a native-confidence gate at the same
  coverage"), then give AURC as the aggregate. Pull the exact (coverage, purity) pairs from
  `e4_selective_utility.json`; do not invent them.
- *Gate.* Every quoted operating point must be an actual point on the risk-coverage curve in the
  artifact. *Risk.* None beyond number fidelity.

---

## 4. Elevate reliability drift as a deploy-time diagnostic (reframing + small analysis)

**Priority: medium, differentiates the contribution. Effort: ~1 day, no GPU.**

- *Current state.* Reliability drift D(nu) is computed (`results/e12_reliability_drift.json`) and used
  in `sec:break` to diagnose the shift as concept (not covariate). It is presented as an explanation,
  not as a tool. But it is arguably a standalone contribution: it tells a practitioner, before
  deployment and without target labels needed for the decision, WHETHER weighted conformal will work
  (small drift -> covariate-dominated -> weighted CP repairs; large drift -> concept-dominated ->
  weighted CP will abstain, use group-conditional). This is the actionable "which repair do I reach
  for" decision.
- *The change.* Add a short paragraph (or a boxed decision rule) framing drift as the deploy-time
  selector between the two repairs, tied to the theorem: drift is the empirical proxy for the
  accept-region concept gap the impossibility theorem names. Optionally add a small validation: on the
  strata where drift is small, show weighted CP repairs; where large, show it abstains, matching the
  E3b result already in `RESULTS.md`. Ground the thresholds in the measured drift values (structural
  +0.47 to +0.63 vs temporal <=0.07).
- *Gate.* The diagnostic must actually predict the E3b weighted-CP outcome per stratum, not just
  correlate loosely. *Risk.* Do not overclaim it as a guarantee; it is a heuristic selector. State so.

---

## 5. De-emphasize the combiner, elevate the theory (framing / restructure)

**Priority: medium, sharpens the STS "what is genuinely yours." Effort: ~half a day, prose only.**

- *Current state.* The gradient-boosted combined score is presented as a co-equal contribution. It is
  the least novel part (a small learned model) and it invites the STS/reviewer question "so you did
  train something after all," which the paper answers honestly but defensively.
- *The change.* Reframe the combiner as engineering that improves the operating point, and let the two
  genuinely novel pieces carry the contribution weight: (a) the impossibility+achievability theorem
  and its feasibility-frontier empirical shadow, (b) the measured exchangeability break with the
  concept-shift identification. In the contribution list and abstract, lead with the theorem, present
  the combiner as "an optional calibration-only score that improves coverage, not a new model." This
  is a repositioning, not a cut; keep all the combiner results.
- *Gate.* The theorem framing must stay exactly as rigorous as `THEOREM_RECONCILED.md` (coverage-
  pinned, neighborhood-conditional, the ratio-risk caveat). *Risk.* None; it strengthens the interview
  posture.

---

## 6. Head-to-head against the Hou AF3-VS baseline (the strongest comparator)

**Priority: medium-high, directly answers "what do you beat." Effort: ~1-2 days, no GPU (data on disk).**

- *Current state.* The screening arm cites the Hou group's AF3-VS work (Chem Sci D5SC06481C, the
  released-screen source, `\citep{shen2026vs}`) and the paper's own screening results are in
  `results/e16_selective_screening.json`, `e20_screening_broad.json`, `e23_screening_honest.json`,
  `e24_screening_baseline.json`. The comparison is currently in prose, not a head-to-head on identical
  targets. The Hou line is the designated "baseline to beat" (`RELATED_WORK.md`): it uses ipTM as a VS
  ranking signal and documents novelty degradation, but supplies NO guarantee and NO abstention.
- *The change.* Build one figure/table that puts, on the SAME screen targets, (i) the Hou-style
  ipTM-ranking-without-abstention, against (ii) the layer's guaranteed accept/abstain gate, measured on
  the metric the guarantee governs (pose-correctness selective risk / retained-active enrichment under
  abstention), with the random-abstention control (`e23`/`e24` already implement the honest baselines).
  The honest framing must stay: the headline EF rides on the ungoverned Boltz-2 affinity head, so the
  head-to-head is on the POSE signal the layer certifies, not the affinity head. See
  `docs/theory/DATA_VERIFICATION.md` (J2) for the exact pose-vs-affinity split so this is not
  overclaimed.
- *Gate.* The comparison must be on the pose signal the layer actually governs, with CIs (screening
  EF CIs are wide: LIT-PCBA n=5, GPCR n=16 -- report them). *Risk.* If the pose signal does not clearly
  beat naive ipTM ranking under abstention, report it as a null; the contribution is the guarantee, not
  a bigger EF.

---

## 7. The ceiling: a full second dataset, ideally prospective

**Priority: highest scientific value, but gated on GPU or on an embargo. This is the Nature-tier item.**

Two routes, neither cheap:

- **7a. Strengthen FoldBench to the full model panel (GPU).** Regenerate the other FoldBench models
  (AF3/Boltz/Chai) the way E15b did for Protenix, so the cross-dataset transfer is a five-model result
  matching RNP rather than Protenix-only. This converts the "0.5 dataset" into a full second dataset.
  Needs the J1-style GPU regeneration pipeline (the repo has it from the Protenix run) on ~558
  FoldBench protein-ligand targets x K models. Bounded GPU cost. Also archive these to the same Zenodo
  dataset (item 1). This is the highest-value item that is actually actionable now if GPU is available.
- **7b. Mac1 prospective validation (blocked, watch).** The 557-target Mac1 co-folding benchmark
  (eLife reviewed-preprint 110475) is the ideal: genuinely prospective, RMSD-labeled, the gold standard
  for a reliability claim. But the crystal ground-truth is **embargoed** ("released after a small delay
  to preserve blind prediction"; `D1_D2_EXECUTION_PLAN.md` Section 0). Set a watch on the coordinate
  release (github.com/jongbin99/Cofolding); the moment coords drop, this is the single most valuable
  experiment in the whole project and should preempt everything else.
- *Gate for 7a.* The regenerated multi-model FoldBench gate must transfer (AURC beats the matched
  control) on the full panel, and the regen success gap must be disclosed per model as in E15b.
  *Risk.* GPU cost; regeneration fidelity (disclose the success gap honestly, as E15b already does).

---

## Recommended execution order

1. **Item 1 (Zenodo)** and **Item 2 (repair figure)** -- same day, cheap, close real gaps.
2. **Item 3 (operating points)** and **Item 5 (de-emphasize combiner)** -- prose repositioning, half a day.
3. **Item 4 (drift as diagnostic)** -- one day, adds a differentiated contribution.
4. **Item 6 (Hou head-to-head)** -- one to two days, answers "what do you beat."
5. **Item 7a (full FoldBench panel)** -- when GPU is available; the biggest scientific lift that is
   actionable now.
6. **Item 7b (Mac1)** -- watch the embargo; execute the instant coords release.

Items 1-6 are all CPU / prose / existing-data and can land in a week. Item 7 is the ceiling and the
one that most separates a strong method paper from a finalist / high-impact one.

## The honest ceiling statement (keep it in the paper)

Even after items 1-6, the co-folding pose guarantee is validated on one full dataset plus one partial
regenerated one, all retrospective. State this plainly. The theorem is dataset-independent and is the
strongest, most defensible contribution; the empirical reliability claim is bounded by the validation
breadth until item 7 lands. Do not let the paper imply broader validation than it has.
