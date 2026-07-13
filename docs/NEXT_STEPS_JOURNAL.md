# From MoML short paper to journal / top-tier: a merit-and-prestige roadmap

This plan takes `foldgate` from the current 4-page MoML short paper (E1--E19, the reliability
layer plus the released-screen decision number) to an archival paper that a strong journal or a
top-tier ML venue would accept. Items are ordered by prestige-per-unit-effort. Decide the venue
first, because it changes what counts as "done".

## 1. Venue strategy (decide first)

The work now spans two audiences, and the strongest move is to split it into two papers rather
than force one venue to value both halves.

- **Applied / tool paper (recommended primary): Digital Discovery (RSC) or J. Cheminformatics.**
  Narrative: a training-free, model-agnostic reliability layer for co-folding that gives
  risk-controlled accept/abstain decisions, exposes the novelty-shift failure, and lifts
  virtual-screening enrichment on novel targets. This audience rewards the decision number (E16),
  the released tool, and the honesty. Digital Discovery is gold open access with a fast first
  decision and fits the method-plus-software shape. This is the archival home for the full study.
- **Methods / theory paper (reach, higher prestige): a top ML or stats venue (ICLR / NeurIPS /
  AISTATS / a stats journal).** Narrative: selective risk control under concept shift, the
  reliability-drift diagnostic, the label-free worst-subpopulation (CVaR) certificate, localized
  conformal, and the certified concept-shift floor, with co-folding as the driving application.
  This needs the theory tightened (Sec. 3 below) and a second dataset beyond RNP.
- **Non-archival signals to collect on the way (do all):** MoML 2026 (Sept 1, the current PDF),
  MLSB @ NeurIPS 2026 extended abstract (~Oct). These do not burn journal novelty and build
  reviewer familiarity.

Prestige ceiling: if the prospective validation (item 2a) lands, a **Nature Communications /
Nature Methods** submission for the combined story becomes credible, because a guaranteed
accept/abstain layer that demonstrably changes a prospective screening decision is a
general-interest result, not a niche one.

## 2. The highest-prestige additions (do these to reach top-tier)

**(a) A prospective validation.** The single biggest lever. Two routes: (i) the Mac1 / Cofolding
prospective screen once its crystal coordinates and per-compound co-folding confidences release
(currently embargoed), or (ii) partner on any live screen where the reliability gate is applied
before assay and the retained hit rate is measured. A prospective "the gate changed which
compounds we tested and the hit rate went up" result is what separates a top-tier paper from a
strong one.

**(b) A true feature-matched cross-dataset positive on FoldBench.** The current E15 transfer is
feature-parity-limited because FoldBench does not release the confidence fields the combined score
needs. Regenerate predictions for the 558 protein-ligand FoldBench targets with a single wired
model (Protenix's `make_predictions.sh` is already integrated and its summary_confidence JSON
emits `chain_pair_iptm`), recover interface-ipTM, and run the frozen calibrate-on-RNP /
deploy-on-FoldBench test at feature parity. Budget the GPU explicitly and keep it bounded to one
model. This closes the "single benchmark" weakness with a real positive rather than an honest
negative.

**(c) Finish and stress the depth trio.** Localized (randomly-localized) conformal keyed
continuously on novelty (E18) as the principled continuous replacement for discrete Mondrian
bins; a distributionally-robust extension that turns the worst-subpopulation number into a
certified robustness radius (bracketing the achievable region together with the concept-shift
floor E19); and simultaneous multi-model certificates. Validate each on synthetic data with known
conditional risk, as we already do for the CVaR bound. Frame the localized guarantee as
approximate neighborhood-conditional (Barber et al. 2021 forbids exact distribution-free
conditional coverage), never as exact.

## 3. Statistical and experimental hardening (reviewer-proofing)

- **Broaden the screening study (E16) to full strength.** All three models (AF3, Boltz-2,
  Protenix), affinity-head vs pose-confidence gates as decoupled tasks, a docking head-to-head
  with property-matched controls, BEDROC and EF with bootstrap CIs over targets, and the
  active-to-training similarity axis reported as a proper shift curve. Pre-register the abstention
  operating point (fixed by LTT on held-out pose correctness, never on test enrichment) as a
  hashed commitment so no reviewer can claim tuning on the enrichment.
- **Second, affinity-framed screening set** (Zenodo 18669539, Boltz-2 affinity with binder
  labels) for a decoupled potency task, showing the layer is not specific to one screen.
- **Multiplicity and estimator care.** Report AURC and BEDROC with confidence intervals and the
  known estimator caveats; apply a simultaneous correction when certifying across models and
  strata together.
- **Leakage discipline everywhere.** The pooled-split leakage audit (95.6% of test targets shared
  with calibration under a naive split) should gate every pooled analysis; keep the target-grouped
  (LOTO) protocol as the default, not an add-on.

## 4. Tool, reproducibility, and positioning

- Mint a Zenodo DOI from a tagged `v0.1.0` GitHub release; wire the screening download into the
  data pipeline (DVC stage) so `make repro` rebuilds every figure end to end.
- Ship the reliability layer as a small `pip`-installable package with the model and data cards
  already drafted, and a one-command quickstart notebook (calibrate, break, repair, screen).
- **Scoop re-sweep immediately before submission**: a targeted search over conformal + co-folding
  / pose / binding-mode / abstention, plus the adjacent 2026 preprints (Confidence Gate Theorem,
  residue-level conformal risk control, robust-validation DRO conformal). The cell
  (training-free x guaranteed accept/abstain x shift-robust x pose-correctness label x screening
  decision) is currently unoccupied; confirm that before submitting.

## 5. Suggested sequence

1. Broaden E16 + pre-register the operating point, add the second affinity set (weeks, no GPU).
2. Finish E18 localized + the DRO radius; tighten the theory writeup (weeks, no GPU).
3. Regenerate Protenix FoldBench confidences for the feature-matched transfer (bounded GPU).
4. Submit the applied paper to Digital Discovery; in parallel prepare the methods paper.
5. Pursue the prospective validation as it becomes available; if it lands, escalate the venue.
