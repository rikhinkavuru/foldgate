# Methods (as implemented)

This documents the exact statistical construction the code implements, so the
paper and the repo never drift. See `PLAN.md` for the thesis and `CLAUDE.md`
for grounded facts.

## Unit of prediction

The decision-relevant unit is the **delivered pose**: for each (system, ligand,
model) we take the top-1 sample by the model's own `ranking_score` — what a
practitioner actually uses. Correctness label `Y = 1` iff BiSyRMSD ≤ 2 Å.
Source: released Runs N' Poses predictions (`data/raw/predictions/`), proper
ligands only. 13,536 delivered poses across 6 models.

## Confidence / nonconformity score

- **Native (baseline):** `ranking_score` (fully model-native; for AF3,
  `0.8·ipTM + 0.2·pTM + 0.5·frac_disordered − 100·has_clash`).
- **Combined (primary):** a calibration-only `ScoreCombiner`
  (HistGradientBoosting → P(correct)) over native confidence + interface
  chain-pair ipTM + PoseBusters validity + intra-model ensemble spread across the
  5 diffusion samples + ligand difficulty (MW, rotatable bonds, heavy atoms).
  Training-set novelty is deliberately **excluded** from the score; novelty
  enters only through calibration, keeping the score and the shift variable
  separate. Higher score = more likely correct, so it plugs into the same
  threshold machinery as a raw confidence.

## Selective risk control (the gate)

Accept iff score ≥ τ. Target: error among accepted ≤ α with confidence 1 − δ
(default α = 0.20, δ = 0.10; the paper also reports α = 0.10 and the full curve).

τ is chosen by **Learn-then-Test with fixed-sequence testing** (Angelopoulos,
Bates, Candès, Jordan, Lei 2021). Over a pre-specified, data-independent coverage
grid (smallest accept set first), each H0(τ): P(error | score ≥ τ) ≥ α is tested
with an exact binomial p-value P(Bin(n_accept, α) ≤ errors); the accept set grows
while H0 keeps being rejected at level δ and stops at the first failure.
Fixed-sequence testing controls the family-wise error at δ without a Bonferroni
penalty, giving P(selective risk ≤ α) ≥ 1 − δ, finite-sample and distribution-free.

The binomial test conditions on the accept count — the correct tool for
error-among-accepted. A full-sample Hoeffding/RCPS upper bound over-penalises small
accept sets (its penalty can exceed α before anything is accepted); `hb_upper_bound`
(Hoeffding-Bentkus) and `naive_threshold` (no correction, the practitioner
baseline) are provided for comparison. Validity is unit-tested empirically
(`tests/test_conformal.py`).

## Shift-robust repair

- **Group-conditional / Mondrian (E3):** a separate LTT threshold per novelty
  stratum. Needs only stratum labels (RNP ships these). Rigorously restores the
  per-stratum guarantee, at an honest coverage cost where the model is unreliable.
- **Weighted covariate-shift (E3b):** calibrate on the familiar (low-novelty)
  source, reweight calibration points by the likelihood ratio
  w(x) = p_target(x)/p_source(x) estimated with a logistic source-vs-target
  classifier on the novelty covariates, and deploy on the novel target **without
  target labels** (Tibshirani et al. 2019). Coverage is approximate under
  estimated weights (the known limitation); group-conditional is the rigorous
  fallback.

## Novelty (the covariate-shift variable)

From RNP `annotations.csv`, pre-computed similarity-to-nearest-training-system:
`morgan_tanimoto` (ECFP4, ligand), `sucos_shape_pocket_qcov` (pocket),
`target_release_date` (temporal). Scaled to [0,1]; **NaN similarity = no training
analog = maximally novel** and forms its own top stratum. Strata are quantile bins
of similarity plus the no-analog stratum.

## Splits

- **E1/E2/E3 (native score):** random 50/50 calibration/test, 300 repeats.
- **E4 / combined score:** 3-way 40/30/30 train/cal/test — the combiner is fit on
  TRAIN, τ calibrated on CAL (out-of-sample combiner scores, exchangeable with
  test), evaluated on TEST. This preserves conformal validity for a learned score.
- **E3b:** low-novelty source split into combiner-train + calibration; high-novelty
  target split into (unused-for-weighted) labels + evaluation.

## Metrics

Risk-coverage curve, AURC (area under it, lower = better), realized selective risk
and coverage at certified α, per-stratum conditional coverage, all with percentile
bootstrap CIs (`foldgate.selective`).
