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
Bates, Candès, Jordan, Lei 2021). The hypotheses are ordered by a **pre-specified,
data-independent coverage grid** (accept the top-k fraction, smallest first); the
mapped threshold τ is the corresponding empirical quantile of the calibration
scores, so the *order* is fixed a priori even though the threshold values are
data-dependent (equivalent to a pre-specified sequence of top-k **ranks**). Each
H0(τ): P(error | score ≥ τ) ≥ α is tested with the binomial tail p-value
P(Bin(n_accept, α) ≤ errors); the accept set grows while H0 keeps being rejected at
level δ and stops at the first failure. Fixed-sequence testing controls the
family-wise error at δ without a Bonferroni penalty, giving P(selective risk ≤ α) ≥
1 − δ.

The binomial p-value is **valid, and exact for a homogeneous error rate**; when the
accepted poses have heterogeneous per-pose error probabilities (a Poisson-binomial
sum) the test is **conservative**, because a Poisson-binomial with mean ≤ α is
stochastically dominated in the lower tail by Bin(n_accept, α) (a convex-order /
majorization argument), so P(Bin ≤ errors) upper-bounds the true tail. Conditioning
on the accept count is the correct tool for error-among-accepted; a full-sample
Hoeffding/RCPS upper bound over-penalises small accept sets (its penalty can exceed α
before anything is accepted). `hb_upper_bound` (Hoeffding-Bentkus) and
`naive_threshold` (no correction, the practitioner baseline) are provided for
comparison. Validity is unit-tested empirically on synthetic data where the true risk
is known (`tests/test_conformal.py`).

## Shift-robust repair

- **Group-conditional / Mondrian (E3):** a separate LTT threshold per novelty
  stratum. Needs only stratum labels (RNP ships these). Restores **per-stratum
  marginal** validity (each stratum at 1 − δ); a simultaneous "every stratum holds"
  statement follows by calibrating each at δ/K. Honest coverage cost where the model
  is unreliable.
- **Weighted covariate-shift (E3b):** calibrate on the familiar (low-novelty)
  source, reweight calibration points by the likelihood ratio
  w(x) = p_target(x)/p_source(x), and deploy on the novel target **without target
  labels** (Tibshirani et al. 2019). Weights are estimated **out-of-fold** with a
  probability-calibrated source-vs-target classifier (`estimate_weights_cv`,
  isotonic + StratifiedKFold), which the in-sample logistic fit
  (`estimate_weights`) does not do. Two certifiers: the plug-in Hájek estimator
  (`weighted_threshold`, approximate coverage) and an **importance-weighted LTT**
  (`weighted_ltt_threshold`) that tests the reweighted-source null
  E_source[w(L−α)1_accept] ≥ 0 — equal to the target null R_target ≥ α under
  covariate shift — with a WSR betting p-value on the rescaled weighted losses
  (Almeida et al. 2025). Fixed-sequence testing then gives P(R_target ≤ α) ≥ 1 − δ,
  finite-sample but **conditional on correct weights**. A concept-shift diagnostic
  (`concept_shift_diagnostic`) checks whether P(correct | confidence) itself moves;
  when it does (novel pockets), pure reweighting cannot restore validity and the
  group-conditional certificate is the operative guarantee.

## Continuous-RMSD gate (E9)

Beyond the binary 2 Å label, `continuous_risk_threshold` certifies a gate whose mean
bounded loss min(RMSD, B)/B (B = 4 Å) among the accepted is ≤ a target, with
P(mean ≤ target) ≥ 1 − δ. The default certifier is a **WSR betting bound**
(`wsr_betting_pvalue`, Waudby-Smith & Ramdas 2024): a predictable-plug-in
empirical-Bernstein capital process whose reciprocal max is a valid p-value by Ville's
inequality. It is variance-adaptive and uses the whole sample, so it certifies
non-trivial coverage where a distribution-free Hoeffding bound certifies almost none.
Maurer-Pontil empirical-Bernstein, Hoeffding, and exact-binomial bounds are provided
for comparison; the binomial mode on a 0/1 loss reproduces the binary LTT gate exactly.
The acceptance fraction is co-reported with a Clopper-Pearson interval.

## Pose-agreement features (W1, needs the structure tarball)

`build_pose_features.py` streams the 39.5 GB RNP `prediction_files.tar.gz` once (no disk
extraction) and computes, with spyrmsd + gemmi, structural-consensus signals orthogonal to
the confidence features:

- **intra-model pose diversity** across a model's 25 diffusion predictions (5 seeds x 5
  samples): each sample's ligand-RMSD to the delivered (top ranking_score) pose after
  pocket-Calpha superposition, then `intra_model_pose_std`, `intra_model_pose_median`,
  `pose_consensus_frac` (fraction in the delivered binding mode). A model's own samples share
  ligand atom order, so no symmetry matching is needed.
- **cross-model pose agreement**: the delivered pose's symmetry-corrected ligand-RMSD to the
  other models' delivered poses (`xmodel_pose_rmsd_median/min`, `pose_consensus_cluster_size`),
  matching ligand atoms across models by name and superposing on the shared pocket. Consensus
  is a feature, not independent validation, because co-folding models make correlated errors.

The delivered ligand chain is selected by matching `ligand_num_heavy_atoms` (predicted CIFs
carry no hydrogens), which resolves the drug ligand from cofactors/ions in multi-ligand CIFs.
Superposition uses `minimize=False` so the ligand RMSD measures binding *mode*, not conformer.

## Interaction-fingerprint recovery (E6b / W2, non-circular downstream)

E6 purity equals 1 - selective risk by construction, so it cannot demonstrate downstream value
beyond the guarantee. `interactions.py` computes a genuinely different quantity: the set of
receptor residues within 4.5 A of the delivered ligand (keyed on (seqid, resname), since
predicted and crystal CIFs share numbering but rename chains), and its recall / Jaccard against
the crystal contact set from `ground_truth.tar.gz`. Contact recovery is only correlated with,
not equal to, the 2 A RMSD label, so the accepted-vs-rejected recovery lift (E6b) is a
non-circular downstream result. A screening-enrichment harness (`selective/enrichment.py`:
EF@k%, BEDROC, selective-EF at fixed retained-library size, coverage-enrichment and
active-retention curves, random-abstention control) is implemented and tested, ready for a
screening dataset (DEKOIS 2.0 / LIT-PCBA / the Mac1 prospective screen when released).

## Baselines (E11)

On the same splits: raw `ranking_score`, native chain-pair ipTM (LTT gate),
PoseBusters-pass filter, and Platt/isotonic calibration of the combined score
thresholded at P(correct) ≥ 1 − α. The decisive comparison isolates the *guarantee*
from the *features*: the same combined score as a plain calibrated classifier + fixed
threshold vs. inside the conformal layer. Reported i.i.d. AND under the novelty shift —
calibration carries no finite-sample coverage and breaks under shift exactly as naive
conformal does; only the shift-robust conformal variants repair it.

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
- **E3b:** low-novelty source split into combiner-train + calibration (the weighted
  gates use source labels only); high-novelty target split into a held-out label fold
  (for the rigorous target-calibrated reference and the concept-shift diagnostic) and
  an evaluation fold. Weights are estimated from the target's novelty covariates,
  never its labels.
- **E11:** i.i.d. 3-way split for the ranker/gate comparison; for the shift panel,
  train + calibrate on the familiar source stratum and evaluate on a held-out fold of
  the novel target strata (group-conditional gate calibrated per target stratum).

## Metrics

Risk-coverage curve, AURC (area under it, lower = better), realized selective risk
and coverage at certified α, per-stratum conditional coverage, all with percentile
bootstrap CIs (`foldgate.selective`).
