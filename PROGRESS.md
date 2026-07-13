# PROGRESS.md — SOTA Expansion Ledger (durable memory)

**This file is the single source of truth for the multi-tier SOTA expansion. Re-read at the
start of every work session. Update after every task. Never lose a result number: write it here
the moment it is produced.**

Last updated: 2026-07-12 (session start).

---

## 0. The goal (pin — do not drift)

Turn `foldgate` (training-free conformal accept/abstain reliability layer for co-folding pose
correctness; RNP-only study, drafted MoML short paper) into a **stronger, SOTA-grounded paper**
by executing 4 tiers of improvements, each Exa-grounded and adversarially vetted, then deliver:
1. A **4-page MoML 2026 short paper PDF** (refs/acks/appendices excluded from the 4pp) explaining the work.
2. A **journal / top-tier-conference next-steps plan** (decided on merit + prestige, scope-agnostic).

**Paper thesis (do not wander):** co-folding confidence -> risk-controlled accept/abstain with a
finite-sample selective-risk guarantee; the guarantee **collapses under novel-pocket/chemotype
shift** (E2, the money figure); **repaired** by group-conditional conformal keyed on training
similarity; utility via AURC + non-circular downstream (E6b). Honesty ethos: report where the
guarantee is vacuous; never fabricate a number.

**Hard constraints:** training-free at inference; reuse-first data (no unbounded GPU); torch-free
analysis package; group-conditional is the operative shift guarantee (weighted CP abstains under
the documented concept shift); every claim states where it is vacuous. Prose: no em-dashes, no
rule-of-three, no negative parallelism.

**Env:** `.venv/bin/python` = 3.12 (np 2.5.1, pd 3.0.3, scipy 1.18, sk 1.9). NOT pixi. NOT system python.

---

## 1. Ground truth (verified 2026-07-12)

- Delivered table `data/processed/rnp_delivered.parquet` = **13,535 x 44**, ALL features already
  joined: iface_iptm, ens_* (intra-model confidence spread), pb_valid, ligand/pocket novelty +
  strata, temporal_stratum, xmodel_iptm_*, intra_model_pose_std/median, pose_consensus_frac,
  ifp_recall/precision/jaccard, xmodel_pose_rmsd_median/min, pose_consensus_cluster_size.
- `rnp_pose_features.parquet` also on disk (537 KB). RNP raw: annotations.csv, ground_truth.tar.gz
  (413 MB), prediction_files.tar.gz (39.5 GB, already distilled into parquet -> do NOT re-stream),
  posebusters, predictions.tar.gz.
- FoldBench local (`data/external/foldbench/`): `foldbench_protein_ligand_confidence_rmsd.csv`
  (55,050 rows, cols = pdb_id/seed/sample/chains/ligand/lrmsd/**ranking_score**/model — feature-poor,
  confirms the E10 confound); `foldbench_protein_ligand_rmsd_lddtlp.csv` (62,808 rows, has
  **is_unseen_protein** novelty flag + lddt-lp); `foldbench_source_data.xlsx` (Nature-Comms figure
  source sheets). ONE example `summary_confidence_sample_0.json` present -> schema HAS
  `iptm, chain_iptm, chain_pair_iptm, plddt, ptm, has_clash, ranking_score`. **Full JSON tree NOT
  downloaded** -> Tier 1 full transfer is fetch-gated.
- 14 results JSON + 7 figures committed. 5 src subpackages. 23 tests green (per HANDOFF).
- 5 analyzed models: af3, boltz1, boltz1x, chai, protenix. boltz2 excluded (n=933 < MIN_METHOD_N 1200).
- Operating point: alpha=0.20, delta=0.10.

---

## 2. Scope decisions (merit-driven; marginal proposals pruned)

Kept the 11 adversarially-**worthwhile** proposals, in their **scoped/de-risked** form. Dropped all
13 marginal ones and the risky headline variants the panel flagged (classifier-radius DRO certificate
[unsound under concept shift], headline "impossibility theorem" [likely-vacuous], GPU-weeks screening
[infeasible + null-risk], multi-risk FWER grid, PAC-Bayes combiner, PLINDER re-axis, FEP campaign).

---

## 3. Task board  (status: TODO / RESEARCH / PLAN / IMPL / RUN / DONE / BLOCKED)

### RESEARCH (Exa-grounded, gates everything) — workflow R
- [ ] R.A Tier-3 conformal prior-art + SOTA construction details (CVaR/QRC finite-sample UCB;
      randomly-localized CP Hore-Barber/Guan; concept-vs-covariate decomposition; WSR betting).
- [ ] R.B FoldBench release contents: is the per-prediction summary_confidence JSON tree public +
      fetchable? download route. (gates Tier 1 full)
- [ ] R.C Screening data: any released co-folded virtual screen (Mac1 eLife AmpC/D4/sigma2 status;
      any public co-folded VS set; DEKOIS/LIT-PCBA co-folded). (gates Tier 2)
- [ ] R.D MoML 2026 short-paper template/format guidelines (page limit rules, style files).

### TIER 0 — honesty + validity (all on-disk, unblocked)
- [x] T0.1 DONE. CLAUDE.md pixi->uv (lines 61,65); conformal/__init__ docstring updated + robust.py exports
      wired; robust.py + tests/test_robust.py (5 pass). Ruff clean, 23+5 tests green.
- [x] T0.2 DONE in PLAN.md:17 + RELATED_WORK.md:30 (group-conditional operative, weighted=complement,
      + CVaR worst-subpop). Paper-body overclaim to fix during D.1 PDF rewrite.
- [x] T0.3 DONE. `experiments/e12_reliability_drift.py` + `results/e12_reliability_drift.json` +
      `results/figures/e12_reliability_drift.png`. Result: structural drift large + CI excludes 0 all
      5 models (see §5); temporal ~0, CI covers 0. Admissibility rule attached.
- [x] T0.4 DONE. `experiments/e13_loto_validity.py` + `results/e13_loto_validity.json`. GroupKFold on
      system_id (leakage-free), pooled-split leakage audit, cluster vs row bootstrap. Results in §5.

### TIER 1 — breadth (FoldBench)  [R.B: JSON tree UNAVAILABLE -> feature-limited, no GPU]
- [ ] T1.1 Frozen transfer (calibrate-on-RNP / freeze / deploy-on-FoldBench) on SHARED feature
      (ranking_score) -> improved E10 that fixes within-FoldBench overfit confound. Novelty axis =
      is_unseen_protein. Honest feature-parity note (iface_iptm unrecoverable w/o GPU). `e15_foldbench_transfer.py`.
- [x] T1.2 RESOLVED by research: iface_iptm not fetchable at scale; do NOT GPU-regenerate. Frozen
      transfer on shared features is the honest test. Folded into T1.1.

### TIER 2 — decision number (screening)  [R.C: DROP-IN DATA AVAILABLE -> FLAGSHIP, runnable]
- [ ] T2.1 Download Zenodo 17568813; parse per-compound co-folding confidence + labels (DEKOIS2.0,
      LIT-PCBA, GPCRrecent). Three-gate selective enrichment (native docking baseline vs co-folding
      confidence vs LTT-calibrated abstain gate). EF@k/BEDROC, random-abstention control, bootstrap CIs.
      Novelty axis = molecular_sim/scaffold_sim + GPCRrecent post-2022. Honest: pose-reliability gate is a
      HEURISTIC transfer (drop "guaranteed"); pre-register null. `e16_selective_screening.py`.
  [x] T2.1 DONE. Data at data/external/screening/ (Zenodo 17568813). Results in §5. FLAGSHIP.

### TIER 3 — depth (theory/methods, on-disk)
- [x] T3.1 DONE. `src/foldgate/conformal/robust.py` (CVaR binary exact + DKW continuous; validated
      synth cov 0.910) + `experiments/e17_worst_subpop.py` + json + fig. Results in §5.
- [ ] T3.2 Covariate-vs-concept decomposition PROPOSITION (signed, accept-region-restricted) + per-bin
      WSR CIs on concept_shift_diagnostic. `src/foldgate/conformal/shift_decomp.py` + e19. [delegated]
- [x] T3.3 DONE. `src/foldgate/conformal/localized.py` (RLCP, randomized anchor + inf mass) + e18. Synthetic
      validity: RLCP 0.907 vs non-randomized 0.892 (target 0.90) -> randomization load-bearing. Recovers more
      coverage than Mondrian on moderate novelty (chai S0 0.59 vs 0.02), abstains on no-analog tail. §5.
- [x] T3.4 DONE. `experiments/e14_disagreement_strata.py` + json. Honest partial (§5).

### DELIVERABLES
- [x] D.1 DONE. `paper/moml2026_foldgate.{tex,pdf}` — 4 pages (tectonic, log_2022-equivalent preamble:
      Times 10pt, US Letter, 1.5in margins, single column, non-anonymous, booktabs). Refs on p4 (excluded).
      Fig 1 = E16 screening; Table 1 = AURC/LOTO/drift/m*. Compliant.
- [x] D.2 DONE. `docs/NEXT_STEPS_JOURNAL.md` — venue strategy (2-paper: Digital Discovery applied +
      ML/stats methods; Nature-tier if prospective lands) + prestige-ranked additions.

### E18 localized RLCP [T3.3] — recorded above in T3.3. All 4 tiers + both deliverables COMPLETE.
### Final state: 32 tests pass, ruff clean repo-wide, 23 conformal exports, PDF 4pp. Screening data on disk.

---

## 3b. RESEARCH FINDINGS (Exa, 2026-07-12) — GATES THE PLAN

- **R.A Tier-3 SOTA methods (grounded):**
  - **CVaR/worst-beta (T3.1):** Snell et al. Quantile Risk Control (arXiv:2212.13629). Build a
    (1-delta) LOWER CDF band on ACCEPTED selective losses via Truncated-Berk-Jones (Beta(i,n-i+1)
    order-stat marginals). Theorem 4.1: CDF dominance -> risk dominance for any QBRM; CVaR_beta is the
    QBRM psi(p)=1/(1-beta)1[p>=beta], so R_psi(Fhat_lower) is a valid UCB on CVaR_beta for ALL beta at
    once. beta* = largest beta with UCB(CVaR_beta)<=alpha. CVaR-DRO duality: CVaR_beta<=alpha certifies
    EVERY subpop of mass>=(1-beta) has risk<=alpha -> label-free worst-subpopulation cert. Binary loss ->
    exact Beta band. MUST co-report coverage/abstention. Refs: 2212.13629, Thomas&Learned-Miller ICML19,
    RCPS 2101.02703. Pitfall: report where n_accepted too thin -> vacuous beta*.
  - **Localized (T3.3):** Hore & Barber Randomly-Localized CP (arXiv:2310.07850, JRSSB 2024) — draw a
    RANDOM reference point ~kernel, reweight, keep +inf point mass -> EXACT finite-sample marginal +
    neighborhood-conditional. Use over Guan LCP (2106.08460). Localizer = ligand Tanimoto / pocket sim,
    bandwidth h. Ceiling: Barber-Candes-Ramdas-Tibshirani 2021 (1903.04684) — exact pointwise conditional
    impossible distribution-free -> frame as approximate/neighborhood-conditional, never exact.
  - **Decomposition (T3.2):** signed, accept-region-restricted; Ben-David lambda term; weighted-CP
    (Tibshirani 2019) assumption is pure covariate shift. (full RA text in output file; read when impl.)
  - **Scoop:** Cauchois Robust Validation (f-div DRO conformal), Confidence Gate Theorem 2603.09947,
    residue-level CRC 2509.20345 — none occupy our cell. Cite/scope; do targeted re-sweep before submit.
- **R.B FoldBench JSON tree = UNAVAILABLE.** Bring-your-own-predictions harness; Zenodo 17180806 = 3.8 MB
  code mirror; public per-pose CSV has only ranking_score/rmsd/lddt (no iptm). Only a 4-target Protenix
  DEMO ships JSONs. => Tier 1 CANNOT get iface_iptm without GPU regen. DECISION: frozen transfer on
  SHARED features only (ranking_score), honest feature-parity note. No GPU. (RNP already ships chain-pair ipTM.)
- **R.C Screening = AVAILABLE, DROP-IN (flagship unblock).** Zenodo **10.5281/zenodo.17568813** (Shen et al.,
  Chem Sci 2026 D5SC06481C, CC-BY-4.0, ~302 MB). Per-compound co-folding confidence + active/decoy LABELS:
  DEKOIS2.0 (79 targets, 40 act/1200 decoy each), LIT-PCBA (5 targets, confirmed inactives), GPCRrecent
  (16 post-2022 NOVEL targets). Score CSV cols: boltz = lid,label,ptm,iptm,confidence_score,
  affinity_pred_value,affinity_probability_binary,mpae ; protenix = ...,plddt,gpde,ptm,iptm,ranking_score ;
  glide/gnina docking baselines. **Built-in novelty axis:** molecular_sim.csv + scaffold_sim.csv
  (active-to-train similarity) + GPCRrecent post-cutoff. Pre-computed AUROC/BEDROC80.5/EF0.005-0.05/NEF in
  *_stat. Direct: https://zenodo.org/api/records/17568813/files/{dekois,lipcba,gpcr}_{scores,stat}.tar.gz/content
  Mac1 = NOT drop-in (embargoed). Secondary Boltz-2 affinity: Zenodo 18669539. THIS IS TIER 2. No GPU.
- **R.D MoML template = confirmed.** 2-4 pp; refs + appendices EXCLUDED (acks ambiguous -> keep short/in body);
  \documentclass{article}+\usepackage{log_2022} (LoG 2022 style, log_2022.sty), Times 10pt, US Letter,
  1.5in L/R margins, single column, SINGLE-BLIND (author names shown, NOT [review] option), booktabs tables.
  Overleaf: sxmntmtvjttx ; LoG spec arXiv:2309.09045v2 ; submit moml.mit.edu/submit. Deadline Sept 1 AoE.

## 4. Decisions log (append-only; the "why")

- 2026-07-12: Adopted scoped forms only; dropped 13 marginal + risky-headline variants (see §2).
- 2026-07-12: Do NOT re-stream 39.5 GB tarball; pose features already in parquet.
- 2026-07-12: Never fabricate FoldBench/screening numbers; run what is runnable, build+test harnesses
  for fetch-gated parts, be transparent in the PDF (matches project honesty ethos).
- 2026-07-12 (post-research): FoldBench iface_iptm NOT fetchable -> Tier1 feature-limited, no GPU regen.
  Screening data IS drop-in (Zenodo 17568813) -> Tier2 is now the flagship, promote it. Tier-3 methods
  locked to SOTA: QRC/Truncated-Berk-Jones for CVaR beta*; Hore-Barber RLCP for localized.
- 2026-07-12: New experiment numbering to avoid clobbering E1-E11: E12 drift, E13 LOTO, E14 disagreement,
  E15 foldbench-transfer, E16 selective-screening, plus new src modules robust.py / localized.py / shift_decomp.py.

### E14 disagreement-keyed strata [T3.4] — honest partial (deployable-without-training-set)
- Calibrate group-conditional gate on intra_model_pose_std bins; evaluate realized risk PER SIMILARITY
  stratum vs global (E2) and similarity-oracle (E3).
- Spearman(disagreement, novelty) WEAK: boltz1 +0.16, boltz1x +0.17, chai +0.20, protenix +0.05 -> a
  weak novelty proxy (honest limit).
- On MODERATE novelty (S1/S2) disagree-keyed pushes risk toward alpha (e.g. boltz1x S2 0.208->0.140);
  on EXTREME (S3/S4) it fails to control (af3 S3 0.236, chai S3 0.415), same wall as the oracle.
- Correlated-error signature: mean disagreement dips at S3 for some models (protenix S2 8.96 -> S3 6.47)
  before rising at S4 -> models agree confidently-wrong on novel targets. Frame as: recovers moderate-
  novelty control training-free, cannot cover extreme. File: results/e14_disagreement_strata.json.

### E17 worst-subpopulation certificate m*=1-beta* (CVaR, label-free) [T3.1]
- robust.py validated finite-sample: CVaR_beta UCB coverage 0.910, r_ucb coverage 0.907 (>= 0.90 target).
- Combined score certifies m*<=0.5 (every half-subpop of accepted at error<=alpha, NO labels/weights) at
  coverage: af3 0.45, chai 0.40, protenix 0.25, boltz1 0.20, boltz1x 0.25. Native often None (weaker).
- m*<=0.25 unreachable at any coverage (honest ceiling: error rates too high to protect quarter-subpops).
- **Deploy-on-novel (S3-S4): m*<=0.5 vacuous at every coverage** -> honest boundary reframing the
  weighted-CP null as a number (worst-subpop cert on novel deployment vacuous; principled abstention only).
- File: results/e17_worst_subpop.json ; fig e17_worst_subpop.png ; module src/foldgate/conformal/robust.py

### E16 selective virtual screening [T2.1 FLAGSHIP] — Boltz-2, Zenodo 17568813, target-level boot CIs
- **Co-folding confidence >> docking, biggest gap on NOVEL targets**: DEKOIS(79) EF@1% affinity=31.0 (ceiling)
  vs Gnina=10.3, BEDROC 0.867 vs 0.321, AUROC 0.964; GPCRrecent(16 post-2022) affinity EF=26.6 vs docking
  **1.97** (docking ~random on novel GPCRs); LIT-PCBA(5) affinity 5.12 vs 2.97.
- **Selective abstention by ipTM lifts EF above random on NOVEL targets**: GPCR selective EF@50%cov=27.1 vs
  random-abstention 19.05, beats random-95%-band in **75% of targets**; DEKOIS only 13% (EF saturated at ceiling,
  no headroom); LIT-PCBA 40%. Retention@50%: decoys drop faster than actives (GPCR act 0.79 vs decoy 0.49).
- **Shift in screening (E2 analogue)**: LIT-PCBA mean EF0.01 collapses 8.7 (sim1.0, train-similar actives) ->
  0.0 (sim0.3, novel-only actives). DEKOIS saturated (co-folding near-perfect) so trend masked. VERIFY mol_sim
  direction semantics before final paper wording; cross-dataset GPCR-vs-DEKOIS contrast is the cleaner shift evidence.
- Honest framing: pose-reliability gate is a HEURISTIC transfer (calibrated on pose-correctness, not activity);
  reported EF lift with random-abstention control, no coverage guarantee claimed. File: results/e16_selective_screening.json.

## 5. Results captured (append the moment produced)

### E12 reliability drift D_signed(nu) [T0.3] — S0 reference, target-mass-weighted P(correct|conf) gap, 90% boot CI
- **Structural novelty (ligand + pocket): large, monotone, CI excludes 0 for ALL 5 models.** Examples:
  - protenix pocket S3 = +0.626 [0.589,0.660], S2 +0.318, S1 +0.141; ligand S3 +0.520.
  - chai pocket S3 = +0.520 [0.468,0.568]; boltz1x pocket S3 +0.466; ligand S3 ~+0.40 all models.
  - S4 (no-analog, ~76/model) slightly below S3 and wider CI (still excludes 0), consistent with thin-n.
- **Temporal novelty: near 0, CI covers 0** (e.g. boltz1x temporal S1/S2/S3 = +0.042/+0.011/+0.004,
  all "covariate-only admissible"); a few small-but-nonzero for chai/protenix (honest nuance).
- Verdict rule: structural -> "concept drift (group-conditional / abstain)"; temporal -> "covariate-only
  (reweighting admissible)". This is WHY weighted CP abstains and group-conditional is operative.
- Reframes the E3b honest negative as a measurable first-class finding; corroborates E7 with a new lens.
- File: results/e12_reliability_drift.json ; fig e12_reliability_drift.png ; script experiments/e12_reliability_drift.py

### E13 target-grouped (LOTO) validity + leakage audit [T0.4]
- LOTO gate on native ranking_score (GroupKFold-5 on system_id, leakage-free), realized selective risk:
  af3 0.185 [0.168,0.203] cov 0.52 folds_hold 5/5; boltz1 0.157 cov0.15 5/5; boltz1x 0.168 cov0.25 5/5;
  protenix 0.191 cov0.08 2/2; **chai 0.304 cov0.01 0/1 (honest edge: native chai barely certifies).**
- Out-of-fold AURC native->combined (leakage-free, replicates E4): af3 0.185->0.118, boltz1 0.234->0.162,
  boltz1x 0.215->0.154, chai 0.247->0.146, protenix 0.281->0.166.
- Cluster (target) bootstrap CI ~= row bootstrap CI within a model (~1 pose/target/model) -> honest nuance:
  within-model clustering does not widen CIs; the real hazard is POOLING.
- **Leakage audit (headline): pooled random row-split leaks 95.6% of test rows (share a cal target,
  target recurs across 5 models); grouped split shares 0.** Justifies grouped splitting for any pooled work.
- File: results/e13_loto_validity.json ; script experiments/e13_loto_validity.py

### E15 FoldBench frozen transfer [T1.1] — RNP-calibrated tau deployed unchanged on FoldBench (feature-limited)
- Frozen RNP LTT tau on ranking_score, deployed on FoldBench top-1 (441 poses/model, 388 seen/53 unseen).
- af3 HOLDS <=alpha on FoldBench (seen risk 0.149, unseen 0.158); protenix UNDER-controls (seen 0.255>alpha,
  unseen 0.222>alpha) = genuine dataset-level guarantee break; boltz1/chai abstain (near-0 cov); chai tau=None.
- AURC (ranking quality) transfers stably: af3 0.199 (seen) ~ 0.200 (unseen). Combiner UNTESTABLE (FoldBench shares
  no combiner features) -> feature-parity-limited by construction, not a generalization failure. seen/unseen
  contrast underpowered (unseen n=53). Honest mixed transfer. Files: results/e15_foldbench_transfer.json + fig.
  io/foldbench.py gained load_foldbench_novelty() (load_foldbench for E10 untouched).

### E19 covariate-vs-concept decomposition [T3.2] — signed accept-region floor, WSR/boot CI
- af3 S3: gap_total 0.328 = gap_concept 0.309 + gap_covariate 0.018; concept CI [0.268,0.350] EXCLUDES 0
  (nonvacuous) -> ~94% of the target risk gap is CONCEPT shift, which covariate reweighting cannot close =>
  weighted CP must abstain, group-conditional is NECESSARY. af3 S4: concept 0.164 CI [0.047,0.285] nonvacuous but wide.
- Proposition framing (score-measurability assumption), not headline theorem. Files: results/e19_shift_decomp.json,
  src/foldgate/conformal/shift_decomp.py. (validity checks in subagent output.)

## 5b. JOURNAL-HARDENING results (Tier 4, 2026-07-12)

### E20 broadened selective screening [J3] — multi-model, CIs, pre-registered gate, self-computed shift
- EF@1% median [90% target-bootstrap CI] vs Gnina docking, per dataset:
  DEKOIS: Boltz-2 affinity 31.0 [28.4,31.0], AF3 20.7 [18.1,23.3], Protenix 15.5 [12.9,18.1], docking 10.3.
  GPCRrecent(novel): Boltz-2 affinity 26.6 [19.3,35.7] vs docking 1.97; Protenix 1.80 (weak). LIT-PCBA ~5 vs 2.97.
  BEDROC: boltz2 0.867 > af3 0.614 > protenix 0.447 (DEKOIS).
- **Pre-registered gate** (RNP-calibrated per-model ipTM tau via LTT: boltz2 0.661, af3 0.875, protenix 0.989),
  applied model-matched to screen with NO label peeking: Boltz-2 ipTM barely filters (cov~1.0, ipTM high across
  screen); high-tau models (protenix 0.989) over-abstain (cov 0.01-0.04). Honest transfer.
- Selective ipTM-abstain@50% beats random-95%-band: GPCR 75%, DEKOIS 13% (saturated), LIT-PCBA 40%.
- **Self-computed shift (actives binned by ec_sim to train, all decoys)**: GPCRrecent EF@1% MONOTONE 42.3(sim>0.7)
  -> 27.2(sim<0.3) = clean enrichment decay on novel actives; DEKOIS non-monotone/saturated (property-matched).
- File: results/e20_screening_broad.json. script experiments/e20_screening_broad.py.

### E21 selective affinity prediction [J3, decoupled 2nd task] — Zenodo 18669539 (ChEMBL Boltz-2 affinity)
- 9218 compounds/99 targets. Spearman(confidence, -|error|) = +0.034 -> Boltz-2 AFFINITY confidence is a WEAK
  selective signal (unlike pose ipTM for pose correctness; matches "don't conflate the two Boltz-2 heads").
- MAE full 0.946 -> confidence-abstain@50% 0.927 (barely beats random 0.946). BUT novelty-degradation HOLDS:
  MAE by compound-train similarity 0.99(0.3-0.5) -> 0.83(0.7-1.0). Honest: shift generalizes, gate signal weak.
- File: results/e21_affinity_selective.json. Data: data/external/screening_affinity/ (Zenodo 18669539).

### E22 DRO robustness radius + simultaneous multi-model certificate [J2]
- robust.py + chi2_dro_risk_ucb / robustness_radius / simultaneous_certificate. Research (Cauchois Robust
  Validation JASA24, Duchi-Namkoong, QRC Thm4.6) CONFIRMS my construction is the EXACT 2-point Pearson
  worst-case r+sqrt(rho*r(1-r)), finite-sample (not asymptotic), tight. Synth cov 0.909. 3 new tests (35 total).
- rho* inter-converts with CVaR m* via rho=(1-m*)/m*. Per-model rho*(coverage) curves (af3 cov0.2 rho*0.37).
- **NEW headline: SIMULTANEOUS certificate across all 5 models (Bonferroni delta/K=0.02): at top-20% coverage
  joint m*=0.522, joint rho*=0.098, ALL 5 certified.** At aggressive LTT gate margin ~0 (honest).
- Honesty (from research): DRO radius IS label-aware; caveat is translating rho* to novelty-coordinate units.
- File: results/e22_robust_certificates.json. script experiments/e22_robust_certificates.py.

### J1 research verdicts (web-grounded, Exa out -> WebSearch/WebFetch)
- DRO: my construction validated + refined (exact binary Pearson, finite-sample). See E22.
- SCOOP: **cell STILL OPEN** (no 2025-26 paper has all 5 properties). Baseline-to-beat = Chem Sci D5SC06481C
  (the E16/E20 data source, uses ipTM for VS, no guarantee/abstention). Adjacent to cite: CONFIDE 2512.02033,
  GESPI 2509.20345, SCoRE 2603.24704, ConfHit 2603.07371, Confidence Gate Theorem 2603.09947 (all out of cell).
  Fixed CLAUDE.md scoop-watch (2509.20345=GESPI not 'residue CRC'). Updated RELATED_WORK.
- SCREEN thread returned degenerate output (WebSearch failed for it); non-critical, E20 already bakes best practice.

### J4 reproducibility + paper
- scripts/download_data.py --screening (Zenodo 17568813 + 18669539); Makefile experiments -> E1-E22.
- paper/moml2026_foldgate.{tex,pdf} updated: DRO+simultaneous cert, broadened multi-model screening +
  pre-registered gate, decoupled affinity dataset, scoop citations. 5pp total, MAIN TEXT 4pp (refs on p4-5, excluded).
- Blocked (honest, need user/GPU): prospective validation (Mac1 embargoed), FoldBench Protenix GPU regen, Zenodo DOI.

## 6. New files created

- PROGRESS.md (ledger). Experiments: e12_reliability_drift, e13_loto_validity, e14_disagreement_strata,
  e15_foldbench_transfer, e16_selective_screening, e17_worst_subpop, e18_localized(pending), e19_shift_decomp.
  Modules: conformal/robust.py, conformal/localized.py, conformal/shift_decomp.py. Tests: tests/test_robust.py.
  Data: data/external/screening/ (Zenodo 17568813). Figures: e12,e16,e17 (+ e15,e18 from agents).
