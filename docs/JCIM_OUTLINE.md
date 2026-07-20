# JCIM Draft Blueprint — Know When to Fold

Target: J. Chem. Inf. Model. (ACS). 9–11 pp main + SI. Three contributions, not five.
Every audit fix mapped to a location. Numbers from `docs/REVISION_NUMBERS.md`.

## Framing
- Contributions (3): (1) the break + per-stratum feasibility frontier; (2) the exact
  selective-risk decomposition as a deployable drift diagnostic; (3) the honestly-priced
  group-conditional repair with a loss-matched certifier.
- Ranking utility → a subsection of §7. Screening → SI-C (cost result, not enrichment win).
- Every headline number is target-grouped (leakage-free). Pose-level splits → SI only.

## Abstract (~200 words, 4 numbers) [R4.17]
Numbers: break 0.55 vs 0.20; robust zero-coverage cells 35/47; leakage-free repair
(AF3 combined 73% at risk ≤0.20 nested-LOTO, native 20%); label price median 38 (familiar).
Drop: EF 9.3 [R3.3], "0 of 300" as headline [R4.6], +0.63/0.46–0.61 crowding, theorem-as-headline.
Add one clause: the operative stratifier is uncomputable for closed-training models → proxy result [R2.2].

## §1 Introduction
- 3 contributions (cut combined-score + screen as "contributions") [Part IV].
- Prior work: CoDrug (scalar), Shen2026 (ranking, no guarantee), crcgupta (concurrent,
  LLM — flag preprint + date) [R1.9]. Conformal selection (Jin/Bai — flag preprint) complementary.
- State the deliverable: certificate cards per model×stratum [III.10].

## §2 Data and definitions  [absorbs App. B; new material]
- Loss L = 1[ligand-RMSD > 2Å]; use L throughout, reserve Y for nothing [R1.3].
- Notation table (Table 0): α, δ, c, τ, ν, S_ν, η_P, η_Q, R_ref, Δ̄_c, D, m*, β*, ρ*, L [R1.3].
- RMSD convention: symmetry-corrected BiSyRMSD as shipped; state alignment [R2.4].
- Novelty axes (methods subsection) [R2.1]: ligand ECFP4 Morgan Tanimoto, pocket SuCOS×qcov,
  to nearest training system, quartile bins S0–S3 + NaN=S4 no-analog; SI table of n +
  sim-range per (model,axis,stratum) from e26. Note quartile edges are data-defined a priori.
- CONSORT flow diagram (Fig 1) reconciling all counts [R4.1] → consort_flow.png.
- Provenance table [R2.5, A.3]: every result = shipped BiSyRMSD (primary) vs recomputed
  geometry (danger floor/consensus only, single-chain n=6,163). Promote the homodimer
  protomer-trap integrity note to main methods (Spearman 0.620→0.995) — a service finding.
- Composition [R2.6]: drug-like dominated (55%), diverse (2,412 receptors, 78 classes,
  no monoculture), from e40.
- Deployment note [R3.6]: native-score gate = CPU, one inference; combined score = ensemble
  + cross-model features → N inferences per target. Native-only guarantee stated prominently.
- Pre-registration pointer [R1.6]: coverage grid arange(0.05,1,0.01), FST order, binning,
  endpoints frozen in a timestamped repo file before reanalysis.

## §3 The break — S3-led, accepted-n annotated, CIs throughout  [R4.3, R4.4]
- Marginal 0.177 masks per-stratum: name R_marg,S3 = 0.375 [R4.5]. Lead on S3; S4 only in frontier.
- Deploy-on-novel: R_transfer,S3 = 0.55 (AF3; 0.46–0.61 range). Resampling scheme stated
  (300 grouped resamples), effect size + one CI, not "0 of 300" as the headline [R4.6].
- Every risk annotated (accepted n / stratum n) + Clopper–Pearson; cells n<10 greyed [R4.4].
- Reliability drift D: structural axis large (Protenix pocket +0.63, robust — e40 bins ≥103
  [A.2]); temporal axis reframed [R2.7]: RNP is wholly post-cutoff (e25), so the temporal
  axis is recency among out-of-training structures, NOT in/out — a benchmark property, not
  temporal robustness. Boltz-2's genuine 2023 boundary shows drift +0.005; structural break
  holds under the correct 2023 reference. Structural similarity is the operative axis.

## §4 The feasibility frontier  [PROMOTED to lead impossibility result]  [R4.8]
- Per-stratum c*_g; AF3 ligand 1.00/0.85/0.75/0.20/0.00. 21/50 zero at α=0.20, 26/50 at 0.10;
  **47 = the α∈{.10,.20} union**, robust 35/47 headlined [R4.8]. 50 = 2 axes × 5 models × 5 strata.
- Binning-robustness [III.3]: zero-frontier fraction ~0.4 across n_bins∈{2,4,6}+fixed (e26).
- Deployed view: R_gate,S3 = 0.538, margin −0.338, certificate inverted [R4.5].
- Consensus/danger-floor escape: include the metric construction (½ pg) or move numbers to
  SI with definition — do not forward-reference out [R2.9]. Single-chain provenance stated.

## §5 The decomposition as a deployable diagnostic  [RETITLED from "impossibility"]  [R1.1]
- Theorem → "Exact selective-risk decomposition under concept shift". Part (b) = one-line
  remark (accept set fixed at fixed coverage). Weight on (c): making Δ̄_c measurable [R1.1].
- "Exact identity" honest [R1.2]: ties@τ = 0 on ranking_score (exact); ≤0.7%-cov bracket on
  ipTM (footnote), from e35.
- Synthetic: relabel the 9-reweighting invariance as a regression test of the identity, not
  evidence; add the D_q sweep locating the α-crossing [R1.4].
- Achievability slack for the ratio: cite precise LTT-ratio result OR call it empirical
  calibration — and point to the label-cost curve as its empirical shadow [R1.8].
- Proof in SI-A with the ties bracket widths [R1.2].
- Drift selects the repair; small-drift covariate regime vs large-drift concept regime.
  A.4 fix: weighted conformal's ONLY lever is threshold selection (a coverage move), state
  the coverage at which naive 0.27 was measured.
- Disclosure [R3.7]: the COMBINED score is moderately novelty-correlated (xmodel_iptm |ρ|
  up to 0.35, e36) → it partly re-scores on ν = the achievability escape; the impossibility
  governs the frozen NATIVE score and is unaffected. State plainly.

## §6 The repair, priced  [R3.5, R3.10; add Fig 4 + baseline table]
- Group-conditional calibration restores per-stratum control (AF3 combined S1 52%, S2 39%).
  Uses shipped strata → deployment-computable proxy [R2.2/III.2] from e29: report proxy vs
  oracle control; frame as deployable / degrades / uncomputable per the result.
- Label-cost curve (Fig 4) [III.1]: certified coverage vs n_g per stratum (combined-score
  variant from e28). Median 38 is a familiar-stratum price [R3.5]; novel strata label-starved
  on native (impossibility empirical), unlocked by the combined score (achievability).
- CVaR/DRO label-free certificate: fix prose to worst-(1−β) [A.1]; state m*→1 vacuous on
  deploy-to-novel as a FINDING in the conclusion, not a parenthetical [R3.10]. Anchor ρ*=0.10
  to a chemotype/pocket-shift magnitude [R3.9].
- Match the certifier to the loss: exact binomial 38 vs HB 60 vs Hoeffding 102; graded-loss
  betting bound 43%@1Å (Hoeffding 0.04%) — PROMOTE to main text [A.5]. Betting validation:
  120 reps, pose-level unit stated [R4.12].
- Certificate ledger table [R1.7]: one row per claim (global gate, LOTO, CVaR/DRO, betting,
  drift grid) with δ, multiplicity method, n, joint/per-model.
- Baseline table [R1.10/III.7] from e31: fixed-ipTM, Venn–Abers, localized-continuous,
  accept-all/abstain-all vs group-conditional. If Venn–Abers matches the GBM, note the GBM
  is optional.
- Multiplicity: enumerate the 11 axis-strata + S0-reference exclusion; 40 structural all
  survive = effect-dominated family [R4.7].

## §7 Utility  [leakage-free only; screening → SI]
- Ranking utility (leakage-free): the nested-LOTO matched pair — AF3 combined 73% vs native
  20% at α=0.20, certified HB bounds; α=0.10 points; Chai/Protenix native abstain [R3.1, e34].
  Replace 71/22 everywhere. LOTO table gets a certified-UB column + pass/fail + Chai n [R4.11].
- Decision curve (Fig): conformal wins λ∈[0.65,4.55], dominates fixed-ipTM [III.6, e30] —
  retires the α=0.20 defense [R3.8].
- IFP recovery [R2.8]: replace the "0.14–0.21" gap with the RMSD-conditioned residual —
  within-correct gap +0.02–0.04, OLS gate coef +0.03–0.07 (CI clear of 0), e32. State the
  confound explicitly; this is the honest non-circular lift.
- PoseBusters-joint [R2.3]: accepted-set PB-validity > rejected for every model (strengthening,
  free); joint-cert cost stated, e27.

## §8 Generality and limitations
- Pseudo-prospective [III.4/e33]: time-split holds risk ≤ α for 4/5 native, 3/5 combined.
- FoldBench [R2.10/e38]: realized risk at frozen τ, low-homology (3/52, coverage collapse,
  CP-UB 0.63) vs train-similar separately; AURC 0.380 vs 0.454 CIs overlap → directional;
  0.40/0.57 caveat. Second cross-dataset: if none with confidence fields is available, state
  the concrete community ask (release ipTM/PB per pose) [III.5].
- Limitations: affinity head is the best ranker but ungoverned — name as top open problem
  [R3.4]. no-analog tail uncertifiable. all retrospective.

## §9 Conclusion  [three honest bottom lines]
- Certifiable on familiar strata; ~38 novel-target structures per stratum to extend one step;
  label-free CVaR vacuous exactly on novelty [R3.5, R3.10].
- Purity = 1 − selective risk; no summary implies independence [A.6].

## Front/back matter  [R4.14, R4.15]
- Funding (state), competing interests (none), author contributions, broader impact/limitations
  paragraph. Data/code: pinned env (uv.lock), seeds, wall-clock + cores for make repro, opt-in
  artifact sizes, license (Apache-2.0/MIT). Zenodo DOIs (RNP, FoldBench-regen).
- Bib hygiene [R4.16]: complete rnp authorship; split Boltz-1/1x/2 with versions; mark
  preprints inline (crcgupta, incoherence, credal, bai2026, protenix, chai).

## SI
- SI-A proof + ties bracket widths. SI-B synthetic + elec2/ACS (reframed). SI-C screening as
  a COST result [R3.2]: gate halves inspection at ~no enrichment cost; EF discreteness +
  BEDROC + Wilson intervals [R4.9,R4.10, e37]; LIT-PCBA n=5 footnote. SI-D certificate-card
  grid [III.10]. SI-E pose-level (non-grouped) split sensitivity.

## Figures
- Fig 1 CONSORT flow (done). Fig 2 the break (regen: split-spread relabeled, bootstrap CIs,
  accepted-n annotated, S3-led) [R4.13]. Fig 3 feasibility frontier (annotate accepted n;
  replace area encoding) [R4.13]. Fig 4 label-cost curve (new). Fig 5 repair (direct coverage
  labels, not marker area) [R4.13]. Fig 6 decision curve. Screening fig → SI.
