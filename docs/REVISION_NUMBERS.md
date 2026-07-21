# Revision Numbers Ledger

Corrected/new numbers for the JCIM rewrite. Every entry is from a committed experiment
JSON. Old paper numbers noted where they change. Used to rewrite `foldgate_journal.tex`
in one coherent pass.

## Dataset counts (R4.1) — CONSORT chain (verified)
- 13,535 raw delivered poses (6 models, per system×method×ligand-instance, top-1 ranking_score).
- − 933 boltz2 (ungoverned affinity comparator) = **12,602 governed poses / 2,425 systems**.
- dedup to one pose per (system,method): **11,254 independent target-labels** (5 governed).
- all-6 dedup = **12,125 target-labels** (d2 unit).
- d1 distance track (separate): 13,215 pairs → **13,146 frame checks**; valid-frame subset (single-chain + unique copy) 6,223, of which 1,115 retain all K=5.

## Novelty axes (R2.1) — quartile bins + NaN=no-analog S4
- ligand = `morgan_tanimoto` (ECFP4 Morgan Tanimoto to nearest train ligand, /100).
- pocket = `sucos_shape_pocket_qcov` (SuCOS shape × pocket qcov, /100).
- Both: quartiles S0–S3 + NaN (no analog) = S4. NaN fraction ligand 30.4%, pocket 33.5%.
- Ligand n per (model,stratum): af3 692/654/629/658/76; boltz1 563/581/577/592/71; boltz1x 552/566/568/577/68; chai 636/625/614/624/72; protenix 649/633/618/635/72.
- Pocket n: af3 667/646/642/675/79; (full table in results/e26_*.json when ready).

## e25 — temporal per-model (R2.7) — DONE
- RNP release-date span 2021-10-06 .. 2024-06-05.
- **Every 2021-era model is 100% post-cutoff** (af3/boltz1/boltz1x/protenix in-era n=0; chai in-era n=137, 94.7% post). So NO in/out temporal split for the panel — the temporal axis ranks recency among out-of-training structures.
- Boltz-2 (2023-06-30 cutoff) is the only genuine within-RNP boundary: n_in=77, n_out=856; base correct 0.688 (in) → 0.740 (out); reliability drift in→out **+0.005 (CI90 [−0.075, 0.079])** — negligible.
- Boltz-2 structural (pocket) break holds under BOTH references: S3 correctness 0.364 (2021 ref) / 0.367 (2023 ref) — the break is not an artifact of the reference set.
- **Reframe:** the temporal null is an artifact of RNP being wholly post-cutoff, not evidence of temporal robustness; the structural-similarity axis is the operative novelty variable. Replaces the paper's "temporal-vs-structural magnitude contrast" claim with this honest statement.

## e34 — leakage-free nested-LOTO matched pair (R3.1, R4.11, R3.8) — DONE
Nested target-grouped LOTO (GroupKFold outer, grouped 50/50 fit/cal inner). α=0.20:
| model | native cov (risk, HB-ub) | combined cov (risk, HB-ub, folds) |
|---|---|---|
| af3 | 0.20 (0.189, 0.223) | **0.73** (0.176, 0.192 ✓, 4/5) |
| boltz1 | 0.21 (0.188, 0.222) | 0.60 (0.181, 0.201, 4/5) |
| boltz1x | 0.20 (0.164, 0.197) | 0.48 (0.182, 0.203, 3/4) |
| chai | **0.00 (abstains)** | 0.64 (0.178, 0.196, 4/5) |
| protenix | **0.00 (abstains)** | 0.57 (0.182, 0.201, 4/5) |
- **New headline pair (leakage-free): AF3 combined 73% vs native 20% at α=0.20** — replaces the leaky 71%/22% (e4). Even stronger and honest.
- α=0.10 combined coverage: af3 0.13 (risk 0.066, HB-ub 0.094 ✓), chai 0.22 (0.088, 0.112), protenix 0.08 (0.050, 0.086 ✓), boltz1 abstains, boltz1x 0.05.
- Chai & Protenix native gates ABSTAIN entirely (0% coverage) under leakage-free LOTO — the honest native limit the combined score recovers.

## e35 — ties bracket (R1.2) — DONE
- On `ranking_score` (the primary native gate) ties@τ = **0.0** for every model (max atom mass ~0.001) → the exact-identity equalities are literally exact for the ranking-score gate.
- On `iface_iptm` the worst bracket is 0.7% coverage (AF3 atom mass 0.079, ties@τ 0.00701).
- **Statement:** keep "exact identity" for the ranking-score gate; footnote a ≤0.7%-coverage bracket for the ipTM feature.

## e27 — PB joint label (R2.3) — DONE
- Accepted-set PB-validity > rejected for every model under the combined RMSD gate (α=0.20): af3 0.774 vs 0.582; boltz1 0.724 vs 0.538; boltz1x 0.998 vs 0.989; chai 0.849 vs 0.657; protenix 0.843 vs 0.636. **Strengthening at no cost.**
- Certifying the JOINT label (RMSD≤2 ∧ PB-valid) directly is expensive because the combined score ranks for correctness not validity: coverage collapses (af3 0.73→0.02, chai 0.64→0.17, protenix 0.57→0.16, boltz1→0); boltz1x (base pb_valid 0.995) pays ~nothing (0.484→0.481, HB-ub 0.203, holds 3/4). Honest: report accepted-set PB-validity as the strengthening; note joint-cert cost tracks confidence↔validity decoupling.

## e32 — RMSD-conditioned IFP (R2.8) — DONE
- Unconditioned e6b gap recomputes to AF3 +0.196 (acc 0.909 vs rej 0.713).
- **Within-correct (sub-2Å) gap collapses to +0.02–0.04** (af3 +0.028 [0.015,0.045], chai +0.038, protenix +0.020) — ~80–85% of the pooled gap was the RMSD selection confound.
- OLS ifp_recall ~ rmsd + accepted: **gate coefficient +0.034 to +0.072, CI clear of 0 for all 5** — a small but robust non-circular lift.
- **Reframe:** replace the paper's "0.14–0.21 contact gap" headline with the RMSD-conditioned residual (+0.03–0.07 gate coefficient); state the confound explicitly.

## e36 — ensemble-novelty correlation (R3.7) — DONE
- Combined score DOES leak novelty: `xmodel_iptm_mean` vs `ligand_novelty` |ρ| 0.31–0.35 in all 5 models (max 0.345 protenix), above the 0.30 substantial bar; intra-model ensemble spread weak (|ρ| 0.00–0.22).
- **Statement:** soften "novelty excluded from the score" → "novelty is not a direct input, but ensemble/cross-model features are moderately correlated with it (|ρ| up to 0.35), placing the COMBINED score in the achievability regime". The impossibility theorem is unaffected (it governs the frozen native score); the combined score is the operating-point improver and may legitimately re-score on ν.

## e26 — strata table + binning sensitivity (R2.1, III.3, within-stratum drift) — DONE
- AF3 ligand stratum sim-ranges: S0 0.782–1.000 (corr 0.880), S1 0.384–0.779 (0.777), S2 0.173–0.384 (0.731), S3 0.000–0.172 (0.439), S4 NaN (0.553). Base correctness monotone S0→S3.
- **Zero-frontier FRACTION stable across binnings**: n_bins=2 → 7/15 (0.467), n_bins=4 → 11/25 (0.440), n_bins=6 → 12/30 (0.400), fixed-edge [.2,.4,.6,.8] → 10/30 (0.333). Novel-tail infeasibility is not a quartile artifact.
- Within-ligand-stratum residual drift (pocket-median split) is SUBSTANTIAL: signed gap +0.07 to +0.32 (median 0.243), all 20 cells familiar-pocket-half more correct → ligand bins do NOT absorb pocket novelty → motivates the two-axis stratification. (Honest: gaps are large, not small.)

## e28 — label-cost curve (III.1, R1.8, R3.5) — DONE (native; combined variant pending)
- Native score: labels buy usable certified coverage ONLY on familiar S0 (AF3 ~40 labels → 0.56, all → 1.00); S1/S2/S3 stay ~0 at ANY budget (S2 0.092 at all, S1/S3 0.00). Coverage rises ~1−c/√n_g (S0 corr −0.57) but the asymptote is near zero on novel chemotypes because the RULE fails there, not the estimate.
- **Reframe:** "38 labels certifies the repair" is a familiar-stratum number; the native-score design curve shows novel strata are label-starved regardless of budget (impossibility made empirical). The combined-score variant (pending) is what unlocks S1/S2 — the achievability escape (cf. e36).

## e33 — pseudo-prospective time split (III.4) — DONE
- T = 2023-05-31 (60th pct); pre/post targets per model ~1300/900. No stratum info at calibration.
- Native gate holds realized post-T risk ≤ α (or CP-UB does) for 4/5 models; combined for 3/5. Realized−expected combined risk ∈ [−0.042, +0.003] → temporal deployment does not silently break control. AF3 combined: cov 0.54, risk 0.130, CP-UB 0.157. Honest abstentions (chai native, boltz1x/protenix combined = LTT non-certification, not failure).

## e37 — screening stats (R4.9, R4.10) — DONE
- EF@1% discreteness: DEKOIS ipTM realizes only **15 distinct EF values** across 79 targets (20.667 ×11, 23.25 ×10, 18.083 ×9) → the median lands on a grid point = the CI endpoint. GPCR 16 distinct, LIT-PCBA 5. Explains the "20.7 [18.1,20.7]" endpoint CIs (discreteness, not instability).
- BEDROC (α=80.5) as continuous secondary: DEKOIS affinity 0.867 [0.831,0.886], dock 0.321; GPCR affinity 0.505 [0.375,0.646], dock 0.048; LIT-PCBA affinity 0.155.
- Beat-random proportions with Wilson95: DEKOIS **10/79 = 12.7% [7.0,21.8]**; GPCR **12/16 = 75.0% [50.5,89.8]**; LIT-PCBA **2/5 = 40% [11.8,76.9]** → footnote, underpowered.

## e38 — FoldBench as risk control (R2.10) — DONE
- Frozen RNP interface-ipTM gate τ=0.9888 on FoldBench: low-homology 52 → accepts **3**, risk 0.000 but CP-UB **0.63** (uninformative, coverage collapse to 5.8%); train-similar 384 → accepts 18, risk 0.111 [0.02,0.31]. Failure mode is coverage starvation, not risk overshoot.
- AURC interface-ipTM **0.380 [0.331,0.432]** vs ranking_score **0.454 [0.403,0.505]** — **CIs OVERLAP** → ranking advantage directional, not significant at n=436.
- Self-scored top-1 0.401 vs released 0.567 → directional transfer check only (feature/label self-consistent within the run).

## e40 — composition + drift-bin occupancy (R2.6, A.2) — DONE
- 2,425 systems: MW 76/396/797 (median 396); **fragment <300 = 24%, drug-like 300–500 = 55%, large >500 = 22%**; ~0 ions, 15 peptide/flexible, 2,410 drug-like. Target class (PDB `entry_keywords`, no UniProt/EC shipped): broad, top TRANSFERASE 20.6%, no class >21%; **2,412 distinct receptors, 1,685 distinct CCD ligands** → no monoculture.
- **Protenix pocket-S3 +0.63 drift is robust, not thin-bin:** every ranking_score quintile has ≥103 novel-pocket targets; top-confidence bin (>0.987) still only 0.544 correct vs 0.947 in S0. AF3 same pattern (high-conf S3 tops at 0.72 vs 0.95). No thin-bin flag fires.

## e30 — decision curve / net-benefit (III.6, retires R3.8) — DONE
- NB(λ)=(TP−λ·FP)/N, leakage-free (nested LOTO). Conformal gate wins for **λ∈[0.65, 4.55]** (AF3); dominates fixed-ipTM≥0.8 at EVERY λ (TP 1636/FP 356 vs 1500/432). Accept-all best only for λ<0.65; abstain-all best only for λ>~4.6.
- All 5 models same shape (win ranges: af3 0.65–4.55, boltz1 0.80–3.80, boltz1x 0.85–4.20, chai 0.55–3.65, protenix 0.75–4.60).
- **Statement:** the layer serves the high-cost regime; no single α to defend (retires R3.8).

## consort_flow.png — DONE (counts verified live from parquet).

## e28 combined-score variant (III.1) — DONE
- AF3 combined-score certified coverage: S1 0.217 at n_g=80 → 0.829 at full pool; S2 0.480 at 80 → 0.769; S3 0.332 only at full pool. Native = 0 at every budget on S1–S3.
- Combined score raises the CEILING (unlocks S1/S2), not the min_accept=20 FLOOR. Design tool: ~80 in-stratum labels certify moderate-novelty S1 with the combined gate; S2/S3 need near the full pool. S1 crosses 0.2 at n_g=80 for boltz1x/chai/protenix too.

## e29 proxy stratifier (R2.2/III.2) — DONE
- Public proxy = `num_training_systems_with_similar_ccds` (CCD count): Spearman to oracle 0.59, adjacent-agreement 0.83 (af3); recency-date and protein seqsim are at chance.
- Graceful degradation: proxy matches oracle control on low/mid novelty (S0–S2 within ±0.05) but loosens the novel tail (true S3 risk 0.23→0.37, +0.142).
- Load-bearing: even the ORACLE group-conditional gate leaves S4 at 0.60 (>α). So **concept shift, not the un-enumerable training set, is the binding limit** on the novel tail — stratification alone can't certify it regardless of proxy vs oracle. Governance line for the abstract.

## e31 extra baselines (R1.10/III.7) — DONE
- AF3 under shift (risk/cov): fixed-ipTM≥0.8 **0.301** (breaks), naive conformal **0.412** (breaks), accept-all 0.417 — vs group-conditional 0.190/0.18, RLCP-localized 0.187/0.44, per-stratum Venn-Abers 0.166/0.24 (all HOLD by abstaining).
- Fixed-ipTM across models: erratic (boltz 0.04 cov, protenix 0.93 cov 0.511 risk) — never a controlled field baseline.
- **Venn-Abers matches the GBM** (0.166/0.24 vs 0.190/0.18) → the shift repair does NOT require the gradient-boosted combiner; a per-stratum Venn-Abers on the native score reaches the same guarantee. Simplification to flag (drops R1.5/R3.7 GBM concerns entirely if adopted).

## Money figure (Fig 2 rebuild, R4.13/R4.4) — DONE
- Per-stratum realized risk of the global gate (300 grouped resamples, mean + 90% resample interval, median accepted-n), S3-led, S4 greyed. AF3 S3 0.367; chai S3 0.609 (native gate erratic).
- deploy-to-novel (calibrate S0–S1, deploy S3–S4, CP interval): AF3 **0.549**, boltz1 0.491, boltz1x 0.478, chai 0.500, protenix 0.664. Range ~0.48–0.66; matches E2's 0.547. Cite break_money_numbers.json.

## ALL EXPERIMENTS + FIGURES COMPLETE. Next: paper rewrite → JCIM structure.

## RE-AUDIT FIXES (2026-07-20, second pass)
- **B1 frontier:** 50 cells include 10 S0-reference (trivially feasible). Over the **40 non-reference cells**: α=0.20 → **21 zero-frontier, 14 CP-robust**; α=0.10 → 26 zero-frontier, 21 CP-robust; pooled both α → 47 zero-frontier, **35 CP-robust**. Abstract "35 of 50" was wrong; headline 21/40 (14 CP-robust) at α=0.20.
- **B2 S4 size:** no-analog S4 = **2.8% of poses** (af3 76, pooled 359/12,602), NOT 30-34%. "small-sample, greyed" correct; fix the percentage.
- **B3:** the 0.60 is base/point-threshold error on S4, NOT the folding LTT gate (which abstains at 0 coverage). Reframe.
- **B4:** valid certificate = per-fold LTT + folds_holding, not pooled CP-UB. Leakage-free α=0.20: combined af3 73% cov, folds 4/5, HBub 0.192 (certified); native af3 20% cov, folds 1/2, HBub 0.223 (NOT certified); chai/protenix native abstain. Fixed-cap certified-native scan is post-hoc/invalid — not cited. Use folds_holding column + LTT framing; drop pooled-CP-as-certificate.
- **B5:** IFP n = **12,475** (not 11,711). Per-structure contact set-comparison keyed on (seqid,resname), no cross-frame superposition → not subject to the protomer trap; homodimer copies share numbering. Verified: single-chain-subset gate coef +0.041 to +0.056 (vs full +0.034 to +0.072) — survives for all 5 models.
- **M5 receptor diversity:** RNP `cluster` column → **954 receptor clusters** (largest 170 = 7%), not 2,412 per-PDB. Top PDB class ~17%.
- **M4:** median 38 labels is over the **23 feasible cells** (d2_certify); the ~80-label figure is the one-stratum-out design-curve budget — conclusion should use ~80.
- **M6 pseudo-prospective failures:** native — boltz1 CP-UB 0.207 (overshoots), chai abstains; combined — chai CP-UB 0.202 (overshoots), boltz1x + protenix abstain (LTT non-certification). abstain = holds vacuously.
- **M1 RLCP:** dominates on coverage (0.44 vs 0.18) but its guarantee is randomized + marginal-over-localization, not stratum-conditional finite-sample — state the guarantee difference in the table caption.

## ROUND-3 FIXES (2026-07-20)
- **Venue → Digital Discovery** (RSC, gold OA, ML-for-chem + honest-negative friendly; J.Cheminformatics backup). Best chance for the statistical framing; avoids JCIM's full ACS chemist-rewrite.
- **#5 cluster-grouped LOTO (e41):** AF3 combined headline robust 0.72 (vs 0.73 system-grouped), still certified HB-UB 0.192. Boltz-1 0.60→0.24, Protenix 0.57→0.41 drop under stricter homolog grouping (reported).
- **#7 frontier multiplicity (e42):** 21 zero-frontier at α=0.20 → 13 survive Holm/BY, 15 BH; α=0.10 → 19 family-wise. Corrected count ≈ per-cell (14). Abstract now says "per-cell tests; 13 survive family-wise correction".
- **#4 stratifier (e44):** 2021 vs 2023 pocket similarity Spearman **0.14** (2023 mean 87 vs 2021 64) → reference year matters → JUSTIFIES cutoff-matching (5 governed share ~2021; Boltz-2 re-keyed to 2023). Turned the objection into a handled point.
- **#8 cards:** 3-way verdict FEASIBLE / ABSTAIN-infeasible (abandon) / ABSTAIN-underpowered (collect labels); native-score-labeled. AF3: S3 underpowered, S4 infeasible.
- **#2 abstract:** 73% now attributed to the marginal combined-feature LOTO gate, separate from group-conditional 52/39%.
- **#3 abstract:** CPU caveat — raw-confidence gate CPU-only; combined-feature consumes 5 samples + cross-model runs.
- **#1 cards figure:** native-score labeled + protocol reconciled with the combined-score repair figure (AF3 S3 native ABSTAIN vs combined 5%).
- **#6:** proposition → lemma; §5 leads with the diagnostic contribution.
- **#12:** label-ground-truth limitation + graded-loss pointer + PoseBusters citation; resolution check deferred (no per-entry resolution in RNP annotations).
- **V2:** chemist-facing abstract + positive practitioner recommendation with the ~80-label congeneric-series worked example.
- **Minors:** Table 5 ✓ caption (pooled-bound property, fold-count-independent), CRC orphan cited, BiSyRMSD/SuCOS/qcov defined, foldgate capitalization consistent.
- Compiles clean, 14pp.

## ROUND-4 EXPERIMENTS (2026-07-20)
- **A5 (e45) within-stratum AUROC:** native score NOT anti-selective — S3 AUROC 0.71 AF3 / 0.64 Protenix / 0.56 Chai (all >chance); break is base-rate (S3 correctness 0.44) not signal-collapse; collapses to chance only on S4 (AF3 0.54, Protenix 0.50). New Sec 3 paragraph.
- **A4 (e47) common score:** break SURVIVES on interface ipTM (Protenix +0.63 pocket/+0.50 ligand, Chai +0.56, CIs far from 0) → not a score-definition artifact; cross-model top-2 ranking stable. Frontier is SCORE-DEPENDENT: ipTM 6/20 ligand zero-cells vs ranking_score 10/20 → 4 cells flip to feasible. Impossibility must be stated as the intersection across the score family (compute with D1 ligand-pLDDT).
- **A3 (e48) S4 characterization:** S4 is MIXED — similarity missing for whole system, not ligand; ~53% multi-ligand-chain, ~18% cofactors/sugars (ADP/FAD/NAG = coverage gap not novelty); topological_tanimoto NOT a fallback (NaN in lockstep, recovers 2/76). Genuine drug-like no-analog core = ~29/76 (accuracy 0.44 vs 0.67). Softened S4 to "no-computable-analog" + disclosure; lead on S3. External ECFP4 recompute = fuller fix (deferred).
- **D22 (e49) crystallography:** NO resolution confound — median resolution flat across strata (S0 1.98, S3 2.00 Å); S3 break survives high-res-only (0.42→0.46) and 5%/10% label flips (S3 >α in 100%/99%). Defends the break as reliability, not label noise. pdb_resolution.csv cached (2,412 entries).
- **§I fixes:** I1 cluster 954 (governed) not 1005; I2 boltz2-excluded clause; I3 "8,277 pooled across 5 models"; I5 S0 c*=1.00/0.65; I6 "0.00" typo → range [-0.04,+0.003]; I7 coverage-map extended (native frontier 0.72/0.58, native group-cond 0.08/0.12, combined 0.52/0.39 all mapped) + 38-vs-80 label-budget rows.
- **PENDING:** D1 (e46 ligand pLDDT/PL-PAE, streaming 37GB) — big one, in flight. Frontier intersection rewrite waits on it.

## ROUND-4 TEXT TIER (2026-07-20, D3 running on GPU)
- A5 (e45) within-stratum AUROC + reliability diagram (Fig) added: break is base-rate not signal-collapse.
- A4 (e47) common-ipTM drift + A3 (e48) S4 softening + D22 (e49) crystallography defense integrated.
- D1 (e46) ligand-pLDDT + e50 score-family frontier: impossibility = INTERSECTION 13/40 (ranking 22, ipTM 16, pLDDT 18); PL-PAE unavailable in dump.
- A1/A2 theory reframe (sigma(s), 'concept shift'->score-conditional drift, part-d demoted); B1 numbered achievability proposition + binning caveat; B3/B4 (Ben-David 'analogous', crcgupta concurrency).
- B2 (e52) closed-form label lower bound n_g>=ceil(ln(1/delta)/KL(r||alpha)); D19 (e53) seq-axis redundant; D25 (e54) seed-invariant; D26 (e55) Boltz-2 same break under 2023 ref.
- D2 (e51) typed ProLIF IFP: gate lift LARGEST on polar/directional (H-bond +0.07/+0.08, pi-stack +0.06/+0.07).
- G citations (Barber-beyond-exchangeability, multicalibration, Gibbs-Candes, applicability-domain, Chow/Madras/Mozannar, Vickers-Elkin, Duchi-Namkoong).
- C7/C8/C10/C12/C13 stats-conventions note; C11 fold-power; C15 quartile-marginal; C16 via D25.
- E: analysis_table.csv artifact, App C integrity callout, stratifier fragility (e44).
- F: card-as-proposal, documented novelty search. D32 economic pricing. H: MIT license, determinism, pinned versions, card API, repro-by-default.
- I1-I8 internal-consistency fixes. Paper 18pp, compiles clean.
- **D3 (GPU, in flight):** Boltz-2 + Chai-1 on PoseBusters-V2 (308) + PLINDER-test (307), DS=5, --no_kernels. Awaiting d3_package.tar.gz.
- **PENDING local:** e56-e59 bundle (C14 min-accept sensitivity, D20 joint-2D, D28 trained ceiling, RMSD sweep). Deferred: D21 apo/holo, D29 physics-rescore, D31 expert-study (inflated), J de-hedge/Fig-matrix presentation pass, SI restructuring (move screening/synthetic to SI).
