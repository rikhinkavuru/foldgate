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

## Pending (agents running)
e26 strata+binning · e28 label-cost curve · e29 proxy stratifier · e30 decision curve · e31 extra baselines · e33 pseudo-prospective · e37 screening stats · e38 foldbench risk · e40 composition.
