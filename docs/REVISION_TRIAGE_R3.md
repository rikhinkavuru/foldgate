# Round-3 Audit Triage

Verdict legend: **FIX-NOW** (text/presentation), **EXPERIMENT-CHEAP** (CPU, existing tabular data),
**EXPERIMENT-HEAVY** (needs the 39.5 GB structure stream, external datasets, or a tool install),
**DECISION** (needs the author to choose), **DEFEND** (rebut with a clarification, finding is softer than stated).

## The things a reviewer will catch

| # | Item | Verdict | Action |
|---|------|---------|--------|
| 1 | Fig 3 / Fig 5 (cards) / coverage-map disagree on AF3 S3 (5% vs ABSTAIN vs 0.18) | **FIX-NOW** | Cards are built on the NATIVE score + d2 feasibility (→ ABSTAIN on S3); Fig 3 is the COMBINED-score group-conditional gate (→ 5% on S3); coverage-map deploy-to-novel is a third protocol. Real, not contradictory. Make cards score-specific (label "native"/"combined") + add a protocol line to every card/figure caption. |
| 2 | Abstract misattributes 73% to group-conditional repair | **FIX-NOW** | 0.73 is the marginal combined nested-LOTO gate; the group-conditional repair is 0.52/0.39 on S1/S2. Fix the abstract sentence (Sec 7 is already correct). |
| 3 | "CPU" sold in abstract, but the 73% needs the combined score = N inferences | **FIX-NOW** | State in the abstract: native-gate is CPU/one inference (but certifies little leakage-free); combined score buys the coverage at N co-folding inferences per target. |
| 4 | One similarity annotation across five different training corpora | **DEFEND + EXPERIMENT-CHEAP** | RNP similarity is to the public pre-cutoff PDB, the SHARED ~2021 training era for AF3/Boltz-1/1x/Chai/Protenix, so one annotation is appropriate for the governed cohort; Boltz-2 (2023) is the only divergent cutoff and is already re-keyed. Exact per-model sets aren't all enumerable (AF3 closed) — the deployment-proxy experiment (Sec 5) already owns that. CHEAP add: report the 2021-vs-2023 stratum-assignment agreement to bound how much re-keying moves it. |
| 5 | GroupKFold on system_id, not sequence cluster → homolog leakage | **EXPERIMENT-CHEAP** | `cluster` column ships (1005 clusters). Re-run the nested-LOTO headline (e34) grouped on cluster; report whether 73% moves. High value, cheap. |
| 6 | Proposition thinner than "impossibility" billing | **FIX-NOW (reframe)** | Already retitled to "exact decomposition" and (b) called a one-line remark. Sharpen: lead §5 with "Δ̄c is measurable from a labeled probe → repair-selection diagnostic"; demote the proposition to a supporting lemma explicitly. |
| 7 | Frontier 21/14 headline uncorrected for multiplicity | **FIX-NOW + EXPERIMENT-CHEAP** | Run Romano-Wolf (or Benjamini-Yekutieli) on the 40-cell frontier grid, quote the corrected count, and state "per-cell uncorrected" in the abstract. Ledger row already added. |
| 8 | ABSTAIN conflates infeasible vs underpowered | **EXPERIMENT-CHEAP** | Split the card verdict three ways using the CP lower bound: CP-lower > α → ABSTAIN-infeasible (abandon); not CP-robust and n small → ABSTAIN-underpowered (collect labels); else FEASIBLE. Regenerate cert cards. |

## Chemistry-side (where JCIM reviewers live)

| # | Item | Verdict | Action |
|---|------|---------|--------|
| 9 | Gate on ipTM/ranking_score, not ligand pLDDT / PL-PAE (the field's pose-triage score) | **EXPERIMENT-HEAVY** | Predicted ligand pLDDT + PAE-interaction block live in the CIF B-factors / NPZ in the 39.5 GB `prediction_files.tar.gz` (the stream the W1 pose features already use). Extract ligand-atom pLDDT + PL-PAE, recompute drift Δ̄c and the gate on a ligand-LOCAL score. Likely the single most-requested experiment; could narrow the concept gap. |
| 10 | 4.5 Å any-atom "IFP" is not an IFP; use ProLIF/PLIP with interaction typing | **EXPERIMENT-HEAVY** | Redo interaction recovery with ProLIF (per class: H-bond, salt bridge, hydrophobic, π-stack) on the structure files. It is the only non-circular quality result, so it should be the most chemically credible. |
| 11 | External validity = one benchmark (FoldBench, starved 3/52) | **EXPERIMENT-HEAVY** | Add PLINDER (ships similarity splits) and/or PoseBusters as a 2nd/3rd dataset — needs their co-folding predictions with confidence fields. "Community ask" framing reads as an unmet requirement. |
| 12 | 2 Å BiSyRMSD vs a single crystal pose ignores resolution / alt-confs | **EXPERIMENT-CHEAP + FIX-NOW** | Resolution-stratified sensitivity of the break (RNP annotations carry resolution?); cite the RMSD-threshold literature; surface the graded-loss betting bound earlier as the partial answer. |

## Venue & presentation

| # | Item | Verdict |
|---|------|---------|
| V1 | JCIM vs J. Cheminformatics / Digital Discovery | **DECISION** — reviewer argues the latter two tolerate the statistical framing better; JCIM is publishable but needs a chemist-facing rewrite. Author must choose. |
| V2 | NeurIPS voice → ACS style (no contributions list, real Methods section, TOC graphic, SI as separate PDF, chemist abstract, positive practitioner recommendation with an ~80-label worked example) | **FIX-NOW (large)** — conditional on V1. |

## Minor (all FIX-NOW)
- [26] angelopoulos2024crc orphan reference (defined, not cited) — cite in App A or remove.
- Table 5 (LOTO) ✓ column: caption must say ✓ is a pooled-bound property independent of the fold count.
- Verify author lists + arXiv IDs on preprints (rnp, crcgupta, boltz, bai2026, incoherence, credal) — copyedit.
- Define BiSyRMSD, SuCOS, query-coverage at first use (chemists know SuCOS, not BiSyRMSD).
- `foldgate` capitalization consistency.

## Proposed execution order
1. FIX-NOW batch (1,2,3,6,7-text,12-cite, all minors) + EXPERIMENT-CHEAP (5 cluster-LOTO, 7-RW frontier, 8 verdict split, 4-agreement, 12-resolution).
2. Present heavy experiments (9 ligand-pLDDT, 10 ProLIF, 11 datasets) + venue decision (V1) to the author — these need a scope/venue call before spending the structure-stream / external-data budget.
