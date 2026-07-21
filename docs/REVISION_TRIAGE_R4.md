# Round-4 Audit Triage

Environment established: 37 GB RNP structure tarball **local**; GPU is **Apple M4 (Metal, not CUDA)**
so co-folding *inference* cannot run here; **network works** (PDB API 200); ProLIF/PLIP not yet installed.

Tags: **CPU-NOW** (do locally, tabular/analysis) · **STREAM** (CPU, parse the 37 GB tarball) ·
**NET** (needs network fetch, no GPU) · **GPU** (needs a cloud CUDA GPU to *generate* co-folding
predictions) · **TEXT** (writing/framing/citation) · **INFLATED** (reviewer-maximalism; not blocking).

## A. Blocking correctness

| # | Verdict | Feasibility | Note |
|---|---------|-------------|------|
| A1 σ-algebra: Δ̄ is a within-score-fiber gap, "concept shift" is a misnomer | **TRUE, reframe** | CPU-NOW + TEXT | Real. Δ̄ is measured on binned s (D(ν) sums over score bins), so the label law given the FULL covariate is fixed; what moves is the label rate within a score fiber = residual covariate shift. Reframe "concept shift" → "score-fiber reliability drift"; add Δ̄ binning-sensitivity; separate the σ(s)-invariance (fiber) from the coverage-pinning argument; state η̂_P extrapolation error + the CI's resampling unit. |
| A2 Prop (d) is a tautology | **TRUE** | TEXT | Drop (d) or replace with the frontier (empirical, over the whole coverage grid) as the non-trivial achievability statement. Already have the frontier. |
| A3 S4 may be parse-failure not novelty | **TRUE, must characterize** | CPU-NOW (+STREAM/NET for fallback sim) | Characterize S4 membership (MW/heavy-atom/element/peptide/covalent/metal flags, CCD freq, receptor class); attempt fallback similarity (MCS/shape) for S4 members; if a chunk gets a similarity, re-stratify. Until done, soften S4 claims in abstract. |
| A4 ranking_score differs across 5 models | **TRUE** | CPU-NOW | Re-run the drift grid + frontier on a COMMON score (interface ipTM, already have) as the primary cross-model comparison; native-score as secondary per-model. May change cell verdicts. |
| A5 gate anti-selective on S3? | **DONE — resolved** | CPU-NOW | Native score is NOT anti-selective: within-stratum AUROC S3 = 0.71 (AF3), 0.56 (Chai), 0.64 (Protenix), all > chance. Break is base-rate, not signal-collapse. Signal collapses to chance only on no-analog S4 (AF3 0.54 CI[0.43,0.66], Protenix 0.50 CI[0.38,0.62]). Add the (model×stratum) accept-all-err / gate-err / within-stratum-AUROC table with CIs. Sharpens the paper. |

## B. Theory to complete

| # | Verdict | Feasibility |
|---|---------|-------------|
| B1 numbered achievability proposition (within-stratum exchangeability is an approximation under binning) | **TRUE** | TEXT + CPU (quantify degradation with bin width) |
| B2 information-theoretic label-cost lower bound (min n_g from exact binomial) | **TRUE, valuable** | CPU-NOW (closed form) — turns "~80" into a formula |
| B3 Ben-David connection precise or drop | **TRUE** | TEXT ("analogous in spirit") |
| B4 relationship to Kotte [19] formal | **TRUE** | TEXT (state strengthening + separating example, or concede concurrent) |
| B5 unique τ_c on the 1/n lattice | TRUE, minor | TEXT |
| B6 ratio-functional LTT: conditional vs marginal | **TRUE** | TEXT (spell out what LTT delivers, over which randomness) |

## C. Statistical methodology (all CPU-NOW or TEXT)

C7 FST direction+stop archived · C8 infeasibility test is IUT (conservative, not exact) · C9 min-detectable-gap on amber cards · C10 label resample vs CI distinctly · **C11 fold counts have no power → demote to descriptive, lead pooled HB (TRUE, important)** · C12 bootstrap unit/replicates/clustered everywhere (cluster-bootstrap default) · C13 Romano-Wolf spec · C14 ≥20-accept threshold sensitivity (10/20/30/50) · C15 quartile-edge marginal source (promote fixed cuts) · C16 top-1 max-selection sensitivity (random-sample gate). All CPU-NOW.

## D. Experiments (reviewer reprioritizes #9/#11 to blocking)

| # | Verdict | Feasibility | What I need |
|---|---------|-------------|-------------|
| **D1 ligand pLDDT + PL-PAE** (blocking) | **TRUE, do it** | **STREAM (CPU-local, no GPU)** | Tarball is local; parse CIF B-factor pLDDT + NPZ PAE ligand block; re-run frontier+cards on ligand-local scores. If zero-frontier holds → impossibility much stronger; if flips → better gate. Either way a win. **Feasible here.** |
| **D3 PLINDER + PoseBusters** (blocking generality) | **TRUE, gated** | **GPU (likely)** | Needs co-folding predictions WITH confidence for these sets. PLINDER/PoseBusters don't ship them; must find released predictions OR **generate on a cloud CUDA GPU**. First I investigate whether released predictions exist (NET); if not → **this is the GPU ask**. |
| D2 typed IFP (ProLIF/PLIP) | **TRUE** | STREAM + pip install | Local; pip install prolif. CPU. |
| D22 crystallographic quality (resolution/R-free/RSCC/PDB-REDO) — "don't defer" | **TRUE** | **NET (no GPU)** | Fetch per-entry from RCSB + PDB-REDO APIs for ~2,425 entries; stratify break by resolution; label-noise sensitivity. Feasible here. |
| D19 sequence-novelty axis (3rd axis) | TRUE | CPU-NOW (protein_seqsim_max in annotations) |
| D20 joint 2-D novelty grid | TRUE | CPU-NOW |
| D21 apo/holo pocket conformational novelty | TRUE | NET/STREAM (needs holo refs) |
| D25 native-gate seed stability | TRUE, cheap | CPU-NOW (5 seeds in data) |
| D26 Boltz-2 fully governed (2023 ref) | TRUE | CPU-NOW (data present) |
| D-current-gen new models | TRUE | **GPU** (generate) or find released |
| D28 trained-on-novelty ceiling | TRUE, valuable | CPU-NOW |
| D29 physics rescoring fallback (Vina/GNINA/MM-GBSA) | strong | STREAM + tool install (GNINA CPU-ok) |
| D31 expert baseline (chemist eyeballs S3) | **INFLATED** (needs human experts) | defer/label as proposal |
| D32 economic pricing ($/calendar) | TEXT | CPU-NOW |
| RMSD-threshold sweep (frontier vs 1.0–5.0 Å) | TRUE | CPU-NOW |
| label-noise sensitivity | TRUE | CPU-NOW |
| consensus depth K=2..5 | TRUE | CPU-NOW |

## E. Data / integrity (CPU-NOW or NET)

E-single-chain-subset characterization · E-system-composition disclosure (metals/cofactors/covalent/peptide/NA) · E-tautomer/protonation handling · **E-promote App C protomer trap to a numbered section (TRUE)** · E-stratifier fragility investigation (2021-vs-2023 driver; ligand-axis re-key) · E-shared-cutoff evidence per model · **E-release per-(model,system) analysis CSV (TRUE, highest-value artifact)**. All CPU-NOW/NET.

## F. Claims / framing (TEXT)

F-title/abstract scope to one benchmark until D3 · F-split training-free/CPU vs 73% operating points more sharply · **F-certificate card unvalidated → label as PROPOSAL (TRUE)** · F-documented novelty search (databases/dates/terms) · F-2.7× next to A5 number · F-abstention needs a next-action (D29). 

## G. Related work — material omissions (TEXT + NET to verify)

**G-Barber-Candès-Ramdas-Tibshirani "Conformal beyond exchangeability" 2023 (most conspicuous gap)** · applicability-domain/OOD cheminformatics (Sheridan, Sahigara, Norinder) · **multicalibration (Hébert-Johnson; Gupta-Jung-Noarov-Pai-Roth) — add as a Table-3 baseline (substantive, CPU to run)** · adaptive conformal (Gibbs-Candès ACI; Podkopaev-Ramdas label shift) · AF2/AF3 confidence-calibration lit · selective-prediction lineage (Chow; Madras; Mozannar-Sontag) · Vickers-Elkin (decision curve) · Duchi-Namkoong (DRO) · flag preprint dependence in-text.

## I. Internal inconsistencies (all CPU-NOW / TEXT — MUST fix)

I1 cluster 954 vs 1005 (dedup subset vs full — reconcile) · I2 feasibility-map 6-model dedup (12,125) feeding 5-model cells (reconcile) · I3 IFP "AF3 8,277 across models" self-contradictory phrasing · I4 abstract 21/40 (+13) vs conclusion 35/47 denominators (unify) · I5 S0 "feasible by construction" fails at α=0.10 (c*=0.65) · I6 Sec 8 "by more than 0.00" typo · I7 AF3 S1/S2 coverage in ≥3 forms (extend coverage-map table to card values) · I8 "38 vs 80" both in abstract/conclusion without distinction · I9 decimal/percent consistency · I10 m*/β*/ρ* used once (inline or use).

## J. Presentation (TEXT + figures)

J-de-hedge prose (biggest readability win) · J-Fig 5 cards → color matrix, full cards to SI · J-Table 5 redesign (separate HB≤α and folds columns) · J-Fig 2 S4 hatching · J-Sec 5 split Lemma/Interpretation/Defense blocks · **J-add reliability diagram (score-bin vs accuracy per stratum) — basic, absent (TRUE)** · J-move practitioner box to front.

## H. Reproducibility (TEXT + config)

H-invert make repro opt-in defaults · H-ship analysis CSV · H-machine-readable card API · H-pin model versions in a table · **H-independent pre-registration timestamp (OpenTimestamps/OSF, not git) — TRUE** · H-compute/energy reporting · H-license/terms per model output · H-determinism statement.

---

## What I need from you (resource asks)

1. **A cloud CUDA GPU (e.g. the Lambda H100 used for the FoldBench regen)** — needed ONLY to *generate* co-folding predictions where none are released: **D3 (PLINDER/PoseBusters external validity)** if released predictions don't exist (I will check first), and any current-generation-model run. The M4 here cannot run co-folding inference. This is the single hard dependency.
2. **Everything else I can do locally now** — D1 (ligand pLDDT/PL-PAE) and D2 (typed IFP) from the local 37 GB tarball; D22 (crystallographic quality) over the network; all of A/B/C/E/F/G/I/J/H text + tabular work. No GPU.

## Honest read on scope
~80 items. Genuinely blocking-for-correctness: A1–A5 (A5 done), the entire §I checklist, C11. Blocking-for-scope-of-claim: D1 (feasible here), D3 (GPU-gated). Strongly-expected-and-cheap: A3, A4, B2, C-tier, D19/D20/D25/D26, E-CSV, G-Barber/multicalibration, J-reliability-diagram. Inflated/defer: D31 expert study, card user-study. Recommended path: do all CPU/NET items + D1 locally now; investigate D3 released-prediction availability; if absent, spin up the GPU for D3.
