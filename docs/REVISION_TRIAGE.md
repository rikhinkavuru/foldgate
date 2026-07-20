# Revision Triage — Four-Reviewer Audit → JCIM Draft

Source audit: `~/Downloads/foldgate_review_and_revision_plan.md` (R1 theory, R2 struct-bio, R3 practitioner, R4 reproducibility, Appendix A).
Target venue: **JCIM (ACS)** — decided 2026-07-20 (supersedes Bioinformatics Advances / Digital Discovery).

Verdict legend:
- **TRUE** — the finding is correct against the actual paper/code; must fix.
- **PARTIAL** — partly correct or already partly handled; fix the residual.
- **FALSE** — the finding is wrong against the evidence; rebut in cover letter / no change (still often worth a clarifying sentence).
- **PENDING** — awaiting code verification.

Action type: `text` (prose/bib) · `compute` (derive from existing results/data, no new experiment) · `experiment` (new run) · `figure` (regenerate) · `reframe` (restructure).

---

## R1 — Conformal / statistical ML theory

| ID | Verdict | Evidence | Action |
|----|---------|----------|--------|
| R1.1 theorem set-theoretic not shift-theoretic | **PARTIAL** | Part (b) IS a tautology (accept set fixed at fixed coverage); paper already frames (c) as load-bearing and the frontier as "strictly stronger". | `reframe`: retitle Theorem → "Exact selective-risk decomposition under concept shift"; demote (b) to one-line remark; foreground making Δ̄_c measurable (the drift selector). |
| R1.2 "exact identity" is a bracket under ties | **TRUE** | Abstract+Sec8 say "exact identity"; App proof admits ties → ≤/≥ brackets under boundary randomization. | `compute`: report empirical ties mass at each deployed τ per model (poses + coverage points); keep "exact" only if <0.5% coverage, else soften. |
| R1.3 Y redefined mid-paper | **TRUE** | Sec 4: "For this section only we take Y=1 to mean error." | `text`: define loss L=1[RMSD>2Å] in Sec 2, use L throughout; add notation table (Table 0). |
| R1.4 synthetic validation validates a tautology | **TRUE** | "invariant across nine reweightings (residual <0.001)" is exactly part (b) — a code regression test. | `text`+`experiment`: relabel as regression test; use synthetic generator (b7) to sweep D_q and locate the α-crossing vs drift magnitude. |
| R1.5 LTT validity vs fitted combiner | **FALSE (no bug) → presentation fix** | Confirmed disjoint everywhere: e4 combiner.fit on `tr`(40%), LTT on `cal`(30%), eval `te`(30%), `tr∩cal=∅` (`e4:39-43,65,76-79`); e13 LOTO refits combiner per fold OOF, **gate uses native score not combiner** (`e13:116-118`); e11 shift combiner on `s_tr` disjoint from `s_cal`/`t_cal` (`e11:165-168`). Coverage grid hard-coded (`risk.py:94`). No regeneration needed. | `figure`: data-flow diagram (n per node per model) + one sentence stating disjointness. Rebut the "correctness bug" reading in cover letter. |
| R1.6 FST ordering must be pre-registered | **PARTIAL (resolved mechanism)** | Confirmed hard-coded `coverage_grid = np.arange(0.05, 1.0, 0.01)` ascending, data-independent, default used by every call site (`risk.py:94-95,99`; e4/e13/e11 pass no grid). | `text`: state the explicit grid + direction; commit a timestamped pre-registration file (grid, FST order, binning, endpoints). |
| R1.7 δ bookkeeping inconsistent | **PARTIAL** | δ=0.10 default; δ/K=0.02 is the Bonferroni split of the joint claim (not an inconsistency). | `text`: add a **certificate ledger** table (one row per claim: δ, multiplicity method, n, joint/per-model). |
| R1.8 achievability slack asserted for a ratio | **TRUE** | Paper cites O(√(log(1/δ)/n_g)) LTT slack for a ratio functional without a precise reference. | `text`+`experiment`: cite precise LTT-ratio result OR call it empirical calibration; the label-cost curve (III.1) is the empirical version — show it tracks the rate. |
| R1.9 positioning of concurrent [crcgupta] + preprint flags | **PARTIAL** | crcgupta framed as concurrent; incoherence/credal/bai2026 are unrefereed 2026 preprints. | `text`: add dated preprint record for the independence claim; flag preprint status inline for load-bearing citations. |
| R1.10 missing baselines | **PARTIAL** | Have Platt/isotonic/native-ipTM-LTT-gate/PoseBusters (E11). Missing: plain fixed-ipTM-threshold field baseline, Venn–Abers, continuous-novelty localized conformal (hore2024rlcp cited not run), accept-all/abstain-all. | `experiment`: add the four missing baselines (feeds III.7 + decision curve). |

## R2 — Structural biology / computational chemistry

| ID | Verdict | Evidence | Action |
|----|---------|----------|--------|
| R2.1 novelty axes never defined | **TRUE** | Confirmed: ligand=`morgan_tanimoto` (ECFP4 Morgan Tanimoto to nearest train ligand, RNP-precomputed max-agg, /100); pocket=`sucos_shape_pocket_qcov` (SuCOS shape × pocket qcov, /100); both **quartile** bins S0–S3 + NaN=no-analog S4 (`novelty.py:39-67`, `pd.qcut(q=4)`). NaN fraction large (ligand 30.4%, pocket 33.5%) so S4 is real. n per (model,stratum) obtained (ligand af3 692/654/629/658/76; pocket 667/646/642/675/79; full table in agent report). ABSENT from paper. | `text`+`compute`: methods subsection (descriptor, similarity fn, reference set=nearest train system, aggregation, quartile edges, NaN=S4) + supp table of n + similarity range per (model, axis, stratum). |
| R2.2 stratifier not computable at deployment for closed models | **TRUE (important)** | RNP ships labels; AF3 training corpus not publicly enumerable → operative repair uncomputable for the flagship. | `experiment`: **deployment-computable proxy** — stratify by similarity to a public PDB snapshot at each model's cutoff; report per-stratum control vs shipped labels. (III.2) |
| R2.3 2Å label insufficient; add PB-validity | **TRUE** | Label is RMSD≤2 only; PoseBusters data present (`data/raw/posebusters`). | `experiment`: joint label Y=1[RMSD≤2 ∧ PB-valid] as secondary target; report accepted-vs-rejected PB-validity. (III.8) |
| R2.4 alignment convention unstated | **TRUE** | Sec 2 gives no RMSD alignment convention. | `text`: state global/pocket/ligand-in-place; confirm matches shipped BiSyRMSD. |
| R2.5 App B integrity contaminates unknown set | **PARTIAL** | Primary selective-risk uses shipped BiSyRMSD (safe). Recomputed geometry only in W1 pose-agreement (D1) + E6b interactions. | `text`+`compute`: provenance table (per result: shipped vs recomputed; if recomputed, single-chain subset + n); promote App B to main methods. |
| R2.6 dataset composition black box | **TRUE** | Sec 2 gives no target-class / ligand-property breakdown. | `compute`: report target-class composition + ligand property distributions + cofactor/ion/peptide/covalent counts. |
| R2.7 temporal null artifact of shared cutoff | **TRUE (confirmed, worse than stated)** | `build_features.py:37` bins a SINGLE pooled `release_date` (2021-10-06→2024-06-05) into 4 quantiles for all models. Every RNP system is already post-AF3-cutoff (2021-09-30) and post-Chai (2021-12-01), so temporal strata rank *recency among already-out-of-training targets*, not in/out. Boltz-2's earliest is 2023-06-07 → pooled edges dump all its rows into bins 2–3 (artifact). Per-model metadata (`release_date_before_cutoff`, `target_release_date`, `sucos_shape_pocket_qcov_2023`) EXISTS in annotations, UNUSED. | `experiment`: recompute temporal axis per model at each model's own stated cutoff (in/out-of-training split, not recency quantiles). Report whether the null survives. High value — either strengthens or kills the "temporal-vs-structural magnitude contrast" the paper calls its one robust contrast. |
| R2.8 fingerprint method unspecified + uncontrolled | **TRUE (confirmed confound)** | `interactions.py`: custom numpy+gemmi (not PLIP/ProLIF), 4.5Å min-heavy-atom-distance contact shell, NO angles/typing/protonation, key=(seqid,resname), crystal from ground_truth.tar.gz. e6b comparison pools ALL accepted vs ALL rejected (`e6b:59-66,85-90`), never RMSD-binned → the 0.15–0.20 gap partly reflects accepted-set being low-RMSD-enriched. | `text`+`experiment`: name method/cutoffs; condition on RMSD — accepted vs rejected WITHIN sub-2Å set, or regress recovery on RMSD with gate status as covariate. Report only the residual gate coefficient. (III.9) |
| R2.9 consensus metric forward-referenced out | **TRUE** | Sec theory: "we develop it and its metric construction separately." Construction exists in D1/E14. | `text`: include the construction (~½ page) or move numbers to SI with definition. |
| R2.10 FoldBench compares wrong quantity | **PARTIAL** | Reports AURC (ranking) not realized risk; 0.40 self-scored vs 0.57 released top-1 gap. | `compute`: report realized selective risk at frozen τ on the 52 low-homology separately from 384 train-similar; CIs on both AURCs; state directional. |

## R3 — Industrial drug-discovery practitioner

| ID | Verdict | Evidence | Action |
|----|---------|----------|--------|
| R3.1 headline coverage is the leaked one | **TRUE** | Confirmed: e4 `three_way` split is random at pose-ROW level, NOT target-grouped; up to 3 rows per (system,method) → same-target poses straddle tr/cal/te (exactly the leak e13 quantifies). 71%/82% is from this leaky split; leakage-free = e13 LOTO (GroupKFold on system_id) 52%@0.185. Abstract+conclusion headline 71/22. | `experiment`+`text`: headline the leakage-free LOTO pair everywhere; compute the native-gate coverage under LOTO as the matched 22%-comparator (currently only from the leaky split). |
| R3.2 screening does not beat baseline | **TRUE** | Table: +ipTM gate ties affinity-no-abstention on DEKOIS, ~ on GPCR, worse than random on LIT-PCBA; only beats 50% random discard. | `reframe`: recast as a **cost result** (halves inspection at no enrichment cost); move screening arm to SI. |
| R3.3 abstract EF 9.3 is Hou's ungoverned baseline | **TRUE** | Table row "Pose ipTM, no abstention (Hou)" = 9.3; governance contributes nothing. | `text`: remove from abstract+conclusion or restate as "[10]-established signal our layer governs" + give the incremental number. |
| R3.4 best method (affinity head) has no guarantee | **TRUE** | Table affinity head 31.0/26.6 marked ‡ ungoverned. | `text`: confront in discussion; name affinity-head certification as the top open problem (state what label is needed). |
| R3.5 label cost lands where labels don't exist | **TRUE** | 38 median in-stratum labels = crystals of novel targets a live program lacks. | `text`+`experiment`: "what this costs a program" paragraph; honest bottom line in conclusion; label-cost curve (III.1). |
| R3.6 "no GPU" true of audit not deployment | **TRUE** | Combined score uses ensemble spread + cross-model agreement → 5 co-folding runs per target. | `text`: split the claim — native-score gate CPU-only, single inference; combined needs N inferences (report GPU cost); surface the native-only guarantee prominently. |
| R3.7 ensemble features reintroduce novelty into score | **PARTIAL** | METHODS says novelty excluded from score; but ensemble/cross-model features plausibly correlate with novelty. | `compute`: report correlation of combined-score ensemble features with novelty label; disclose achievability-regime implication if substantial. |
| R3.8 α=0.20 loose | **PARTIAL** | α=0.10 numbers exist; 1Å graded result in Sec 5. | `text`+`experiment`: main-text operating points at α=0.10 and 0.05; promote 1Å graded result; decision curve (III.6) retires the level defence. |
| R3.9 ρ* uninterpretable | **TRUE** | ρ*≥0.10 given with no physical anchor. | `compute`: anchor ρ=0.10 to a chemotype/pocket shift magnitude estimated from the empirical strata. |
| R3.10 label-free repair vacuous where needed | **TRUE** | Paper admits m*→1 on deploy-to-novel parenthetically. | `text`: state as a finding in the conclusion, not a mid-section parenthetical. |

## R4 — Methodology & reproducibility

| ID | Verdict | Evidence | Action |
|----|---------|----------|--------|
| R4.1 four irreconcilable counts | **TRUE (chain resolved)** | Verified: 13,535 raw (6 models, per system×method×ligand-instance, top-1 ranking_score) − 933 boltz2 = 12,602 governed / 2,425 systems → dedup one-per-(system,method) = 11,254 (5 governed target-labels); all-6 dedup = 12,125 (d2, `d2:116-123`); d1 track separate = 13,146 frame checks (13,215 pairs ∩ non-null). Gaps are multi-ligand-instance collapse. | `figure`: CONSORT-style flow diagram, n at every node, which results use which node. |
| R4.2 S4 n stated two ways | **TRUE** | Fig 1 "n≈68–76"; Sec 7 "n≈63". | `compute`: reconcile, give exact per-model n. |
| R4.3 S4 simultaneously headline and disclaimed | **TRUE** | Sec 3 quotes 0.427 on S4; Fig 1 caption calls S4 estimation noise. | `text`: drop S4 from headline claims; keep only in the frontier (zero coverage is the point); lead the break on S3. |
| R4.4 no accepted-set n on any selective risk | **TRUE** | d2 already carries `n_accepted_targets` + exact CP `R_Q_ci` per cell (free to surface). e17 has only model-level N (reconstruct k=round(cov·N)); e9 persists no per-cell N (needs light re-run to store n_accept). | `compute` (d2, e17) + `experiment` (e9): annotate every risk with (accepted n / stratum n) + Clopper–Pearson; suppress/grey cells with accepted n<10. |
| R4.5 three different "AF3 on S3" numbers | **TRUE** | 0.375 (marginal global), 0.38 (fig), 0.538 (transfer); abstract 0.55 vs Sec 4 0.538 (raw 0.547). | `text`: name distinctly (R_marg,S3 / R_gate,S3 / R_transfer,S3); reconcile 0.55↔0.538, consistent rounding. |
| R4.6 "0 of 300 runs" overstates evidential weight | **PARTIAL** | With point est 0.55 vs 0.20 the all-300 violation is near-arithmetic; scheme (300 random 50/50 splits, e2) unstated in paper. | `text`+`compute`: specify resampling scheme + unit; replace headline with effect size + one CI. |
| R4.7 Romano–Wolf grid doesn't add up (40 vs 50) | **TRUE (resolved)** | 11 axis-strata = ligand 4 (S1–S4, S0=ref) + pocket 4 + temporal 3 (S1–S3). ×5 models = 55 = 40 structural (2 axes × 4 × 5) + 15 temporal (1 × 3 × 5). The 40-vs-50 gap is the S0 reference stratum excluded from cells (`e12:144-148`). | `text`: enumerate the 11 axis-strata + state the reference exclusion; note all-40-structural-surviving = effect-dominated family, not a survival rate. |
| R4.8 frontier headline omits robust subset; 50 vs 47 | **TRUE (resolved)** | 21/50 zero-frontier at α=0.20, 26/50 at α=0.10; **47 = 21+26 pooled across BOTH α levels** (not 3 cells dropped); 35/47 survive the CP lower bound (`R_Q_ci[0]>α`). 50 = 2 axes × 5 models × 5 strata. | `text`: headline the robust 35; state 47 explicitly as the α∈{.10,.20} union; reword the currently-ambiguous sentence. |
| R4.9 EF CIs at endpoint / discreteness | **TRUE** | Table 2: 10.3 [10.3,15.5], 20.7 [18.1,20.7], 31.0 [28.4,31.0] — point at an endpoint. | `compute`: report actives-in-top-1% counts per set + number of distinct achievable EF values; add BEDROC / log-AUC secondary metric. |
| R4.10 proportions without intervals | **TRUE** | 13%/75%/40% (=12/16, 2/5 etc.) no intervals. | `compute`: n/N + Wilson intervals inline; LIT-PCBA (n=5) → footnote not column. |
| R4.11 LOTO risks reported as realizations not bounds | **TRUE** | Table 1 LOTO realized .185/.157/.168/.30†/.191 — reader can't tell if certificate held. | `compute`: add certified upper-bound column at δ + pass/fail per model; report Chai † coverage n. |
| R4.12 betting-bound validation depth unstated | **TRUE (resolved)** | e9 `n_repeats=120`, pose-level (non-grouped) 3-way splits; AF3 1Å "0.90" = 46/51 usable splits meeting target. | `text`: give reps (120) + unit; note the pose-level caveat and that grouped resampling is the primary certificate (e13/d2). |
| R4.13 figure encoding problems | **TRUE** | Fig 1 bars = "5th–95th pct over splits" (spread, mislabeled CI); Fig 3b coverage as marker area + ▽ glyph. | `figure`: relabel Fig 1 as split spread + add true bootstrap CIs; Fig 3b direct coverage labels; Fig 3a annotate accepted n. |
| R4.14 reproducibility artifact underspecified | **TRUE** | Data/code statement lacks pinned env, seeds, wall-clock, artifact sizes, license. | `text`: add all; state license. |
| R4.15 missing front/back matter | **TRUE** | No funding, competing-interests, author-contribution, broader-impact statements. | `text`: add all (single author → competing interests "none"). |
| R4.16 citation hygiene | **TRUE** | [rnp] no authors; [boltz] bundles Boltz-1/1x/2; preprints unmarked. | `text`: complete rnp; split boltz with versions; mark preprint status inline. |
| R4.17 abstract & prose density | **TRUE** | Abstract ~400 words, ~15 numbers; theory sentences >60 words. | `text`: cut abstract to ~200 words / 4 numbers; ~35-word sentence ceiling in theory. |

## Appendix A — initial-read items

| ID | Verdict | Evidence | Action |
|----|---------|----------|--------|
| A.1 CVaR convention mismatch | **TRUE** | Prose "worst β-fraction"; formula min(1,r/(1−β)) and m*=1−β* use worst-(1−β). | `text`: fix prose to worst-(1−β). |
| A.2 reliability-drift bin occupancy | **TRUE** | D(ν) bins on s; novel strata may empty the high-s bins; +0.63 is load-bearing. | `compute`: report per-bin counts behind D. |
| A.3 consensus provenance | **TRUE** (= R2.5) | Cross-model floor uses recomputed geometry. | `text`: state single-chain subset + size (6,163). |
| A.4 weighted-conformal tension Sec4↔Sec5 | **TRUE** | Sec 5 "weighted pulls 0.27→0.19 at 70% cov" reads as fixing the conditional; by (b) it is purely a coverage move. | `text`: state the coverage at which naive 0.27 was measured; say weighted's only lever is threshold selection. |
| A.5 graded-loss result buried | **TRUE** | 43%@1Å (Hoeffding 0.04%) mid-Sec 5. | `text`/`reframe`: promote to main text. |
| A.6 purity = 1−selective risk | **PARTIAL** | Paper states it correctly. | `text`: ensure no summary/abstract implies purity is independent evidence. |

---

## Part III additions (do at SOTA)

| ID | Description | Type | Feasible locally? |
|----|-------------|------|-------------------|
| III.1 | Label-cost curve (Fig 4): certified coverage vs n_g ∈ {5,10,20,40,80,all} per stratum, band over resamples | `experiment` | Yes (CPU, RNP labels) |
| III.2 | Deployment-computable stratifier (= R2.2): proxy via public PDB snapshot at model cutoff vs oracle | `experiment` | Yes (annotations ship temporal + similarity) |
| III.3 | Stratifier-misspecification sensitivity: 3/5/7 bins, quantile vs fixed edges; within-stratum residual drift | `experiment` | Yes |
| III.4 | Pseudo-prospective eval: freeze calibration pre-T, evaluate post-T, no stratum info | `experiment` | Yes |
| III.5 | Second cross-dataset check (PoseBusters / Astex / PDBbind time-split) | `experiment` | Check data availability; else state the concrete community ask |
| III.6 | Decision-curve / net-benefit vs cost ratio λ; gate vs fixed-ipTM vs accept-all vs abstain-all | `experiment` | Yes |
| III.7 | Missing baselines: fixed ipTM threshold, per-stratum Venn–Abers, localized-continuous conformal, accept-all/abstain-all | `experiment` | Yes (= R1.10) |
| III.8 | Joint PoseBusters label (= R2.3) | `experiment` | Yes (PB data present) |
| III.9 | RMSD-conditioned fingerprint analysis (= R2.8) | `experiment` | Yes |
| III.10 | Certificate card (per model×stratum): α, δ, cal n, accepted n, certified risk UB, realized risk, coverage, drift D, repair, FEASIBLE/ABSTAIN | `compute`+artifact | Yes |
| III.11 | Software packaging: pip API, container digest + lockfile + seeds, Zenodo DOI, tutorial notebook, license | release-eng | Yes |

## Part IV — restructure (JCIM, 9–11 pp + SI)

Contributions 5 → **3**: (1) break + per-stratum feasibility frontier; (2) exact decomposition as a deployable drift diagnostic; (3) honestly-priced group-conditional repair with loss-matched certifier. Ranking utility → subsection; screening → SI (cost result). ~200-word abstract, 4 numbers (break 0.55 vs 0.20; robust zero-coverage cells; repair coverage@risk≤0.20 leakage-free; label price 38). Sections per audit Part IV table.

## Cross-cutting findings (surfaced during verification)

- **Pose-level vs target-grouped resampling.** e4 (3-way), e9 (continuous risk), and the e11 i.i.d. panel resample at the pose/row level (up to 3 rows per (system,method)), so same-target poses can straddle folds. The primary CERTIFICATE results (e13 LOTO, d2 feasibility) are correctly grouped by `system_id`. For the JCIM draft, every headline number must come from a target-grouped protocol; pose-level splits move to SI or are re-run grouped. This is the root of R3.1 and touches R4.12.
- **Temporal axis is a recency ranking, not in/out-of-training** (R2.7) — the single most impactful methodological fix; it may change the "temporal null" narrative.
- **All four "irreconcilable" counts reconcile** via one dedup step (per-(system,method)) + the separate d1 distance track; a CONSORT diagram closes R4.1 cleanly.
- **Verdict tally:** TRUE/confirmed 33 · PARTIAL 6 · FALSE 1 (R1.5, no bug). Zero findings require discarding a result; the corrections are stronger-honesty + missing-analysis + presentation.

## Execution order

1. Triage sign-off (this doc) — **DONE**, all PENDING resolved via code agents (2026-07-20).
2. New experiments (III + the compute items) → JSON under `results/`.
3. Regenerate figures with CIs + accepted-n (R4.13) + CONSORT flow (R4.1).
4. Rewrite `paper/foldgate_journal.tex` → JCIM structure; certificate ledger + notation + provenance tables; leakage-free headlines; abstract cut.
5. Pre-registration file + front/back matter + bib hygiene + repro block.
6. Hand back for re-audit.
