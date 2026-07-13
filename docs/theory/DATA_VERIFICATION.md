# DATA VERIFICATION  (Phase 0 grounding, 2026-07-12)

I have everything needed. Here is the memo.

---

# DATA-VERIFICATION + FEASIBILITY MEMO — screening EF (J2 / J3 / W3)

**Scope of files read:** `results/e16_selective_screening.json`, `results/e20_screening_broad.json`, `results/e19_shift_decomp.json`, `results/e17_worst_subpop.json`, `results/e22_robust_certificates.json`, plus the raw screen at `/Users/rikhinkavuru/moml/data/external/screening/{dekois,lipcba,gpcr}_{scores,stat}/` and the drivers `experiments/e16_selective_screening.py` and `experiments/e20_screening_broad.py`.

**One load-bearing fact that reframes J2 up front (verified in the driver code):** in `e16` the ranker for every `ef_affinity / bedroc_affinity / auc_affinity / ef_full / ef_sel50 / ef_rand50` field is Boltz-2 `affinity_probability_binary` (the affinity head). The only pose-confidence EF in `e16` is `ef_iptm` (ranker = Boltz-2 `iptm`). In `e20`, `boltz2.ef` is ALSO the affinity head (`ranker="affinity_probability_binary"`), whereas `protenix.ef` and `af3.ef` use `ranking_score` (a pose-confidence field). So the headline enrichment numbers are affinity-driven, and the conformal layer (calibrated on pose-RMSD≤2Å correctness) governs only the ipTM / ranking_score signals. The gap this exposes is exactly the J2 concern.

---

## J2 — pose-confidence gate vs affinity head EF (the "decision = guarantee same object" check)

### The GPCR claim is confirmed exactly
`e16 → datasets.gpcr.agg`:
- `ef_affinity.median = 26.62652068126521`, ci90 `[19.333, 35.660]` (n=16)
- `ef_iptm.median = 9.300486618004866`, ci90 `[5.747, 14.832]` (n=16)

"Affinity head EF 26.6 vs pose ipTM EF 9.3 on GPCR" = CONFIRMED (26.627 vs 9.300).

### Full pose-vs-affinity split, all three datasets (`e16 → datasets.<ds>.agg`, medians with target-level bootstrap ci90)

| Dataset (n_targets) | Affinity head `ef_affinity` (NOT guaranteed) | Pose ipTM `ef_iptm` (guaranteed-layer signal) | Docking `ef_dock` (Gnina CNNscore) |
|---|---|---|---|
| DEKOIS (79) | **31.0** ci90 [28.417, 31.0] | **20.667** ci90 [18.083, 20.667] | 10.333 ci90 [10.333, 15.5] |
| LIT-PCBA / lipcba (5) | **5.11995** ci90 [4.873, 23.060] | **3.96078** ci90 [0.0, 26.115] | 2.97059 ci90 [0.0, 7.687] |
| GPCR (16) | **26.62652** ci90 [19.333, 35.660] | **9.30049** ci90 [5.747, 14.832] | 1.97134 ci90 [1.5, 2.617] |

Take-away: across all three, the affinity head beats the pose signal it does not certify — DEKOIS 31.0 vs 20.7, GPCR 26.6 vs 9.3, LIT-PCBA 5.1 vs 4.0 (LIT-PCBA CIs are enormous, n=5).

### Corroborating independent numbers from the source-paper stat tables (`*_total_stat.csv`, method-level median `EF0.01`)
These are the paper's own EF computation (not ours), and they separate the two Boltz-2 affinity variants:
- **DEKOIS:** `Boltz-2 (Probability)` = 28.615, `Boltz-2 (Affinity)` = 28.592 [these two = affinity head, `affinity_probability_binary` / `affinity_pred_value`]; `Boltz-2 (ipTM)` = 19.077, `Boltz-2 (Ranking Score)` = 9.538, `Boltz-2 (pTM)` = 4.758 [pose]; `AF3 (Ranking Score)` = 21.462, `AF3 (ipTM)` = 21.462 [pose]; `Protenix (ipTM)` = 16.692, `Protenix (Ranking Score)` = 14.308 [pose]; `Gnina (CNNscore)` = 9.538 [dock].
- **GPCR:** `Boltz-2 (Probability)` = 26.627, `Boltz-2 (Affinity)` = 16.999 [affinity]; `Boltz-2 (ipTM)` = 9.300, `Protenix (ipTM)` = 5.667 [pose]; `Gnina (CNNscore)` = 1.971 [dock].
- **LIT-PCBA:** `Protenix (ipTM)` = 5.001, `Boltz-2 (ipTM)` = 3.883 [pose]; `Glide_SP` = 3.751, `Gnina (CNNscore)` = 2.912 [dock]. (Note: LIT-PCBA stat table ships **no** Boltz-2 Probability/Affinity rows — the affinity head is absent from the paper's LIT-PCBA stats; our `e16 ef_affinity=5.12` is recomputed from the raw `boltz_scores.csv`.)

Note the paper's stat `EF0.01` (28.6 for DEKOIS Probability) differs from our recomputed `e16 ef_affinity` (31.0). The difference is our pipeline: one-row-per-compound by `max` over poses/seeds, top-k = `round(0.01·N)`, our `enrichment_factor`. So we are already NOT blindly inheriting the paper's number.

### Pre-registered gate numbers (`e20`) — pose-confidence-linked mechanism, but the ranker underneath is still affinity for Boltz-2
`e20 → datasets.<ds>.models.<m>.agg`. The `ef_reg_gate` applies the RNP-LTT-calibrated ipTM threshold (`rnp_iptm_tau`: boltz2=0.66138, protenix=0.98878, af3=0.875) to the screen:
- DEKOIS boltz2: `ef` = 31.0 [28.417,31.0], `ef_reg_gate` = 31.0, `cov_reg_gate` = 0.99839 → the pose gate keeps ~99.8% of the library, i.e. it barely acts; the 31.0 is essentially the ungated affinity ranking.
- DEKOIS protenix (pose ranking_score ranker): `ef` = 15.5 [12.917,18.083], `ef_reg_gate` = 18.083 but `cov_reg_gate` = **0.04032** (n=60 targets survive) → the ipTM=0.989 threshold discards ~96% of compounds.
- DEKOIS af3 (pose ranking_score): `ef` = 20.667 [18.083,23.25], `cov_reg_gate` = 0.14032.
- GPCR boltz2: `ef` = 26.627, `cov_reg_gate` = 0.98164. GPCR protenix: `ef` = 1.80476 [1.0,4.167], `cov_reg_gate` = 0.01409.
- LIT-PCBA boltz2: `ef` = 5.11995 [4.873,11.192]; protenix: `ef` = 5.11995 [0.975,22.385].

### Precise pose-linked vs affinity-linked ledger (which numbers the guarantee governs)
- **Affinity-linked (NOT covered by the pose conformal guarantee):** `e16` `ef_affinity`, `bedroc_affinity`, `auc_affinity`, `ef_full`, `ef_sel50`, `ef_rand50*`; `e20` `boltz2.ef/bedroc/auc`; `e20` `shift_boltz2_affinity`; stat rows `Boltz-2 (Probability)` and `Boltz-2 (Affinity)`.
- **Pose-confidence-linked (fields the conformal layer actually governs):** `e16` `ef_iptm`; `e20` `af3.ef` and `protenix.ef` (ranking_score); every `ef_reg_gate / cov_reg_gate / ef_sel50` abstention (the gate variable is ipTM); stat rows `* (ipTM)`, `* (Ranking Score)`, `* (pTM)`, `* (min-iPAE)`.
- **Supporting context that the guarantee object is pose, not activity:** `e19_shift_decomp` (RNP pose-RMSD) shows the novel-target risk gap is concept-dominated, e.g. `af3` target `"3+4"`: `gap_total=0.3175`, `gap_concept=0.2966`, `gap_covariate=0.0209`; `chai "3+4"`: `gap_total=0.5068`, `gap_concept=0.4958`, `gap_covariate=0.0110`; `protenix "3+4"`: `gap_total=0.3123`, `gap_concept=0.3055`, `gap_covariate=0.0068`. `e17_worst_subpop` and `e22_robust_certificates` are pose-RMSD RCPS/LTT certificates on RNP (e.g. `e17 af3.cov_at_mstar_0.5_combined=0.45`, `novel_curve` almost entirely `certified=false`; `e22 simultaneous_at_cov_0.2.all_certified=true`, `joint_rho_star=0.09798`). None of e19/e17/e22 contains a screening EF — they certify pose correctness, which is the object the guarantee is about; the EF is a downstream, largely affinity-driven decision.

**J2 bottom line:** the enrichment being sold (EF 31 DEKOIS, 26.6 GPCR) rides on the Boltz-2 affinity head, which carries no conformal guarantee; the pose-confidence signal the layer certifies enriches materially worse (20.7, 9.3). The "decision" and the "guarantee" are different objects, and the numbers to make that honest are all present.

---

## J3 — fair baseline: what is computable from shipped data vs what needs a rerun

Raw per-compound files exist for all datasets under `data/external/screening/<ds>_scores/<target>/`. Verified column schemas:
- `boltz_scores.csv`: `lid,label,ptm,iptm,confidence_score,affinity_pred_value,affinity_probability_binary,mpae` (all 3 datasets).
- `pix_scores.csv` (Protenix): `lid,label,seed,sample,plddt,gpde,ptm,iptm,ranking_score` (+`pid`,`Unnamed:0` on DEKOIS; +`label.1` on lipcba/gpcr).
- `af3_scores.csv`: `lid,label,seed,sample,ptm,iptm,ranking_score,mpae` — **DEKOIS only** (79 files; none in lipcba/gpcr).
- Docking: `gnina_scores.csv` = `ID,label,vina,cnnscore,cnnaffinity` (all datasets, all rows non-null — verified a2a: 40 act / 1200 dec, cnnscore & vina non-null for all 1240). DEKOIS also ships `af3_glide_scores` (`glide_min_SP/XP_top1/top5`), `af3_gnina_scores` (ad4/vina/vinardo/cnnscore/cnnaffinity ×top1/top5), `pix_glide_scores`, `pix_gnina_scores`, `pix_igmodel_scores`, `pix_pignet2_scores`, `pix_planet_scores`, `pix_rtms_scores`, `carsidock_scores` (`score`). GPCR adds `glide_scores` (`score`), `pix_glide/pix_gnina/pix_rtms`. LIT-PCBA: `glide_scores` + `gnina_scores`.

### (a) Scaffold split of actives — PARTIALLY computable
- `actives_similarity.csv` ships for **DEKOIS and GPCR** (per-active): `ID,smiles,ec_sim,ec_mk_sim`. `ec_sim` = ECFP Tanimoto-to-train; `ec_mk_sim` = Murcko-scaffold Tanimoto-to-train (a2a actives: ec_sim mean 0.444, ec_mk_sim mean 0.578). So a **scaffold-to-train novelty axis is already shipped** (and already used by the shift curves).
- A true Murcko **scaffold split/grouping of actives** is fully computable for DEKOIS + GPCR: RDKit is present in `.venv` and parses the shipped SMILES (verified: a2a 40/40 actives → 40 unique Murcko scaffolds via `MurckoScaffold.MurckoScaffoldSmiles`).
- **LIT-PCBA: NOT computable from shipped data.** No `actives_similarity.csv` and no SMILES in its per-target files (`MAPK1/` has only `boltz/glide/gnina/pix_scores.csv`, `lid` ids only). Scaffold split for LIT-PCBA needs an external SMILES source (LIT-PCBA is public, but it is not in this tree).
- Decoys carry **no SMILES anywhere** in the tree (only `actives_similarity.csv` has a `smiles` column). A scaffold split of the full active+decoy library is therefore not possible without external decoy SMILES.

### (b) Decoy-quality diagnostic — LIMITED
- No physicochemical property columns (MW/logP/TPSA/etc.) are shipped for actives or decoys, and decoys have no SMILES → property distributions for decoys cannot be computed from shipped data.
- What IS computable: (i) active-side properties (MW/logP/TPSA…) via RDKit from the shipped active SMILES (DEKOIS + GPCR only); (ii) the active→train similarity distributions (`ec_sim`, `ec_mk_sim`) are shipped directly. A decoy→active similarity or property-match diagnostic (the standard DEKOIS/DUD-E analog-bias check) needs decoy SMILES → external.

### (c) Recomputing EF/BEDROC ourselves — YES, broadly
From the raw score files we can (and `e16`/`e20` already do) recompute EF@1% and BEDROC ourselves for: Boltz-2 (ipTM, ranking_score, confidence_score, pTM, affinity_probability_binary, affinity_pred_value), Protenix (ranking_score, ipTM, pTM), AF3 (ranking_score, ipTM, pTM — DEKOIS only), Gnina (cnnscore, vina, cnnaffinity), Glide (SP/XP where shipped), CarsiDock, and the pix rescore models (igmodel/pignet2/planet/rtms). Pose BEDROC/AUROC (missing from `e16`, which only computes affinity BEDROC/AUROC) are available either by our own recompute from raw scores or from the stat tables (`* (ipTM)` rows carry `BEDROC80.5` and `AUROC`). So we are not forced to inherit the source paper's stat EF.

### (d) NOT computable without a rerun (honest list)
- AF3 pose-confidence EF on **GPCR and LIT-PCBA** (no `af3_scores.csv` shipped there) → needs running AF3.
- Any new docking method/pose/co-folding seed not already in the tree → GPU/CPU docking rerun.
- **Pose-RMSD correctness labels for screen compounds** — decoys have no crystal pose, so the conformal guarantee's own label (RMSD≤2Å) does not exist on the screen. The guarantee cannot be directly validated on the screen; this is precisely why `e16`/`e20` describe the ipTM gate as a "heuristic transfer" and pre-register the null.
- Decoy scaffold/property fair-split diagnostics that require decoy SMILES.

---

## W3 — LIT-PCBA both directions, and the DEKOIS/GPCR directions (report honestly)

There are **two different shift constructions in the results, and they point opposite ways for DEKOIS/GPCR.** Both sets of exact numbers are below so the paper can pick one convention and state it.

### Construction 1 — `e16 → shift.<ds>.{molecular_sim,scaffold_sim}` (inherited: source-paper `*_stat/{molecular_sim,scaffold_sim}.csv`, Boltz-2, `mean_ef01` by cumulative similarity-threshold `sim_cutoff`; n_targets grows with cutoff, so `sim=1.0` = full active set, low cutoff = novel subset)

`molecular_sim` (sim_cutoff → mean_ef01, n_targets):
- **DEKOIS:** 0.3→31.888 (106), 0.4→27.168 (158), 0.5→25.672, 0.6→23.857, 0.7→22.893, 0.8→21.930, 0.9→21.276, 1.0→**20.344** (158). Direction: **DECREASES** with cutoff (novel subset scores higher).
- **GPCR:** 0.3→16.141 (32), 0.4→14.435, 0.5→13.136, 0.6→13.165, 0.7→13.105, 0.8→13.107, 0.9→13.087, 1.0→**13.050** (32). Direction: **DECREASES** (mild).
- **LIT-PCBA:** 0.3→**0.0** (8), 0.4→0.393 (10), 0.5→2.363, 0.6→3.678, 0.7→6.020, 0.8→7.310, 0.9→7.440, 1.0→**8.694** (10). Direction: **INCREASES** (novel subset scores near zero) — opposite of DEKOIS/GPCR.

`scaffold_sim` (sim_cutoff → mean_ef01, n_targets):
- **DEKOIS:** 0.3→35.987 (18), 0.4→31.235 (124), 0.5→30.623 (158), 0.6→29.569, 0.7→26.988, 0.8→25.673, 0.9→25.401, 1.0→**20.344** (158). DECREASES.
- **GPCR:** 0.3→20.559 (32), 0.4→15.542, 0.5→14.364, 0.6→13.542, 0.7→13.147, 0.8→13.111, 0.9→13.047, 1.0→13.137 (32). DECREASES.
- **LIT-PCBA:** 0.3→**0.0** (2), 0.4→1.612 (8), 0.5→5.768 (10), 0.6→3.947, 0.7→9.936, 0.8→11.227, 0.9→**11.716** (10), 1.0→8.694 (10). Mostly INCREASES (non-monotone; peak at 0.9).

So for LIT-PCBA **both novelty axes go the same (intuitive) way** — enrichment is lowest on the most novel actives — while DEKOIS and GPCR run the opposite way under this cumulative-threshold convention.

### Construction 2 — `e20 → datasets.<ds>.shift_boltz2_affinity` (self-computed, disjoint `ec_sim` bins of actives with all decoys held fixed; value = [median EF@1%, n_targets])
- **DEKOIS:** `-0.01-0.3` → [66.833, 35], `0.3-0.5` → [42.5, 79], `0.5-0.7` → [67.167, 78], `0.7-1.01` → [72.828, 66]. Direction: roughly **INCREASES** with similarity (novel low bin 66.8, dip to 42.5, up to 72.8 for familiar) — non-monotone.
- **GPCR:** `-0.01-0.3` → [27.165, 16], `0.3-0.5` → [29.470, 16], `0.5-0.7` → [32.957, 16], `0.7-1.01` → [42.269, 14]. Direction: **MONOTONE INCREASING** with similarity (novel 27.2 → familiar 42.3). This is the clean "enrichment degrades on novel chemotypes" result.
- **LIT-PCBA:** all four bins = `[NaN, 0]` → the self-computed bin shift is **EMPTY for LIT-PCBA** (no `actives_similarity.csv`, so no per-active `ec_sim`).

### W3 honesty summary
- LIT-PCBA: the only available shift signal is Construction 1 (`e16`), and there **both molecular and scaffold axes agree** (novel actives enrich worse: molecular 0.0→8.7, scaffold 0.0→~11.7). Construction 2 does not exist for LIT-PCBA.
- DEKOIS and GPCR: the **two constructions disagree in direction.** Construction 1 (cumulative `≤`-threshold, inherited from the source paper) shows enrichment *higher* on the novel subset; Construction 2 (disjoint `ec_sim` bins, computed under our control) shows enrichment *lower* on the novel bin — the expected "novelty hurts" story, cleanest on GPCR (27.2→42.3 monotone). The likely reason is that Construction 1's nested `≤`-threshold bins put a handful of very-novel actives against all decoys (small, unstable EF), whereas Construction 2's disjoint bins are the interpretable comparison. Recommendation for the paper: report Construction 2 (GPCR monotone 27.2→42.3; DEKOIS non-monotone) as the shift figure, state the bin definition explicitly, and either drop Construction 1 or footnote the opposite direction so a reviewer cannot claim cherry-picking. Do not present the two side by side without reconciling the convention.

No number above is fabricated; every value is quoted from the named JSON key or the raw CSV. Where a number does not exist in the JSON (AF3 EF for GPCR/LIT-PCBA; decoy properties; LIT-PCBA self-computed shift; screen pose-RMSD labels), it is listed as not computable rather than estimated.