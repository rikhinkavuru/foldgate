# Experiments (E1–E11)

Each script produces one figure and its backing tables. Inputs come from released prediction sets (see `../data/DATASETS.md`); outputs land in `../results/`. `make experiments` runs the full E1–E11 suite; `make repro` runs download → features → experiments → figures → test end to end.

| Script | Claim | Inputs | Output |
|--------|-------|--------|--------|
| `e1_iid_validity.py` | Native-confidence conformal hits nominal coverage on random splits — the method isn't broken. | RNP predictions + confidence, random cal/test split | coverage vs α, calibration plot |
| `e2_exchangeability_break.py` | **Money figure.** Coverage systematically under-covers on high-novelty pockets/chemotypes. | RNP + novelty strata (ligand Tanimoto, pocket sim, temporal) | coverage vs novelty stratum |
| `e3_shift_repair.py` | Weighted + group-conditional conformal restore coverage; report residual gap. | RNP + novelty features + density-ratio weights | coverage vs stratum, before/after |
| `e4_selective_utility.py` | Conformal gate dominates native-confidence thresholding on AURC. | scores + labels + gates | risk-coverage curves, AURC (with CIs), operating points, per-stratum coverage |
| `e3b_weighted_repair.py` | Weighted conformal (plug-in) repairs coverage under shift without target labels; the finite-sample weighted-LTT gate (WSR betting) is honest-conservative and the concept-shift diagnostic shows why. | RNP + novelty covariates + combined score + cross-fit weights | risk on target: naive vs weighted plug-in vs weighted-LTT vs target-cal; n_eff, concept-shift gap |
| `e3c_combined_conditional.py` | Full method: combined score + group-conditional recovers usable coverage on novel strata, not just abstention. | RNP + combined score + strata | per-stratum risk + coverage |
| `e6_downstream.py` | Abstaining cleans the delivered pose set (higher purity / interface quality) inside RNP; Mac1 screen arm later. | RNP + combined score | purity, enrichment, correct-retained, LDDT-PLI with vs without the gate |
| `e7_shift_axes.py` | The break generalizes across ligand / pocket / temporal axes; it is structural-chemical novelty, not recency. | RNP + 3 stratum columns | per-axis, per-stratum realized risk |
| `e8_interface_task.py` | Task-agnostic: the gate transfers to interface quality (LDDT-PLI ≥ 0.5), not just pose-RMSD. | RNP + combined score | AURC native vs combined on the interface label |
| `e9_continuous_risk.py` | Continuous-RMSD risk beyond the 2 Å convention; certified mean-RMSD gate via a variance-adaptive WSR betting bound (dominates Hoeffding). | RNP + combined score | mean-RMSD risk-coverage + certified continuous gate (WSR vs Hoeffding, Clopper-Pearson coverage) |
| `e10_foldbench.py` | Cross-dataset honest negative: FoldBench ships only ranking_score, so the feature-rich combiner does not transfer. | FoldBench per-pose table | AURC + validity vs raw ranking_score |
| `e15b_foldbench_iptm_transfer.py` | Cross-dataset positive: regenerated FoldBench Protenix to recover interface-ipTM, self-scored ligand-RMSD; frozen RNP interface-ipTM gate transfers (AURC 0.380 vs ranking_score 0.454). | Regenerated FoldBench Protenix (25 poses/target) + deposited assemblies | AURC seen/unseen vs matched ranking_score control |
| `e11_baselines.py` | Baselines a reviewer demands + calibration-vs-conformal: native ipTM / PoseBusters / Platt / isotonic, i.i.d. and under shift; only shift-robust conformal repairs the break. | RNP + combined score + strata | AURC table; gate validity i.i.d. and under shift |
| `e5_ablations.py` | Threshold robustness (1.5–3.0 Å) + cumulative feature ablation. | RNP | per-threshold break + AURC; ablation AURC |
| `make_figures.py` | Multi-model summary figures (E2 all models, E4 AURC bars, E7 axes). | results/*.json | results/figures/*.png |

## Ablations

- Feature importance: novelty vs ensemble disagreement vs native confidence.
- Marginal vs conditional coverage.
- Calibration-set-size sensitivity for rare strata (report where guarantees become vacuous).
- RMSD threshold sweep (not just 2 Å) + continuous-RMSD risk.

## Falsifiable outcome

E2 shows a clear coverage collapse on novel strata; E3 closes most of it; E4 shows AURC dominance; E6 shows a downstream lift. If E2 shows **no** collapse, that is a publishable positive surprise about co-folding calibration — worth running either way.
