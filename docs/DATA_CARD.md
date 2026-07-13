# Data Card — foldgate

## Summary

foldgate is trained and evaluated **only on released predictions**, never on data it
generates. The single dataset used for every experiment in this repo is the tabular
release of **Runs N' Poses (RNP)**.

## Source dataset: Runs N' Poses (RNP)

| Field | Value |
|-------|-------|
| Provider | plinder-org / RNP |
| DOI | 10.5281/zenodo.14794785 (concept); record 18366081 (tabular artifacts) |
| License | Apache-2.0 |
| Access | `python scripts/download_data.py` (~52 MB tabular, CPU-only, no GPU) |
| Content used | per-prediction confidences, accuracy labels, training-similarity metadata |

### Structure tarballs (W1/W2, optional)

`python scripts/download_data.py --structures` additionally fetches the predicted structures
(`prediction_files.tar.gz`, 39.5 GB) and crystal references (`ground_truth.tar.gz`, 0.41 GB).
These drive the pose-agreement features (W1) and the interaction-recovery downstream metric
(E6b / W2). They are streamed, not extracted (~40 GB free disk suffices), and remain CPU-only.
Layout: `prediction_files/{model}/{system}/seed-*_sample-*.cif` (25 diffusion samples per
model) and `ground_truth/{system}/system.cif` + `ligand_files/{instance}.sdf`.

RNP contains ~2,073 PDB protein-ligand complexes released after 2021-09-30, with
released predictions from six co-folding models (AlphaFold3, Boltz-1, Boltz-1x,
Boltz-2, Chai-1, Protenix). The **training structures themselves are not redistributed**;
what RNP ships (and what foldgate consumes) is *similarity-to-training* metadata, which is
exactly the covariate the shift analysis needs.

## Processed table

`experiments/build_features.py` joins the released CSVs into
`data/processed/rnp_delivered.parquet`: **one row per (system, ligand, method) top-1
delivered pose** (ranked by `ranking_score`), 13,535 rows across five models with enough
labelled poses (Boltz-2 has 933 labelled rows, below the 1,200 threshold, so it is dropped
from per-model results).

Columns consumed:

| Group | Columns | Role |
|-------|---------|------|
| Native confidence | `ranking_score`, `iface_iptm` (chain-pair ipTM) | nonconformity score inputs |
| Ensemble spread | `ens_ranking_std/mean/range`, `ens_iptm_std`, `ens_n_samples` | intra-model disagreement (5 diffusion samples) |
| Physical validity | `pb_valid` (PoseBusters) | filter/feature (absent for Boltz-2) |
| Cross-model | `xmodel_iptm_mean/std`, `xmodel_n_models` | cross-model confidence agreement |
| Ligand difficulty | `ligand_molecular_weight`, `ligand_num_rot_bonds`, `ligand_num_heavy_atoms` | difficulty features |
| Novelty (the shift variable) | `ligand_similarity`, `ligand_novelty`, `novelty_stratum`, `pocket_similarity`, `pocket_novelty_stratum` | stratifier + weight source |
| Temporal | `release_date`, `temporal_stratum`, `target_release_date` | recency axis (E7) |
| Label | `rmsd` (BiSyRMSD), `lddt_pli`, `correct` (= `rmsd <= 2 Å`) | ground truth |

## Label definition

Primary label: **binding-mode correct iff BiSyRMSD ≤ 2 Å** (`correct`). The continuous
`rmsd` supports the threshold-robustness (E5) and continuous-risk (E9) analyses.

## Known limitations / biases

- **Model coverage is uneven** — Boltz-2 lacks PoseBusters results and has fewer labelled
  rows; results are reported per model, never pooled across models with different fields.
- **Novelty strata are imbalanced** — the most-novel ligand stratum (S4) is small
  (~392 poses), so its per-stratum estimates are noisier (reported with CIs).
- **RNP is retrospective** — genuinely prospective evaluation (Mac1 557) is blocked by
  release-delayed crystal coordinates and is deferred to a follow-up.
- **Cross-dataset transfer is a stated negative** — FoldBench ships only `ranking_score`,
  so the feature-rich combiner does not transfer to it (E10, an honest negative).

## What is NOT committed

Raw predictions, structure blobs, and processed parquet are git-ignored. Reproduce with
`scripts/download_data.py` + `make features`.
