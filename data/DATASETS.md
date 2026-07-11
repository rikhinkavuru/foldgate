# Datasets

Reuse-first. Raw data is not committed (`.gitignore`); download into `data/raw/`, write derived tables to `data/processed/`. Track with DVC + a Zenodo mirror for release. Keep the novelty-similarity definition consistent across all sets (see `src/foldgate/features/`).

## Runs N' Poses (RNP) — PRIMARY

- **What:** ~2,073 high-resolution PDB protein-ligand complexes released after 2021-09-30, with released predictions from AlphaFold3, Boltz-1, Boltz-2, Chai-1, Protenix, RoseTTAFold-All-Atom.
- **Ships:** `predictions.tar.gz` (ranking scores, chain-pair ipTM confidence, accuracy labels: LDDT-PLI, BiSyRMSD/ligand-RMSD, pocket F1), `ground_truth.tar.gz`, `msa_files.tar.gz`, `posebusters_results.tar.gz`, `annotations.csv`, `all_similarity_scores.parquet`.
- **Novelty metadata (pre-computed):** Morgan/topological Tanimoto to train, SuCOS pocket similarity + coverage (`qcov`), count of >0.9-Tanimoto train ligands, temporal cutoff bins (30-Sep-2021; separate 1-Jun-2023 bins for Boltz-2). Training set itself is NOT redistributed, but similarity-to-train is.
- **Get it:** GitHub `plinder-org/runs-n-poses`; Zenodo DOI **10.5281/zenodo.14794785**; ML-ready on Polaris Hub.
- **License:** Apache-2.0.
- **VERIFY EARLY:** confirm the prediction dumps expose the raw per-prediction confidence fields (ranking_score / ipTM / chain-pair ipTM / pLDDT) the nonconformity score needs. This is the biggest reuse-plan risk.

## FoldBench

- **What:** 1,522 low-homology all-atom biological assemblies (README lists ~1,819 targets — reconcile before quoting), 558 protein-ligand interfaces. Evaluates AF3, Boltz-1/2, Chai-1, HelixFold3, Protenix, OpenFold3, RoseTTAFold3.
- **Novelty:** low-homology vs PDB pre-2023-01-13; defines an "unseen ligand" subset (<0.50 Tanimoto to train).
- **Get it:** GitHub `BEAM-Labs/FoldBench` (targets CSV in `/targets`; reference CIFs via Google Drive/Zenodo).
- **License:** MIT. Paper: Nature Comms s41467-025-67127-3.
- **Note:** homology cutoff (2023-01-13) differs from each model's true training cutoff; recompute model-specific temporal flags.

## Mac1 557 — flagship prospective shift test

- **What:** 557 SARS-CoV-2 Nsp3 Mac1 macrodomain ligand complexes, genuinely prospective; 3 virtual screens (AmpC, D4, sigma-2). Reported pose accuracy (RMSD<2 Å): AF3 ~70%, Chai-1 ~70%, Boltz-2 52%.
- **Novelty:** model-specific cutoffs (AF3 2021-09-30, Chai-1 2021-12-01, Boltz-2 2023-06-30). Ligand Tanimoto / pocket similarity must be **computed** (not shipped) — use PLINDER tooling keyed to each model's training corpus.
- **Get it:** code at GitHub `jongbin99/Cofolding`. eLife reviewed preprint 110475 / bioRxiv 2025.12.25.696505.
- **CAVEAT:** crystallographic coordinates are **release-delayed** to preserve blind prediction, so RMSD ground-truth labels may not be fully reproducible yet. Track which rows are label-available.

## PoseBusters

- **What:** physical-validity benchmark + `posebusters` package (~30 RDKit checks: bond lengths/angles, planarity, clashes, strain/energy, stereochemistry). Benchmark sets V1=428, V2=308.
- **Use:** compute the PB-valid flag + RMSD≤2 Å success as a secondary label/feature.
- **Get it:** `pip install posebusters` (MIT, v0.6.5). Data Zenodo 8278563. Paper: Chem. Sci. 2024 doi:10.1039/D3SC04185A.
- **Note:** no ligand-Tanimoto-to-train metadata built in.

## PLINDER (similarity engine)

- **What:** 449,383 protein-ligand systems with 500+ annotations and similarity metrics (ECFP4 Tanimoto, pocket Jaccard, PLIP-feature, sequence identity); PL50 split machinery.
- **Use:** compute missing novelty features for any external set (notably Mac1) so all datasets share one similarity definition.
- **Get it:** `pip install plinder`; data in the `gs://plinder` GCS bucket (multi-TB — use pre-computed annotations, don't recompute at scale).

## Suggested split axes (respect the shift)

Build calibration/test partitions along three axes, plus a novelty-stratified test grid so E2 is directly measurable:
1. **Ligand novelty** — max Tanimoto (ECFP4) to nearest train ligand, quantile-binned.
2. **Pocket novelty** — pocket sequence identity + structural TM-score (foldseek/US-align), binned.
3. **Temporal** — post-training-cutoff flag, model-specific.
