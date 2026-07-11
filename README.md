# Know When to Fold

**Distribution-shift-aware selective prediction for protein–ligand co-folding.**

`foldgate` is a model-agnostic, training-free reliability layer that turns co-folding confidence (AlphaFold3 / Boltz / Chai) into risk-controlled **accept / abstain** decisions with conformal coverage guarantees. It shows that standard conformal guarantees **collapse** under the novel-pocket / novel-chemotype shift central to drug discovery, and **restores** them with shift-robust conformal keyed on training-set similarity.

> Status: early scaffold. The science plan is in [`PLAN.md`](PLAN.md); working conventions and grounded facts in [`CLAUDE.md`](CLAUDE.md); literature and novelty defense in [`RELATED_WORK.md`](RELATED_WORK.md).

## Why

Co-folding models emit confidence (ipTM, PAE, ligand-pLDDT, Boltz-2 affinity), and prior work shows it *correlates* with pose accuracy. But correlation is not a decision rule, and the correlation degrades exactly where drug discovery operates: novel pockets and novel chemotypes. `foldgate` gives you a calibrated gate with a guarantee ("at 50% coverage, accepted poses are 95% correct") and repairs that guarantee under shift.

## What it does

1. Wrap a frozen model's outputs (or reuse released prediction sets like Runs N' Poses).
2. Compute novelty features (ligand Tanimoto-to-train, pocket similarity, temporal), ensemble/cross-model agreement, and PoseBusters validity.
3. Calibrate a conformal gate on held-out data — split (baseline), then weighted + group-conditional (shift-robust).
4. Decide accept/abstain on new predictions and report risk-coverage / AURC and per-novelty-stratum coverage.

Primary correctness label: ligand-RMSD ≤ 2 Å.

## Install (planned)

Environment is managed with [pixi](https://pixi.sh) (conda-forge + bioconda), because the stack mixes compiled comp-bio tools (RDKit, foldseek, US-align, posebusters) with pure-Python ML:

```bash
pixi install          # creates the env from pixi.toml, commits pixi.lock
pixi run test
```

The reliability layer alone is torch-free and pip-installable:

```bash
pip install -e .      # installs the `foldgate` package
```

## Data

Reuse-first. See [`data/DATASETS.md`](data/DATASETS.md) for sources, DOIs, licenses, and what novelty metadata each ships:

- **Runs N' Poses** (primary) — released multi-model predictions + pre-computed training-similarity metadata.
- **FoldBench** — low-homology multi-task benchmark.
- **Mac1 557** — prospective, post-cutoff single-target depth (crystal coords release-delayed).
- **PoseBusters** — physical-validity checks (`pip install posebusters`).
- **PLINDER** — similarity engine to compute missing novelty features.

## Experiments

Figures E1–E6 map to `experiments/e*.py`; see [`experiments/README.md`](experiments/README.md). `make repro` runs the full DVC pipeline.

## Citing

See [`REFERENCES.bib`](REFERENCES.bib). If you use `foldgate`, cite the paper (in prep) and the datasets/methods it builds on.

## License

Code: MIT (see `LICENSE`, to be added). The shippable layer is built to run on permissively-licensed models (Boltz — MIT; Chai-1 — Apache-2.0). AlphaFold3 support is for non-commercial research use only, per AF3's weights terms.
