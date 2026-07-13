# Know When to Fold

**Distribution-shift-aware selective prediction for protein–ligand co-folding.**

`foldgate` is a model-agnostic, training-free reliability layer that turns co-folding confidence (AlphaFold3 / Boltz / Chai) into risk-controlled **accept / abstain** decisions with conformal coverage guarantees. It shows that standard conformal guarantees **collapse** under the novel-pocket / novel-chemotype shift central to drug discovery, and **restores** them with shift-robust conformal keyed on training-set similarity.

> Status: full study complete on real data (Runs N' Poses, 13,536 delivered poses, 6 models). Findings in [`RESULTS.md`](RESULTS.md); method in [`METHODS.md`](METHODS.md); science plan in [`PLAN.md`](PLAN.md); grounded facts in [`CLAUDE.md`](CLAUDE.md); literature/novelty in [`RELATED_WORK.md`](RELATED_WORK.md); paper draft in [`paper/`](paper/). Reproduce with `make repro` (download → features → E1–E11 → figures → tests).

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

The reliability layer alone is torch-free and pip-installable (uv or pip):

```bash
uv venv --python 3.12 .venv && uv pip install -e .   # or: pip install -e .
```

One-command reproduction (downloads the ~52 MB RNP tabular bundle, no GPU):

```bash
make repro            # download -> features -> E1..E11 -> figures -> tests
```

Or step through it in [`notebooks/quickstart.ipynb`](notebooks/quickstart.ipynb):
calibrate a gate, watch it break under novelty shift, repair it with group-conditional
conformal — in a few CPU seconds.

## Data

Reuse-first. See [`data/DATASETS.md`](data/DATASETS.md) for sources, DOIs, licenses, and what novelty metadata each ships:

- **Runs N' Poses** (primary) — released multi-model predictions + pre-computed training-similarity metadata.
- **FoldBench** — low-homology multi-task benchmark.
- **Mac1 557** — prospective, post-cutoff single-target depth (crystal coords release-delayed).
- **PoseBusters** — physical-validity checks (`pip install posebusters`).
- **PLINDER** — similarity engine to compute missing novelty features.

## Experiments

Twelve experiments (`experiments/e*.py`, see [`experiments/README.md`](experiments/README.md)):
E1 i.i.d. validity, E2 the exchangeability break, E3/E3b/E3c shift-robust repair
(group-conditional, weighted, full method), E4 selective utility (AURC), E5 threshold
robustness + feature ablation, E6 downstream pose-set purity, E7 shift axes
(ligand/pocket/temporal), E8 interface-quality task, E9 continuous-RMSD risk with a
certified (WSR betting) gate, E10 FoldBench cross-dataset honest negative, E11 baselines
(native ipTM / PoseBusters / Platt / isotonic) and calibration-vs-conformal.
`make experiments` runs them all.

**Headline:** the conformal gate is valid i.i.d. but under-controls error 2–3× on
novel ligands (deploy-on-novel: 0.55 error vs a 0.20 target, guarantee holds 0% of
runs); group-conditional + weighted conformal restore it; a combined reliability
score cuts AURC 24–40%, and a structure-based pose-agreement upgrade (W1) pushes it to
38–51% (pose Δ(AURC) CI excludes 0 for all models). Abstention lifts pose-set purity
63–70%→82–94% and, on a non-circular downstream metric, raises crystal-contact recovery
to 0.90 vs 0.72 on accepted vs rejected poses (E6b). Under shift, ordinary calibration
(Platt/isotonic) breaks exactly as naive conformal does — only the shift-robust conformal
repair holds (E11).

New in this release: an importance-weighted Learn-then-Test gate with a finite-sample
guarantee conditional on the weights and a concept-shift diagnostic (E3b), a certified
continuous-RMSD gate via a variance-adaptive WSR betting bound (E9), and a full
baselines suite (E11).

## Citing

See [`REFERENCES.bib`](REFERENCES.bib). If you use `foldgate`, cite the paper (in prep) and the datasets/methods it builds on.

## License

Code: MIT (see `LICENSE`, to be added). The shippable layer is built to run on permissively-licensed models (Boltz — MIT; Chai-1 — Apache-2.0). AlphaFold3 support is for non-commercial research use only, per AF3's weights terms.
