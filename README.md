# Know When to Fold

**A training-free reliability layer that turns protein-ligand co-folding confidence into risk-controlled accept/abstain decisions, and a theorem for when that guarantee can hold under distribution shift.**

`foldgate` wraps a frozen co-folding model (AlphaFold3, Boltz, Chai, Protenix) and converts its confidence into a calibrated gate: accept a predicted pose or abstain, with a conformal coverage guarantee such as "among accepted poses, 95% are correct." The correctness label is binding-mode accuracy, ligand-RMSD within 2 Angstrom.

The layer never retrains a co-folding model. It consumes frozen models' outputs, so it runs on released prediction sets with no GPU.

## The result

Three findings, in order.

1. **An impossibility.** Any label-free certificate built from covariate-measurable weights recovers only the source conditional, so on a shifted test set it silently under-reports realized risk by a concept-gap term that no reweighting can detect. The construction and its matching achievability floor are in [`docs/theory/THEOREM_RECONCILED.md`](docs/theory/THEOREM_RECONCILED.md), and the theorem generalizes beyond co-folding to standard tabular shift benchmarks (`experiments/b1`-`b7`).

2. **The break, on real data.** A conformal gate calibrated i.i.d. hits its nominal coverage, then under-controls error by 2 to 3 times on novel ligands. Deployed on the novel-chemotype stratum it realizes 0.55 error against a 0.20 target, and the guarantee holds in 0% of runs. Ordinary calibration (Platt, isotonic) breaks the same way.

3. **The repair.** Weighted and group-conditional conformal keyed on training-set similarity restore per-stratum coverage. A combined reliability score cuts AURC by 24 to 40%, and a structure-based pose-agreement upgrade pushes that to 38 to 51%. Abstention lifts pose-set purity from 63-70% to 82-94% and raises crystal-contact recovery to 0.90 on accepted versus 0.72 on rejected poses.

Full numbers with confidence intervals are in [`RESULTS.md`](RESULTS.md).

## Install

The environment is `uv` + a Python 3.12 virtualenv, and the lockfile is committed. The shippable layer is torch-free (numpy, pandas, scikit-learn, scipy only).

```bash
uv venv --python 3.12 .venv
uv pip install -e .
```

`pixi.toml` and `environment.yml` remain as conda fallbacks for the optional structure-feature pipeline (RDKit, gemmi, spyrmsd). They are not the primary environment.

## Reproduce

One command, no GPU. It downloads the Runs N' Poses tabular bundle, builds features, runs the experiment suite, and rebuilds the figures.

```bash
make repro          # download -> features -> experiments -> tests
```

Individual stages are `make download`, `make features`, `make experiments`, `make figures`, and `make test`. Run `make help` for the full list.

## Repository map

```
src/foldgate/
  io/          parse AF3/Boltz/Chai/Protenix + RNP/FoldBench into Prediction records
  features/    novelty (Tanimoto, pocket similarity, temporal), ensemble + cross-model agreement, PoseBusters
  scores/      nonconformity scores from confidence and derived signals
  conformal/   split, weighted, group-conditional (Mondrian), RCPS, Learn-then-Test
  selective/   accept/abstain gate, risk-coverage, AURC
  eval/        experiment drivers and figures
  bench/       the general shift benchmark behind the theorem
experiments/   E1-E24 co-folding studies + b1-b7 benchmark (see experiments/README.md)
docs/theory/   the theorem, its reconciliation, and the D1/D2 research plan
paper/         MoML 2026 short paper (tex, pdf) + the extended draft
data/          reuse-first datasets, not committed (see data/DATASETS.md)
```

The experiment arc: E1 i.i.d. validity, E2 the exchangeability break, E3 the shift repair, E4 selective utility, E5 through E14 robustness and shift-axis studies, E15/E15b the FoldBench cross-dataset transfer, E16 through E24 screening honesty and robust certificates. See [`experiments/README.md`](experiments/README.md) for the full table and [`METHODS.md`](METHODS.md) for the method.

## Data

Reuse-first. Sources, DOIs, licenses, and the novelty metadata each ships are in [`data/DATASETS.md`](data/DATASETS.md).

- **Runs N' Poses** (primary) released multi-model predictions plus pre-computed training-similarity metadata. Near-zero GPU.
- **FoldBench** low-homology multi-task benchmark. We regenerated its Protenix predictions to recover the interface-ipTM field the public table withholds, giving a feature-matched cross-dataset positive (E15b).
- **Mac1 557** prospective, post-cutoff single-target depth. Crystal coordinates are under release embargo as of July 2026, so it is deferred.
- **PoseBusters** physical-validity checks.
- **PLINDER** similarity engine for computing missing novelty features.

## Paper and citing

The MoML 2026 short paper is in [`paper/`](paper/); an extended journal version is in progress. Bibliography is [`REFERENCES.bib`](REFERENCES.bib). The novelty defense and literature map are in [`RELATED_WORK.md`](RELATED_WORK.md).

## License

Code is MIT. The shippable layer runs on permissively-licensed models (Boltz is MIT, Chai-1 is Apache-2.0). AlphaFold3 support is for non-commercial research use only, per AF3's weights terms.
