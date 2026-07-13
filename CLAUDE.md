# CLAUDE.md — Know When to Fold

Working conventions and **grounded facts** for this repo. Read this before touching code.
`PLAN.md` holds the science (RQs, experiments, novelty). `RELATED_WORK.md` holds the literature and novelty defense. This file holds how to work here and the facts you must not re-derive from memory.

**One-liner:** a model-agnostic, training-free reliability layer that turns protein–ligand co-folding confidence (AF3 / Boltz / Chai) into risk-controlled **accept/abstain** decisions with conformal coverage guarantees, shows standard conformal coverage **collapses under novel-pocket / novel-chemotype shift**, and **repairs it** with shift-robust conformal keyed on training-set similarity. Primary label: binding-mode correctness, ligand-RMSD ≤ 2 Å.

---

## Golden rules

1. **Training-free at inference.** This layer never retrains a co-folding model. It consumes frozen models' confidence outputs. Keep the shippable package (`src/foldgate/`) torch-free.
2. **Reuse-first data.** Consume released predictions (RNP, FoldBench, Mac1) before generating anything. Generating predictions is the only real GPU cost and must be bounded and justified.
3. **Novelty is the conceptual center.** Training-set structural/chemical similarity is both the *stratifier* (group-conditional conformal) and the *weight source* (weighted conformal). Keep its definition auditable and consistent across datasets.
4. **Don't hard-code confidence field names from memory.** They differ per model and some "obvious" ones don't exist (AF3 emits no PDE — see below). Verify against a real output file before parsing.
5. **Report where the guarantee is vacuous** rather than hiding it. Thin strata, weight-estimation error, and concept shift all break guarantees; state assumptions and test them.
6. **Marginal coverage is not enough.** The whole thesis is that marginal validity hides per-group under-coverage. Always report conditional (per-novelty-stratum) coverage.
7. Prose style: no em-dashes, no rule-of-three, no negative parallelism (`humanizer` house style). Code/commits/PRs: write normally.

---

## Repo layout

```
moml/
  CLAUDE.md            this file
  PLAN.md              the science north-star
  RELATED_WORK.md      literature + novelty defense + scoop risks
  REFERENCES.bib       citable bibliography
  pyproject.toml       pip-installable torch-free `foldgate` package
  pixi.toml            env manager (conda-forge + bioconda); commit pixi.lock
  environment.yml      conda fallback
  requirements.txt     pure-pip subset for the analysis package
  Makefile             task runner (setup / features / calibrate / figures / test)
  src/foldgate/
    io/                parse AF3/Boltz/Chai + RNP/FoldBench -> Prediction records
    features/          novelty (Tanimoto, pocket sim, temporal), ensemble/cross-model agreement, PoseBusters
    scores/            nonconformity scores from confidence + derived signals
    conformal/         split / weighted / Mondrian(group-conditional) / RCPS / LTT
    selective/         accept/abstain gate, risk-coverage, AURC
    eval/              E1-E6 drivers + figures
  data/{raw,processed,external}/   NOT committed (see data/DATASETS.md)
  configs/             YAML configs
  experiments/         figure-producing scripts (E1-E6)
  results/figures/     committed small figures only
  notebooks/           exploration only, not pipeline
  scripts/             one-off utilities, data fetch
  tests/
```

Import package = `foldgate`; PyPI distribution name = `foldgate`.

---

## Environment & tooling

The stack splits into two worlds that must stay apart:
- **Compiled comp-bio tools** (RDKit, foldseek, US-align, biotite, posebusters) live in conda-forge/bioconda.
- **Pure-Python ML/conformal** (numpy, pandas, scikit-learn, crepes, crepes-weighted, mapie, matplotlib, lightgbm) can come from PyPI but are safest from conda-forge too.

**Live environment: `uv` + `.venv` (Python 3.12), and `uv.lock` is committed.** The shippable package is torch-free (numpy/pandas/scikit-learn/scipy only), so the analysis env needs no compiled comp-bio stack. Run everything with `.venv/bin/python`. `pixi.toml` / `environment.yml` remain as conda fallbacks for the optional structure-feature pipeline (RDKit, gemmi, spyrmsd), but no `pixi.lock` exists and pixi is not the primary env.

**Segfault gotcha (load-bearing):** lightgbm + torch in one env can segfault on macOS from two OpenMP runtimes (LLVM `libomp` vs Intel `libiomp`). Fixes: install everything from conda-forge (patched OpenMP), **or** isolate any torch co-folding inference in a separate env/process from the lightgbm+conformal analysis. `KMP_DUPLICATE_LIB_OK=TRUE` is NOT a safe workaround (can crash or silently corrupt results). Because the reliability layer is training-free, the default is: no torch in the analysis env at all.

**Local machine note:** system `python3` here is homebrew 3.14 (too new for some wheels). Use the project env at `.venv/bin/python` (created by `uv venv --python 3.12`); do not rely on system python.

---

## Grounded facts (verified mid-2026 — cite `REFERENCES.bib`)

### Co-folding models: confidence outputs, licenses

| Model | License (code / weights) | Confidence output (where) | Key fields | Notes |
|-------|--------------------------|---------------------------|------------|-------|
| **AlphaFold3** | Apache-2.0 / **weights request-gated, non-commercial** | mmCIF (pLDDT in B-factor) + summary & full confidence **JSON** | `ptm`, `iptm`, `ranking_score`, `chain_pair_iptm`, `chain_ptm`, `chain_iptm`, `pae` (token×token), `atom_plddts` (0-100), `contact_probs`, `has_clash` | **No PDE field** (PDE is internal-only). Ligand = own chain → per-ligand confidence via chain_pair_iptm row + ligand-atom pLDDT. `ranking_score = 0.8·ipTM + 0.2·pTM + 0.5·frac_disordered − 100·has_clash`. Ligand pLDDT uses only ligand↔polymer distances. |
| **Boltz-1 / Boltz-2** | **MIT / MIT** (code+weights) | `confidence_*.json` + `pae/plddt/pde_*.npz` | `confidence_score`, `ptm`, `iptm`, `ligand_iptm`, `protein_iptm`, `complex_plddt`, `complex_iplddt`, **`complex_pde`, `complex_ipde`**, `chains_ptm`, `pair_chains_iptm` | Per-atom PAE/pLDDT in **NPZ**, not JSON. **Boltz-2 affinity** (`affinity_*.json`): `affinity_pred_value` (log10 IC50) + `affinity_probability_binary` (binder prob, 0-1), each with per-member `…1`/`…2`. The two heads differ (hit-discovery vs potency); do not conflate. |
| **Chai-1** | **Apache-2.0 / Apache-2.0** (code+weights) | `pred.model_idx_*.cif` + `scores.model_idx_*.npz` + ranking JSON | `aggregate_score`, `ptm`, `iptm`, `per_chain_ptm`, `per_chain_pair_iptm`, clash flags; per-token `pae/pde/plddt` logits in NPZ | MSA-free/single-sequence mode removes the MSA-search cost. `aggregate_score` composition undocumented — read from source before using as a principled feature. |

All three default to **5 diffusion samples** → intra-model disagreement is free. **Cross-model agreement** (pairwise ligand-RMSD across AF3/Boltz/Chai) is a third, model-agnostic signal.

**Licensing fork (decides how the tool ships):** build the shippable reliability layer on **Boltz (MIT) + Chai (Apache-2.0)**; treat **AF3 as a non-commercial research comparator only** (weights request-gated, outputs non-commercial). Normalize the differing score fields (`ranking_score` vs `confidence_score` vs `aggregate_score`) before pooling into one nonconformity score. Raw ipTM is known-miscalibrated for some interfaces (basis for ipSAE) — a weak feature without correction.

### Datasets (reuse-first)

| Dataset | What / size | Novelty metadata | Access / license | Role |
|---------|-------------|------------------|------------------|------|
| **Runs N' Poses (RNP)** | **PRIMARY.** ~2,073 post-2021-09-30 PDB complexes; released predictions for AF3, Boltz-1, Boltz-2, Chai-1, Protenix, RoseTTAFold-All-Atom; ships `predictions.tar.gz` (ranking scores, chain-pair ipTM, LDDT-PLI, BiSyRMSD/ligand-RMSD, pocket F1), `ground_truth`, `msa_files`, `posebusters_results` | **Pre-computed**: Morgan/topological Tanimoto to train, SuCOS pocket similarity + `qcov`, count of >0.9-Tanimoto train ligands, temporal cutoff (30-Sep-2021; separate 1-Jun-2023 bins for Boltz-2) in `annotations.csv` + `all_similarity_scores.parquet` | github.com/plinder-org/runs-n-poses; Zenodo **10.5281/zenodo.14794785**; Polaris. **Apache-2.0** | Calibration + eval; near-zero GPU. Train set NOT redistributed but similarity-to-train IS. |
| **FoldBench** | 1,522 low-homology assemblies (README lists ~1,819 targets — reconcile), 558 protein-ligand; evaluates AF3, Boltz-1/2, Chai-1, HelixFold3, Protenix, OpenFold3, RoseTTAFold3 | Low-homology vs PDB pre-2023-01-13; "unseen ligand" subset < 0.50 Tanimoto | github.com/BEAM-Labs/FoldBench (CIFs via Google Drive/Zenodo). **MIT**. Nature Comms s41467-025-67127-3 | Cross-dataset generalization check |
| **Mac1 557** | 557 SARS-CoV-2 Nsp3 Mac1 macrodomain complexes, genuinely prospective; 3 virtual screens (AmpC, D4, σ2). Pose accuracy AF3 ~70% / Chai ~70% / Boltz-2 52% | Model-specific cutoffs: AF3 2021-09-30, Chai 2021-12-01, Boltz-2 2023-06-30. Ligand Tanimoto / pocket sim must be **computed** (not shipped) | github.com/jongbin99/Cofolding; eLife reviewed preprint 110475 / bioRxiv 2025.12.25.696505 | Flagship prospective shift test. **Crystal coords release-delayed → RMSD labels not fully reproducible yet.** |
| **PoseBusters** | V1=428, V2=308 complexes; `posebusters` pkg = ~30 RDKit checks | none built-in (no Tanimoto-to-train) | `pip install posebusters`, MIT, v0.6.5 | PB-valid flag + RMSD≤2Å labels |
| **PLINDER** | 449,383 systems; similarity engine (ECFP4 Tanimoto, pocket Jaccard, PLIP, seq identity), PL50 split | full | `pip install plinder`; GCS bucket (multi-TB) | Compute missing novelty features (esp. Mac1); use pre-computed annotations, don't recompute at scale |

**Early verification TODO:** confirm the RNP/FoldBench prediction dumps expose the **raw per-prediction confidence** (ranking_score / ipTM / chain-pair ipTM / pLDDT) that the nonconformity score needs — this is the single biggest risk to the reuse-first plan. (RNP verified to ship ranking scores + chain-pair ipTM; confirm the exact columns.)

### Conformal / selective-prediction methods + libraries

- **Split (inductive) conformal** — guarantee `1−α ≤ P(Y∈C(X)) ≤ 1−α + 1/(n+1)`, **marginal**, needs only exchangeability. Baseline (E1).
- **Covariate shift** (`P_test(X) ≠ P_train(X)`, `P(Y|X)` stable) breaks exchangeability → under-covers on the novel region that matters. This is E2.
- **Weighted conformal** (Tibshirani, Barber, Candès, Ramdas 2019, arXiv:1904.06019) — likelihood-ratio weights `w = dP_test/dP_train` + **point mass at +∞** (dropping it breaks validity). Estimate weights via a probabilistic train-vs-test classifier `w ∝ c/(1−c)` or KLIEP/uLSIF. E3.
- **Mondrian / group-conditional** (arXiv:2306.09335) — per-stratum quantile → per-group coverage. Needs only stratum labels, not weights. E3 fallback.
- **RCPS** (Bates et al., JACM 2021, arXiv:2101.02703) and **Learn-then-Test** (arXiv:2110.01052) — control the RMSD-threshold risk directly, `P(risk ≤ α) ≥ 1−δ`. RCPS needs monotone risk; LTT handles non-monotone/multi-risk.
- **Selective prediction** (Geifman & El-Yaniv, NeurIPS 2017) — gate `(f,g)`, coverage, selective risk, risk-coverage curve, **AURC**. Native-confidence thresholding = the baseline to beat (RQ4). AURC has known estimator pitfalls (arXiv:2407.01032, 2410.15361) — report with CIs.

**Library map (corrected — this matters):**

| Need | Use | Note |
|------|-----|------|
| Mondrian / group-conditional | **crepes** (v0.9.1, June 2026, BSD-3, `pip install crepes`) | actively maintained; `class_cond=True` / `MondrianCategorizer` |
| **Weighted CP under covariate shift** | **crepes-weighted** (predict-idlab, v0.1.3, `pip install crepes-weighted`) | **early-stage (0.1.x)** — validate it; budget a thin wrapper |
| RCPS / LTT / CRC risk control | **MAPIE** (v1.4.1) | **MAPIE does NOT do weighted covariate-shift CP** — its `sample_weight` leaves conformity scores uniformly weighted. Do not cite MAPIE for weighted CP. |
| Reference weighted-CP (alt) | TorchCP `WeightedPredictor` | PyTorch/GPU-heavier; alternative to crepes-weighted |
| RCPS reference code | github.com/aangelopoulos/rcps | Hoeffding-Bentkus UCB |

The project needs to **fuse weighted + group-conditional** conformal keyed on similarity, so expect a thin `src/foldgate/conformal/` wrapper over crepes + crepes-weighted, with the +∞ mass verified.

---

## The method in one screen

1. `io` → parse released predictions into `Prediction` records (native confidences per model).
2. `features` → compute novelty (max ligand Tanimoto, pocket seq-identity + TM-score via foldseek/US-align, temporal flag), intra-model ensemble disagreement, cross-model agreement, PoseBusters PB-valid + strain.
3. `scores` → nonconformity score (native confidence, optionally combined via a calibration-only combiner).
4. `conformal` → **baseline** split conformal (E1) → show it under-covers on high-novelty strata (E2) → **repair** with weighted + group-conditional keyed on similarity (E3).
5. `selective` → accept/abstain gate, risk-coverage curve, AURC vs native-confidence threshold (E4).
6. `eval` → generality across AF3/Boltz/Chai & tasks (E5); downstream screening-enrichment / FEP-starting-structure lift (E6).

Open method question to resolve early: is the shift **pure covariate shift** (weighted CP applies) or does confidence reliability itself degrade on novel chemotypes (**concept shift** — weighted CP won't fix; lean group-conditional / RCPS)? Test this, don't assume.

---

## Experiments → files

| Fig | Claim | Script | Module |
|-----|-------|--------|--------|
| E1 | i.i.d. conformal hits nominal coverage | `experiments/e1_iid_validity.py` | `eval.e1_iid_validity` |
| E2 | coverage collapses on novel strata (money figure) | `experiments/e2_exchangeability_break.py` | `eval.e2_exchangeability_break` |
| E3 | weighted + group-conditional repair coverage | `experiments/e3_shift_repair.py` | `eval.e3_shift_repair` |
| E4 | conformal gate beats native threshold on AURC | `experiments/e4_selective_utility.py` | `eval.e4_selective_utility` |
| E5 | holds across AF3/Boltz/Chai & tasks | `experiments/e5_generality.py` | `eval.e5_generality` |
| E6 | abstaining lifts screening enrichment / FEP quality | `experiments/e6_downstream.py` | `eval.e6_downstream` |

---

## Data handling & reproducibility

- **Never commit** `data/{raw,processed,external}`, model weights, or large intermediates (`.gitignore` enforces this).
- Version data + released artifacts with **DVC** (`dvc.yaml` stages: features → novelty → posebusters-labels → calibrate → evaluate); mirror released artifacts to a **Zenodo DOI** for citation.
- **Mac1 caveat:** crystal coords are release-delayed; RMSD labels may not be reproducible yet. Track which Mac1 rows are label-available.
- **Seeds:** one global RNG helper (numpy `Generator` + `PYTHONHASHSEED`) imported everywhere. Log resolved config + git SHA + `pixi.lock` hash into each run's output dir.
- `make repro` should run `dvc repro` end to end so a reviewer reproduces every figure with one command.

---

## Publication timeline (from 2026-07-10)

- **NeurIPS 2026 main track is CLOSED** (was May 6, 2026). A main-conference selective-prediction framing must target NeurIPS 2027 / ICLR 2027 — do not block on it.
- **Mid-Aug 2026:** preprint to arXiv (**q-bio.BM** primary + cs.LG + stat.ML) and bioRxiv.
- **Sept 1, 2026:** MoML 2026 short paper (2-4 pp, non-archival, MIT Oct 14).
- **~late Sept 2026:** MLSB @ NeurIPS 2026 extended abstract (non-archival; CFP not yet posted — watch mlsb.io / @workshopmlsb).
- **~early Oct 2026:** full method + tool paper to **Digital Discovery** (RSC; ~45-day first decision, gold OA, APC ~£2100) → Dec-Jan acceptance. Backup: **J. Cheminformatics**. PLOS Comp Bio only if a biology-methods narrative is preferred (slower).

Workshops are non-archival → they don't burn journal novelty; the journal paper is still the archival record.

---

## Scoop watch (re-sweep near submission)

**Re-swept 2026-07-12: our cell is STILL OPEN** (training-free + guaranteed conformal accept/abstain + similarity-keyed shift-robust + co-folding pose-correctness label + virtual-screening abstention decision). No 2025-26 paper satisfies all five. Closest prior art = **CoDrug** (conformal for scalar molecular property under shift, KDE weights — scalar, not structured pose). Nearest threats, all confirmed OUT of cell: **Chem Sci D5SC06481C** (Hou group, Dec 2025 — uses AF3/Boltz/Protenix ipTM as a VS ranking signal and documents novelty degradation, but NO conformal guarantee and NO abstention; this is our baseline-to-beat and the E16/E20 data source); **GESPI** (arXiv:2509.20345 — general synthetic-data conformal risk control with an AlphaFold residue demo, not our label); **Confidence Gate Theorem** (arXiv:2603.09947 — generic ranked-abstention theory); adjacent conformal-selection methods **CONFIDE** (2512.02033), **SCoRE** (2603.24704), **ConfHit** (2603.07371). Re-sweep near submission; watch the Tingjun Hou group (AF3-VS) and Ying Jin group (SCoRE/ConfHit).
