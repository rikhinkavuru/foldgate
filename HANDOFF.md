# foldgate — Execution Handoff

**Purpose.** Everything needed to continue *Know When to Fold* / `foldgate` from its current
state (a complete, reviewed reuse-first study with a drafted MoML short paper, plus finished
W1–W6 and five CP-precision fixes) to a submission-ready paper + released tool. Written so a
competent ML-for-science engineer (or a future agent) can execute each remaining workstream
without re-deriving context.

Repo: **github.com/rikhinkavuru/foldgate** · local: `/Users/rikhinkavuru/moml` ·
Env: **uv + `.venv` (py3.12)** — `.venv/bin/python` is 3.12; system `python3` is homebrew 3.14
(too new for some wheels), do not use it. Version **0.1.0**.

---

## 0. TL;DR — status at a glance

| Workstream | State | Acceptance evidence |
|---|---|---|
| Core study E1–E11 + E6b | **done** | 14 experiment scripts, 14 `results/*.json`, 7 figures |
| W3 rigorous weighted conformal | **done** | `weighted_ltt_threshold` (WSR betting), cross-fit calibrated weights, concept-shift diagnostic; E3b |
| W4 tight certified continuous gate | **done** | WSR betting bound; E9 certifies AF3 43% coverage at 1 Å where Hoeffding gives 0% |
| W5 baselines + calibration-vs-conformal | **done** | E11: ipTM/PoseBusters/Platt/isotonic; calibration breaks under shift like naive conformal |
| W6 release engineering | **done** | CI, pre-commit, `uv.lock`, download script, notebook, CITATION/`.zenodo.json`, cards |
| W1 pose-agreement features | **done** | E4 pose ΔAURC CI excludes 0 for all 5 models (AURC → 38–51%) |
| W2 interaction-recovery (E6b) | **done** | accepted 0.90 vs rejected 0.72 crystal-contact recall, gap CI excludes 0 for all 5 |
| W2 screening enrichment (headline #) | **open (PI)** | harness built + tested; needs a benchmark decision + data |
| 5 CP-precision reviewer fixes | **done** | binomial-conservative wording, LTT ranks, E1 CI + demonstrated proxy-bias, E3 marginal, training-free clarified |
| W7 submission | **open (PI)** | authorship, accounts, venue priority |
| W8 full-paper expansion | **open** | fold W1/W2 into a Digital Discovery paper |

**Tests: 23 green** (test_conformal 15, test_enrichment 4, test_pose_agreement 3, test_import 1).
**Ruff clean.** `.venv/bin/ruff check src/ experiments/ tests/ scripts/` passes.

---

## 1. One-page context

**Thesis.** Co-folding models report confidence that correlates with pose accuracy, but a
correlation is not a decision rule, and it is weakest on the novel pockets / chemotypes drug
discovery faces. `foldgate` turns confidence into a **risk-controlled accept/abstain** decision
with a **finite-sample conformal guarantee**, shows the guarantee **breaks under novelty shift**,
and **repairs** it with shift-robust conformal keyed on training-set similarity.

**Data.** Released Runs N' Poses (RNP): **13,535** delivered poses (top-1 by `ranking_score` per
system/ligand/model), 6 co-folding models. Ships `ranking_score` + interface chain-pair ipTM +
BiSyRMSD + LDDT-PLI, PoseBusters results, and pre-computed training-similarity. Label: **correct
iff BiSyRMSD ≤ 2 Å**. Zero GPU; the tabular study runs on CPU from a 52 MB download. The W1/W2
pose features additionally stream the 39.5 GB structure tarball (CPU-only, no GPU).

---

## 2. Repo map

```
moml/
  CLAUDE.md          working conventions + grounded facts  (⚠ stale: still mandates pixi; repo uses uv)
  PLAN.md            the science north-star (RQs, experiments)
  RELATED_WORK.md    literature + novelty defense + scoop risks
  RESULTS.md         all results, real numbers (E1–E11 + W1 + E6b)
  docs/METHODS.md    exact statistical constructions (kept in sync with code)
  docs/{DATA_CARD,MODEL_CARD}.md
  paper/moml2026_shortpaper.{md,pdf,html}   MoML short paper + build_pdf.py
  pyproject.toml (0.1.0, hatchling) · uv.lock (committed) · requirements.txt
  pixi.toml / environment.yml   conda fallbacks (⚠ NOT the live env; no pixi.lock exists)
  Makefile · .github/workflows/ci.yml · .pre-commit-config.yaml
  scripts/download_data.py   RNP tabular (+ --structures for the 39.5 GB tarballs)
  notebooks/quickstart.ipynb   calibrate → break → repair, CPU seconds
  src/foldgate/  io/ features/ scores/ conformal/ selective/ eval/
  experiments/   e1..e11, e3b, e3c, e6b, build_features.py, build_pose_features.py, make_figures.py
  tests/         test_conformal.py test_enrichment.py test_pose_agreement.py test_import.py
  data/{raw,processed}/   NOT committed; rebuild via download_data.py + make features
  results/       *.json (committed) + figures/*.png (committed)
```

Import package = `foldgate`. **Torch-free**: only numpy/scipy/pandas/scikit-learn are used by the
tabular layer; `gemmi`/`spyrmsd` are imported lazily inside the pose/interaction functions, so
`import foldgate` needs no compiled comp-bio libs.

---

## 3. The method (one screen) + public API

- **`conformal/risk.py`** — distribution-free selective risk control.
  - `ltt_threshold(scores, correct, alpha, delta=0.1, min_accept=20)` — Learn-then-Test with
    fixed-sequence testing. Tests H0(τ): P(error|score≥τ) ≥ α with the exact binomial p-value,
    grows the accept set along a data-independent coverage grid, stops at first failure.
    **P(selective risk ≤ α) ≥ 1 − δ**, finite-sample, distribution-free. `rcps_threshold` = alias.
  - `continuous_risk_threshold(scores, loss, target, bound="wsr")` — certifies mean [0,1] loss
    ≤ target. bounds: `wsr` (default, tightest), `bernstein`, `hoeffding`, `binomial` (reproduces
    the binary LTT gate — the degenerate check).
  - `wsr_betting_pvalue(losses, target, delta)` — Waudby-Smith & Ramdas (JRSSB 2024) betting
    p-value, variance-adaptive; valid by Ville's inequality. Vectorized.
  - `hb_upper_bound`, `naive_threshold` (practitioner baseline, no correction).
- **`conformal/weighted.py`** — covariate-shift CP.
  - `estimate_weights_cv(...)` — cross-fitted, probability-calibrated density-ratio weights.
  - `weighted_ltt_threshold(...)` — importance-weighted LTT (WSR betting on rescaled weighted
    losses); **exact finite-sample conditional on correct weights** (Almeida et al. 2025).
  - `weighted_threshold` (plug-in Hájek), `effective_n` (Kish), `concept_shift_diagnostic`.
- **`scores/combiner.py`** — `ScoreCombiner` (calibration-only HistGradientBoosting → P(correct)).
  `DEFAULT_FEATURES` = 12 tabular signals (the cheap primary). `POSE_FEATURES` = 6 structure-based
  (W1 opt-in upgrade). Novelty deliberately excluded (it enters via calibration, not the score).
- **`selective/metrics.py`** — `risk_coverage_curve`, `aurc`, `evaluate_gate`,
  `conditional_coverage` (per-stratum), `bootstrap_ci`, `clopper_pearson`.
- **`selective/enrichment.py`** (W2) — `enrichment_factor` (EF@k%), `bedroc`, `roc_auc`, `log_auc`,
  `selective_enrichment_curve`, `active_retention_curve`, `random_abstention_ef`.
- **`features/`** — `novelty` (the shift variable), `agreement` (cross-model *confidence*),
  `pose` (ensemble spread + PoseBusters), `pose_agreement` (W1 structural: `parse_pose`,
  `select_ligand`, `intra_model_pose_features`, `cross_model_pose_features`),
  `interactions` (E6b: `contact_fingerprint`, `ifp_metrics`, `load_true_contacts`).
- **`io/rnp.py`** — `load_rnp` → delivered top-1 table; `RNP_METHODS` maps method key ↔ CSV/dir
  basename (**parquet `boltz1` ↔ tarball dir `boltz`** — load-bearing for W1).

**⚠ Group-conditional / Mondrian has no package function.** E3/E3c and E11's `combined_groupcond`
implement it **inline in the experiment scripts** by looping `ltt_threshold` per novelty stratum
(needs ≥40 cal points/stratum). The `conformal/__init__.py` docstring still says Mondrian/weighted
are "Next" via crepes — that is stale; weighted.py already ships them in-house, and crepes /
crepes-weighted are installed but **unused** for the guarantee.

**Guarantees table.**

| Setting | Method | Guarantee |
|---|---|---|
| i.i.d. | LTT (binomial) | P(risk ≤ α) ≥ 1 − δ, finite-sample, distribution-free |
| Novelty shift, labelled strata | group-conditional (inline Mondrian) | per-stratum **marginal** P(risk ≤ α) ≥ 1 − δ (δ/K for simultaneous) |
| Novelty shift, unlabelled target | weighted LTT (WSR) | P(R_target ≤ α) ≥ 1 − δ, **conditional on correct weights** |
| Continuous RMSD | WSR betting bound | P(mean bounded-RMSD ≤ target) ≥ 1 − δ |

---

## 4. Experiments → files → current numbers

α = 0.20, δ = 0.10; 5 analyzed methods (af3, boltz1, boltz1x, chai, protenix; boltz2 has 933
labelled rows, below `MIN_METHOD_N`=1200, so excluded from per-model results).

| Fig | Claim | Current headline number |
|-----|-------|-------------------------|
| E1 `e1_iid_validity` | valid i.i.d. | split-holding fraction w/ CP CI reaches 0.90 for powered models (boltz1x 0.91 [0.88,0.93]); rigorous validity = synthetic test |
| E2 `e2_exchangeability_break` | **the break** | AF3 marginal 0.177 hides S3 0.38/S4 0.43; deploy-on-novel realized **0.55**, guarantee holds **0%** (all models 0.46–0.61) |
| E3 `e3_shift_repair` | group-conditional repair | per-stratum risk ≤ α (native score abstains on hard strata) |
| E3c `e3c_combined_conditional` | full method | combined + group-cond recovers usable coverage (AF3 S1 0.52, S2 0.39) at risk ≤ α |
| E4 `e4_selective_utility` | combined ≫ native | AURC native→combined **33/28/25/40/38%**; **+pose (W1): 42/41/38/47/51%**, pose ΔAURC CI excludes 0 all 5 |
| E3b `e3b_weighted_repair` | weighted repair (label-free) | plug-in pulls risk toward α; weighted-LTT abstains (0% — honest, see §9); concept-shift gap 0.08→0.29 |
| E5 `e5_ablations` | robustness + ablation | threshold sweep 31–34%; ablation native 0.193 → tabular 0.123 → **+pose 0.106** (biggest jump after ipTM) |
| E6 `e6_downstream` | pose-set purity | base 63–70% → 82% (α=.2) / 91–94% (α=.1). **Circular by construction (=1−risk); cite E6b instead** |
| E6b `e6b_interaction_recovery` | **non-circular downstream** | crystal-contact recall accepted **0.90** vs rejected **0.72**, gap CI excludes 0 all 5 |
| E7 `e7_shift_axes` | structural not temporal | break strong on ligand/pocket novelty, weak on temporal |
| E8 `e8_interface_task` | task-agnostic | interface-quality (LDDT-PLI) AURC AF3 0.085→0.048 (44–55% across models) |
| E9 `e9_continuous_risk` | certified continuous gate | WSR certifies AF3 **43%** coverage at 1 Å target where Hoeffding gives **0%** |
| E10 `e10_foldbench` | honest negative | FoldBench ships only ranking_score → combiner does not transfer (AURC −23% to +9%) |
| E11 `e11_baselines` | calibration ≠ conformal | under shift, Platt/isotonic/naive-conformal all break; only group-conditional repairs (risk ≤ α) |

`build_features.py` → `data/processed/rnp_delivered.parquet` (13,535 rows; conditionally joins
`rnp_pose_features.parquet` when present). `make_figures.py` → the 3 summary figures.

---

## 5. W1 / W2 — the pose-feature pipeline (structures)

**Already built. Do NOT regenerate unless you changed the feature code.** `rnp_pose_features.parquet`
(11,711 rows, 93–97% coverage) and `rnp_delivered.parquet` are on disk.

- **`experiments/build_pose_features.py`** streams the 39.5 GB `prediction_files.tar.gz` ONCE (no
  disk extraction; ~2–3 h single-core) and computes, with spyrmsd + gemmi:
  - **intra-model pose diversity** across the 25 diffusion samples (pocket-superposed,
    symmetry-corrected ligand-RMSD to the delivered pose): `intra_model_pose_std/median`,
    `pose_consensus_frac`. Median std: AF3 0.61, Boltz-1 0.47, Boltz-1x 0.28, Chai-1 0.41,
    **Protenix 1.90 Å** (its samples disagree most).
  - **cross-model pose agreement**: `xmodel_pose_rmsd_median/min`, `pose_consensus_cluster_size`.
    Median: AF3 2.93, Boltz-1 2.22, Boltz-1x 2.09, **Chai-1 5.23** (structural outlier), Protenix 3.39.
  - **interaction-fingerprint recovery** (E6b) vs `ground_truth.tar.gz`: `ifp_recall/jaccard`.
- **Five real bugs the driver had to solve** (all fixed, unit-tested): ligand atom *names* differ
  across models → cross-model uses spyrmsd graph-isomorphism (no name matching); tarball dirs ≠
  parquet methods (`boltz`→`boltz1`) → `DIR_TO_METHOD`; Protenix emits explicit H → heavy-atom-only
  ligand selection; receptor chain names differ (`A` vs `A0`) → pocket keyed on chain *ordinal*;
  homodimer symmetry inflates cross-model RMSD → `XMODEL_CAP=10 Å`.
- **Reap-resilient run mechanism** (the harness reaps long background *tasks* ~hourly): the driver
  **checkpoints every 400 groups** (`_pose_intra_ckpt.parquet` + `_pose_cache_ckpt.pkl`, auto-deleted
  on completion) and resumes; an **flock lock** (`_pose_driver.lock`) prevents a second driver from
  racing the checkpoint; launch **orphaned** (`nohup … & disown`, not via a tracked background task)
  so it survives reaps. `ANALYZED_DIRS` filters to the 5 analyzed models (~30% less compute).
  **`--limit N` disables the lock AND checkpointing — never treat a `--limit` parquet as real.**

Rebuild (only if needed): `python scripts/download_data.py --structures` then `make pose-features`
(runs the driver + re-joins into the delivered table). `make experiments` re-runs everything.

---

## 6. Reproduce from scratch

```bash
cd ~/moml
make setup                       # uv venv --python 3.12 .venv + deps (torch-free)
python scripts/download_data.py  # RNP tabular (52 MB) into data/raw
make features                    # -> data/processed/rnp_delivered.parquet (13,535 rows)
make experiments                 # E1..E11 + E6b -> results/*.json + figures
make paper                       # -> paper/moml2026_shortpaper.pdf
make test                        # 23 tests
# optional W1/W2 (heavy):
python scripts/download_data.py --structures   # + 39.5 GB prediction_files + ground_truth
make pose-features               # -> rnp_pose_features.parquet, re-joins into delivered
```

Env facts: `crepes`/`crepes-weighted` are installed but the primary certifier is the in-house LTT
(exact binomial) — no heavy conformal dependency. `scipy>=1.11` is a direct dep (binomial/beta tails).

---

## 7. Release engineering (W6, done)

`.github/workflows/ci.yml` — GitHub Actions on py3.11/3.12: `uv` install of a lean torch-free
toolchain (numpy/pandas/scikit-learn/scipy/pytest/ruff), `ruff check`, `pytest`. `.pre-commit-config.yaml`
(ruff + hygiene). `uv.lock` committed (62 packages). `CITATION.cff` + `.zenodo.json` ready for a
GitHub-Release→Zenodo DOI (still to mint — needs a Zenodo account, W7). `docs/DATA_CARD.md` +
`docs/MODEL_CARD.md`. `notebooks/quickstart.ipynb` (verified executable: calibrate → break under
shift → group-conditional repair, CPU seconds). Ruff config: line-length 120, select E/F/I/W/UP/B.

---

## 8. Reviewer findings (adversarial panel) + the CP-precision fixes

A 6-lens reviewer panel + adversarial verification ran on the paper. Of the reviewers' "big
weaknesses", **18 were OVERSTATED, 1 WRONG, only 4 REAL** — and those reduce to **two**:
1. **E6 purity is circular** (= 1 − selective risk). **Fixed by E6b** (interaction recovery, a
   metric distinct from the RMSD label; accepted 0.90 vs rejected 0.72, CI excludes 0 all 5).
2. **Single-benchmark (RNP)**; the one external test (FoldBench) is a negative, Mac1 deferred.
   **Partly addressed**: W1 recomputes rich features on the real structures; a true cross-dataset
   positive still needs richer external data or Mac1 coords (W8/PI).

**Five precision fixes applied** (paper + METHODS + tests): (a) binomial p-value stated as *valid,
exact for homogeneous, conservative for heterogeneous* (convex-order), not "exact"; (b) LTT framed
as pre-specified top-k *ranks*, not data-independent thresholds; (c) E1 real-data validity now
reported with a Clopper-Pearson CI **and** a synthetic test that *demonstrates* the finite-fold
proxy is downward-biased; (d) E3 stated as per-stratum *marginal* (δ/K for simultaneous);
(e) "training-free" clarified (structure-predictor-frozen; the combiner is calibration-only).

---

## 9. Where results can mislead (keep these prominent)

- **E6 purity = 1 − selective risk** (circular). Use **E6b** for downstream value.
- **E1 per-split realized-risk fraction is a downward-biased proxy** (tight certifier → true risk at
  α → finite-fold crosses α ~half the time even when valid). The rigorous validity is the synthetic
  test in `tests/test_conformal.py`; the RNP fraction is reported *with a CP CI* to make it checkable.
- **`weighted_ltt` abstains in every cell (E3b, 0% coverage) BY DESIGN** — on real data even the
  plug-in barely clears α, so the finite-sample-conditional certificate has no margin. Never report
  it as a positive. The label-free *plug-in* is the practical repair; **group-conditional (E3) is the
  rigorous headline** where stratum labels exist. Concept-shift diagnostic explains why (P(correct|conf)
  moves 0.08→0.29 into the novel regime).
- **The combined score comes in two tiers**: tabular (cheap, structure-free, the consistent primary
  across all experiments) and **+pose (W1, needs structures, reported only in E4/E5 as an upgrade)**.
  Do not conflate their AURC numbers.
- **E10 is an honest negative** (feature-poverty, not method failure), not a bug.

---

## 10. Known limitations

1. No cross-dataset *positive*; the one external test (FoldBench) is a negative; Mac1 (prospective)
   deferred until coordinates release.
2. Weighted-CP finite-sample guarantee is conditional on correct weights and abstains on real data
   (concept shift); group-conditional is the operative shift guarantee.
3. Screening-enrichment number not yet produced (harness ready; needs a benchmark + data).
4. Boltz-2 affinity-probability baseline (an honest-negative comparator) not in the tabular dump.
5. Reviewer LOW items remaining: LTT uses empirical-quantile thresholds (framed as ranks; empirically
   valid), `>=` tie handling (negligible on continuous scores).
6. **Doc drift to fix**: `CLAUDE.md` still mandates pixi + `pixi.lock` (repo is on uv, no pixi.lock);
   `conformal/__init__.py` docstring calls Mondrian/weighted "Next" (already shipped in-house).

---

## 11. Remaining work

### W2b — screening-enrichment number (PI decision)
- **Goal:** show abstention lifts virtual-screening enrichment (the "changes a real decision" number).
- **Method (ready):** `selective/enrichment.py` (EF@0.5/1/5%, BEDROC α=80.5, selective-EF at fixed
  retained size, coverage-enrichment + active-retention curves, random-abstention control). Rank by
  chain-pair ipTM / min-iPAE. Calibrate the abstention threshold with LTT on held-out, never on test
  enrichment. Bootstrap over ligands.
- **Needs (PI):** a benchmark/decoy decision — **DEKOIS 2.0** (property-matched) primary + a clean
  LIT-PCBA subset; avoid raw DUD-E. Confidence tables from released supplementary (no GPU) or a
  co-fold run. The Mac1 prospective screen (github `jongbin99/Cofolding` is pipeline-only; the eLife
  screen data is the drop-in target when released).

### W7 — preprint + submission (PI decisions)
- MoML 2026 short paper **Sept 1 2026 AOE** (non-archival, MIT Oct 14); MLSB @ NeurIPS 2026 (~Oct 1,
  Sydney Dec 6–12); journal extended version → **Digital Discovery** or **J. Cheminformatics**.
  arXiv q-bio.BM + cs.LG + stat.ML; bioRxiv. **Needs:** author list/order, arXiv/bioRxiv/Zenodo/journal
  accounts, venue priority. Mint the Zenodo DOI from a tagged v0.1.0 GitHub Release.

### W8 — full-paper expansion (reach)
- Fold W1/W2 into a Digital Discovery paper; add per-model figures; surface the single-model-deployable
  config (E5 ablation "+ ensemble spread" = tabular one-model number) explicitly; a possible DRO/
  distributionally-robust bound turning the weighted-CP null into a theorem (panel's "think-big").
- **Scoop re-sweep before submission:** targeted "conformal" + "co-folding / pose / binding mode"
  search; watch arXiv 2509.20345 (residue-level CRC), 2603.09947 (Confidence Gate Theorem), and any
  abstention layer bolted onto RNP/FoldBench. Novelty cell (unoccupied): model-agnostic × training-free
  × guaranteed accept/abstain × shift-robust × pose-correctness label. Closest prior = CoDrug (scalar,
  KDE weights, asymptotic).

---

## 12. Load-bearing gotchas (read before touching)

- **Env is uv + `.venv` (py3.12)**, not pixi. Ignore CLAUDE.md's pixi mandate.
- **The harness reaps background *tasks*** (~hourly). Long jobs must be launched **orphaned**
  (`nohup … & disown`) and be **resumable** — see `build_pose_features.py`.
- **`--limit` on the pose driver disables the lock + checkpointing.** Never ship a `--limit` parquet.
- **Group-conditional is inline in experiments**, not a package function (yet).
- **Torch stays out of the analysis env** (libomp/lightgbm+torch segfault on macOS from dual OpenMP;
  `KMP_DUPLICATE_LIB_OK=TRUE` is NOT safe). The layer is training-free, so no torch is needed.
- **Confidence field names differ per model** — verify against a real output file before parsing
  (AF3 emits no PDE). RNP `boltz1` = tarball dir `boltz`.
- **Prose style** (CLAUDE.md humanizer rules): no em-dashes, no rule-of-three, no negative parallelism
  in docs/paper. Code/commits/PRs: normal.
