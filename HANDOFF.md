# foldgate — Execution Handoff

**Purpose.** Everything needed to take *Know When to Fold* / `foldgate` from its current state (a complete, reviewed reuse-first study with a drafted MoML short paper) to a SOTA, submission-ready paper + released tool. Written so a competent ML-for-science engineer (or a future agent) can execute each workstream without re-deriving context. Best-method guidance per workstream is grounded in a mid-2026 methods sweep (see §6).

Repo: **github.com/rikhinkavuru/foldgate** · local: `/Users/rikhinkavuru/moml` · Python venv: `.venv` (py3.12).

---

## 1. One-page context

**Thesis.** Co-folding models (AlphaFold3, Boltz, Chai, Protenix) report confidence that correlates with pose accuracy, but a correlation is not a decision rule, and it is weakest on the novel pockets / chemotypes that drug discovery actually faces. `foldgate` is a model-agnostic, **training-free** reliability layer that turns confidence into a **risk-controlled accept/abstain** decision with a **finite-sample conformal guarantee**, shows the guarantee **breaks under novelty shift**, and **repairs** it with shift-robust conformal keyed on training-set similarity.

**Data.** Released Runs N' Poses (RNP): 13,536 delivered poses (top-1 by `ranking_score` per system/ligand/model), 6 co-folding models. Ships `ranking_score` + interface chain-pair ipTM + BiSyRMSD + LDDT-PLI per method, PoseBusters results, and pre-computed training-similarity (Morgan Tanimoto, pocket similarity, release date). Label: binding-mode correct iff BiSyRMSD ≤ 2 Å. Zero GPU; everything runs on CPU from a 52 MB download.

**RQ → answer (all on real data):**

| RQ | Answer | Experiment |
|----|--------|-----------|
| Native confidence → valid conformal coverage i.i.d.? | Yes (certifier validated on synthetic; mean risk ≤ α on RNP) | E1 |
| Holds under novelty? | **No** — per-stratum risk 0.07→0.43; deploy-on-novel 0.55 error, guarantee holds 0% of runs | E2 |
| Shift-robust methods restore it? | Yes — group-conditional (E3), weighted (E3b), full method recovers coverage (E3c) | E3/E3b/E3c |
| Beat native thresholding + help downstream? | Yes — combined score AURC −24–40% (ΔAURC CIs exclude 0); downstream purity 63–70%→82–94% | E4/E6 |
| Model- and task-agnostic? | Yes — 6 models; interface-quality task (E8); robust across RMSD thresholds (E5) | E5/E8 |

**Extra findings.** E7: the break is **structural/chemical novelty, not recency** (strong on pocket-novelty, weak on temporal). E9: combined score orders by continuous RMSD (accepted mean-RMSD 2.1→1.1 Å). E10: FoldBench cross-dataset is an **honest negative** — it ships only `ranking_score`, so the feature-poor combiner does not transfer; this corroborates the ablation.

---

## 2. Current state

- **Package** `src/foldgate/`: `io` (RNP + FoldBench loaders), `features` (novelty strata, pose/PoseBusters, cross-model agreement, temporal), `scores` (`ScoreCombiner`), `conformal` (LTT `ltt_threshold`, `weighted_threshold`, `continuous_risk_threshold`, `naive_threshold`, `hb_upper_bound`), `selective` (risk-coverage, AURC, conditional coverage, bootstrap), `eval`.
- **11 experiments** `experiments/e*.py` (E1, E2, E3, E3b, E3c, E4, E5, E6, E7, E8, E9, E10) + `make_figures.py`. Results in `results/*.json`, figures in `results/figures/`.
- **Tests** `tests/` — 6/6 green, including an empirical validity test of the LTT certifier.
- **Paper** `paper/moml2026_shortpaper.{md,pdf,html}` (MoML short, drafted + audited via paper-audit).
- **Docs** `RESULTS.md`, `METHODS.md`, `RELATED_WORK.md`, `CLAUDE.md`, `PLAN.md`, `experiments/README.md`, `data/DATASETS.md`.

**Independent review outcome (already applied):** the LTT certifier and 3-way-split hygiene are **sound** — no leakage, validity confirmed by null-hypothesis Monte Carlo (false-cert ≤ δ). Fixed: AURC significance now a **paired data-bootstrap over test poses** (ΔAURC 90% CI excludes 0 for all models); E1 reframed honestly (tight certifier → realized-risk indicator is a noisy proxy; the synthetic test is the rigorous evidence); E3b stated as a point estimate not a certified gate; NaN-RMSD handling + a dead novelty check fixed; empirical-Bernstein continuous bound added.

**Honest known limitations (state them; some are the work below):**
1. Weighted CP (E3b) is approximate under estimated weights — no finite-sample 1−δ (workstream W3).
2. Certified *continuous* gate is loose with distribution-free bounds (W4).
3. No cross-dataset *positive* (FoldBench is feature-poor; needs richer external data — W1/W5).
4. Mac1 downstream arm blocked (delayed coords) — substitute a screening-enrichment proxy (W2).
5. Baselines beyond raw `ranking_score` not yet run (W5).
6. Reviewer LOW items still open: LTT uses empirical-quantile thresholds (not a pre-specified grid; empirically valid) and `>=` tie handling (negligible on continuous scores).

---

## 3. Reproduce from scratch

```bash
cd ~/moml
make setup           # uv venv + deps  (or: uv venv --python 3.12 .venv && uv pip install ...)
# download RNP tabular artifacts (52 MB) into data/raw/ if absent — see data/DATASETS.md:
#   zenodo record 18366081: predictions.tar.gz, annotations.csv, posebusters_results.tar.gz
make features        # -> data/processed/rnp_delivered.parquet (13,535 rows)
make experiments     # E1..E10 -> results/*.json + results/figures/*.png
make paper           # -> paper/moml2026_shortpaper.pdf
make test
```

Env facts: `.venv/bin/python` is py3.12 (system python3 is 3.14, too new). Everything is torch-free and CPU-only. `crepes`/`crepes-weighted` installed but the primary certifier is the in-house LTT (exact binomial) — no heavy conformal dependency.

---

## 4. Remaining workstreams (execution plan)

Each workstream: **goal · why · best method (§6) · steps · code pointers · acceptance · effort · needs**. Ordered by impact-per-effort.

### W1 — Cross-model POSE agreement + pose-ensemble features  ★ biggest quality lever, no GPU
- **Goal:** add structural consensus features (pairwise cross-model ligand-RMSD, intra-model pose diversity across the 5 diffusion samples) to the combined score.
- **Why:** the ablation shows the AURC gain comes from ipTM + ensemble + cross-model *confidence* agreement; *pose* agreement is a stronger, orthogonal signal expected to push AURC down another ~10–20% and possibly rescue the novel strata.
- **Needs:** download the **39 GB RNP structure tarballs** (`prediction_files.tar.gz` + `ground_truth.tar.gz`, Zenodo 18366081). Disk + CPU only, **no GPU**. ~40 GB free on the machine (OK — 79 GB free). **PI approval for the download.**
- **Best method / tools:** spyrmsd `symmrmsd` (pocket-superpose the two predictions, then symmetry-corrected ligand RMSD with `minimize=False`); build the ligand graph from the reference SDF (RDKit `AssignBondOrdersFromTemplate`); parse mmCIF with gemmi. Full recipe + the PAE-availability caveat in **§6.W1**.
- **Steps (skeleton):** (1) extract predicted ligand poses (mmCIF/SDF) per (system, ligand, model, seed, sample); (2) compute symmetry-corrected pairwise ligand-RMSD across models (delivered poses) and across the 5 samples within a model; (3) features per delivered pose: `xmodel_pose_rmsd_median/min`, `pose_consensus_cluster_size`, `intra_model_pose_std`; (4) add to `src/foldgate/features/agreement.py` + `combiner.DEFAULT_FEATURES`; (5) rebuild table, re-run E4/E5 ablation, E3c.
- **Acceptance:** ablation shows pose-agreement lowers AURC beyond confidence-agreement; ΔAURC CI still excludes 0; document the feature's marginal contribution.
- **Effort:** M (download + CPU RMSD over ~100k poses + wiring).

### W2 — Downstream screening-enrichment arm (E6b)  ★ turns statistics into practice
- **Goal:** show that abstaining on unreliable poses **improves virtual-screening enrichment** vs using all predictions. Substitute for the delayed Mac1 set.
- **Why:** reviewers will ask "does the guarantee change a real decision?" E6 (RNP-internal purity) is suggestive; a screening-enrichment result is the "changes practice" number.
- **Best method / benchmark / metrics:** DEKOIS 2.0 primary + a clean LIT-PCBA subset (avoid raw DUD-E); rank by chain-pair ipTM / min-iPAE (not pTM/affinity); EF@0.5/1/5%, BEDROC α=80.5, logAUC; calibrate the abstention threshold with LTT and trace a coverage-enrichment curve. Details + pitfalls in **§6.W2**.
- **Steps (skeleton):** (1) pick benchmark + targets; (2) obtain or co-fold actives+decoys, score by confidence/affinity; (3) rank all vs rank-accepted-only (abstain the rest); (4) report EF@1%/BEDROC with vs without the reliability gate; bootstrap CIs.
- **Acceptance:** measurable enrichment lift (EF or BEDROC) from abstention, with CI; honest about decoy bias.
- **Effort:** M–L (data + a screening pipeline). **PI decision:** which benchmark/decoy set.

### W3 — Make weighted conformal (E3b) rigorous
- **Goal:** replace the plug-in weighted point estimate with a defensible finite-sample (or clearly-scoped) guarantee under estimated weights, with out-of-fold weight estimation.
- **Best method:** importance-weighted LTT with a WSR betting p-value (Almeida et al. 2025) → exact `P(R_target≤α)≥1−δ` *conditional on correct weights*; the risk-control analogue of Tibshirani-2019, landing on the existing E1 LTT. Full recipe in **§6.W3**.
- **Steps:** implement the ~25-line WSR p-value in `conformal/weighted.py`; **cross-fit** the source-vs-target density-ratio out-of-fold (calibrated classifier); clip weights; run weighted-LTT with the E1 fixed-sequence; add n_eff + a weight-sensitivity sweep + a Mondrian fallback; re-run E3b; test the covariate-shift assumption and state the exact scope.
- **Acceptance:** either a genuine P(risk≤α)≥1−δ on the target, or an explicit, cited approximate-coverage statement + sensitivity plot. Effort: M.

### W4 — Tight certified continuous-RMSD gate (E9)
- **Goal:** the certified continuous-mean gate should certify meaningful coverage (currently ~0 with Hoeffding; empirical-Bernstein added but verify).
- **Best method:** WSR predictable-plug-in empirical-Bernstein (PrPl-EB, closed-form) as the default certifier; WSR betting UCB for the tightest. Both beat Hoeffding-Bentkus. Formulas in **§6.W4**.
- **Steps:** swap the bound in `conformal/continuous_risk_threshold` to PrPl-EB (already scaffolded with empirical-Bernstein); use the smallest defensible RMSD cap B; re-enable + verify the certified gate in `experiments/e9_continuous_risk.py`; co-report the acceptance fraction (Clopper-Pearson); degenerate check must reproduce the E1 binomial number. Acceptance: non-trivial certified coverage with a stated 1−δ guarantee. Effort: S.

### W5 — Baselines a reviewer will demand
- **Goal:** compare the combined score against the confidence baselines practitioners/reviewers expect, not just raw `ranking_score`.
- **Best baseline set:** native chain-pair ipTM threshold, PoseBusters-pass, temperature/Platt/isotonic calibration, Boltz-2 affinity-probability (Boltz rows only — honest negative), ipSAE/pDockQ2 if PAE ships, and the **key ablation**: the learned score as a plain classifier + fixed threshold vs the same score in the conformal layer (isolates guarantee value from feature value). Details in **§6.W5**.
- **Steps:** compute each on the *same* split, report AURC + coverage vs combined **iid AND under the E2 shift** (so "calibration breaks under shift, conformal doesn't" is shown, not asserted); cite the delta vs CalPro + the TCR-pMHC calibrated-abstention paper. Acceptance: combined dominates or matches each; calibration-vs-conformal framing explicit; scoop-check clean. Effort: M.

### W6 — Release engineering (pip + CI + DOI + docs)
- **Goal:** a top-tier open-source method release.
- **Best practice:** uv + hatchling + ruff + type checker (mypy/`ty`) + GitHub Actions CI + Zenodo-DOI-from-GitHub-Release; scripted data download (not the pose blob in git). Full checklist in **§6.W6**.
- **Steps:** add CI running ruff + type + `make test`; pre-commit; a quickstart notebook (calibrate → accept/abstain with guarantee); `scripts/download_data.py`; seeded reproduction of E1/E4/E7; tag `v0.1.0` → mint Zenodo DOI; DATA/MODEL cards; `CITATION.cff` + `.zenodo.json`. Acceptance: green CI, clean-clone `uv sync`/`pip install -e .` works, DOI minted + cited. Effort: S–M. **Needs:** Zenodo account.

### W7 — Preprint + submission
- **Goal:** post preprint, submit to workshops/journal.
- **Confirmed logistics:** MoML 2026 short paper **Sept 1 AOE** (non-archival, MIT Oct 14); MLSB @ NeurIPS 2026 ~Oct 1 (CFP pending; NeurIPS is Sydney Dec 6–12); journal extended version → Digital Discovery or J. Cheminformatics. arXiv q-bio.BM + cs.LG + stat.ML. Full timeline in **§8**.
- **Needs:** author list/affiliations, arXiv + bioRxiv + journal accounts. **PI decisions:** venue priority, authorship.

### W8 — Full-paper expansion + remaining review nits (optional / reach)
- Expand the MoML short paper to a full Digital Discovery / J. Cheminformatics paper (add W1/W2/W5 results, the marginal-coverage-of-RMSD-interval robustness framing, per-model figures). Address reviewer LOW items: pre-specified LTT score grid; tie handling `>` vs `>=`. Effort: L.

---

## 5. Decisions the PI (Rikhin) must make

1. **Approve the 39 GB structure download** for W1 (biggest quality lever; no GPU). y/n.
2. **Screening benchmark** for W2 (recommendation in §6 W2). Which one.
3. **Venue priority** — MoML short now vs. straight to a Digital Discovery / J. Cheminformatics full paper. Sets how much of W1/W2/W5 is in-scope before submission.
4. **Authorship** (list + order) and **accounts** for arXiv / bioRxiv / Zenodo / journal.
5. Optional: a **new GPU slice** (only if a *third feature-rich dataset* is wanted; A100-80 GB, Colab Pro+ marginal, Kaggle too small). Recommendation: skip; do W1 instead.

---

## 6. Best current methods (mid-2026 research)

Grounded in a mid-2026 methods sweep. Each subsection is the recommended approach + tools + the non-obvious gotchas.

### 6.W1 — pose features (structures)
- **RMSD tool:** **spyrmsd v0.9.0** `symmrmsd()` (`pip install spyrmsd rustworkx rdkit gemmi`) — symmetry-corrected ligand RMSD via graph isomorphism; ensure the **rustworkx** backend (not the slow NetworkX fallback), `cache=True`. `obrms` (OpenBabel) is the ~10× speed fallback; DockRMSD for pathological symmetry.
- **Correctness-critical:** ligand bonds in predicted mmCIF are unreliable. Build the molecular graph **once** from the RNP reference SDF/SMILES (`RDKit AssignBondOrdersFromTemplate`), cache its adjacency/atomic-props, reuse for every predicted pose. Parse coords with **gemmi**.
- **Cross-model agreement = binding mode, not conformer:** superpose the two predictions on the shared **receptor pocket** (Cα/pocket-atom Kabsch) first, then symmetry-corrected ligand RMSD with **`minimize=False`**. `minimize=True` Kabsch-fits the ligand onto itself and throws away the placement signal you want.
- **Features:** per delivered pose — `xmodel_pose_rmsd_{median,min}` (across the other models' delivered poses), `pose_consensus_cluster_size`, `intra_model_pose_std` (spread across the 5 diffusion samples).
- **PAE features (conditional):** **first confirm PAE ships in the 39 GB tarball** — predictions may be mmCIF + summary scalars only, and AF3/Boltz store PAE differently. If present: interface-PAE (mean/min over protein↔ligand token pairs) and ligand-pLDDT are the best cheap correlates; `ipsae.py` (DunbrackLab) emits **ipSAE + pDockQ2 + LIS** in one numpy script (adapt the protein↔ligand block). Validate your RMSD pipeline against RNP's own BiSyRMSD with **OpenStructure**.
- **Honest caveat to state:** co-folding models make **correlated errors** (AF3/Chai RMSD r≈0.72; big errors missed by all models together), so consensus is a *feature*, not independent validation, and can be overconfident on shared blind spots. CONFIDE (Dec 2025) beats pLDDT but needs AF3 diffusion internals → out of scope for a training-free layer.

### 6.W2 — screening enrichment (E6b)
- **Benchmark:** **DEKOIS 2.0** primary (property-matched decoys, what the leading AF3-for-VS study used, tractable) + a small **clean LIT-PCBA** subset as honest-hard secondary (real inactives — but cite its known train/val leakage, arXiv:2507.21404). **Do not headline raw DUD-E** (analog/decoy bias inflates enrichment); use only as a bias-sensitivity control.
- **Score by interface confidence** (chain-pair ipTM / **min-iPAE**), which ranks far better than global pTM or raw Boltz-2 affinity for pose separation (RSC d5sc06481c). Get confidence tables from released supplementary data — **no GPU co-folding needed**. The eLife Mac1 screen (AF3 prospective AUC 0.46–0.61) is both the motivation and drop-in data.
- **Metrics:** EF@0.5%/1%/5%, **BEDROC (α=80.5)**, logAUC/ROC-AUC. Bootstrap over **ligands** for CIs; paired test across targets.
- **Abstention plug-in:** (v1 hard) drop poses below a calibrated reliability threshold then re-rank the accepted subset; (v2 soft) push abstained to the bottom. **Calibrate the threshold with LTT on a held-out split, never on test enrichment.** Trace a **coverage-enrichment curve** (the screening analogue of risk-coverage).
- **Pitfalls:** EF's denominator shifts when you abstain → define a **selective-EF at fixed retained-library size**. Abstention can silently discard true actives (models are least confident on novel-chemotype actives) → **show an active-retention-vs-coverage curve** (decoys removed faster than actives). Confidence is not calibrated across targets → use per-target / group-conditional thresholds. Controls: random-abstention at matched coverage.

### 6.W3 — rigorous weighted conformal (E3b)
- **The exact upgrade, on the existing LTT machinery:** **Almeida et al. 2025, "High Probability Risk Control Under Covariate Shift"** (PMLR v266, almeida25a). Test the reweighted-source null `E_source[w(X)·L(λ)] > α`, which equals the target null `R_target(λ) > α` under covariate shift → plug a valid weighted-loss p-value into LTT/FWER → **exact finite-sample `P(R_target(λ̂) ≤ α) ≥ 1−δ`**, the risk-control analogue of Tibshirani-2019.
- **Use the WSR betting p-value** (~25 lines: capital process `K_i = Π(1 − ν_j(L_j − α))`, `ν_i = min(1, sqrt(2 log(1/δ)/(n·σ̂²_{i-1})))`, `p = 1/max_i K_i`, on rescaled weighted losses). It dominates Hoeffding/Clopper-Pearson and uses the whole calibration set (unlike Park-2022 rejection sampling).
- **Weights out-of-fold (mandatory):** split calibration — fold A fits the source-vs-target density ratio (`densratio` uLSIF/RuLSIF, KMM, or `CalibratedClassifierCV(isotonic)`; the classifier **must be probability-calibrated** or weights are biased), fold B computes weighted losses. Clip weights, set `B=clipped max`, report **n_eff = (Σw)²/Σw²** + a weight-model sensitivity sweep.
- **Honest scope + fallback:** the 1−δ is exact **only conditional on correct weights** — say so. Under novel-pocket/chemotype shift `P(correct|confidence)` likely also moves (label shift), so pure covariate reweighting controls an aligned distribution and true risk can exceed α by up to the TV distance (Angelopoulos 2024) — **test it** (compare `P(correct|conf)` source vs target). When n_eff is low / overlap poor, **fall back to the E3 group-conditional (Mondrian) certificate** as the operative guarantee.

### 6.W4 — tight continuous-risk certification (E9)
- **Default gate:** **WSR predictable-plug-in empirical-Bernstein CI (PrPl-EB)** (Waudby-Smith & Ramdas, JRSSB 2024) — closed-form, one numpy expression, variance-adaptive, matches conjugate-mixture EB to O(1/n), >500× faster than root-finding. `μ̂_t=(½+Σ_{i≤t}L_i)/(t+1)`, `σ̂²_t=(¼+Σ(L_i−μ̂_i)²)/(t+1)`, `λ_t=min(sqrt(2 log(1/δ)/(n·σ̂²_{t-1})), ¾)`. **Tightest option:** WSR hedged-betting UCB (grid over m). Both beat Hoeffding-Bentkus (worst-case-binomial, no variance adaptation). Reference code: `confseq`.
- **Loss:** `L_i = min(RMSD_i, B)/B ∈ [0,1]` with the **smallest scientifically defensible cap B** (e.g. 4 Å) — a large B re-inflates variance and undoes the EB advantage.
- **Selective validity:** the accept threshold must be **independent of the certified points' RMSDs** — pick τ on one fold, certify on a fresh fold (or LTT multiple-testing over a τ grid). **Always co-report the acceptance fraction** (Clopper-Pearson) — a low certified risk from accepting almost nothing is otherwise a free lunch. Degenerate check: binarizing the loss must reproduce the E1 exact-binomial number.
- **Not** Conformal Risk Control (ICLR 2024): it gives `E[loss] ≤ α` in expectation, not a 1−δ certificate — use RCPS/LTT.

### 6.W5 — baselines a reviewer will demand
- **Must-have comparison table** (same calibration/test split as foldgate, reported **iid AND under the E2 shift**): raw `ranking_score`, **native chain-pair ipTM** (Genz 2025: the best single discriminator), **PoseBusters-pass**, **temperature-scaling / Platt / isotonic** calibration, **Boltz-2 affinity-probability** (Boltz rows only — likely a useful honest negative), and — if PAE ships — **ipSAE / pDockQ2** (`pip install ipsae`).
- **The key ablation:** the learned combined score **as a plain classifier + fixed threshold** (SiteAF3 / Genz-merged-score style) vs the **same score wrapped in the conformal layer** — this isolates the value of the *guarantee* from the value of the *features*.
- **Framing a reviewer will hammer:** calibration (temp/Platt/isotonic) fixes only marginal probability, carries **no finite-sample coverage**, and **breaks under shift**; conformal gives a distribution-free finite-sample guarantee and the weighted/group-conditional variants repair the exchangeability break. Do not conflate them.
- **Scoop status: white space intact** (no conformal/selective-prediction accept-abstain with finite-sample guarantees on co-folding *pose* reliability). **Cite with an explicit delta:** CalPro (conformal for residue-level *monomer* structure, not poses) and **Calibrated Abstention for TCR-pMHC under epitope shift** (arXiv:2604.13254 — near-identical selective+conformal-under-shift template, but binding *classification*, not pose geometry). Run PoseBusters/GBM in **separate processes** (libomp/OpenMP segfault).

### 6.W6 — release engineering + venues (confirmed mid-2026)
- **Release stack:** `pyproject.toml` (hatchling), **uv** + committed `uv.lock`, **ruff** (lint+format), a type checker (mypy or Astral `ty`), `.pre-commit-config.yaml`, **GitHub Actions CI** (install + ruff + type + `pytest` on py3.11/3.12), a scripted data download (not the 13.5k-pose blob in git), a seeded one-command reproduction of E1/E4/E7, an example notebook, MODEL + DATA cards, `CITATION.cff` + `.zenodo.json`, **Zenodo DOI** from a tagged GitHub Release. Verify the archived tarball reproduces from a clean clone.
- **Venues (confirmed):** **MoML 2026** — short paper, **deadline Sept 1, 2026 AOE**, decisions Sept 8, event Oct 14 (MIT), 2–4 pp, **non-archival**. **MLSB @ NeurIPS 2026** — 5th ed, NeurIPS is **Sydney, Dec 6–12**; 2026 CFP not yet posted (prior pattern ~Oct 1, ~5 pp, double-blind, non-archival; spotlights reward open code/data). **MLCB 2026 already closed (Jul 1–3) — missed this cycle.** Journals for the extended version: **Digital Discovery** (RSC, gold OA, APC £2200, requires Data Availability Statement + deposited code/data) or **Journal of Cheminformatics** (OA CC-BY, APC £1690/$2390, will only publish fully third-party-reproducible work). arXiv **q-bio.BM** primary + cross-list **cs.LG, stat.ML**; bioRxiv. Non-archival workshops don't conflict with a later journal; don't dual-submit to two *archival* venues.

### 6.W7 — submission timeline is folded into §8.

---

## 7. Risk register

- **39 GB download** fails / disk pressure → stream-extract per-target, delete after feature computation; keep only the feature table.
- **Weighted-CP guarantee** may stay approximate even with SOTA (estimated weights) → keep group-conditional as the rigorous headline; scope weighted as the label-free complement.
- **Screening decoy bias** (DUD-E) → prefer LIT-PCBA / property-matched decoys; report the caveat.
- **Scoop** — re-sweep "conformal + co-folding pose" before submission (see `RELATED_WORK.md`).
- **Mac1** stays blocked → do not gate submission on it; land it as a v2/rebuttal addition when coords release.

## 8. Submission timeline (from 2026-07-11)

| window | action |
|---|---|
| Jul 11–25 | Release engineering (W6): pyproject/uv/ruff/CI/pre-commit; scripted data download; seeded reproduction of E1/E4/E7. In parallel, W4 (continuous certifier, ~1 day) and W3 (weighted-LTT, ~2–3 days) — both land on existing machinery. |
| Jul 25 – Aug 8 | W5 baselines (table + the learned-score-without-conformal ablation). If PI approves, W1 (39 GB download → pose features → re-run E4/E5). MODEL/DATA cards, example notebook. |
| Aug 8–15 | Tag **v0.1.0** → mint **Zenodo DOI**; verify clean-clone reproduction. Draft Data/Code Availability paragraph. Start W2 screening arm if PI picks a benchmark. |
| Aug 15–28 | Write/tighten the MoML 2–4 pp short paper (method + E1/E4/E7 headline + artifact DOI). Internal reproducibility dry-run on a fresh machine. |
| **Sept 1, 2026 (AOE)** | **Submit MoML short paper.** Post arXiv (q-bio.BM + cs.LG + stat.ML) + bioRxiv at/after submission. |
| ~late Sept / ~Oct 1 | Submit to **MLSB @ NeurIPS 2026** once its CFP posts (watch mlsb.io). |
| Oct → Nov | Fold W1/W2/W5 results into the extended paper; submit to **Digital Discovery** or **J. Cheminformatics** → target Dec–Jan acceptance. |

**Fast path if the goal is only the MoML short paper:** W3 + W4 + W5 + W6 are enough (all doable now, no external data). W1 (39 GB) and W2 (screening) are the differentiators for the extended journal version.
