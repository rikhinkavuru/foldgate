# HANDOFF — foldgate, session 2 of 2026-07-12 (theorem + benchmark + workshop paper)

**For the next session. Read this first, then `docs/EXECUTION_2026-07-12b.md` (the granular ledger),
then `PROGRESS.md` (session-1 ledger). The auto-memory `know-when-to-fold-project.md` has a
compressed version.** The one live task for the next session is the GPU run (J1); its runbook is §6.

---

## 0. TL;DR

This session turned the workshop paper into a **theorem-backed, twice-audited MoML submission** and did
the two non-GPU journal tasks. Concretely:

- **Final MoML workshop paper shipped:** `paper/moml2026_foldgate.{tex,pdf}`, 4pp main text + appendix
  (theorem proof + benchmark), non-anonymous (Rikhin Kavuru, Independent Researcher). Passed a
  6-reviewer audit and a 3-lens re-audit, every number verified against the result files.
- **The theorem** (the "journal 10" work): a real impossibility + achievability result, proved and
  validated against ground truth and on public non-co-folding data.
- **General benchmark** (`b1`-`b7`, elec2, ACS) + **multiplicity control** (Romano-Wolf, joint LOTO,
  Bonferroni delta/K) implemented.
- **Journal J2/J3 done** (screening decision = guaranteed object; scaffold-split fair baseline).
- **J1 (FoldBench Protenix regen) is GPU-blocked** and is the next session's job. User is renting a
  **Lambda 1x H100 80GB PCIe** (Path A: user launches, hands us SSH key + IP). Runbook in §6.
- **62 tests pass, ruff clean.** Everything committed + pushed this session (see §5).

---

## 1. Environment (unchanged, load-bearing)

- **`uv` + `.venv` (Python 3.12).** Run everything with `.venv/bin/python`. NOT pixi, NOT system python.
- Torch-free analysis package (numpy 2.5, pandas 3.0, scipy 1.18, sklearn 1.9, **rdkit present**).
  lightgbm and folktables were absent; the benchmark uses sklearn HistGradientBoosting, and b5 pip-installs
  folktables on demand.
- `tectonic` builds the paper (`cd paper && tectonic moml2026_foldgate.tex`). `pdfinfo`/`pdftotext` for checks.
- Data on disk (all gitignored): `data/processed/rnp_delivered.parquet` (13,535 x 44), `data/external/screening/`
  (Zenodo 17568813), `data/external/screening_affinity/` (18669539), `data/2018/` (folktables ACS cache, 828MB),
  RNP raw incl. 39.5GB `prediction_files.tar.gz` (already distilled -> do NOT re-stream).

---

## 2. What was built this session

### The theorem (Sec 4 of the paper + App A; full construction in `docs/theory/THEOREM_RECONCILED.md`)
For the deployed rule "accept iff frozen score `s >= tau`", with covariate-measurable label-free weights:
- **Impossibility (Thm 1):** at fixed coverage `c`, realized target selective risk `R_Q(tau_c) =
  R_ref(tau_c) + Delta_bar_c`, where `Delta_bar_c` (accept-region concept gap) is **invariant to the
  weights**. So the oracle-weighted certificate under-reports realized risk by `Delta_bar_c` (silent
  violation), and level `alpha` is unachievable at that coverage once `Delta_bar_c > alpha - R_ref`.
  Proof is a one-line tower identity (a covariate weight recovers only the source conditional `eta_P`).
- **DRO lower edge (Thm 2):** no label-free certificate beats the worst case over the ambiguity ball.
- **Achievability (Thm 3):** in-stratum recalibration attains the floor given `n_g` target-stratum
  labels, with slack `1/(n_g+1)` for coverage and the joint accept-and-err rate, and
  `O(sqrt(log(1/delta)/n_g))` for the **ratio** selective risk (the clean `1/(n_g+1)` does NOT apply to
  the ratio -- this correction was the single most important audit catch).
- Specializes the Tibshirani weighted-CP covariate-shift boundary and the Ben-David irreducible term;
  frames neighborhood-conditional (Barber impossibility forbids exact pointwise conditional coverage).

### General benchmark (`src/foldgate/bench/`, `experiments/b1`-`b7`, `results/bench/`)
- `synth.py`: generator with known `P(Y|s,nu)`; knobs `D` (concept), `E` (covariate-in-score), tilt `T`,
  plus **`Dq`** (genuine target concept drift so `Delta_bar>0`) and **`eps_floor`** (irreducible error
  floor). `oracle_concept_gap`, `oracle_coverage_threshold` added. Backward compatible (Dq=eps=0).
- `certificates.py`: worst-stratum RCPS UCB + f-divergence/CVaR ball (chi2 const `sqrt(2 rho Var)`).
- `realdata.py`: `electricity_triple` (OpenML), `acs_income_triple` (folktables; ACS `group` bug fixed --
  it was RAC1P/race not state), grouped leakage-free splitter.
- Validation (all PASS against ground truth): b1 decomposition; b3 reweighting-invariance + silent
  violation; b2 Mondrian-LTT + thin-strata abstain + joint-vs-ratio; b4 certificate coverage; **b7**
  (genuine Dq>0): R_ref 0.273 + Delta_bar 0.205 = R_Q 0.478, invariance residual 0.0002, crossing at
  Dq=1.47, source-label Mondrian fails (exceedance 1.0) vs target-label controls (0.12), eps>alpha ->
  100% abstention. Real data: b6 elec2 (concept drift 0.36) marginal 0.31 / weighted-misled 0.34 /
  group-conditional 0.20; b5 ACS (covariate 0.05) weighted repairs 0.23->0.20.

### Multiplicity (`experiments/_common.py` + recomputed e12/e13/e1/e4)
- `_common.py`: `holm`, `bh`, `romano_wolf_stepdown`, `iut_all`, `tost_equivalence`, `DELTA_JOINT=0.02`.
- E12: Romano-Wolf step-down on the 55-cell drift grid (42 survive, all structural); temporal TOST does
  NOT certify equivalence -> claim only the magnitude contrast. E13: joint LOTO at delta/K=0.02
  (af3/boltz1/boltz1x hold with coverage, chai/protenix vacuous-by-abstention, surfaced). E4/E1 IUT flags.
- Full spec: `docs/theory/MULTIPLICITY_SPEC.md`.

### Screening honesty (J2/J3): `experiments/e23_screening_honest.py`, `e24_screening_baseline.py`
- e23: pose-confidence (guaranteed) EF@1% DEKOIS 20.7 / GPCR 9.3 / LIT 4.0 beats docking 10.3/2.0/3.0;
  affinity head 31/26.6/5.1 reported SEPARATELY as ungoverned. W3 shift Construction 2 (disjoint ec_sim
  bins): GPCR affinity 27->42, pose 11->18.
- e24: Murcko scaffold-novelty split (DEKOIS novel<familiar all signals; GPCR mixed, honest); all methods
  recomputed; decoy analog-bias NOT computable (no decoy SMILES) -> stated limitation.
- Data-verification memo: `docs/theory/DATA_VERIFICATION.md` (every screening number with its source key).

### Paper (the deliverable)
- `paper/moml2026_foldgate.{tex,pdf}`. Reframed around the theorem (Sec 4), guaranteed pose signal as the
  screening headline, concept-shift mechanism, multiplicity paragraph, honest limitations. **Two audit
  rounds, all findings fixed** (see `docs/EXECUTION_2026-07-12b.md` for the full fix list). 4pp main text.

---

## 3. Audit trail (what the reviewers caught, so you trust the numbers)

Round 1 (6 reviewers + chair): theorem confirmed SOUND, numbers reproduce; 5 blockers + 4 majors, all
labeling/honesty, ALL FIXED. Notable catches: 2 wrong Table-1 drift cells (+.51/+.44 -> +.47/+.47); the
"42->27 collapse" was the AFFINITY head not the governed pose signal; the abstract over-called the screen
"guaranteed"; a false "temporal intervals cover zero" claim. Round 2 (3-lens re-audit): confirmed all
resolved; caught ONE defect the fixes introduced (a garbled+false theorem clause) -> deleted. Minor fixes:
n~76->~72, CVaR m*=1 -> m*->1 for most models, residual house-style, abstract 2-3x -> about twice.
Full fix list: `docs/EXECUTION_2026-07-12b.md`.

---

## 4. Reproduce from scratch

```bash
cd ~/moml
make setup                                     # uv venv + torch-free deps
python scripts/download_data.py                # RNP tabular
python scripts/download_data.py --screening    # screens + affinity (E16/E20/E21/E23/E24)
make features                                  # -> data/processed/rnp_delivered.parquet
.venv/bin/python -m pytest -q                  # 62 tests
for e in experiments/e*.py experiments/b*.py; do .venv/bin/python "$e"; done   # regen results/
cd paper && tectonic moml2026_foldgate.tex     # -> the 4-page PDF
```
b5 (ACS) pip-installs folktables and caches to `data/2018/` (gitignored). b6 (elec2) fetches OpenML.
Both degrade gracefully if offline.

---

## 5. Git state

- Remote: `git@github.com:rikhinkavuru/foldgate.git` (origin). Pushed this session.
- Branch: `session-2026-07-12-theorem-journal` (branched off `main` at `a1c7fb7`; see the commit for the
  exact SHA). `.gitignore` updated to exclude `data/2018/` and `data/[0-9]{4}/` census caches.
- Committed: all of session 1 (E11-E22) + session 2 (bench, theorem, multiplicity, e23/e24, paper, docs).
  Data (`data/`), parquet, npz stay ignored.

---

## 6. THE GPU RUNBOOK (J1 -- FoldBench Protenix feature-matched transfer)  <<< NEXT SESSION'S JOB

**Goal:** regenerate Protenix predictions for the 558 FoldBench protein-ligand targets, recover
interface-ipTM (`chain_pair_iptm`), and run the frozen calibrate-on-RNP / deploy-on-FoldBench transfer at
**feature parity** (the combined score, not just shared `ranking_score`). This closes the single-real-dataset
weakness and turns the honest E15 negative into a real positive.

### 6.0 The instance (user launches -- Path A)
- **1x H100 (80 GB PCIe), Lambda Stack 22.04, no attached filesystem, SSH key `morpheus-lambda-nopass`.**
- User provides: the **nopass private key file** + the **instance public IP**. Connect: `ssh ubuntu@<IP>`.
- 80 GB VRAM clears every target at default config (Protenix peak ~78 GB at 4000 tokens). 1 TB local SSD.
- **Kill the instance when done** (user does this; billing is ~$3.29/hr, expect ~$35-50 total).

### 6.1 On the box: verify + set up Protenix (its OWN env; never mix with the analysis venv)
```bash
nvidia-smi                                     # confirm H100 80GB, driver OK
# Protenix pins torch/CUDA; use its instructions:
git clone https://github.com/bytedance/Protenix && cd Protenix
# create a fresh conda/venv per its README, install pinned torch+deps, download weights
# (see Protenix/docs/training_inference_instructions.md -- has the exact install + a VRAM/latency table)
```
Protenix inference peak VRAM by size (from its docs): 500 tok=6GB, 1000=18GB, 2000=67GB, 4000=78GB.
Most FoldBench protein-ligand targets are ~300-800 tok (fit easily); the big assemblies need the 80GB.
Do NOT reduce `pairformer.nblocks`/`diffusion_batch_size` to save memory -- it changes outputs and breaks
feature parity. Size the GPU instead (that is why 80GB).

### 6.2 Get the 558 FoldBench protein-ligand inputs
- FoldBench: `github.com/BEAM-Labs/FoldBench`, CIFs via Google Drive/Zenodo (see repo README). Fetch the
  **protein-ligand subset** (558 targets). We already have locally at
  `data/external/foldbench/`: `foldbench_protein_ligand_rmsd_lddtlp.csv` (has `is_unseen_protein` novelty
  flag + pdb ids + lrmsd -- the LABELS + novelty axis) and `foldbench_protein_ligand_confidence_rmsd.csv`
  (ranking_score only). Use these for the target list, labels, and novelty axis; you still need the INPUT
  structures/sequences to build Protenix inputs.
- Build a Protenix input JSON per target: protein sequence(s) + ligand (CCD code or SMILES) + (optional)
  precomputed MSA. Extract sequence + ligand from the FoldBench input CIFs. MSA: Protenix takes precomputed
  MSAs (`--use_seeds_in_json true`); precompute once on CPU (colabfold search) or use FoldBench-provided if
  present. This input-prep is the fiddly part -- budget time for it.

### 6.3 Run + parse
```bash
# run Protenix inference over the 558 (batch script; ~17-60s each for small, minutes for big; ~10-15 GPU-hr)
# each target emits summary_confidence*.json with: iptm, chain_iptm, chain_pair_iptm, plddt, ptm, ranking_score
# parse chain_pair_iptm -> per-target interface-ipTM (ligand-chain <-> protein-chain entry), like RNP
```
Confirm the JSON schema first on ONE target (we verified a FoldBench Protenix demo JSON HAS
`iptm, chain_iptm, chain_pair_iptm, plddt, ptm, has_clash, ranking_score`).

### 6.4 Pull back + run the feature-matched transfer (on local analysis venv, no GPU)
- `scp` the recovered per-target confidences (small JSON/parquet) back to `~/moml/data/external/foldbench/`.
- Extend `experiments/e15_foldbench_transfer.py` (currently shared-feature `ranking_score` only) or add
  `e15b`: build the combined score on FoldBench using the now-recovered interface-ipTM (+ whatever features
  parity allows), deploy the frozen RNP-calibrated gate, report per-`is_unseen_protein` realized selective
  risk and AURC at feature parity. `src/foldgate/io/foldbench.py` has `load_foldbench_novelty()`.
- Update `results/e15_foldbench_transfer.json` + the paper's cross-dataset section (workshop + journal).
- **Terminate the Lambda instance.**

### 6.5 Honest guardrails for J1
- Feature parity is limited by what Protenix emits vs the RNP combined-score features (interface ipTM
  recovered; PoseBusters/cross-model-agreement may not be reconstructable on FoldBench -- state the parity
  level). The point is to test the combined-score transfer with interface-ipTM RECOVERED, not to claim full
  parity if it is not there.
- The `is_unseen_protein` subset is the novelty axis; report seen vs unseen realized risk.

---

## 7. What remains after J1

- **Extended journal manuscript** (superset of the 4pp workshop: full in-body proofs, full benchmark +
  screening tables, the E21 ChEMBL second task trimmed for space, related work, reproducibility). The user
  paused this mid-start; resume when they ask. `docs/NEXT_STEPS_JOURNAL.md` has the venue strategy
  (Digital Discovery applied + a methods paper).
- **Top-tier ML version** (deferred by the user): the theorem as a standalone AISTATS/ICLR methods paper
  with a general benchmark; the impossibility+achievability bracket is the contribution.
- Submission logistics: `docs/SUBMISSION_GUIDE.md` (arXiv/bioRxiv/Zenodo steps, venue ranking). MoML deadline
  Sept 1 2026 AoE. Zenodo: add ORCID to `.zenodo.json` (currently missing) before tagging v0.1.0.

## 8. Load-bearing gotchas

- Env is uv + `.venv` (py3.12), not pixi. Run Protenix in a SEPARATE env from the analysis venv (libomp
  segfault + version conflicts).
- The synthetic benchmark's default (Dq=0) has near-zero POOLED concept gap; the concept effect shows at the
  worst stratum. b7 (Dq>0) is the one that validates the sharp impossibility (Thm 1c/1d). Keep this straight.
- The clean `1/(n_g+1)` slack is for coverage and joint error ONLY; the ratio selective risk needs
  LTT/QRC `O(sqrt(log(1/delta)/n_g))`. Do not regress this in any writeup.
- Screening: pose-ipTM is the GUARANTEED object; the affinity head is an ungoverned comparator. The 75/13
  selective result is ranked-by-affinity / gated-by-pose -- always disclose that.
- Paper is exactly 4pp main text; any edit can push the Conclusion to p5. Rebuild and check
  `pdftotext -f 4` for the Conclusion before considering it final.
- Confidence field names differ per model; verify against a real file before parsing (AF3 emits no PDE).
