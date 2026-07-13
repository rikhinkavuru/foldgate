# BENCHMARK SPEC  (Phase 0 grounding, 2026-07-12)

I have what I need to ground the spec. Here is the full benchmark specification.

---

# BENCHMARK SPEC: Selective Risk Control Under Concept Shift (general, torch-free)

## 0. Purpose and mapping to the theorem

Co-folding (RNP) stays the flagship application. This benchmark exists to show the three theoretical claims are dataset-general, not artifacts of one pose dataset:

- **(T1) Impossibility.** A single marginal accept/abstain threshold cannot control per-stratum selective risk once the confidence's reliability degrades on novel strata (concept shift). There is a lower bound on worst-stratum excess risk that grows with the concept-shift magnitude `D`.
- **(T2) Mondrian achievability.** Group-conditional (Mondrian) calibration keyed on the novelty coordinate restores per-stratum risk control, up to a finite-sample slack that shrinks with per-stratum count `n_k`.
- **(T3) CVaR/DRO certificate.** A distributionally-robust certificate (worst-stratum RCPS UCB, and an f-divergence / CVaR ball over strata) upper-bounds realized worst-case selective risk, and is valid whenever the ambiguity radius covers the true shift.

The synthetic generator (Section 2) has a **known closed-form `P(Y=1 | s, nu)`**, so T1's lower bound, T2's achievability, and T3's certificate can each be checked against ground truth. The real datasets (Section 3) show the same qualitative separation holds on public data where the conditional is unknown.

Everything below is numpy / pandas / scikit-learn / scipy only, CPU, and runs from `/Users/rikhinkavuru/moml/.venv`. No torch anywhere.

---

## 1. Common problem formalization (shared by synthetic and real)

Every example `i` reduces to a triple `(s_i, y_i, nu_i)`:

- `s_i ∈ [0,1]` = a base classifier's **confidence** (its self-reported probability that its own answer is right). This is the analog of a co-folding native confidence.
- `y_i ∈ {0,1}` = the **selective label**, `y=1` means the base answer is CORRECT / acceptable. In co-folding, `y = 1[ligand-RMSD ≤ 2 Å]`. On a real tabular classifier, `y = 1[argmax f(x) == true label]`.
- `nu_i` = the **novelty / shift coordinate** (scalar or categorical). In co-folding it is training-set similarity. Here it is state, time block, or a domain-distance score.

**Gate.** Accept iff `s ≥ τ`. Selective risk at threshold `τ` on a population `P` is
`R_P(τ) = P(y = 0 | s ≥ τ)` (the error rate among accepted).
Target: choose `τ` so `R(τ) ≤ α` with confidence `1−δ`. Strata `k = 1..K` are bins/values of `nu`; per-stratum risk is `R_k(τ) = P(y=0 | s≥τ, nu ∈ k)`.

**Four methods compared everywhere:**
1. `MARGINAL` (split conformal / RCPS): one `τ̂` from pooled calibration to hit risk `α` marginally.
2. `MONDRIAN` (group-conditional): one `τ̂_k` per stratum from that stratum's calibration scores.
3. `WEIGHTED` (covariate-shift CP): importance weights `ŵ ∝ p_tgt(nu)/p_cal(nu)` on calibration scores (weights known exactly on synthetic, estimated by a cal-vs-test classifier on real).
4. `DRO/CVaR` (certificate, T3): worst-stratum RCPS upper-confidence bound, plus an f-divergence-ball certificate over strata.

`WEIGHTED` vs `MONDRIAN` is the load-bearing contrast: reweighting repairs covariate shift, only conditioning repairs concept shift.

---

## 2. Synthetic generator (known `P(Y=1 | s, nu)`)

### 2.1 Parametric form (discrete strata, everything closed-form)

Choose `K` strata with novelty levels `ν_k = (k−1)/(K−1) ∈ [0,1]` (`ν_1 = 0` seen, `ν_K = 1` most novel). Per example:

1. **Stratum draw (nu-marginal, the covariate axis at stratum level).**
   `p_cal(k) ∝ exp(−c_cal · ν_k)`, `p_tgt(k) ∝ exp(−c_tgt · ν_k)`.
   Large `c_cal` concentrates calibration on seen strata; small/negative `c_tgt` puts more test mass on novel strata. **Tilt knob** `T := KL(p_tgt ‖ p_cal)` (closed form over the K-simplex).

2. **Confidence draw (covariate axis in score space).**
   `z_i := logit(s_i) = m0 − E·ν_{k_i} + ε_i`, `ε_i ~ N(0, σ_s²)`, so `s_i = sigmoid(z_i)`.
   **Covariate-shift-in-score knob** `E`: `E=0` makes `P(s | nu)` identical across strata; `E>0` makes novel strata systematically lower-confidence with calibration intact.

3. **Label draw (concept axis).**
   `y_i ~ Bernoulli( π(z_i, ν_{k_i}) )`, `π(z, ν) = sigmoid(β0 + β1·z − D·ν)`.
   **Concept-shift knob** `D`. Defaults `β0 = 0, β1 = 1` so that **when `D = 0`, `π = sigmoid(z) = s` exactly** — the confidence is a perfectly calibrated correctness probability, identical across strata. With `D>0`, on novel strata the true correctness probability sits below the reported confidence by `D·ν` in logit units: the confidence over-states correctness exactly where it matters. This is concept shift: `P(y | s)` is no longer stable across `nu`.

Default constants: `σ_s = 1.3`, `m0 = 0.6` (base accept rate near 0.6), `β0=0`, `β1=1`.

### 2.2 The three knobs, isolated

| Knob | Meaning | Isolate covariate-only | Isolate concept-only |
|---|---|---|---|
| `D` | concept-shift slope (reliability decay) | `D = 0` | `D > 0` |
| `E` | covariate shift in score dist given stratum | sweep `E` | `E = 0` |
| `T` | cal-vs-test tilt over the nu-marginal | sweep `c_tgt` | keep `p_cal = p_tgt` (`T=0`) |

- **Pure covariate shift:** `D = 0`, sweep `E ∈ {0,1,2}` and/or `T` via `c_tgt`. Expected: marginal conformal on `s` stays conditionally valid because `P(y|s)` is truly stable; `WEIGHTED` corrects any marginal miscoverage; Mondrian is also fine (harmless).
- **Pure concept shift:** `D ∈ {0.5,1,2,4}`, `E = 0`, `T > 0`. Expected: `MARGINAL` under-covers the novel strata; `WEIGHTED` fixes the marginal number but NOT the worst stratum; only `MONDRIAN` controls every stratum.

### 2.3 Ground-truth quantities (closed / oracle)

Let `t = logit(τ)`, `μ_k = m0 − E·ν_k`. Then per stratum, in closed 1-D-integral form:

- Accept rate: `P(s ≥ τ | k) = 1 − Φ((t − μ_k)/σ_s)`.
- Selective risk:
  `R_k(τ) = [ ∫_t^∞ (1 − π(z, ν_k)) · (1/σ_s) φ((z−μ_k)/σ_s) dz ] / [ 1 − Φ((t−μ_k)/σ_s) ]`,
  evaluated on a fine `z`-grid (treat as ground truth; also cross-check with an `N_oracle = 5×10⁶` Monte-Carlo draw).
- Oracle per-stratum threshold `τ_k* = min{ τ : R_k(τ) ≤ α }` by root-find.
- Mixture risk under any nu-distribution `p`:
  `R_mix(τ; p) = Σ_k p(k) P(s≥τ|k) R_k(τ) / Σ_k p(k) P(s≥τ|k)`.

Because these are exact, the theorem's bounds have a ground-truth reference.

### 2.4 Exact quantities to measure

- **(T1 check) Impossibility gap.** Let `τ̂_∞` be the population marginal threshold solving `R_mix(τ; p_cal) = α`. Define the analytic gap `Δ(D,T) = max_k R_k(τ̂_∞) − α`. Plot empirical `max_k R_k(τ̂)` from finite-sample MARGINAL against `Δ(D,T)`, over the `D` and `T` sweeps. **Verify:** `Δ` increases monotonically in `D` and in `T`, and `Δ → 0` as `D → 0`; finite-sample worst risk tracks `Δ` and stays `≫ α`.
- **(T2 check) Mondrian achievability + finite-sample slack.** MONDRIAN realized `max_k R_k(τ̂_k)`. **Verify:** stays `≤ α + O(1/√n_k)`; the slack shrinks as `n_cal` grows; and **breaks in the thin-strata regime** (large `K`, small `n_k`) — report this honestly per CLAUDE.md rule 5.
- **(covariate vs concept separation) WEIGHTED.** Realized marginal risk and worst-stratum risk. **Verify:** with `D=0` both are controlled; with `D>0` marginal is controlled but worst-stratum is not.
- **(T3 check) Certificates.**
  - Worst-stratum RCPS: `U = max_k UCB_δ(R_k(τ_k))` via Hoeffding–Bentkus on the `n_k` accepted calibration points. Measure certificate **validity rate** `P(max_k R_k(τ_k) ≤ U) ≥ 1−δ` over seeds and **slack** `U − max_k R_k`.
  - f-divergence / CVaR ball: certified worst risk `sup_{q: D(q‖p_cal) ≤ ρ} R_mix(τ; q)` (convex program over the K-simplex; equivalently a CVaR_η dual with `η` tied to `ρ`). **Verify:** certificate `≥ R_mix(τ; p_tgt)` exactly when `ρ ≥ KL(p_tgt‖p_cal)`; track slack vs `ρ`.
- **(utility, E4-analog) AURC / risk–coverage.** MONDRIAN gate vs MARGINAL gate at matched marginal coverage; report AURC with bootstrap CIs.

### 2.5 Sweeps and sample sizes

- `α ∈ {0.05, 0.10, 0.20}`; `δ = 0.10`.
- `D ∈ {0, 0.5, 1, 2, 4}`; `E ∈ {0, 1, 2}`; tilt via `c_cal = 3`, `c_tgt ∈ {3 (T=0), 1, 0, −1}`.
- `K ∈ {4, 8, 16}` (thin-strata stress on the largest `K`).
- `n_cal ∈ {200, 500, 1000, 2000, 5000, 10000}` (convergence curve); `n_test = 50000`.
- `N_oracle = 5×10⁶` for ground-truth integrals cross-check.
- **Repeats:** 300 seeds per config; report mean and 95% bootstrap CIs. One global RNG helper (numpy `Generator` + `PYTHONHASHSEED`), git SHA logged per run, matching the repo's reproducibility convention.
- Runtime target: full sweep under a few minutes on one CPU core (pure numpy/scipy).

*(Continuous-nu variant, optional: draw `nu ~ Beta(a,b)`, keep the same `π(z,ν)`, and bin nu into K quantile strata for Mondrian. Discrete strata is the primary design because it makes every theorem quantity exactly analytic and makes the thin-strata knob a clean `E[n_k] = n_cal · p_cal(k)`.)*

---

## 3. Real public tabular datasets (2 primary + 1 optional)

Same `(s, y, nu)` reduction. Base classifier `f` = LightGBM or logistic regression (both torch-free) trained on the SOURCE domain only; `s =` predicted probability of the predicted class (max-softmax confidence); `y = 1[f correct]`; `nu =` the natural shift coordinate. Leakage-free means the split unit is the natural group, and calibration rows for a stratum come from that stratum and are disjoint from both training and test rows.

### 3.1 PRIMARY — Folktables ACSIncome (spatial + temporal shift; documented decomposition)

- **What:** US Census ACS PUMS income prediction (`income ≥ $50k`), 10 features, hundreds of thousands of rows per state-year. Ding et al. (NeurIPS 2021) established it as a distribution-shift benchmark across states and years.
- **`s`:** `f`'s predicted P(correct-class); **`y`:** `1[f's prediction correct]`; **`nu`:** target **state** (spatial) or **survey year** (temporal). For a continuous novelty axis, use a source-vs-target domain-classifier score, or rank strata by the DISDE Y|X-shift magnitude from WhyShift (Section 3.3).
- **Leakage-free grouped split:** train `f` on source-state rows only (e.g. CA 2018). Calibration = a held-out labeled slice drawn from each stratum you will certify (for Mondrian) plus a source-domain slice (for Marginal), all disjoint from test. Test = target-state rows (e.g. SD, PR, WY, MS). ACS PUMS rows are distinct persons; split by state (and year), never shuffle individuals across splits.
- **Why it supports the decomposition:** WhyShift/DISDE attributes each source→target degradation to Y|X (concept) vs X (covariate). Pick a covariate-dominated pair and a concept-dominated pair, then show WEIGHTED repairs the former while only MONDRIAN repairs the latter. This is the flagship real-data result.
- **Loader / license:**
  ```python
  from folktables import ACSDataSource, ACSIncome
  ds = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
  df = ds.get_data(states=['CA'], download=True)     # cached after first fetch
  X, y_true, group = ACSIncome.df_to_pandas(df)
  ```
  `pip install folktables`; package **MIT**; underlying ACS PUMS is US Census public microdata (Census terms of use). Dependencies numpy/pandas/scikit-learn only, **torch-free**. `download=True` caches locally; predownloaded CSVs make it fully offline.

### 3.2 PRIMARY — Electricity / elec2 (documented temporal concept drift, fully offline-able)

- **What:** Australian (NSW) electricity market, 45,312 rows, 8 features, binary target (price UP vs DOWN), ordered in time. This is the canonical concept-drift benchmark: `P(Y|X)` drifts as the market expands over the two-year recording window.
- **`s`:** `f`'s predicted P(class); **`y`:** `1[f correct]`; **`nu`:** **time block** (period index). Strata = contiguous temporal blocks. Temporal drift is the "novel chemotype" analog.
- **Leakage-free split:** temporal only. Train `f` on block 0; calibrate on a held-out slice of each later block (Mondrian) or on the block-0 tail (Marginal); test on later blocks. **Never shuffle across time** — random shuffling of elec2 is a known evaluation pitfall that destroys the drift and inflates accuracy; respect the ordering.
- **Loader / license:**
  ```python
  from sklearn.datasets import fetch_openml
  d = fetch_openml(name='electricity', version=1, as_frame=True)   # OpenML data_id 151
  ```
  Fetched via scikit-learn's OpenML client, cached to `~/scikit_learn_data` (save to disk for full offline use). **Torch-free.** License: confirm the exact field on `openml.org/d/151` before redistributing — I could not render that page this session, so treat the license line as UNVERIFIED and check it (OpenML commonly lists this set as freely usable; the `name='electricity', version=1` route is robust to the id).

### 3.3 OPTIONAL — WhyShift task (built-in covariate-vs-concept attribution; second task)

- **What:** `namkoong-lab/whyshift` ships five shift benchmarks (ACS Income, ACS Public Coverage, ACS Mobility, Taxi, US-Accident) plus the **DISDE** method that decomposes each source→target degradation into Y|X (concept) and X (covariate) parts. Using it grounds the benchmark's decomposition claim in an external, published attribution rather than only our synthetic labels. Public Coverage or US-Accident gives a non-income second task for a generality check across tasks.
- **`s`/`y`/`nu`:** same reduction; `nu` = target state; use the DISDE Y|X-share to label each stratum as covariate- vs concept-dominated.
- **Loader / license:**
  ```python
  from whyshift import get_data
  X, y = get_data('pubcov', 'CA')      # tasks: 'income','pubcov','mobility','accident','taxi'
  ```
  `pip install whyshift` (or GitHub `namkoong-lab/whyshift`); **MIT**; dependencies numpy/pandas/scikit-learn/xgboost/lightgbm, **no torch**. ACS tasks auto-download via folktables; Taxi and US-Accident need a one-time Kaggle download (CC BY-SA 4.0), so prefer the ACS tasks for a frictionless offline run.

**Coverage of the decomposition requirement:** ACSIncome (via folktables or WhyShift) and WhyShift Pub-Coverage give datasets with *documented, method-attributed* covariate-vs-concept shift. Electricity gives a clean, fully-offline temporal concept-drift set. Together that is minimal but principled: one exact synthetic checker plus two-to-three real sets spanning spatial, temporal, and task shift.

---

## 4. Protocol, metrics, deliverables

**Headline metrics** (reported for all four methods, on synthetic and real):
- Realized selective risk: marginal, per-stratum, and **worst-stratum** (the money number) vs target `α`.
- Coverage/risk gap `R − α` per stratum; fraction of strata controlled.
- Certificate validity rate (`≥ 1−δ`) and certificate slack (T3).
- AURC / risk–coverage of MONDRIAN vs MARGINAL at matched coverage, with bootstrap CIs.
- Synthetic-only: empirical worst-stratum risk vs the analytic lower bound `Δ(D,T)`; Mondrian slack vs `n_k`; thin-strata breakdown.

**Statistical protocol:** 300 seeds (synthetic) / stratified resamples (real); 95% bootstrap CIs on every reported number (AURC estimators have known pitfalls, so CIs are mandatory). Log resolved config + git SHA per run.

**Suggested repo layout** (mirrors existing `experiments/eN_*.py` convention, all torch-free, all runnable from `.venv/bin/python`):
```
src/foldgate/bench/
  synth.py        # generator: draw_stratum, draw_score, draw_label; closed-form R_k, tau_k*, R_mix
  certificates.py # RCPS worst-stratum UCB (Hoeffding-Bentkus); f-divergence/CVaR ball over strata
  realdata.py     # loaders: acs_income(), electricity(), whyshift_task(); grouped leakage-free splitter
experiments/
  b1_synth_impossibility.py   # T1: worst-stratum risk vs D, T  (vs analytic Delta)
  b2_synth_mondrian.py        # T2: achievability + n_k slack + thin-strata breakdown
  b3_synth_weighted_vs_cond.py# covariate-only (D=0) vs concept-only (D>0) separation
  b4_synth_certificate.py     # T3: certificate validity + slack vs rho
  b5_real_acs.py              # ACSIncome spatial/temporal, DISDE-labelled strata
  b6_real_electricity.py      # elec2 temporal concept drift
  b7_real_whyshift_pubcov.py  # optional second task
results/figures/bench/        # committed small PNGs
```

**Runtime budget:** synthetic full sweep + all certificates under a few CPU-minutes; real sets bounded by the base-classifier fit (LightGBM on ~10⁵ rows, seconds to low-minutes). No GPU, no torch, no compiled comp-bio stack needed.

---

## Verification status of grounded claims

- Verified via web: folktables loader API, MIT license, torch-free deps, and its role as the Ding-et-al. distribution-shift benchmark; WhyShift's five tasks, DISDE Y|X-vs-X decomposition, `get_data` API, MIT license, torch-free deps; elec2's 45,312×8 shape, binary price target, and canonical concept-drift status; the general finding that concept shift on `P(Y|X)` dominates in tabular data (TableShift, Gardner et al. 2023) which motivates using tabular sets for a concept-shift benchmark.
- **UNVERIFIED (flag):** the exact OpenML license field for electricity `d/151` — the OpenML page did not render this session; confirm on `openml.org/d/151` before redistributing. The `data_id=151` / `name='electricity', version=1` route itself is standard; use the name-keyed form for robustness.

Sources:
- [socialfoundations/folktables (GitHub)](https://github.com/socialfoundations/folktables)
- [namkoong-lab/whyshift (GitHub)](https://github.com/namkoong-lab/whyshift)
- [sklearn.datasets.fetch_openml docs](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_openml.html)
- [ELEC2 dataset reference (CRAN dynaTree)](https://search.r-project.org/CRAN/refmans/dynaTree/html/elec2.html)
- ["How good is the Electricity benchmark for evaluating concept drift adaptation" (ResearchGate)](https://www.researchgate.net/publication/234131070_How_good_is_the_Electricity_benchmark_for_evaluating_concept_driftadaptation)
- [Benchmarking Distribution Shift in Tabular Data with TableShift (NeurIPS 2023)](https://proceedings.neurips.cc/paper_files/paper/2023/file/a76a757ed479a1e6a5f8134bea492f83-Paper-Datasets_and_Benchmarks.pdf)
- [Exploring Covariate and Concept Shift for OOD Detection (arXiv 2110.15231)](https://arxiv.org/abs/2110.15231)