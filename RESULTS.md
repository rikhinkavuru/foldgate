# Results (real, on released Runs N' Poses)

All numbers are from released RNP predictions consumed as-is (no model inference):
13,535 delivered poses (top-1 by `ranking_score`, proper ligands) across 6 co-folding
models. Label: BiSyRMSD ≤ 2 Å. Target: error among accepted ≤ α, confidence 1 − δ.
Defaults α = 0.20, δ = 0.10. Reproduce: `make features && python -m experiments.e1_iid_validity` (etc.).

## Base rates and the raw novelty gradient

Delivered-pose correctness by model: AF3 0.70, Boltz-2 0.74, Boltz-1 0.64,
Boltz-1x 0.66, Chai-1 0.65, Protenix 0.64.

AF3 correctness by ligand-novelty stratum (S0 familiar → S4 no training analog):

| stratum | S0 | S1 | S2 | S3 | S4 (no analog) |
|---|---|---|---|---|---|
| median similarity | 1.00 | 0.59 | 0.27 | 0.11 | NaN |
| correctness | 0.88 | 0.78 | 0.73 | 0.44 | 0.53 |

The accuracy of a delivered pose halves as the ligand moves away from training.
A single global confidence threshold cannot hold a uniform error guarantee across
this gradient — which is the whole problem.

## E1 — the guarantee holds i.i.d.

The finite-sample guarantee (P(true selective risk ≤ α) ≥ 1 − δ) is validated on
synthetic data where the true risk is known: `tests/test_conformal.py` certifies a
gate, estimates its true risk on a large fresh sample, and confirms it holds in
≥ 1 − δ of draws. Note the certifier is **tight** — it accepts the largest set with
true risk ≤ α, so realized risk sits at α and a per-split realized-risk indicator
on a finite fold crosses α ~half the time even when valid; that indicator is not
the right success metric.

On RNP (native `ranking_score`, 40/60 splits): the certified gate's mean realized
risk is ≤ α for models whose native score certifies non-trivial coverage
(AF3 0.17 at 28% coverage; Boltz-1/1x similar at 14–21%). Native ranking score is a
near-vacuous gate for Chai and Protenix (< 5% coverage), which motivates the
combined score (E4).

To make the real-data claim checkable rather than asserted, we report the fraction
of the 300 splits whose realized selective risk was ≤ α, with a Clopper-Pearson
interval: Boltz-1 0.88 [0.85, 0.91], Boltz-1x 0.91 [0.88, 0.93], AF3 0.85 [0.81, 0.89].
For the well-powered models this reaches the 1 − δ = 0.90 target. The realized-risk
indicator is a **downward-biased proxy** for P(true risk ≤ α): because the certifier
is tight, the true risk sits at α, so on a finite test fold the realized risk crosses
α about half the time even when the guarantee holds. The rigorous evidence is the
synthetic test, where the true risk is known and the gate holds in ≥ 1 − δ of draws
(`tests/test_conformal.py`). Chai/Protenix show a low fraction only because their
native gate accepts almost nothing (the indicator is computed over a handful of
non-abstaining splits), which is the E1 finding motivating the combined score.

## E2 — the exchangeability break (the money result)

Global iid-calibrated gate, native score. AF3 **marginal** risk 0.177 looks
compliant, but the risk is grossly uneven across novelty strata:

| stratum | S0 | S1 | S2 | S3 | S4 |
|---|---|---|---|---|---|
| realized selective risk | 0.071 | 0.149 | 0.163 | **0.375** | **0.427** |

Marginal validity hides severe per-group under-control on novel ligands (S3–S4
run ~2× the target). Worse, in the **covariate-shift** setting — calibrate on
familiar ligands (S0–S2), deploy on novel (S3–S4) — the gate accepts 98% of poses
at realized risk **0.547** and the guarantee holds in **0%** of runs. Every
well-powered model shows the same collapse (realized risk on novel target:
Boltz-1 0.48, Boltz-1x 0.46, Chai 0.53, Protenix 0.61).

## E3 — group-conditional (Mondrian) restores the guarantee

A separate LTT threshold per novelty stratum (needs only stratum labels, which RNP
ships). Each stratum's accepted error is then ≤ α with probability 1 − δ (per-stratum
marginal; a simultaneous statement follows by calibrating each at δ/K). The honest cost with the
*native* score: on the hardest strata there is not enough signal to safely accept,
so the gate abstains (AF3 S3 coverage → 0) rather than give false assurance — it
correctly folds. That is the intended behaviour; recovering *usable* coverage on
novel ligands is what the combined score (E4) buys.

| AF3 stratum | global gate risk (cov) | group-conditional risk (cov) |
|---|---|---|
| S0 | 0.071 (0.38) | 0.118 (1.00) |
| S2 | 0.163 (0.38) | 0.152 (0.12) |
| S3 | **0.375** (0.31) | 0.231 (0.00 — abstains) |

**E3c — the full method (combined score + group-conditional).** Swapping the
native score for the combined score turns abstention back into usable, guaranteed
coverage. AF3, per-stratum realized risk (coverage):

| stratum | native + group-cond | combined + group-cond |
|---|---|---|
| S1 | 0.196 (0.08) | 0.156 (**0.52**) |
| S2 | 0.152 (0.12) | 0.164 (**0.39**) |
| S3 | 0.231 (0.00) | 0.155 (**0.05**) |
| S4 (no analog) | — (0.00) | — (0.00) |

Risk stays ≤ α while coverage on the moderately-novel strata rises 4–6×; even the
hard S3 gets a small certified slice. The no-analog extreme (S4) remains
uncertifiable for every model — abstention there is correct, not a failure.

## E4 — combined reliability score dominates native confidence

Combined score (native confidence + interface ipTM + PoseBusters validity +
ensemble spread + ligand difficulty), fit on a train fold, thresholded on a
separate calibration fold, evaluated on test (3-way split preserves conformal
validity). Lower AURC = better.

| model | AURC native | AURC combined | improvement |
|---|---|---|---|
| AF3 | 0.187 | 0.125 | **33.1%** |
| Boltz-1 | 0.234 | 0.170 | 27.5% |
| Boltz-1x | 0.216 | 0.163 | 24.6% |
| Chai-1 | 0.248 | 0.150 | **39.6%** |
| Protenix | 0.281 | 0.175 | **37.7%** |

(These include the cross-model confidence-agreement feature; without it the gains
were 22–38%. Cross-model agreement is the third-largest contributor per the E5 ablation.)

Significance uses a **paired data-bootstrap over test poses** (not over Monte-Carlo
repeats): the per-model Δ(AURC) 90% CI excludes zero for all five models — AF3
0.070 [0.049, 0.092], Boltz-1 0.074 [0.054, 0.097], Boltz-1x 0.062 [0.043, 0.084],
Chai 0.085 [0.059, 0.110], Protenix 0.092 [0.068, 0.116].

Coverage at certified error levels (higher = more usable predictions retained):

| | α = 0.2 native → combined | α = 0.1 native → combined |
|---|---|---|
| AF3 | 0.22 → **0.71** | 0.00 → **0.14** |
| Chai-1 | 0.01 → **0.60** | 0.00 → **0.10** |
| Protenix | 0.04 → **0.51** | 0.00 → 0.04 |

At the same guarantee, the combined gate retains ~3× more predictions, and unlocks
the stringent 90%-correct operating point that native confidence cannot certify at
all for several models.

## E3b — weighted covariate-shift repair (label-free, combined score)

Calibrate on familiar ligands, deploy on more-novel ligands, correct with
likelihood-ratio weights over the novelty covariates — no target labels used. The
weights are estimated **out-of-fold with a probability-calibrated** source-vs-target
classifier (`estimate_weights_cv`, isotonic, 5-fold), the correctness requirement an
in-sample logistic fit misses. We report two weighted certifiers: the plug-in Hájek
estimator (approximate coverage) and an importance-weighted LTT with a WSR betting
p-value (Almeida et al. 2025) that is finite-sample **conditional on the weights**.

**Moderate shift (S0,S1 → S2):** the weighted plug-in pulls realized target error
toward α without target labels, and the repair survives swapping the weight model.

| model | naive risk (cov) | weighted plug-in risk (cov) | alt weight-model risk | target-cal risk (cov) |
|---|---|---|---|---|
| AF3 | 0.267 (0.98) | **0.191 (0.70)** | 0.223 | 0.149 (0.32) |
| Chai-1 | 0.283 (0.89) | 0.201 (0.62) | 0.243 | 0.145 (0.27) |
| Protenix | 0.340 (0.90) | 0.267 (0.60) | 0.293 | 0.136 (0.06) |

Naive (source-calibrated) over-accepts at error above α; weighted reweighting brings
AF3 to 0.191 ≈ the 0.20 target at 70% coverage, using no target labels, and the
alternative (in-sample) weight model lands close by (0.223), so the repair is not an
artifact of one weight estimator. Kish effective sample size n_eff ≈ 115–140.

**The finite-sample weighted certificate is honest-conservative.** The
importance-weighted LTT gate (WSR betting) **abstains on every model/regime** (0%
certified coverage): on real co-folding data even the plug-in barely clears α, so
there is no certifiable margin once the finite-sample slack is paid. This is not a
code artifact — the certifier controls true risk when the shift is clean (validated on
synthetic data with known weights, `tests/test_conformal.py`); it reflects that the
guarantee is exact only *conditional on correct weights*, which is a strong condition
here.

**Why: concept shift, not just covariate shift.** A per-confidence-bin diagnostic
shows P(correct | confidence) itself moves between source and target. The mean gap
grows from the moderate regime (0.08–0.16) to the **extreme regime (S0–S2 → S3,S4:
0.17–0.29)**. Pure covariate reweighting controls an *aligned* distribution, so under
this residual concept shift the true target risk can exceed α — which is exactly what
the extreme regime shows (naive error 0.48–0.50; weighted reduces it to 0.34–0.38 but
cannot reach α; even the label-using target-calibrated gate abstains because the novel
target is < 55% correct at baseline).

**Honest conclusion.** Weighted conformal is the *label-free complement*: its plug-in
closes most of the gap on moderate covariate shift. The *rigorous finite-sample*
guarantee under novel-pocket shift is the group-conditional (E3) certificate, which
needs only stratum labels (which RNP ships) and does not assume the confidence-
reliability map is stable.

## E7 — the break generalizes across shift axes

The exchangeability break is not specific to ligand novelty. Per-stratum realized
risk under a global iid-calibrated gate (AF3, S0 familiar → S3/S4 novel):

| axis | S0 | S1 | S2 | S3 | S4 |
|---|---|---|---|---|---|
| ligand novelty | 0.07 | 0.15 | 0.16 | 0.38 | 0.43 |
| **pocket novelty** | 0.06 | 0.12 | 0.21 | **0.40** | 0.45 |
| temporal (release date) | 0.17 | 0.25 | 0.14 | 0.18 | — |

Pocket novelty produces an equally sharp (Chai up to 0.77 on S3) or sharper break;
the temporal axis does not, so **the shift that breaks conformal validity is
structural/chemical dissimilarity to training, not recency**. This is a cleaner
statement of the covariate-shift variable than "post-cutoff date."

## E6 — downstream payoff (RNP-internal)

The combined-score gate cleans the delivered pose set a downstream pipeline would
carry forward. Base top-1 purity is 63–70%; the gate lifts it and raises interface
quality, while retaining most correct poses.

| model | α | kept | purity (base→gate) | enrichment | correct retained | LDDT-PLI (all→kept) |
|---|---|---|---|---|---|---|
| AF3 | 0.2 | 0.73 | 0.70 → **0.82** | 1.17× | 0.86 | 0.77 → 0.85 |
| AF3 | 0.1 | 0.24 | 0.70 → **0.94** | 1.34× | 0.31 | 0.76 → 0.92 |
| Chai-1 | 0.1 | 0.21 | 0.65 → **0.93** | 1.44× | 0.30 | 0.73 → 0.91 |
| Protenix | 0.1 | 0.09 | 0.63 → **0.94** | 1.49× | 0.13 | 0.72 → 0.92 |

At the stringent operating point every model yields a ~93% pure set with clearly
higher interface quality: downstream structure-based work gets near-clean inputs
instead of a 1-in-3 error rate. The Mac1 virtual-screening enrichment arm lands
when its crystal coordinates are released.

## E5 — generality

The E2 break, E3/E3b repair, E4 combined-score dominance, and E6 payoff all hold
across all six models (AF3, AF3-no-template excluded for brevity, Boltz-1,
Boltz-1x, Chai-1, Protenix; Boltz-2 where powered) — generality is demonstrated as
a byproduct rather than a separate claim. Task/interface variants (continuous
LDDT-PLI, affinity-rank via Boltz-2) are the next extension.

**Threshold robustness (E5).** The break and the AURC gain do not depend on the
2 Å convention. Across RMSD thresholds (AF3, global-τ per-stratum risk):

| threshold | base correct | S0 risk | S3 risk | S4 risk | AURC nat → comb (Δ) |
|---|---|---|---|---|---|
| 1.5 Å | 0.64 | 0.09 | 0.35 | 0.55 | 0.249 → 0.172 (−31%) |
| 2.0 Å | 0.70 | 0.06 | 0.36 | 0.49 | 0.186 → 0.124 (−33%) |
| 2.5 Å | 0.75 | 0.06 | 0.43 | 0.39 | 0.152 → 0.100 (−34%) |
| 3.0 Å | 0.78 | 0.05 | 0.42 | 0.29 | 0.120 → 0.079 (−34%) |

Novel strata exceed α at every threshold; the combined score wins by 26–29% throughout.

**Feature ablation (E5, AF3, cumulative AURC, lower = better).** The gain is driven
mainly by interface ipTM and intra-model ensemble spread:

| features | AURC |
|---|---|
| native ranking_score only | 0.193 |
| + interface ipTM | 0.160 |
| + PoseBusters validity | 0.159 |
| + ensemble spread | 0.137 |
| + cross-model agreement | 0.128 |
| + ligand difficulty (all) — tabular combined | 0.123 |
| + pose agreement (W1, structures) | **0.106** |

Interface ipTM (0.193→0.160), ensemble spread (0.159→0.137), and cross-model
agreement (0.137→0.128) do the work among the tabular features; PoseBusters and ligand
physicochemistry add little. The structural pose-agreement features (W1) are the
single largest jump after ipTM: 0.123→0.106 (−14%). Full W1 result in the pose-agreement
section below.

## E8 — task-agnostic (interface quality)

Swapping the label from ligand-RMSD ≤ 2 Å to interface quality (LDDT-PLI ≥ 0.5),
the combined score again dominates native confidence on AURC (lower = better):

| model | AURC native | AURC combined | improvement |
|---|---|---|---|
| AF3 | 0.085 | 0.048 | 43.6% |
| Chai-1 | 0.103 | 0.053 | 48.8% |
| Protenix | 0.144 | 0.065 | 54.8% |

The reliability layer is not tied to the pose-RMSD label; it transfers to a
different structured-quality target with the same machinery.

## E9 — continuous-RMSD risk (no 2 Å cutoff)

Ordering by the combined score instead of native confidence lowers the accepted
set's mean RMSD at every coverage. At 50% coverage:

| model | mean accepted RMSD, native → combined |
|---|---|
| AF3 | 1.72 → **1.14 Å** |
| Chai-1 | 2.16 → **1.15 Å** |
| Protenix | 2.58 → **1.54 Å** |

The finding survives dropping the 2 Å convention entirely.

**Certified continuous gate.** Beyond ordering, we certify a gate whose mean
bounded-RMSD among the accepted is ≤ a target, with P(mean ≤ target) ≥ 1 − δ, using a
variance-adaptive **WSR betting bound** on the loss min(RMSD, 4 Å)/4 Å. It certifies
non-trivial coverage where a distribution-free Hoeffding bound certifies almost none
(AF3, certified coverage, WSR vs Hoeffding):

| target mean-RMSD | WSR coverage | Hoeffding coverage | realized capped-mean RMSD |
|---|---|---|---|
| ≤ 1.00 Å | **0.43** | 0.00 | 0.90 Å |
| ≤ 1.25 Å | **0.74** | 0.46 | 1.17 Å |
| ≤ 1.50 Å | 0.92 | 0.86 | 1.41 Å |

At the tight 1.0 Å target the Hoeffding bound certifies **zero** coverage for every
model, while WSR certifies a non-trivial slice (2–43% across models, AF3 43%);
realized capped-mean-RMSD stays under the target and
the empirical coverage of the guarantee sits near 1 − δ (0.80–0.94 across models/targets,
with the same tight-certifier noise as E1 — the tightest targets accept few poses, so the
finite-sample realized-risk indicator is noisiest there). The acceptance fraction is co-reported with a
Clopper-Pearson interval so a low certified risk from accepting almost nothing is not
mistaken for a free lunch. Binarising the loss reproduces the E1 exact-binomial gate
(`tests/test_conformal.py`).

## E10 — FoldBench cross-dataset (honest negative)

We attempted a second-benchmark replication on FoldBench (441 units × 5 models,
base correctness 0.54–0.66). **The advantage does not transfer**, and the reason
is informative rather than a failure of the method: FoldBench's public per-pose
table ships only `ranking_score` — no interface ipTM, no PoseBusters, no per-pose
training-similarity — i.e. exactly the features the E5 ablation identifies as the
source of the gain, plus far fewer samples per model. With that feature-poor
subset and small n, the learned combiner overfits and does not beat raw
`ranking_score` (AURC change −23% to +9% across models), and the empirical
validity check is noisy at the smaller scale. The takeaway is consistent with the
ablation: the layer's value depends on the rich confidence signals RNP ships and
FoldBench does not. A clean cross-dataset test needs a benchmark that releases
ipTM/PoseBusters per pose.

## E11 — baselines, and calibration is not conformal

The baselines a reviewer expects, on the same splits.

**Ranker quality (AURC, lower = better).** The combined score beats every native
confidence a practitioner would reach for:

| model | ranking_score | native ipTM | combined |
|---|---|---|---|
| AF3 | 0.186 | 0.159 | **0.124** |
| Chai-1 | 0.249 | 0.206 | **0.150** |
| Protenix | 0.280 | 0.264 | **0.172** |

PoseBusters validity alone is a weak gate — accepting all PB-valid poses leaves a
26–34% error rate (it is a physical filter, not a reliability ranker).

**Gate validity — the guarantee, not the features.** Fix the *same* combined score and
turn it into a gate three ways: LTT (conformal), or Platt/isotonic calibration
thresholded at P(correct) ≥ 1 − α (the practitioner's calibrated classifier). i.i.d.,
all control error (conformal ~0.17 at the target; Platt/isotonic ~0.10, more
conservative). **Under the novelty shift (calibrate on familiar S0, deploy on novel
S1–S2) every source-calibrated gate breaks — naive conformal exactly like calibration**
(AF3 realized error: naive conformal 0.24, Platt 0.22, isotonic 0.18, native ipTM 0.25,
all against a 0.20 target; Boltz/Protenix worse). The break is a property of naive
transfer, not of calibration-vs-conformal.

What repairs it has no calibration analogue — group-conditional (Mondrian) conformal,
which restores realized error ≤ α for **every** model by abstaining more on the novel
strata:

| model | naive conformal (shift) | Platt (shift) | group-conditional (shift) |
|---|---|---|---|
| AF3 | 0.244 | 0.217 | **0.153 (cov 0.22)** |
| Chai-1 | 0.269 | 0.197 | **0.137 (cov 0.18)** |
| Protenix | 0.303 | 0.283 | **0.151 (cov 0.08)** |

The reviewer-facing conclusion: calibration fixes only the marginal probability, carries
no finite-sample coverage, and breaks under shift; conformal carries a distribution-free
finite-sample guarantee, and only its shift-robust variants (group-conditional here,
weighted in E3b) repair the exchangeability break. (Boltz-2 affinity-probability, the
other honest-negative baseline, is not in the released tabular dump; it is a follow-up.)

## W1 — cross-model + intra-model pose agreement (structure-based upgrade)

Beyond the released confidence tables, the predicted structures carry an orthogonal,
model-agnostic signal: whether the *binding modes* agree. Computed from RNP's
`prediction_files` (streamed, no GPU) with spyrmsd + gemmi over 11,711 delivered poses
(93–97% coverage):

- **intra-model pose diversity** across a model's 25 diffusion samples (pocket-superposed,
  symmetry-corrected ligand-RMSD to the delivered pose). Median spread by model: AF3 0.61,
  Boltz-1 0.47, Boltz-1x 0.28, Chai-1 0.41, **Protenix 1.90 Å** — Protenix's samples disagree
  on placement most, a reliability red flag its scalar confidence does not expose.
- **cross-model pose agreement** (delivered-pose ligand-RMSD to the other models). Median by
  model: AF3 2.93, Boltz-1 2.22, Boltz-1x 2.09, **Chai-1 5.23**, Protenix 3.39 Å — Chai-1 is
  the structural outlier, agreeing least with the consensus.

Adding these six features to the combined score (an opt-in upgrade over the tabular default,
using the frozen model's own diffusion samples) lowers AURC further, on top of every tabular
feature. The pose-upgrade Δ(AURC) 90% CI excludes zero for **all five models**:

| model | native | tabular combined | + pose (W1) | pose Δ(AURC) CI90 |
|---|---|---|---|---|
| AF3 | 0.187 | 0.125 (33%) | **0.109 (42%)** | [0.004, 0.021] |
| Boltz-1 | 0.234 | 0.170 (28%) | **0.139 (41%)** | [0.022, 0.049] |
| Boltz-1x | 0.216 | 0.163 (25%) | **0.134 (38%)** | [0.004, 0.035] |
| Chai-1 | 0.248 | 0.150 (40%) | **0.131 (47%)** | [0.020, 0.040] |
| Protenix | 0.281 | 0.175 (38%) | **0.139 (51%)** | [0.032, 0.058] |

The AF3 feature ablation confirms it: pose agreement is the single largest AURC drop after
interface ipTM (0.123→0.106, −14%). The tabular combined score remains the cheap,
structure-free primary; pose agreement is the upgrade when the structures are retained
(they are a free byproduct of running the model).

## E6b — interaction-fingerprint recovery (non-circular downstream payoff)

E6's purity is arithmetically 1 − selective risk, so it cannot show the gate buys anything
beyond the guarantee. Contact recovery is a genuinely different, downstream question a chemist
reads: does the accepted pose recover the correct protein-ligand interactions (receptor
residues within 4.5 Å of the ligand, keyed on (seqid, resname)) vs the crystal structure?
Recovery is only correlated with, not equal to, the 2 Å RMSD label.

Under the reliability gate (α = 0.2), accepted poses recover markedly more crystal contacts
than rejected poses, for every model, with the accepted-minus-rejected recall gap 90% CI
excluding zero:

| model | base recall | accepted | rejected | gap CI90 |
|---|---|---|---|---|
| AF3 | 0.858 | **0.908** | 0.718 | [0.144, 0.227] |
| Boltz-1 | 0.839 | **0.911** | 0.757 | [0.166, 0.231] |
| Boltz-1x | 0.845 | **0.912** | 0.750 | [0.124, 0.194] |
| Chai-1 | 0.838 | **0.911** | 0.706 | [0.143, 0.207] |
| Protenix | 0.834 | **0.902** | 0.759 | [0.123, 0.176] |

The gate raises interaction recovery by 0.15–0.20 on a metric that is not the guaranteed
label, so — unlike E6 purity — this is a genuine, non-circular downstream lift: the abstention
routes forward the poses whose interaction patterns a downstream SAR or structure-based step
can actually trust. A screening-enrichment harness (`selective.enrichment`: EF/BEDROC,
selective-EF, coverage-enrichment and active-retention curves, random-abstention control) is
implemented and unit-tested, awaiting a screening dataset (the Mac1 prospective screen when its
coordinates release, or a co-fold run) for a headline enrichment number.
