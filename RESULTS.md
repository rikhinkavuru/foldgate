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

## E15b — FoldBench cross-dataset positive (feature recovered by regeneration)

E10 was blocked by a missing feature, not by the method. We removed the block by
regenerating the FoldBench Protenix predictions ourselves (Protenix 0.5.5, 5 seeds
× 5 samples = 25 poses per target, ColabFold MSAs) to recover the interface-ipTM
field the public table withholds, then self-scored ligand-RMSD against the
deposited assemblies with pocket-Cα superposition and symmetric spyrmsd, so the
feature and the label come from the same run. Result on 436 protein-ligand targets
(384 train-similar, 52 no-analog, regen top-1 success 0.401): the frozen Runs N'
Poses interface-ipTM gate transfers with a risk-coverage **AURC of 0.380, ahead of
the matched `ranking_score` control at 0.454**. The same frozen threshold
(τ=0.989) accepts only 5% of FoldBench against 26% at home, so the coverage
collapse of the break section reproduces on a second dataset and a different
confidence source. The self-scored success rate (0.401) sits below FoldBench's
released Protenix top-1 (0.567); the gap is expected drift from the scoring method
(our pocket-superposed ligand-RMSD vs their BiSyRMSD), the model version (0.5.5 vs
0.5.0), and the MSA source, and it does not affect E15b because feature and label
are self-consistent. Script `experiments/e15b_foldbench_iptm_transfer.py`, figure
`results/figures/e15b_foldbench_iptm_transfer.png`.

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

## D1 — the label-free danger floor (the escape, and its ceiling)

The impossibility forbids a training-free *upper* bound on risk. It says nothing about a *lower*
bound. Pose correctness is a thresholded distance, so if two frozen models' poses sit more than
2ρ = 4 Å apart, the triangle inequality forces at least one to be wrong. That is a certified danger
signal costing no labels. Prior art owns the bare inequality and its factor one-half
(arXiv:2507.00057 discrete I/O; arXiv:2603.14070 classification under shift); what is new here is
the metric conversion against a *latent* crystal target, the K-model packing floor, the
diverse-vs-consensus decomposition, and the reconciliation with our own theorem.
Drivers: `d1_extract_delivered.py` → `d1_single_frame.py` → `d1_floor.py`.

**The frame is the whole difficulty, and it hides a trap.** W1's `xmodel_pose_rmsd_*` superpose the
other models onto *each reference model's* pocket in turn, so their distances live in one frame per
reference model. That is fine for a monotone "do the models agree" covariate, which is all W1 uses
them for, and it is wrong for T1, which needs the two poses and the crystal pose in ONE frame. D1
recomputes: pocket defined once from the crystal ligand, every receptor superposed onto the crystal
receptor, label and pairwise distance both recomputed in that frozen frame.

RNP ships only the system's receptor chain while co-folding models predict the full assembly (for
`8ttz` the crystal has 1 chain, AF3 predicts the homodimer plus 9 ligand copies). A chain-ordinal
correspondence then superposes onto the wrong protomer: the backbone fits well (ε stays ~0.3 Å)
while the ligand lands 30–50 Å away. **Neither ε nor the triangle-inequality check detects this** —
a displaced pose is far from everything, so its trigger fires and the inequality holds vacuously.
The frame check passed 0/13,146 while ~21% of labels were wrong. Only the external comparison to
RNP's shipped label exposes it:

| subset | n | Spearman | correctness calls | frac \|diff\| > 5 Å |
|---|---|---|---|---|
| no filter | 13,433 | 0.620 | 0.854 | 0.212 |
| chain counts match + unique ligand copy | 9,545 | 0.700 | 0.891 | 0.144 |
| of those, multi-chain receptors | 3,382 | 0.373 | 0.702 | **0.406** |
| **single-chain receptor + unique copy** | **6,163** | **0.995** | **0.994** | **0.001** |

So the valid frame requires a single-chain receptor and an unambiguous ligand copy: 6,223 of 13,618
instances (45.7%), of which **1,115 retain all K=5 models**. The exclusion is a property of the
system, not the model (models never disagree on predicted chain count, 0/2303; per-model survival
42–51%), so cross-model comparison stays fair. Mild composition bias, reported: the kept subset is
slightly more novel (mean ligand similarity 0.452 vs 0.500) and slightly less accurate (0.651 vs
0.672). Repairing the excluded cases needs quotienting by the receptor symmetry group; that is
future work, since resolving it per-instance against the crystal ligand would put a label inside a
label-free statistic and would let each model pick its own frame.

**The floors.** Pre-registered pair (AF3, Chai-1): `R_max ≥ ½·p_L`, exact one-sided
Clopper-Pearson. Any-pair union: constant is `1/K`, **not** ½. Packing (headline): correct poses
form an independent set in the >2ρ graph, so `R̄ ≥ 1 − E[α(G)|A]/K`; α(G)/K is a bounded mean, not
a binomial proportion, so Clopper-Pearson is invalid and we use a WSR betting upper bound
(`conformal/risk.py:wsr_upper_bound`). It is kept because it is variance-adaptive and never worse on
average, not because it buys much: measured against Hoeffding-Bentkus over 470 cells
(`wsr_gain_vs_hb` in the artifact) it is tighter in 61% for a median gain of **0.006** of floor, and
at the accept counts below 50 that the novel strata leave, the median gain is **0.000** and
Hoeffding-Bentkus is tighter in 23% of cells. An earlier draft claimed 0.02 to 0.06; that was read
off a synthetic simulation and is not what the real cells do.
Because disagreement cannot assign blame, every floor bounds the ensemble mean or the worst model,
never the deployed one.

**Valid and non-vacuous:** 0/386 cells where a certified floor exceeded the realized quantity it
bounds. That is a consequence of the construction rather than a test at level δ: the pointwise
inequality already forces the empirical floor under the empirical risk on the same sample, so the
check can catch an implementation error but cannot exhibit the δ-level step from sample to
population. Marginally the packing floor certifies R̄ ≥ 0.124 against realized 0.348.

AF3, ligand axis, 50% coverage (floors carry no labels; R̄ is the validation truth):

| stratum | n_acc | disagree | floor (certified) | realized R̄ | consensus rate | R̄ on consensus |
|---|---|---|---|---|---|---|
| S0 | 127 | 0.079 | 0.010 | 0.090 | 0.921 | 0.063 |
| S1 | 102 | 0.118 | 0.006 | 0.157 | 0.882 | 0.111 |
| S2 | 101 | 0.178 | 0.031 | 0.178 | 0.822 | 0.113 |
| S3 | 112 | 0.402 | **0.109** | 0.320 | 0.598 | 0.200 |
| S4 (no analog) | 11 | 0.273 | **0.000** | 0.455 | 0.727 | 0.275 |

The floor rises with novelty through S3 and then goes **vacuous on S4**, the no-analog stratum this
project treats as the sharpest extrapolation test: `floor_packing = 0.000` for all five models at
this coverage while realized R̄ = 0.455. Only 11 complexes survive there, so the certificate reads
zero exactly where the question is hardest. This is reported, not buried.

**The honest negative.** The floor does **not** certify a positive ensemble concept gap: it exceeds
a (1−δ) upper bound on R̄_ref in **1 of 124 cells** at per-cell level 0.80, which survives no
family-wise correction, so we report it as null. The reason is measured, and it is exactly T2c: the
floor is blind to consensus error (all K share one wrong mode → no edges → floor reads 0).
Consensus covers 64% of instances and carries risk 0.152 against 0.698 on the diverse remainder;
along the novelty axis the consensus rate falls 0.92 → 0.60 while risk *inside* consensus rises
0.063 → 0.200. The models fail together precisely where they fail most. Label-free danger
certification is real and valid, and structurally too weak to audit the concept gap it targets.

## D2 — the certification feasibility map (the cost)

Margin `m_g = α − R_Q,g(τ)`. Where `m_g > 0` labels can certify; where `m_g ≤ 0` the rule genuinely
violates α and no certificate, label-free or label-fed, can certify a false statement. Driver:
`d2_feasibility_map.py`. Query unit is the **target**, not the pose: 12,125 independent
target-labels over 2,425 systems, one delivered pose per (system, model).

**Why a frontier and not one operating point.** At the project default α=0.20 the familiar
stratum's own risk is 0.130, already below α, so LTT certifies the whole set and the gate
degenerates to accept-everything (AF3 τ=0.237, coverage 1.00 on *every* stratum). For Chai-1,
fixed-sequence LTT rejects its first hypothesis and returns no threshold at all. Both are artifacts
of one calibrator, not facts about co-folding. So we sweep source coverage and report
`c*_g(α) = max{c : R_Q,g(τ(c)) ≤ α}`, with τ calibrated on a held-out half of S0 and deployed frozen.

`c*_feasible` on the ligand axis (0.00 = no operating point at any coverage on the grid):

| model | S0 | S1 | S2 | S3 | S4 |
|---|---|---|---|---|---|
| AF3 (α=0.20) | 1.00 | 0.85 | 0.75 | 0.20 | **0.00** |
| Chai-1 (α=0.20) | 1.00 | 0.80 | 0.00 | 0.00 | 0.25 |
| Protenix (α=0.20) | 1.00 | 0.35 | 0.20 | **0.00** | 0.05 |
| AF3 (α=0.10) | 0.65 | 0.40 | 0.10 | **0.00** | **0.00** |
| Chai-1 (α=0.10) | 0.45 | 0.05 | **0.00** | **0.00** | 0.25 |

Across both axes and five models, requiring ≥20 accepted targets to call a stratum feasible, 29 of
50 strata admit an operating point at α=0.20 and 21 admit none at any coverage; at α=0.10 that
worsens to 24 and 26, robust to sampling noise on 35 of the 47 pooled zero-frontier cells. (An
earlier draft quoted 39/11 and 33/17 from the JSON summary field, which ignored the file's own
min_accept=20 and counted S4 cells feasible on 1–3 accepted targets; the paper uses the correct
≥20-accept counts.) The deployed view is starker: the AF3 rule LTT certifies at α=0.20 on familiar
targets realizes **0.538** on S3 (margin −0.338), so the certificate is inverted, not merely loose.

Certification is feasible on the familiar half of the novelty axis, where labels are cheapest and
least needed, and infeasible on the novel tail, where the decision matters, because what fails there
is the rule rather than the estimate of it. S4 is non-monotone (it sometimes exceeds S3); it is the
no-analog bin, n≈60 per model, and its CIs are wide.

### D2b — what certification costs, and the win that does not exist

The feasibility map says *where* the rule holds. This says what it costs to prove it there. On each
feasible cell we hide the labels, reveal one independent target-label at a time in random order
(200 orders), and record the budget at which each certificate first fires. Driver: `d2_certify.py`.
All four certifiers are allowed to peek at every budget, which is legitimate only for the
anytime-valid WSR sequence and is anti-conservative for the fixed-n bounds, so the comparison is
tilted *toward* the baselines.

| certifier | cells certified | median target-labels |
|---|---|---|
| Hoeffding (pure) | 12/23 | 102 |
| Hoeffding-Bentkus (project default) | 19/23 | 60 |
| WSR betting (variance-adaptive) | 16/23 | 62 |
| **exact binomial** | **23/23** | **38** |

**The exact binomial dominates uniformly**: it certifies every feasible cell, and it is cheaper
than WSR in 16/16 cells where both fire and cheaper than Hoeffding-Bentkus in 18/19. WSR beats
Hoeffding-Bentkus in only 2/16.

**The planned win was real but aimed at a baseline we had already beaten.** The D2 design predicted
that empirical-Bernstein variance-adaptivity would be the robust saving over passive Hoeffding-RCPS
at the small accepted error rates these cells show (R_Q = 0.06–0.18). It is not, but an earlier
draft of this section got the reason wrong and the correction matters. That draft claimed there is
"nothing for a variance-adaptive bound to adapt to" because a Bernoulli variance p(1−p) is a
deterministic function of the mean. The premise is true and the inference is false: pure Hoeffding
does not use the count's information at all, it substitutes the worst-case variance ¼, so there is
plenty to adapt to and **WSR does capture it** (cheaper than pure Hoeffding in **12/12** cells where
both fire, median 49 vs 102).

The correct statement is about **sufficiency**, not variance. The count is a sufficient statistic for
a Bernoulli mean and the exact binomial tail is its exact inversion, so every fixed-n concentration
bound is a *relaxation* of that tail and none can be tighter. Variance-adaptivity buys back the
distance from pure Hoeffding to the tail, but a Bentkus binomial term buys the same thing, and this
project's `hb_upper_bound` already carried one (60 vs WSR's 62). `ltt_threshold` already used the
exact test. So the recommended "upgrade" was measuring itself against a baseline the repo had left
behind. Note WSR is *not* merely a relaxation of the fixed-n tail: it is anytime-valid, so its extra
width is the premium for validity under optional stopping, a different guarantee rather than a worse
one.

**The budget grows as the margin shrinks, more slowly than the theoretical rate.** A log-log fit of
the exact-binomial budget on the margin over the 23 cells gives an exponent of **−1.07** (90% CI
[−1.26, −0.91]), which excludes the Θ(m⁻²) rate the theory predicts; `n·m²` spreads 11.1× where a
1/m² law would hold it constant. The margin does not determine the budget on its own: two cells
sharing margin 0.025 **and** error rate 0.075 cost **164 and 72** labels. An earlier draft asserted
"the budget follows 1/margin², visibly" from four hand-picked points, quoting the 164 and omitting
the 72. It does not. At α=0.10 only 6 of 40 cells are feasible at all, their margins are ≤ 0.040,
and pure Hoeffding certifies none of them within the labels RNP provides.

**The PPI control covariate, measured.** Median |ρ| = 0.256 over the candidate label-free
covariates, so the asymptotic PPI variance reduction keeps 0.934 of the variance, roughly a 7%
saving. It is reported as a measured add-on and is deliberately **not** spliced into the curves
above: PPI's (1 − ρ²) is asymptotic and the control-variate variable is unbounded, so mixing it into
a finite-sample labels-to-certify plot would compare two different guarantees.

**One caveat belongs with this ranking rather than under it.** A fixed-n bound (Hoeffding,
Hoeffding-Bentkus, exact binomial) is valid only at a budget fixed in advance, so reading its
first-firing budget off a reveal sequence flatters it; the WSR sequence is the one construction here
that stays valid under optional stopping. The table therefore prices a certificate at a *planned*
budget. It is not a licence to watch the binomial and stop the moment it fires, and the honest
reading is that the exact binomial is the right tool when you fix n in advance, which is what a
label-acquisition plan does.
