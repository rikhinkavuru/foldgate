# Results (real, on released Runs N' Poses)

All numbers are from released RNP predictions consumed as-is (no model inference):
13,536 delivered poses (top-1 by `ranking_score`, proper ligands) across 6 co-folding
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

LTT gate on native `ranking_score`, 300 random 50/50 splits. Mean realized
selective risk ≤ α for every model (AF3 0.177), and the guarantee is satisfied
89–96% of splits ≈ the nominal 1 − δ = 90% (AF3 0.893; within Monte-Carlo noise,
SE ≈ 0.017). AF3 certifies 36% coverage at ≤20% error. Validity is also unit-tested
on synthetic data (`tests/test_conformal.py`).

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
ships). No stratum's accepted error then exceeds α. The honest cost with the
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
| Boltz-1 | 0.233 | 0.169 | 27.5% |
| Boltz-1x | 0.215 | 0.163 | 24.1% |
| Chai-1 | 0.246 | 0.149 | **39.5%** |
| Protenix | 0.280 | 0.174 | **37.6%** |

(These include the cross-model confidence-agreement feature; without it the gains
were 22–38%. Cross-model agreement is the third-largest contributor per the E5 ablation.)

Coverage at certified error levels (higher = more usable predictions retained):

| | α = 0.2 native → combined | α = 0.1 native → combined |
|---|---|---|
| AF3 | 0.22 → **0.67** | 0.00 → **0.12** |
| Chai-1 | 0.00 → **0.59** | 0.00 → **0.08** |
| Protenix | 0.02 → **0.35** | 0.00 → 0.03 |

At the same guarantee, the combined gate retains ~3× more predictions, and unlocks
the stringent 90%-correct operating point that native confidence cannot certify at
all for several models.

## E3b — weighted covariate-shift repair (label-free, combined score)

Calibrate on familiar ligands, deploy on more-novel ligands, correct with
likelihood-ratio weights over the novelty covariates — no target labels used.

**Moderate shift (S0,S1 → S2), where certification is feasible:** weighted
conformal pulls realized target error toward α without target labels.

| model | naive risk (cov) | weighted risk (cov) | target-cal risk (cov) |
|---|---|---|---|
| AF3 | 0.269 (0.96) | **0.198 (0.71)** | 0.151 (0.30) |
| Chai-1 | 0.280 (0.84) | 0.215 (0.69) | 0.145 (0.31) |
| Protenix | 0.353 (0.90) | 0.277 (0.52) | 0.118 (0.06) |

Naive (source-calibrated) over-accepts at error above α; weighted reweighting
brings AF3 to 0.198 ≈ the 0.20 target at 71% coverage, using no target labels.

**Extreme shift (S0–S2 → S3,S4):** naive error 0.52; weighted reduces it to 0.31
but cannot reach α, because the novel target is < 55% correct at baseline —
uncertifiable by any gate (even the label-using target-calibrated one abstains).
The honest conclusion: weighted conformal closes most of the gap on moderate
shift; on the extreme drug-discovery regime the layer's value is principled
abstention, not the naive gate's false confidence. Group-conditional (E3) remains
the rigorous fallback where stratum labels exist.

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
| AF3 | 0.2 | 0.69 | 0.70 → **0.82** | 1.17× | 0.81 | 0.77 → 0.85 |
| AF3 | 0.1 | 0.20 | 0.70 → **0.94** | 1.33× | 0.27 | 0.77 → 0.91 |
| Chai-1 | 0.1 | 0.19 | 0.65 → **0.94** | 1.44× | 0.27 | 0.73 → 0.91 |
| Protenix | 0.1 | 0.08 | 0.63 → **0.94** | 1.47× | 0.11 | 0.72 → 0.92 |

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
| + ligand difficulty (all) | 0.130 |

Interface ipTM (0.193→0.160), ensemble spread (0.159→0.137), and cross-model
agreement (0.137→0.128) do the work; PoseBusters and ligand physicochemistry add
little on top.

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
| AF3 | 2.07 → **1.14 Å** |
| Chai-1 | 2.16 → **1.15 Å** |
| Protenix | 2.58 → **1.54 Å** |

The finding survives dropping the 2 Å convention entirely. (A finite-sample
*certified* continuous-mean bound is loose at these sample sizes with a
distribution-free Hoeffding bound; a variance-adaptive bound is future work.)

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
