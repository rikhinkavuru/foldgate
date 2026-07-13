# Know When to Fold: Distribution-Shift-Aware Selective Prediction for Protein–Ligand Co-Folding

*MoML 2026 short paper (draft). Non-archival. Rikhin Kavuru.*

## Abstract

Protein–ligand co-folding models (AlphaFold3, Boltz, Chai) report confidence
scores that correlate with pose accuracy, but a correlation is not a decision
rule, and the correlation is weakest exactly where drug discovery operates: on
novel pockets and novel chemotypes. We recast co-folding reliability as
**selective prediction** and build a model-agnostic, training-free layer that
turns confidence into a risk-controlled accept/abstain decision with a
finite-sample guarantee. On the released Runs N' Poses benchmark (13,535
delivered poses, six co-folding models) we show three things. (1) A conformal
selective-risk gate is valid on i.i.d. data. (2) That guarantee **breaks** under
the novelty shift central to drug discovery: a gate whose marginal error looks
compliant under-controls error 2–3× on novel-ligand strata, and calibrating on
familiar ligands then deploying on novel ones violates the guarantee in 100% of
runs (realized error 0.55 against a 0.20 target). (3) The break is repaired by
group-conditional and weighted conformal keyed on training-set similarity, and a
combined reliability score cuts the area under the risk-coverage curve (AURC) by
roughly a quarter to two-fifths over native confidence (paired data-bootstrap over
test poses excludes zero for every model), roughly doubling to tripling the
predictions retained at a fixed guarantee. Adding structural pose-agreement features
from the model's own diffusion samples pushes the AURC reduction to about a half (pose
Δ(AURC) CI excludes zero for all models), and abstention raises a genuinely downstream,
non-circular metric — recovery of the correct crystal contacts (0.90 vs 0.72 on accepted
vs rejected poses). The layer is released as a pip-installable package that wraps frozen
model outputs.

## 1. Introduction

Co-folding models are increasingly used to prioritise molecules and generate
starting structures for downstream design. Recent work establishes that their
confidence tracks accuracy and that accuracy drops on inputs dissimilar to
training (Runs N' Poses, FoldBench). Neither gives a practitioner what they need:
a rule for **which predictions to trust**, with a guarantee that survives the
distribution shift they actually face.

We provide that rule. Our contribution is (i) selective prediction / conformal
risk control for co-folding **pose** reliability, a structured 3D output rather
than a scalar property. The closest prior work, CoDrug, conformalizes a scalar
molecular property under shift, and we are not aware of a prior conformal
treatment of pose correctness. Further, (ii) a sharp, quantified demonstration that
standard conformal guarantees collapse under novel-pocket / novel-chemotype shift,
using training-set structural and chemical similarity as the shift variable; and
(iii) a shift-robust repair — group-conditional and importance-weighted conformal
keyed on that similarity — plus a combined reliability score, packaged as a
training-free layer over frozen AlphaFold3 (AF3) / Boltz / Chai outputs. We show
ordinary probability calibration is not a substitute: it carries no finite-sample
coverage and breaks under the same shift, whereas the conformal variants repair it.

## 2. Method

**Setup.** The unit is the delivered pose: the top-1 sample per (system, ligand,
model) by the model's own ranking score. The label is Y = 1 iff the
symmetry-corrected ligand root-mean-square deviation (RMSD) is ≤ 2 Å.
The gate accepts a pose when its confidence s ≥ τ and abstains otherwise; we
report selective risk (error among accepted) against coverage (fraction
accepted).

**Certified threshold.** We choose τ by Learn-then-Test with fixed-sequence
testing [Angelopoulos et al. 2021] over a pre-specified, data-independent sequence
of top-k accept ranks (smallest first). Each null H0(τ): P(error | s ≥ τ) ≥ α is
tested with the binomial tail p-value P(Bin(n_accept, α) ≤ errors); the accept set
grows while H0 is rejected at level δ and stops at the first failure, giving
P(selective risk ≤ α) ≥ 1 − δ, finite-sample and distribution-free. The p-value is
exact for a homogeneous error rate and conservative (hence still valid) for the
heterogeneous Poisson-binomial case by a convex-order argument; conditioning on the
accept count is the correct tool for error-among-accepted.

**Novelty and shift-robustness.** Training-set similarity (ECFP4 Tanimoto to the
nearest training ligand, pocket similarity, temporal cutoff), shipped
pre-computed by Runs N' Poses, is the covariate-shift variable. It both
stratifies (group-conditional / Mondrian conformal: a separate τ per novelty
stratum) and weights (weighted conformal [Tibshirani et al. 2019]: reweight
familiar-source calibration points by an estimated likelihood ratio to certify on
a novel target without target labels). Ligands with no training analog form their
own extreme stratum.

**Combined score.** Native ranking score is a weak sorter for some models. We fit
a calibration-only combiner (gradient boosting → P(correct)) over native
confidence, interface chain-pair ipTM, PoseBusters physical validity, intra-model
ensemble spread across diffusion samples, and ligand difficulty. "Training-free"
here means the co-folding model is never retrained; the combiner is a small model fit
only on the calibration fold, so the layer is free of structure-predictor training,
not of all fitting. Novelty is
excluded from the score and enters only through calibration. A 3-way split (fit
combiner / calibrate τ / test) preserves conformal validity for the learned score:
the combiner never sees the test fold and the threshold is calibrated on
out-of-sample combiner scores, so there is no train/test leakage.

## 3. Results

Data: released Runs N' Poses predictions consumed as-is (no inference), 13,535
delivered poses, six models. α = 0.20, δ = 0.10 unless noted.

**Raw novelty gradient.** AF3 delivered-pose correctness falls from 0.88 on
familiar ligands to 0.44 on the most novel with-analog stratum. A single global
threshold cannot hold a uniform guarantee across this gradient.

**E1: valid i.i.d.** The certifier's finite-sample guarantee — true selective risk
≤ α with probability 1 − δ — is validated on synthetic data where the true risk is
known (the gate holds in ≥ 1 − δ of draws). The realized-risk indicator on a finite
test fold is a *downward-biased* proxy for that event, and we demonstrate the bias
rather than assert it: on synthetic data the small-fold indicator dips below 1 − δ
while the large-sample (true-risk) indicator meets it. On RNP the fraction of splits
holding, with a Clopper-Pearson interval, reaches the target for the well-powered
models (Boltz-1x 0.91 [0.88, 0.93], Boltz-1 0.88 [0.85, 0.91], AF3 0.85 [0.81, 0.89]),
and the certified gate's mean realized risk is ≤ α there (AF3 0.17 at 28% coverage).
Native ranking score certifies almost nothing for Chai and Protenix (< 5% coverage):
a near-vacuous gate that motivates the combined score below.

**E2: the break.** AF3's marginal risk (0.177) hides severe per-stratum
under-control:

| novelty stratum | S0 | S1 | S2 | S3 | S4 |
|---|---|---|---|---|---|
| realized selective risk | 0.07 | 0.15 | 0.16 | **0.38** | **0.43** |

Calibrating on familiar ligands and deploying on novel ones accepts 98% of poses
at realized error **0.55** with the guarantee holding in **0%** of runs (violated
in 100%). All six models show the same collapse. The break is a property of
structural/chemical novelty, not recency: it is at least as strong along
pocket-novelty (AF3 S0 0.06 → S3 0.40, Chai up to 0.77) but weak along a temporal
release-date axis, which sharpens what the shift variable must capture.

**E3: repair.** Group-conditional calibration (a separate τ per novelty stratum)
restores per-stratum validity: each stratum's accepted error is ≤ α with probability
1 − δ marginally (a simultaneous across-strata statement follows by calibrating each
at δ/K). With native
confidence the cost is heavy abstention on the hardest strata (the gate folds
rather than assure falsely). With the combined score, group-conditional
calibration instead recovers usable coverage across the gradient while holding
risk ≤ α: AF3 accepts 52% of stratum S1 and 39% of S2 (vs 8% and 12% with native
confidence) and certifies a small fraction of the novel S3 that native confidence
must abstain on. Only the no-training-analog extreme (S4) stays uncertifiable,
where abstention is the correct answer.

**E4: utility.** The combined score dominates native confidence on the
risk-coverage curve:

| model | AURC native | AURC combined | Δ |
|---|---|---|---|
| AF3 | 0.187 | 0.125 | −33.1% |
| Boltz-1 | 0.234 | 0.170 | −27.5% |
| Boltz-1x | 0.216 | 0.163 | −24.6% |
| Chai-1 | 0.248 | 0.150 | −39.6% |
| Protenix | 0.281 | 0.175 | −37.7% |

The AURC reduction is significant per model: a paired bootstrap over test poses
gives 90% CIs on Δ(AURC) that exclude zero (AF3 0.070 [0.049, 0.092], Protenix
0.092 [0.068, 0.116]; Boltz-2 omitted for power, n = 933). At α = 0.2 the combined
gate retains far more predictions at the same guarantee (AF3 coverage 0.22 → 0.71;
Chai 0.01 → 0.60)
and unlocks the stringent α = 0.1 (90%-correct) operating point that native
confidence cannot certify at all for several models. The combined score fuses
native confidence with interface ipTM, PoseBusters validity, intra-model ensemble
spread, and cross-model agreement.

**E3b: weighted repair, label-free.** Calibrating on familiar ligands and
correcting with cross-fitted, probability-calibrated likelihood-ratio weights over
the novelty covariates pulls the realized error on a moderately-novel target toward
α without target labels (AF3 naive 0.27 → weighted 0.19 at 0.70 coverage; same trend
for all models, and robust to the weight-model choice). We also implement an
importance-weighted Learn-then-Test gate (WSR betting p-value [Almeida et al. 2025])
that is finite-sample valid *conditional on correct weights*; on real co-folding data
it abstains, because even the plug-in barely clears α, so no certifiable margin
remains. A concept-shift diagnostic explains why: P(correct | confidence) itself moves
between source and target, and the gap grows from the moderate regime (0.08–0.16) to
the extreme regime (0.17–0.29), so pure covariate reweighting cannot restore validity
on novel pockets. Weighted conformal is therefore the label-free complement;
group-conditional calibration (E3) is the rigorous finite-sample guarantee where
stratum labels exist. On the extreme regime (< 55% correct at baseline) weighted
conformal reduces error (0.50 → 0.35) but no gate can certify a high-coverage low-error
set, so the layer correctly abstains.

**E11: baselines and calibration vs conformal.** The combined score has the lowest
AURC among the confidences a practitioner would reach for (AF3 0.12 vs native ipTM
0.16 vs ranking score 0.19; same ordering for all models), and PoseBusters validity
alone is a poor gate (realized error 0.26–0.34). The decisive comparison isolates the
guarantee from the features: the same combined score as a Platt/isotonic-calibrated
classifier with a fixed threshold versus inside the conformal layer. i.i.d., both
control error; **under the novelty shift the calibrated fixed-threshold gates break
exactly as naive conformal does** (realized error climbs above α), because the break is
a property of naive transfer, not of calibration-vs-conformal. Only the shift-robust
conformal repair — group-conditional, which has no calibration analogue — restores
realized error ≤ α, by abstaining more on the novel strata. Calibration fixes the
marginal probability but carries no finite-sample coverage and no distribution-shift
repair.

**E6: downstream payoff.** The gate cleans the delivered pose set a downstream
pipeline would carry forward: base top-1 purity 63–70% rises to 82% (α=0.2,
retaining 86% of correct AF3 poses) or 91–94% (α=0.1, all models), with interface
LDDT-PLI lifting from ~0.72 to 0.85–0.92.

**E6b: a non-circular downstream payoff.** Purity equals 1 − selective risk by
construction, so E6 cannot show the gate buys anything beyond the guarantee. We add a
genuinely different, downstream metric: protein-ligand *interaction-fingerprint recovery*
(receptor residues within 4.5 Å of the ligand vs the crystal structure), which is only
correlated with, not equal to, the 2 Å label. Under the gate, accepted poses recover 0.90–0.91
of the crystal contacts vs 0.71–0.76 for rejected poses, and the accepted-minus-rejected recall
gap 90% CI excludes zero for all five models (gaps 0.15–0.20). Abstention routes forward the
poses whose interaction patterns a downstream SAR or structure-based step can trust. A
screening-enrichment harness (EF/BEDROC, selective-EF, coverage-enrichment and active-retention
curves, random-abstention control) is implemented and awaits a screening dataset; the Mac1
prospective screen is the drop-in target when its coordinates release.

**E5: robustness and ablation.** The break and the gain do not depend on the 2 Å
convention: across thresholds 1.5–3.0 Å the novel strata always exceed α (AF3 S3
risk 0.35–0.43 vs S0 0.05–0.09) and the combined score cuts AURC 26–29%. A
cumulative feature ablation attributes the gain mainly to interface ipTM (AURC
0.19 → 0.16), intra-model ensemble spread (0.16 → 0.14), and cross-model agreement
(0.14 → 0.13); PoseBusters and ligand difficulty add little on top.

**Pose-agreement upgrade (W1, structures).** Beyond the released confidence tables, the
predicted structures carry an orthogonal signal: whether the binding *modes* agree. From the
model's own 25 diffusion samples we compute intra-model pose diversity, and across models the
delivered-pose ligand-RMSD (spyrmsd, pocket-superposed, symmetry-corrected; over 11,711 poses,
no GPU). Adding these lowers AURC further on top of every tabular feature — the single largest
drop after ipTM (AF3 0.123 → 0.106) — raising the overall reduction to 38–51% across models,
the pose-only Δ(AURC) 90% CI excluding zero for all five. Intra-model pose diversity is a free
byproduct of running the model, so this keeps the training-free framing; the tabular combined
score remains the cheap, structure-free primary.

**E8/E9: task- and label-agnostic.** The layer is not tied to the 2 Å pose label.
On interface quality (local distance difference test on the protein-ligand
interface, LDDT-PLI ≥ 0.5) the combined score again dominates native confidence on
AURC (44–55% lower). Dropping the threshold entirely and ordering by continuous RMSD,
the accepted set's mean RMSD at 50% coverage falls from ~2.1 Å (native) to ~1.1–1.5 Å
(combined). We also certify a *continuous* gate — mean bounded-RMSD among accepted ≤ a
target with probability 1 − δ — using a variance-adaptive WSR betting bound, which
certifies non-trivial coverage where a distribution-free Hoeffding bound certifies
almost none (AF3, target 1 Å mean-RMSD: WSR 43% coverage vs Hoeffding 0%).

## 4. Discussion and limitations

The guarantee is over the chosen correctness definition and the sampled shift
axes, not a universal notion of trust; the findings are robust across RMSD
thresholds 1.5–3.0 Å (E5) and extend to a certified continuous-RMSD gate (E9).
The finite-sample weighted-conformal guarantee is exact only conditional on correct
importance weights, and under novel-pocket shift there is residual concept shift
(P(correct | confidence) moves), so we keep group-conditional as the rigorous
finite-sample guarantee and scope weighted conformal as the label-free complement. A
cross-dataset test on FoldBench did
not replicate the advantage: its public per-pose table ships only ranking_score,
without the interface-ipTM / PoseBusters / ensemble features the ablation
identifies as the source of the gain, so the feature-poor combiner does not beat
raw confidence there. A clean second-benchmark test needs a release that exposes
those confidences per pose. On the extreme
novel-chemotype regime (base correctness < 55%) no gate can certify a
high-coverage low-error set; the layer's value there is principled abstention over
a naive gate's false confidence. Code, calibration tables, and a one-command
reproduction pipeline are provided in the accompanying repository (public release
on acceptance).

## References

Abbreviated; full entries in the repository `REFERENCES.bib`.
Angelopoulos, Bates, Candès, Jordan, Lei. Learn then Test. 2021.
Bates, Angelopoulos, Lei, Malik, Jordan. Risk-Controlling Prediction Sets. JACM 2021.
Tibshirani, Foygel Barber, Candès, Ramdas. Conformal Prediction Under Covariate Shift. NeurIPS 2019.
Škrinjar et al. Have protein–ligand co-folding methods moved beyond memorisation? (Runs N' Poses) 2025.
Buttenschoen, Morris, Deane. PoseBusters. Chem. Sci. 2024.
