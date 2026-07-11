# Know When to Fold: Distribution-Shift-Aware Selective Prediction for Protein–Ligand Co-Folding

*MoML 2026 short paper (draft). Non-archival. Rikhin Kavuru.*

## Abstract

Protein–ligand co-folding models (AlphaFold3, Boltz, Chai) report confidence
scores that correlate with pose accuracy, but a correlation is not a decision
rule, and the correlation is weakest exactly where drug discovery operates: on
novel pockets and novel chemotypes. We recast co-folding reliability as
**selective prediction** and build a model-agnostic, training-free layer that
turns confidence into a risk-controlled accept/abstain decision with a
finite-sample guarantee. On the released Runs N' Poses benchmark (13,536
delivered poses, six co-folding models) we show three things. (1) A conformal
selective-risk gate is valid on i.i.d. data. (2) That guarantee **breaks** under
the novelty shift central to drug discovery: a gate whose marginal error looks
compliant under-controls error 2–3× on novel-ligand strata, and calibrating on
familiar ligands then deploying on novel ones violates the guarantee in 100% of
runs (realized error 0.55 against a 0.20 target). (3) The break is repaired by
group-conditional and weighted conformal keyed on training-set similarity, and a
combined reliability score cuts the area under the risk-coverage curve (AURC) by
roughly a quarter to two-fifths over native confidence (all gaps have disjoint
bootstrap CIs), roughly doubling to tripling the predictions retained at a fixed
guarantee. The layer is
released as a pip-installable package that wraps frozen model outputs.

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
(iii) a shift-robust repair plus a combined reliability score, packaged as a
training-free layer over frozen AlphaFold3 (AF3) / Boltz / Chai outputs.

## 2. Method

**Setup.** The unit is the delivered pose: the top-1 sample per (system, ligand,
model) by the model's own ranking score. The label is Y = 1 iff the
symmetry-corrected ligand root-mean-square deviation (RMSD) is ≤ 2 Å.
The gate accepts a pose when its confidence s ≥ τ and abstains otherwise; we
report selective risk (error among accepted) against coverage (fraction
accepted).

**Certified threshold.** We choose τ by Learn-then-Test with fixed-sequence
testing [Angelopoulos et al. 2021]. Over a pre-specified coverage grid, each
null hypothesis H0(τ): P(error | s ≥ τ) ≥ α is tested with an exact binomial p-value
P(Bin(n_accept, α) ≤ errors); the accept set grows while H0 is rejected at level
δ and stops at the first failure. This gives P(selective risk ≤ α) ≥ 1 − δ,
finite-sample and distribution-free. The binomial test conditions on the accept
count, the correct tool for error-among-accepted.

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
ensemble spread across diffusion samples, and ligand difficulty. Novelty is
excluded from the score and enters only through calibration. A 3-way split (fit
combiner / calibrate τ / test) preserves conformal validity for the learned score:
the combiner never sees the test fold and the threshold is calibrated on
out-of-sample combiner scores, so there is no train/test leakage.

## 3. Results

Data: released Runs N' Poses predictions consumed as-is (no inference), 13,536
delivered poses, six models. α = 0.20, δ = 0.10 unless noted.

**Raw novelty gradient.** AF3 delivered-pose correctness falls from 0.88 on
familiar ligands to 0.44 on the most novel with-analog stratum. A single global
threshold cannot hold a uniform guarantee across this gradient.

**E1: valid i.i.d.** Mean realized selective risk ≤ α for every model (AF3
0.177), guarantee satisfied 89–96% of splits ≈ the nominal 90%. Validity is also
unit-tested on synthetic data.

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
restores the guarantee: no stratum's accepted error exceeds α. With native
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
| Boltz-1 | 0.233 | 0.169 | −27.5% |
| Boltz-1x | 0.215 | 0.163 | −24.1% |
| Chai-1 | 0.246 | 0.149 | −39.5% |
| Protenix | 0.280 | 0.174 | −37.6% |

Every gap has disjoint 90% bootstrap confidence intervals across 120 splits
(Boltz-2 omitted here for power, n = 933). At α = 0.2 the combined gate retains far
more predictions at the same guarantee (AF3 coverage 0.22 → 0.71; Chai 0.00 → 0.61)
and unlocks the stringent α = 0.1 (90%-correct) operating point that native
confidence cannot certify at all for several models. The combined score fuses
native confidence with interface ipTM, PoseBusters validity, intra-model ensemble
spread, and cross-model agreement.

**E3b: weighted repair, label-free.** Calibrating on familiar ligands and
correcting with likelihood-ratio weights over the novelty covariates pulls the
realized error on a moderately-novel target toward α without target labels (AF3
naive 0.27 → weighted 0.20 at 0.71 coverage; same trend for all models). On the
extreme novel regime (< 55% correct at baseline) weighted conformal reduces error
(0.55 → 0.31) but no gate can certify a high-coverage low-error set, so the layer
correctly abstains.

**E6: downstream payoff.** The gate cleans the delivered pose set a downstream
pipeline would carry forward: base top-1 purity 63–70% rises to 82% (α=0.2,
retaining 81% of correct AF3 poses) or 93–94% (α=0.1, all models), with interface
LDDT-PLI lifting from ~0.72 to 0.85–0.92. Downstream structure-based work receives
near-clean inputs instead of a one-in-three error rate. The Mac1 virtual-screen
enrichment arm follows when its coordinates release.

**E5: robustness and ablation.** The break and the gain do not depend on the 2 Å
convention: across thresholds 1.5–3.0 Å the novel strata always exceed α (AF3 S3
risk 0.35–0.43 vs S0 0.05–0.09) and the combined score cuts AURC 26–29%. A
cumulative feature ablation attributes the gain mainly to interface ipTM (AURC
0.19 → 0.16), intra-model ensemble spread (0.16 → 0.14), and cross-model agreement
(0.14 → 0.13); PoseBusters and ligand difficulty add little on top.

**E8/E9: task- and label-agnostic.** The layer is not tied to the 2 Å pose label.
On interface quality (local distance difference test on the protein-ligand
interface, LDDT-PLI ≥ 0.5) the combined score again dominates native confidence on
AURC (44–55% lower). Dropping the threshold entirely and ordering by
continuous RMSD, the accepted set's mean RMSD at 50% coverage falls from ~2.1 Å
(native) to ~1.1–1.5 Å (combined).

## 4. Discussion and limitations

The guarantee is over the chosen correctness definition and the sampled shift
axes, not a universal notion of trust; the findings are robust across RMSD
thresholds 1.5–3.0 Å (E5), though a continuous-RMSD risk is left for future work.
Weighted conformal coverage is approximate under estimated weights, so we keep
group-conditional as the rigorous fallback. A cross-dataset test on FoldBench did
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
