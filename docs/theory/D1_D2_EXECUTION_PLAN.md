# D1 + D2 Execution Plan (SOTA-grounded, 2026-07-15)

*Distilled, actionable companion to `FUTURE_WORK_PLAN.md` (the full audited design) and `THEOREM_RECONCILED.md` (the impossibility+achievability theorem these build on). This file is the build sheet: what to do, in what order, with go/no-go gates and the current literature folded in. Author: Rikhin Kavuru. Target: Regeneron STS.*

---

## EXECUTION STATUS (2026-07-15, read first)

**Both Week-1 MVPs are built and both gates PASS.** Numbers live in `RESULTS.md` (sections D1, D2)
and `results/{d1_floor,d1_frame_check,d2_feasibility_map}.json`. Paper sections `sec:danger` (D1)
and `sec:cost` (D2) are written into `paper/moml2026_foldgate.tex`.

Pipeline: `d1_extract_delivered.py` (streams the 39.5 GB tarball once, caches 12,597 delivered poses
to `data/processed/delivered_poses.tar.gz`, ~5 min) → `d1_single_frame.py` (~12 min) →
`d1_floor.py`. `d2_feasibility_map.py` runs on the delivered parquet alone.

Four findings that change what this plan said:

1. **Section 2.4 step 1 was right to worry, and the frame trap is worse than described.** The
   delivered `xmodel_pose_rmsd_*` are indeed not single-frame. But the deeper problem is that RNP
   ships only the system's receptor chain while models predict the full assembly, so chain-ordinal
   keying superposes onto the wrong protomer with a SMALL eps and a 30-50 A ligand displacement.
   **Neither eps nor the triangle-inequality check detects this** (a displaced pose satisfies the
   inequality vacuously: the check passed 0/13,146 while 21% of labels were wrong). Only comparison
   against RNP's shipped label exposes it. Valid frame requires a single-chain receptor and a
   unique ligand copy: 6,223/13,618 instances, **1,115 complete K=5** (D1's effective n), verified
   at Spearman 0.995 / 99.4% correctness-call agreement. See `features/single_frame.py`.
2. **The Week-1 D1 gate passes: the floor is valid (0/386) and non-vacuous and tracks novelty**
   (AF3, 50% coverage: 0.010 on S0 → 0.109 on S3). It is NOT the pairwise-prior-art restatement.
3. **Headline item 2 (the reconciliation) does not fire empirically, as pre-registered.** The floor
   exceeds a (1-delta) upper bound on `R_bar_ref` in 1/124 cells at per-cell level 0.80, null under
   any family-wise correction. Cause is measured and is exactly T2c: consensus covers 64% of
   instances, carries risk 0.152 vs 0.698 on the diverse remainder, and the consensus rate falls
   0.92 → 0.60 with novelty while risk INSIDE consensus rises 0.063 → 0.200. **The escape is real
   and provably narrow, and quantifying the narrowness is the contribution.** FUTURE_WORK_PLAN
   already pre-registered the theory-first framing for exactly this outcome; the paper does not
   hinge on the floor beating the baseline.
4. **D2 must be a COVERAGE SWEEP, not the single LTT operating point Section 3 assumed.** At
   alpha=0.20 the source stratum's own risk (0.130) is below alpha, so LTT certifies everything and
   the gate degenerates to accept-everything; for chai, fixed-sequence LTT returns no threshold at
   all. Both are calibrator artifacts. The frontier `c*_g = max{c : R_Q,g(tau(c)) <= alpha}` is
   robust and gives a stronger result: **c* = 0 on the most novel strata means no operating point
   at any coverage**, strictly stronger than the coverage-pinned Thm 1(c).

Two rigor fixes made while building, both of which went AGAINST the headline and both of which
should not be undone: `R_bar_ref` is compared via a bootstrapped (1-delta) UPPER bound with a
delta/K union bound (comparing a certified floor to a point estimate would have produced a false
positive here), and each model's reference is binned on ITS OWN score, which is the hardest
baseline the reweighting is entitled to. New: `conformal/risk.py:wsr_upper_bound` (validated by
simulation, coverage >= 0.91 at target 0.90), `conformal/shift_decomp.py` now returns
`R_ref` / `R_ref_upper`.

5. **D2's Weeks 2-3 MVR is built (`d2_certify.py`) and its predicted mechanism is wrong.** Median
   independent target-labels to certify on the 23 cells feasible at 50% coverage: pure Hoeffding
   102 (fires on 12/23), Hoeffding-Bentkus 60 (19/23), WSR betting 62 (16/23), **exact binomial 38
   (23/23)**. Section 3.3's "the robust win over passive Hoeffding-RCPS is empirical-Bernstein
   variance-adaptivity for small-p Bernoulli" does not survive contact with the repo's OWN
   baseline. The correct reason is sufficiency, not variance: the count is sufficient for a
   Bernoulli mean and the exact binomial tail is its exact inversion, so every fixed-n bound is a
   relaxation of it. Variance-adaptivity does buy a real 2x saving over PURE Hoeffding (WSR wins
   12/12 there, 49 vs 102), but a Bentkus term buys the same thing and `hb_upper_bound` already had
   one, while `ltt_threshold` already used the exact test. Do NOT "upgrade" the certifier.
   Separately, the budget does NOT visibly follow 1/m^2: log-log slope -1.07 (90% CI [-1.26,-0.91]),
   and two cells sharing margin 0.025 and error rate 0.075 cost 164 and 72 labels.
6. **Two adversarial audits of the write-up caught real overclaims that survived my own review**
   (see the workflow transcripts). Three were blockers: an unsupported "0.02-0.06 floor gain" for
   the betting bound read off a synthetic simulation rather than the cells (true median 0.006, and
   HB is tighter in 23% of them); the 1/m^2 assertion above; and an abstract sentence saying the
   consensus regime "grows where risk is highest" when the consensus RATE falls 0.92 -> 0.60 and it
   is the error hidden INSIDE consensus that grows (0.058 -> 0.120). Numbers now come from stored
   artifact fields (`wsr_gain_vs_hb`, `floor_packing_hb`) rather than from prose.

Not done / next: the receptor-symmetry quotient that would recover the 54% excluded instances; the
DRO ambiguity-radius morsel (Section 3.4 T2); emitting `hidden_error_mass` (consensus rate times
risk inside consensus) from `d1_floor.py` rather than multiplying it in prose. Mac1 stays blocked
per Section 0.

---

## 0. Status of item #1 (Mac1 prospective screen) — BLOCKED, deferred

Checked the primary source on 2026-07-15. The Mac1 co-folding benchmark (eLife reviewed-preprint 110475, bioRxiv `10.64898/2025.12.25.696505` v3, published March 2026) states the 557 X-ray structures "will be released after a small delay to preserve the potential for blind predictions by any new methods." The crystal ground-truth is under embargo, and the co-folding predicted poses are not released either. Only the input-prep and post-process scripts are public (`github.com/jongbin99/Cofolding`).

Consequence: a real prospective, RMSD-labeled Mac1 validation cannot start. The only thing computable today is an unvalidated deployment demo, and even that requires regenerating K models on 557 targets ourselves (heavy GPU, like the FoldBench J1 run) to get multi-model poses, then computing the D1 label-free disagreement floor with no floor-vs-truth check. That is low STS value for real GPU cost.

Decision: defer Mac1. Set a watch on the coordinate release. Do not spend GPU on an unvalidated demo. This is consistent with every caveat already in the repo (`FUTURE_WORK_PLAN.md` lines on the single-dataset validation reality, `DATA_CARD.md`, `THEORY_LENS_domain_adapt.md`). The generalization story leans on the RNP temporal split plus per-stratum breadth until the coords drop.

The good news is that #1 being blocked does not block the STS spine. D1 is the escape theorem and it runs entirely on data already on disk.

---

## 1. Fresh-literature delta (folded into the plan below)

Re-swept 2026-07-15. The Jul-13 plan's citations still hold. Three 2026 papers not in that draft are close enough that a judge would expect them cited. None scoops D1's surviving core.

- **arXiv:2606.08517 — A Joint Finite-Sample Certificate for Adaptive Selective Conformal Risk Control (June 2026).** Closest neighbor for the selective-risk-as-ratio certificate machinery both D1 and D2 lean on (empirical-Bernstein on the loss, Clopper-Pearson on acceptance, closeness bound on the ratio). Cite it as the certificate layer; it strengthens the honest "this part is routine machinery" concession rather than threatening the headline.
- **arXiv:2603.08907 — Cross-Domain Uncertainty Quantification for Selective Prediction, with Transfer-Informed Betting.** Bears on D2's transfer + betting-based certificate. Cite in the D2 scaffolding list.
- **arXiv:2510.16166 — Extending Prediction-Powered Inference through Conformal Prediction.** Bears on D2 Target 1 (PPI risk control). Cite alongside Angelopoulos-Zrnic and the semi-supervised / active-PPI line.

Prior art already load-bearing and unchanged: D1 pairwise floor is published (`2507.00057` Incoherence, `2603.14070` Structured Credal Learning); the global selective-risk floor is `2606.29054`; D2's bare margin rate is CSA `2605.20270` Thm 5.4; PPI is Angelopoulos-Zrnic `2301.09633`, active-RCPS is Csillag `2406.10490`, active-multiple-PPI is `2605.08429`.

The honest posture stands: lead with what survives, cite the neighbors up front, do not oversell the finite-sample certificate.

---

## 2. D1 — Ensemble disagreement as a certifiable label-free LOWER bound on risk

### 2.1 The one-sentence contribution

You cannot certify safety label-free (your Theorem 1). You can certify danger label-free: cross-model pose disagreement gives a one-sided lower bound on accept-region selective risk that Theorem 1 does not forbid, because Thm 1 only bans certified upper bounds obtained by reweighting the source conditional.

### 2.2 What is genuinely yours (say this verbatim in the paper)

The bare inequality "if two frozen models disagree at least one is wrong, so the pairwise-disagreement rate times one-half floors the error rate" is prior art (`2507.00057`, `2603.14070`). Wrapping a disagreement indicator in a one-sided Clopper-Pearson lower bound is textbook binomial inference. The four surviving items, in priority order:

1. **(headline) Diverse-vs-consensus concept-gap decomposition** and its consensus mini-impossibility: when the K models share a training-similarity bias they fail together, so consensus does not certify safety on novel chemotypes. This is the structural link to the concept gap.
2. **(headline) Exact reconciliation with the covariate-weighted Theorem 1**: a certified lower bound on risk is admissible precisely on the side Thm 1 leaves open. Make the lower-vs-upper-bound distinction the spine of the argument.
3. **Metric / triangle-inequality conversion** of a continuous structured pose into a thresholded error count against the latent crystal pose, via the `2ρ` geometric trigger set by the `ρ = 2 Å` label radius. Credal and incoherence work use discrete labels or I/O equality, never a distance-to-latent-target trigger.
4. **Multi-model packing / clique floor**: the K-model generalization of the pairwise floor.

Items 1 and 2 are the theoretical headline. Items 3 and 4 are the mechanism. The finite-sample certificate is routine machinery (cite `2606.08517`).

### 2.3 Target theorems (conjectures with proof strategy, not settled)

- **T1 (pairwise metric floor).** In a single fixed common frame containing `y★`, if `RMSD(x_a, x_b) > 2ρ` then at least one of models `a, b` is wrong, so `Pr(disagree) / 2 ≤ mean per-model error`. Proof: triangle inequality on the symmetry-corrected ligand-RMSD quotient metric. Assumption to defend: the single-frame requirement, handled by superposing every model's protein onto one reference receptor once (Method step 1); state the deployment-time frame-transfer slack honestly and do NOT claim a label-free deployment guarantee (the m0-vs-crystal receptor discrepancy is not label-free).
- **T2 (K-model packing floor).** Generalize T1 to a clique/packing argument over K frozen models: mutually `> 2ρ`-separated poses force a lower bound on the ensemble-mean error that grows with the packing number. This is the multi-model headline mechanism.
- **T3 (diverse-vs-consensus decomposition + reconciliation).** Decompose the ensemble concept gap into a diverse component (models disagree, floor is informative) and a consensus component (models agree and are jointly wrong on novel chemotypes, floor is blind). Prove the consensus blind spot is exactly the covariate-weighted-invariant term of Theorem 1. This closes the bracket.

### 2.4 Method — runs on local RNP, zero new GPU

The load-bearing feasibility fact: RNP ships per-model predicted `.cif` coordinates (5 seeds x 5 samples per model, `data/raw/predictions` + `prediction_files.tar.gz`) AND the cross-model pose-disagreement signal is already extracted in `data/processed/rnp_delivered.parquet` (`xmodel_pose_rmsd_median`, `xmodel_pose_rmsd_min`, `xmodel_n_pose`, `pose_consensus_cluster_size`, `intra_model_pose_std`, `pose_consensus_frac`) next to `rmsd`, `correct`, and the novelty strata. Nothing needs regenerating.

1. **Single common frame.** Confirm the delivered `xmodel_pose_rmsd_*` were computed after superposing each model's protein onto one reference receptor. If not, recompute pose-pairwise RMSD in a fixed frame from the raw `.cif` files. This is the one place a silent frame bug would void T1.
2. **Empirical floor vs realized risk.** For each novelty stratum, compute the disagreement-derived lower bound (T1 pairwise and T2 packing forms) and overlay the realized accept-region selective risk `Pr_Q(error | accept)`. Show the floor tracks realized risk on high-novelty strata where the covariate-weighted certificate under-reports.
3. **Diverse-vs-consensus split.** Bin accepted predictions by ensemble agreement; show the consensus-and-wrong mass concentrates on low-Tanimoto / low-pocket-similarity strata (the T3 blind spot), and that the covariate-weighted reference certificate is blind to exactly that mass.
4. **Finite-sample certificate.** One-sided Clopper-Pearson lower confidence bound on the disagreement rate; the selective-risk ratio uses the `2606.08517` empirical-Bernstein + Clopper-Pearson + closeness construction, not a naive Hoeffding range bound.

### 2.5 Milestones and gates

- **Week 1 — MVP.** Reproduce the delivered pose-disagreement features from the raw `.cif` in a single verified frame on a 50-target sample; confirm the frame is clean. Plot the empirical T1 floor vs realized selective risk per novelty stratum. This single figure is the minimal viable result and is already 80% supported by the delivered parquet.
- **Week 1 go/no-go.** The floor must be non-trivial (strictly positive and tracking realized risk) on at least the top-two novelty strata with enough accepted points for a usable Clopper-Pearson interval. If the floor is vacuous everywhere, D1 collapses to the pairwise-prior-art restatement; stop and reassess. Expectation from the E5/E7 disagreement signals already in the repo: it will not be vacuous.
- **Weeks 2-4 — theory.** Write T1, T2, T3 with assumptions stated as conjecture-with-strategy. Get T3 (decomposition + reconciliation) airtight; it is the headline and the piece a hostile judge will probe.
- **Weeks 5-6 — stress + write-up.** RNP temporal split as the generalization check (a split, not a second dataset, stated plainly). PoseBusters label sanity-check. FoldBench pose-disagreement is out of scope for v1 (needs coordinate regeneration; we now have the regen pipeline from J1 if it becomes worth it). Mac1 stays deferred per Section 0.

### 2.6 Risks

- **Frame bug voids T1.** Mitigation: Week-1 single-frame verification on raw `.cif`, no exceptions.
- **Consensus blind spot is a limitation, not a bug.** Present it as the honest boundary of the method and the exact content of T3. It is a feature of the story, not a hole.
- **Over-claiming the certificate.** Cite `2606.08517`, `2507.00057`, `2603.14070` up front; call the binomial inference routine.

D1 alone is a complete, defensible STS project. Feasibility 4/5, no new compute, and it routes around your own impossibility result, which is the strongest interview hook you have.

---

## 3. D2 — Label-efficient certification under concept shift (the cost)

### 3.1 The reframe (do not skip)

Certifying a fixed rule at a fixed threshold is fixed-mean estimation, and active labeling does not lower the labels needed for an unbiased estimate of a fixed mean. The naive "active learning saves labels" framing is ill-posed and a judge will kill it. D2 is real only as one of: variance-reduced certification (PPI / control-variate) or joint threshold-selection-and-certification. This plan takes the variance-reduced route because it maps to a named published method and to a shippable deliverable.

### 3.2 Honest deliverable vs the one novel morsel

- **Reliable deliverable (never sold as the contribution):** prediction-powered risk control instantiated on the accepted-pose ratio risk. The `σ² → σ²(1 − ρ²)` variance reduction is off-the-shelf PPI (Angelopoulos-Zrnic `2301.09633`; semi-supervised risk control `2412.11174`; active-multiple-PPI `2605.08429`; PPI-through-CP `2510.16166`). The bare margin query-complexity is CSA `2605.20270` Thm 5.4.
- **The single differentiated morsel:** a **domain-grounded DRO ambiguity ball** whose radius is calibrated to the measured feature-conditional Bayes noise of co-folding pose error on RNP. Wasserstein-regularized CP (`2501.13430`) and pseudo-calibrated CP under shift (`2602.14913`) decompose the coverage gap into covariate and concept parts but do not tie the ambiguity radius to a measured domain noise floor and do not ask label query complexity.
- **The empirical headline:** a per-stratum **feasibility map** for co-folding pose certification on RNP — where `m_g > 0` and target labels certify, versus where `m_g ≤ 0` and no label-free certificate can (the impossibility regime the project theorem predicts).

### 3.3 Measured realities to state up front (from the pilot in `FUTURE_WORK_PLAN.md`)

- Query unit is the **target**, not the (target, model) pose. RNP clusters ~5 predictions per complex; pooling them as independent Bernoulli draws is anti-conservative and changes the estimand. Report budgets in independent target-labels, per deployed model.
- The control-variate saving is small on feasible strata (`|ρ_g| ≈ 0.22-0.25`, so variance kept `≈ 0.94-0.95`, a ~5% saving). Do not sell it as a speedup. The robust win over passive Hoeffding-RCPS is empirical-Bernstein variance-adaptivity for small-`p` Bernoulli (accepted error rate `p ≈ 0.08-0.18`), which needs no covariate.
- Disagreement does NOT collapse on high-novelty strata; `ρ_g` is larger there (`≈ 0.43`) than on feasible strata (`≈ 0.22`). The honest finding is a feasibility mismatch: the covariate carries the most signal exactly where `m_g < 0` makes certification impossible. Report the measured profile, do not predict a collapse.

### 3.4 Target theorems

- **T1 (feasibility map).** Characterize, per novelty stratum, the sign of `m_g = α − R_Q,g(τ_c)` and the resulting certifiability, tying `m_g ≤ 0` to the Theorem 1 concept floor.
- **T2 (domain-grounded DRO radius).** Given a Wasserstein ambiguity ball of radius `ε` around the source accept-region loss law, the worst-case certifiable margin shrinks by a computable function of `ε`; fix `ε` to the measured feature-conditional Bayes-noise floor of co-folding pose error on RNP, giving a domain-calibrated rather than generic robustness statement.
- **T3 (control-variate measurement).** Report whether cross-model / ensemble disagreement is a usable PPI control covariate along the training-similarity axis, with its feasibility mismatch, as a measurement rather than a predicted theorem.

### 3.5 Estimator (get this right — it is a common failure point)

The control-variate variable `L − β(c − E[c])` is unbounded, so it cannot go into a WSR empirical-Bernstein bet while keeping the half-width. Build the certificate on PPI / sequential PPI++ for the control-variate estimand and a plain bounded WSR confidence sequence for the label-only estimand. Fix `β` on a held-out split or pre-register it; never estimate it on the same scarce labels it corrects. State `(1 − ρ²)` as an asymptotic upper limit not realized at tiny `n_g`.

### 3.6 Milestones and gates

- **Prerequisite:** D1 lands first. D2 only coheres as the cost side of the bracket once the escape (D1) exists.
- **Week 1 — MVP + go/no-go.** Build the feasibility map on RNP strata (needs only the delivered `correct` + strata + `α`). Gate: there must exist strata with `m_g > 0` and enough independent target-labels to make a certificate non-vacuous AND strata with `m_g ≤ 0` to show the impossibility regime. If the whole axis is one-sided, the map is trivial; reassess.
- **Weeks 2-3 — PPI certificate + Bernstein baseline.** Implement prediction-powered risk control on the accepted-pose ratio; show the empirical-Bernstein small-`p` win over passive Hoeffding-RCPS; report the measured `ρ_g` profile and the mostly-null control-variate saving honestly.
- **Weeks 4-5 — DRO morsel + write-up.** Calibrate the ambiguity radius to measured Bayes noise; state T2 with the reduced, defensible scope. Cut the f-divergence/CVaR generalization to stated future work; do NOT let the paper depend on any lower bound closing.

### 3.7 Risks

- **Ill-posed if not reframed.** Section 3.1 is non-negotiable.
- **The novel morsel is small.** That is fine. The paper stands on the feasibility map plus the PPI certificate; the DRO radius is the differentiated flourish, not the load.
- **Single-dataset certification.** Only RNP ships both coordinates and labels. State the ceiling plainly.

---

## 4. Why D1 + D2 cohere (the STS bracket)

- **Impossibility (done, your paper):** label-free certificates under-report realized risk by the concept gap.
- **D1, the escape:** certify danger label-free via cross-model pose disagreement, on the side the impossibility does not forbid.
- **D2, the cost:** when you must spend labels, map exactly where they certify and how few you need, with the feasibility mismatch that ties back to the impossibility regime.

If D1 and D2 both land and cohere, the impossibility → escape → cost bracket is finalist-grade. If D2 stalls, D1 alone is Top-300-viable with no scramble. The interview strength is that you can say precisely what is not yours (the pairwise floor, the PPI mechanism, the bare margin rate) and defend exactly what is (the metric conversion, the diverse-vs-consensus decomposition, the reconciliation, the domain-grounded feasibility map and DRO radius).

## 5. Immediate next actions

1. D1 Week-1 MVP: single-frame verification on a 50-target raw-`.cif` sample, then the floor-vs-realized-risk figure per novelty stratum. Mostly supported by the delivered parquet already.
2. Add the three fresh citations (`2606.08517`, `2603.08907`, `2510.16166`) to `REFERENCES.bib`.
3. Set a watch on the Mac1 coordinate release; revisit item #1 when it drops.
