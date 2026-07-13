# THEORY LENS conformal  (Phase 0 grounding, 2026-07-12)

Web tools are up; I grounded all six prior-art anchors (verified arXiv/journal IDs inline). Here is the full construction.

---

# Training-free selective risk control under concept shift: an impossibility–achievability bracket

## 0. Notation and scope

Frozen base score `s = s(X) ∈ ℝ` (co-folding ipTM/ranking score; higher ⇒ nominally more reliable). Binary loss `Y ∈ {0,1}`, `Y=1` = error (ligand-RMSD > 2 Å). Novelty coordinate `ν = ν(X) ∈ ℝ` (training-set similarity). Selective rule: **accept iff `s ≥ τ`**, else abstain.

For a distribution `D` and threshold `τ`:
- coverage `c(τ;D) = D(s ≥ τ)` (fraction accepted),
- selective risk `R(τ;D) = E_D[Y | s ≥ τ]`.

`P` = source/calibration distribution (train-similar region where CP is calibrated). `Q` = deployment/target distribution on the novel region `T`. Per-covariate error probabilities:
`η_P(x) = P(Y=1 | X=x)`, `η_Q(x) = Q(Y=1 | X=x)`.

---

## 1. Assumptions (stated precisely; each is used)

- **A1 (frozen score).** `s(·)` is a fixed measurable map, not retrained. The selective family under study is `{accept iff s ≥ τ : τ ∈ ℝ}` — *scalar thresholds of a fixed score*. (This is the deployed foldgate rule.)
- **A2 (atomless score at the operating point).** Under both `P` and `Q`, `s` has no atom at the relevant thresholds, so for every target coverage `c ∈ (0,1)` there is a unique `τ_c^Q` with `Q(s ≥ τ_c^Q) = c` (and likewise `τ_c^P` under `P`). Removes tie-breaking bookkeeping.
- **A3 (label-free covariate reweighting = the "training-free reweighting" class 𝒲).** A member is any weight `w : 𝒳 → ℝ_{≥0}` that is **covariate-measurable** (`σ(X)`-measurable) and does **not** use target labels. Calibration reweights the joint law of `(X, s, Y)` by `w(X)`. This includes exact covariate-shift weights `w* = dQ_X/dP_X`, estimated weights (train-vs-test classifier `w ∝ ĉ/(1−ĉ)`, KLIEP, uLSIF), and the `+∞` point mass of weighted CP. It excludes anything that reads `Y` on the target or changes the accept-region *shape* as a function of `ν` (that is Theorem 2's class).
- **A4 (reliability sufficiency of `(s,ν)`).** `P(Y=1|X) = η_P(s,ν)` and `Q(Y=1|X) = η_Q(s,ν)`; i.e. `(s,ν)` is a sufficient covariate summary for reliability. This lets "concept drift" be a function of `(s,ν)` and makes `ν` the correct stratifier. (When A4 fails, Theorem 2 degrades gracefully — see Limitation L4.)
- **A5 (exchangeable in-stratum labels — Theorem 2 only).** For achievability we additionally observe `n_g` labeled points drawn exchangeably with the test point from `Q_g := Q(· | ν ∈ g)`. This is the price of beating the impossibility; it is *not* training-free in the label sense (it is training-free only of the base model).

**Definition (concept drift).** `Δ(s,ν) := η_Q(s,ν) − η_P(s,ν)`. This is exactly E19's measured quantity: movement of `P(Y=1 | s, ν)` at **fixed `s`** (and fixed `ν`). Its accept-region average at target coverage `c`:
```
Δ̄_c := E_Q[ Δ(s,ν) | s ≥ τ_c^Q ].
```

---

## 2. Core lemma: reweighting only ever uses the source conditional

**Lemma 1.** For any `w ∈ 𝒲` and any threshold `τ`, the reweighted plug-in selective risk that the calibrator computes is
```
R̃^w(τ) = E_P[ w(X) Y 𝟙{s≥τ} ] / E_P[ w(X) 𝟙{s≥τ} ]
        = E_{Q_w}[ η_P(s,ν) | s ≥ τ ],
```
where `Q_w` is the covariate law with density `∝ w · dP_X`. In particular it depends on the labels only through the **source** conditional `η_P`, never through `η_Q`.

*Proof.* `w` and `𝟙{s≥τ}` are `σ(X)`-measurable, so tower over `X`:
`E_P[w Y 𝟙{s≥τ}] = E_P[ w · η_P(X) · 𝟙{s≥τ} ]`. Normalizing by `E_P[w 𝟙{s≥τ}]` gives the `Q_w`-conditional expectation of `η_P`. No target label enters, so `η_Q` cannot appear. ∎

This is the exact converse face of Tibshirani–Barber–Candès–Ramdas (2019, arXiv:1904.06019): weighted CP is valid **iff** `P(Y|X)` is preserved. Lemma 1 says what it computes *when that assumption is false* — it silently substitutes `η_P` for `η_Q`.

Setting `w = w* = dQ_X/dP_X` gives the best case `Q_{w*} = Q`, hence the covariate-optimal certificate
```
R_ref^cov(c) := E_Q[ η_P(s,ν) | s ≥ τ_c^Q ]   (source conditional, target accept region).
```

---

## 3. Theorem 1 — Impossibility / lower bound (scalar-threshold class)

**Theorem 1.** Under A1–A4, fix a target coverage `c`. Let any `w ∈ 𝒲` be used to calibrate a threshold that realizes coverage `c` on `Q`. Then the **realized** target selective risk admits the exact decomposition
```
R_Q(c) = E_Q[Y | s ≥ τ_c^Q] = R_ref^cov(c) + Δ̄_c,                              (★)
```
and the concept term `Δ̄_c` is **invariant across all `w ∈ 𝒲`**. Consequently
```
   inf_{w ∈ 𝒲}  R_Q^realized(coverage = c ; w)  =  R_ref^cov(c) + Δ̄_c
                                               ≥  Δ̄_c,                          (LB)
```
with equality in the first line for exact weights `w*`, and no covariate-measurable reweighting reduces the additive `Δ̄_c`.

**Corollary 1 (unachievability threshold).** Target selective-risk level `α` is **unachievable at coverage `c` by any training-free reweighting of the fixed score** iff
```
        R_ref^cov(c) + Δ̄_c > α      ⟺      Δ̄_c > α − R_ref^cov(c).            (UNACH)
```
"Concept gap exceeds the risk slack remaining after covariate correction." Below this threshold reweighting still *mis-certifies* (see Cor. 2); above it, even realized risk is pinned above `α` for the frozen score.

**Corollary 2 (silent certificate violation).** Under exact weights, weighted CP *certifies* risk `R_ref^cov(c)` but *realizes* `R_ref^cov(c) + Δ̄_c`. When `Δ̄_c > 0` the delivered guarantee is violated by exactly `Δ̄_c` — the accept region is more error-prone on the target than any covariate correction can reveal from source labels.

### Proof of Theorem 1

*Step 1 (decomposition ★).* By A2 the coverage constraint `Q(s≥τ)=c` has the unique solution `τ_c^Q`, so the target accept region `{s ≥ τ_c^Q}` is pinned independent of `w`. By A4,
`R_Q(c) = E_Q[η_Q(s,ν) | s≥τ_c^Q]`. Add and subtract `η_P`:
```
E_Q[η_Q | ·] = E_Q[η_P | ·] + E_Q[η_Q − η_P | ·] = R_ref^cov(c) + Δ̄_c.
```

*Step 2 (invariance of `Δ̄_c`).* `Δ̄_c` depends only on (i) the fixed function `Δ = η_Q − η_P` and (ii) the fixed accept region `{s ≥ τ_c^Q}`. Neither is a function of `w`: the region is fixed by Step 1, and `Δ` is a per-`(s,ν)` conditional-label object that no `σ(X)`-measurable reweighting of the *marginal* can alter (Lemma 1: `w` moves only the covariate law, and reweighting a conditional expectation over a *fixed* region by a covariate weight that integrates to that same region's mass leaves the region's `Δ`-average unchanged whenever realized coverage is held at `c`). Hence `Δ̄_c` is `w`-invariant.

*Step 3 (lower bound LB and no-reduction).* Any `w∈𝒲` that realizes coverage `c` yields realized risk `= R_ref^cov(c) + Δ̄_c` by ★. If `w` misestimates the covariate weights, it either (a) still realizes coverage `c` — then realized risk is unchanged (region pinned), or (b) realizes coverage `c' ≠ c` — outside the constraint. Under the constraint, realized risk is exactly the floor; since `R_ref^cov(c) ≥ 0`, `≥ Δ̄_c`. The additive `Δ̄_c` is present for every `w`, so `inf_{w} = R_ref^cov(c) + Δ̄_c` and reweighting cannot remove it. ∎

*Corollary 1* is (★) with `R_Q(c) > α`. Since the score is frozen (A1), `R_Q(c)` is the smallest selective risk realizable at coverage `c` on `Q` by *anyone* who only thresholds `s` — reweighting included — so `R_Q(c) > α` makes `α` unachievable at that coverage. *Corollary 2* is Lemma 1 (`certificate = R_ref^cov`) minus (★). ∎

### Where Theorem 1 is tight vs loose

- **Tight** exactly when weights are the true `w* = dQ_X/dP_X` (the `inf` is attained) and A4 holds: then `R_ref^cov(c)+Δ̄_c` is not a bound but the realized value.
- **Loose only upward:** weight-estimation error can only *inflate* the covariate part `R_ref^cov` (or break coverage); it never eats into `Δ̄_c`. So the concept floor is the robust, weight-independent content.
- **Honest scope.** This is proven for the **scalar-score-threshold class (A1)**. It does *not* claim "no reweighting whatsoever can help." A method that rescoring by `ν` (changes the accept-region *shape*) — i.e. group-conditional / new score — is outside 𝒲 and is precisely how Theorem 2 escapes.

### Relation to named prior results (cite, don't reinvent)

- **Barber, Candès, Ramdas, Tibshirani (2021), "The limits of distribution-free conditional predictive inference,"** *Inf. Inference* 10(2):455–482 / arXiv:1903.04684 — distribution-free *conditional coverage* is impossible. Theorem 1 is the **risk-control / concept-shift analog** and, unlike the pure non-existence result, gives the *exact bias* `Δ̄_c`.
- **Tibshirani, Barber, Candès, Ramdas (2019),** NeurIPS / arXiv:1904.06019 — weighted CP is valid under covariate shift *only* (`P(Y|X)` fixed). Lemma 1 + Cor. 2 are the sharp **negative complement**: when `P(Y|X)` moves, the uncorrectable residual is exactly `Δ̄_c`.
- **Ben-David, Blitzer, Crammer, Kulesza, Pereira, Vaughan (2010), "A theory of learning from different domains,"** *Mach. Learn.* 79 — target error `≤` source error `+ ½ d_{HΔH} + λ`, where the `λ` (joint-optimal / combined error) term survives any covariate alignment. `Δ̄_c` is the **selective-prediction, per-accept-region incarnation of that irreducible `λ`**.

---

## 4. Theorem 2 — Achievability / matching upper bound

Two regimes, matching two label budgets.

### 4a. In-stratum recalibration hits the floor (needs A5)

**Theorem 2a.** Under A1, A2, A4, A5, run split-conformal / conformal-risk-control **within the test point's `ν`-stratum `g`** on the `n_g` exchangeable in-stratum labels, choosing the largest accept threshold whose in-stratum empirical selective risk meets `α`. Then:

*(coverage-form, the `1/(n_g+1)` slack the brief asks for).* The stratum-conditional selective coverage/risk guarantee holds with the finite-sample split-CP correction,
```
E_{Q_g}[Y | s ≥ τ̂_g]  ≤  α  +  1/(n_g + 1),
```
equivalently the guarantee is exact at the inflated level `⌈(1−α)(n_g+1)⌉ / (n_g)`. (Standard inductive-conformal quantile; the constant is the Mondrian/group-conditional split-CP correction.)

*(high-probability risk-form, RCPS/LTT/QRC).* Using a Hoeffding–Bentkus upper confidence bound (RCPS; Bates, Angelopoulos, Lei, Malik, Jordan, JACM 2021), or Learn-then-Test for the non-monotone case (Angelopoulos et al. 2021, arXiv:2110.01052), or the quantile-risk certificate of Snell et al. (Quantile Risk Control, arXiv:2212.13629, ICLR 2023):
```
P( R_{Q_g}(τ̂_g) ≤ α ) ≥ 1 − δ,   slack width = O( sqrt( log(1/δ) / n_g ) ).
```

**Why 2a beats Theorem 1:** the in-stratum labels are drawn from `Q_g`, so the empirical risk it calibrates against *is* `η_Q` on that stratum. It corrects the concept term, not just the covariate term — because it changes the accept-region *shape across `ν`* (a per-stratum `τ̂_g`), leaving class 𝒲. Formally the achieved floor is
```
R_{Q_g}(c_g) = E_{Q_g}[η_Q | s ≥ τ̂_g] = R_ref^cov,g(c_g) + Δ̄_{c_g,g},
```
now *realized-and-certified together* up to the slack — the concept term is inside the calibrated quantity. So target risk `α` is achievable whenever `α ≥ R_{Q_g}(c_g) + slack`, i.e. whenever `α` is information-theoretically feasible for the frozen score on that stratum.

*Proof sketch.* A5 makes the `n_g+1` in-stratum scores (calibration + test) exchangeable, so the rank of the test nonconformity score is uniform on `{1,…,n_g+1}`; the standard inductive-conformal argument gives the `1/(n_g+1)` two-sided bracket on stratum-conditional coverage, and its selective-risk restatement is conformal risk control on the monotone loss `ℓ(τ)=𝟙{Y=1, s≥τ}` normalized by coverage. The high-probability form replaces the exchangeability quantile by a Hoeffding–Bentkus UCB and inverts the test (RCPS/LTT). Monotonicity of selective risk in `τ` holds when `s` is informative; otherwise use LTT's multiple-testing form (no monotonicity needed). ∎

### 4b. Label-free robust certificate brackets the achievable region (no A5)

Without target labels you cannot pinpoint `Δ̄_c`, but you can *bound* it by assuming a shift-magnitude ball and paying conservativeness. This is the achievability *ceiling* that matches Theorem 1's *floor*.

**Theorem 2b (f-divergence DRO / worst-subpopulation).**

*(i) f-divergence ball — Cauchois, Gupta, Duchi, "Robust Validation," JASA 119(548), arXiv:2008.04267.* For every `Q` with `D_f(Q‖P) ≤ ρ`, inflating the source quantile to the DRO worst-case level yields valid coverage/risk on `Q`. For the `χ²`-ball (`f(t)=(t−1)²`), the worst-case accept-region risk has the closed-form variance-expansion certificate
```
R_Q(c) ≤ E_P[Y | s ≥ τ] + sqrt( ρ · Var_P(Y | s ≥ τ) ),   ∀ Q : χ²(Q‖P) ≤ ρ.
```
This *covers* concept drift as long as the true `(η_Q,Q_X)` lies in the ball — the certificate is valid but conservative, and it becomes **vacuous when `ρ` underestimates the realized concept drift** (Cor. 1's regime).

*(ii) Worst-subpopulation mass — Snell et al. QRC / CVaR.* Controlling `CVaR_β` of the loss certifies that every subpopulation of mass `≥ m = β` has selective risk `≤ α`. This traces the achievable `(coverage c, worst-subpop-mass m, risk α)` frontier.

**Bracket (impossibility ∧ achievability).** Combine (★) and 2a: for the frozen score at coverage `c` on target stratum `g`,
```
 Theorem 1 floor    ≤   min achievable selective risk   ≤   Theorem 2a ceiling
 R_ref^cov + Δ̄_c    =            R_Q(c)                  ≤   R_Q(c) + 1/(n_g+1).
```
The floor (no one thresholding `s` beats it) and the ceiling (in-stratum recalibration attains it) **meet at `R_Q(c)` up to `1/(n_g+1)`** — a tight characterization of the achievable region: covariate reweighting *cannot* certify below `R_ref^cov + Δ̄_c`; in-stratum recalibration *does* certify at `R_Q(c)`; and `α < R_Q(c)` is impossible for everyone (must drop coverage). The DRO/CVaR route (2b) fills the label-free corner with a valid-but-loose ceiling whose looseness equals the assumed-vs-true shift gap.

---

## 5. Honest limitations

- **L1 (scope of impossibility).** Theorem 1 binds only the *scalar-threshold-of-fixed-`s`* class 𝒲 (A1, A3). Rescoring with `ν`, stacking a second score, or per-stratum thresholds escape it — by design, that is Theorem 2.
- **L2 (achievability is not label-free).** 2a needs `n_g` *exchangeable target-stratum labels* (A5). "Training-free" here means training-free *of the base co-folding model*, not free of deployment-region labels. Truly zero-target-label deployment gets only the conservative 2b certificate.
- **L3 (thin strata → vacuous guarantee).** The `1/(n_g+1)` (or `√(log(1/δ)/n_g)`) slack blows up for small `n_g`; report per-stratum `n_g` and flag strata where the guarantee is vacuous (CLAUDE.md rule 5).
- **L4 (stratifier misspecification, A4).** If `ν` does not capture the concept-drift direction, Mondrian on `ν` leaves *within-stratum residual* concept drift `Δ̄_{c,g}^{resid}`; 2a then controls only `R_{Q_g} + Δ̄_{c,g}^{resid}`. The residual is the projection of `Δ` orthogonal to `σ(ν)` — measure it, don't assume it away.
- **L5 (DRO radius).** 2b is only as valid as the assumed `ρ`; an under-set `ρ` re-inherits Cor. 1's violation. `ρ` should be *upper*-estimated to cover measured concept drift.
- **L6 (atoms).** A2 excludes score atoms at the operating threshold; with ties, coverage is set-valued and all statements hold with `≤`/`≥` bracketing rather than equality.

---

## 6. Numerical-validation protocol (known `P(Y|s,ν)`, bound checkable)

**Generator (closed-form `η_P`, `η_Q`, hence exact floor).**
1. `ν_P ~ Beta(2,5)` (source concentrated at low novelty), `ν_Q ~ Beta(5,2)` (target novel). — realizes covariate shift in `ν`.
2. `s | ν ~ N(μ0 − κ·ν, σ²)`, `κ>0` (novel ⇒ lower confidence). — covariate-shift channel in `s`.
3. Labels: `η_D(s,ν) = sigmoid(a − b·s + d_D·ν)`, with `d_P < d_Q`. The term `d_D·ν` at fixed `s` is the **concept channel**; `Δ(s,ν) = sigmoid(a−bs+d_Q ν) − sigmoid(a−bs+d_P ν)` is known in closed form.

**Checks.**
- **(C1) Exact floor & decomposition.** By numerical integration compute `R_Q(c)`, `R_ref^cov(c)=E_Q[η_P|s≥τ_c^Q]`, `Δ̄_c`; verify `R_Q(c) = R_ref^cov(c)+Δ̄_c` (Eq. ★) and reproduce an E19-style split (e.g. total ≈ 0.328 = covariate 0.018 + concept 0.309 for a tuned `(κ,d_P,d_Q)`).
- **(C2) Impossibility.** Grid a family of weights `w` (exact `w*`, plus deliberately misestimated), run weighted split-CP to pick `τ̂(w)`, measure realized risk on a large target sample. Confirm: (a) realized risk ≥ `R_ref^cov(c)+Δ̄_c` for *every* `w`, attaining it at `w*` (LB tight); (b) the *certificate* `R̃^{w*}` undershoots realized by ≈ `Δ̄_c` (Cor. 2); (c) no `w` drops realized risk below the floor.
- **(C3) Achievability.** Mondrian on `ν`-strata: within the test stratum, calibrate `τ̂_g` on `n_g` labeled in-stratum points; verify realized selective-risk violation rate ≤ target and `→ R_Q(c)` with an `O(1/n_g)` gap. Sweep `n_g ∈ {20,50,100,500,2000}` to trace the `1/(n_g+1)` slack; sweep `δ` for the RCPS/QRC high-prob form.
- **(C4) Unachievability crossing.** Sweep `d_Q − d_P` (hence `Δ̄_c`) across `α − R_ref^cov(c)`. Below the crossing: Mondrian hits `α`, covariate reweighting fails to *certify* (violates by `Δ̄_c`). Above it: even Mondrian must lower coverage to hold `α` — the score has no accept region on the target with risk `≤ α`. This empirically locates Corollary 1's threshold.
- **(C5) A4-failure stress.** Add a hidden concept-drift coordinate `ζ ⊥ ν`; confirm Mondrian-on-`ν` now controls only `R_{Q_g} + Δ̄^{resid}` (Limitation L4), and that stratifying on `(ν,ζ)` restores control — a direct test that the *right* stratifier is what buys achievability.

---

## 7. One-paragraph summary for the paper

For the deployed rule "accept a co-folding pose iff its frozen confidence `s ≥ τ`," any training-free covariate reweighting (weighted conformal, importance weights, the `+∞`-mass correction) computes its risk certificate from the **source** error-conditional `η_P` (Lemma 1). The realized target selective risk at coverage `c` therefore splits exactly as `R_Q(c) = R_ref^cov(c) + Δ̄_c` (Thm 1), where the accept-region-averaged concept drift `Δ̄_c` is invariant to the choice of weights; the target risk level `α` is unachievable by any such reweighting precisely when `Δ̄_c > α − R_ref^cov(c)` (Cor. 1), and when it is *not* unachievable the reweighting still silently violates its guarantee by `Δ̄_c` (Cor. 2). Group-conditional (Mondrian) recalibration *within* the target stratum, given `n_g` exchangeable in-stratum labels, attains the same floor up to `1/(n_g+1)` finite-sample slack (Thm 2a), and label-free `f`-divergence-DRO / CVaR-worst-subpopulation certificates bracket the achievable `(c, m, α)` region from above (Thm 2b). Impossibility floor and achievability ceiling meet at `R_Q(c)`: the frozen score's own target accept-region risk is exactly the best selective risk any thresholding method can reach, reachable by in-stratum recalibration and provably not by covariate reweighting. This specializes Ben-David et al.'s irreducible-`λ` domain-adaptation lower bound and the Barber–Candès–Ramdas–Tibshirani conditional-coverage impossibility to structured co-folding pose correctness, and completes the Tibshirani et al. weighted-CP assumption by naming exactly what covariate reweighting cannot fix.

**Sources (grounded this session):** [Barber–Candès–Ramdas–Tibshirani 2021, limits of DF conditional inference](https://arxiv.org/pdf/1903.04684) · [Tibshirani–Barber–Candès–Ramdas 2019, weighted CP under covariate shift](https://arxiv.org/abs/1904.06019) · [Ben-David et al. 2010, learning from different domains](https://www.alexkulesza.com/pubs/adapt_mlj10.pdf) · [Snell et al., Quantile Risk Control, arXiv:2212.13629](https://arxiv.org/abs/2212.13629) · [Cauchois et al., Robust Validation, arXiv:2008.04267 / JASA 2024](https://arxiv.org/abs/2008.04267) · [Bates et al., RCPS / Angelopoulos et al. Learn-then-Test](https://arxiv.org/pdf/2407.17358).