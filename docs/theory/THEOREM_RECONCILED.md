# THEOREM RECONCILED  (Phase 0 grounding, 2026-07-12)

## VERDICT

The core result is **sound and publishable**. It is, at its heart, an exact conditional-expectation identity plus a one-line indistinguishability argument, not a deep theorem — and that is a virtue, because it means it cannot be wrong in the way an intricate bound can be. Both lenses converge on the same correct object. But **both overclaim the achievability side**, and both carry sloppy sub-arguments that a referee would flag. Below is the reconciled construction with the errors named and corrected. The one correction that would otherwise put a *wrong* theorem in the paper is in **§4, Error E1** (the clean `1/(n_g+1)` slack does not hold for a *ratio* selective risk; you must either restrict the guarantee to coverage / joint-error-rate, or move to the LTT/QRC high-probability form).

Everything below is coverage-**pinned** and stated **per stratum**, which is the only fully rigorous framing (both lenses drift between pooled and per-stratum quantities in their "bracket" corollary; see Disagreement D3).

---

## 1. AGREED ASSUMPTIONS AND DEFINITIONS

Source (calibration) law `P`, target (deployment) law `Q` on `(X,Y)`, `Y∈{0,1}`, `Y=1` = error (RMSD > 2 Å). Covariate-measurable coordinates `s=s(X)∈ℝ` (frozen score, higher = nominally safer) and `ν=ν(X)` (novelty). Label conditionals `η_P(x)=P(Y=1|X=x)`, `η_Q(x)=Q(Y=1|X=x)`.

- **A1 (Frozen score).** `s, ν` are fixed measurable maps; nothing retrained.
- **A2 (Rule).** Accept iff `s ≥ τ`. Coverage `c_Q(τ)=Q(s≥τ)`; selective risk `R_Q(τ)=E_Q[Y | s≥τ]`.
- **A3 (Thresholding class 𝒯 — the load-bearing scope).** The accept region is a super-level set `{s≥τ}` of the *fixed scalar* `s` (with boundary randomization for exact coverage). A rule with a *different threshold per ν-stratum* is **not** in 𝒯 — that is the achievability side.
- **A4 (Training-free reweighting 𝒲).** Weights `w=w(X)≥0` are covariate-measurable and estimated **without target labels** (density ratio `dQ_X/dP_X`, classifier `ĉ/(1−ĉ)`, KLIEP/uLSIF, incl. the `+∞` mass). Operationally: the accept set and certificate depend on deployment data **only through `Q_X`**, never through `η_Q`.
- **A6 (No atoms).** `s` has no atom at the operating threshold under `Q` (else boundary randomization; all statements hold with `≤/≥` brackets).
- **A8 (Achievability only).** `n_g` labeled deployment points, exchangeable with the in-stratum test point, from the target stratum `g`.

**Definitions** (all exact conditional expectations, region `A=A_τ={s≥τ}`):
```
R_P(τ)    = E_P[Y|A] = E_P[η_P|A]                (naive source risk)
R_ref(τ)  = E_Q[η_P|A]                            (covariate-corrected reference: Q covariates, P labels)
Δ̄_A(τ)    = E_Q[η_Q − η_P | A] = E_Q[Δ|A]         (accept-region-averaged concept gap)
C_cov(τ)  = R_ref(τ) − R_P(τ)                     (accept-region covariate gap)
Δ(x)      = η_Q(x) − η_P(x)                       (concept drift; E19's fixed-(s,ν) object)
```

**Assumption I demote (see Disagreement D1):** the CP-lens "A4 (reliability sufficiency of `(s,ν)`)" — that `η` depends on `X` only through `(s,ν)` — is **not** needed for the impossibility and should **not** be an assumption of Theorem 1. It is only an *interpretation/stratifier* premise for achievability, and even there it is an approximation (Limitation L4). The DRO lens correctly writes everything at full covariate resolution `Δ(x)`.

---

## 2. AGREED THEOREM STATEMENTS

### Theorem 1 (Impossibility — exact decomposition + reweighting-invariance of realized risk)

Under A1, A2, A3, A6, for every `τ`:

**(a) Exact identity.**
```
R_Q(τ) = R_P(τ) + C_cov(τ) + Δ̄_A(τ) = R_ref(τ) + Δ̄_A(τ).
```
**(b) Reweighting is inert on realized risk.** Any rule in 𝒯 (any `w∈𝒲`) that attains target coverage `c` accepts exactly `{s≥τ_c}` (τ_c pinned by A6), so its **realized** selective risk equals `R_Q(τ_c)`, **independent of `w`**.

**(c) Exact unachievability at coverage `c`.** Selective risk level `α` is achievable at coverage `c` by *some* training-free reweighting in 𝒯 **iff** `R_Q(τ_c) ≤ α`, i.e.
```
Δ̄_A(τ_c) ≤ α − R_ref(τ_c).
```
When the accept-region concept gap exceeds `α − R_ref`, **no** covariate-measurable reweighting of the fixed `s` reaches `α` at coverage `c`.

**(d) Silent certificate violation.** A covariate-reweighted plug-in certificate equals `E_{Q_w}[η_P|A]` (Lemma 1 below); with oracle `w*` it equals `R_ref(τ_c)`, while realized risk is `R_ref(τ_c)+Δ̄_A(τ_c)`. When `Δ̄_A>0` the guarantee is anti-conservative by **exactly `Δ̄_A`**. (Unweighted split-CP certifies `R_P` and is biased by `C_cov+Δ̄_A`; weighted CP removes only `C_cov` from the certificate bias, never `Δ̄_A`.)

**Lemma 1.** For any `w∈𝒲`, the covariate-reweighted plug-in selective risk is
`Ẽ^w(τ)=E_P[w Y 𝟙_A]/E_P[w 𝟙_A]=E_{Q_w}[η_P|A]`, where `dQ_w∝w·dP_X`. It depends on labels only through `η_P` — `η_Q` cannot enter. *(Tower over X; `w,𝟙_A` are σ(X)-measurable.)*

### Theorem 2 (Label-free indistinguishability — the DRO lower edge)

Fix `P`, `Q_X`, and the unlabeled deployment sample. Over the concept-ambiguity set `ℰ={Q : Q_X fixed, η_Q=η_P+Δ, Δ∈ℬ}`, every training-free (A4) certificate is measurable w.r.t. the label-free σ-algebra, hence **constant on ℰ**. For validity on all members,
```
certified(τ) ≥ sup_{Q∈ℰ} R_Q(τ) = R_ref(τ) + sup_{Δ∈ℬ} E_Q[Δ|A].
```
Ball-specific closed forms (state the ball explicitly): `L∞` ball `|Δ|≤B` ⇒ floor `R_ref+B`; worst-subpopulation-of-mass-`m` ball ⇒ floor is `CVaR_m` of accepted error; `χ²`-ball of radius `ρ` ⇒ `R_ref + √(2ρ·Var_Q(η_Q|A))` (Cauchois-type). **No label-free method certifies below this.**

### Theorem 3 (Achievability — in-stratum recalibration, matching upper edge)

Under A1, A2, A8, restrict to the test point's stratum `g` (a per-`ν` threshold — outside 𝒯).

**(a) Coverage, exact.** In-stratum split conformal gives stratum-conditional miscoverage `≤ 1/(n_g+1)`, **regardless of concept drift**, because calibration is on `Q_g` itself.

**(b) Joint accept-error rate, exact (CRC).** For the per-example bounded monotone loss `ℓ_τ(x,y)=y·𝟙{s≥τ}∈[0,1]`, conformal risk control certifies `E_{Q_g}[Y·𝟙{s≥τ̂}] ≤ α` with slack `≤ 1/(n_g+1)`. **This controls the joint "accept-and-err" probability, not the conditional selective risk.**

**(c) Conditional selective risk (the ratio), high-probability.** To certify the *ratio* `E_{Q_g}[Y|s≥τ̂] ≤ α`, use **Learn-then-Test** over a threshold grid (valid super-uniform p-values from an empirical-Bernstein / Hoeffding bound on the ratio) or **Quantile Risk Control**, giving `P(R_{Q_g}(τ̂) ≤ α) ≥ 1−δ` with slack `O(√(log(1/δ)/n_g))`. Equivalently: fix in-stratum coverage `c` via the empirical `s`-quantile (coverage controlled to `1/(n_g+1)` by (a)), and the realized in-stratum selective risk concentrates on `R_{Q_g}(τ_c)` at rate `O_p(1/√n_g)`.

**(d) Label-free ceiling (no A8).** QRC / CVaR on any labeled deployment sample, or the Cauchois `f`-divergence DRO certificate, upper-brackets every subpopulation of mass `≥ m`. This meets Theorem 2's lower edge.

### The bracket (per stratum `g`, coverage `c`) — corrected

```
    covariate reweighting        best achievable selective risk        in-stratum recalibration
    certifies R_ref,g            realized floor = R_{Q_g}(τ_c)          certifies R_{Q_g}(τ_c)
    (silently short by Δ̄_{A,g})  = R_ref,g + Δ̄_{A,g}                    up to O_p(1/√n_g) [ratio]
                                                                        or +1/(n_g+1) [coverage/joint]
```
The floor (nobody in 𝒯 beats it) and the ceiling (in-stratum recalibration attains it) **meet at `R_{Q_g}(τ_c)` = the frozen score's own in-stratum target accept-region risk.** The price of closing the *certification* gap `Δ̄_{A,g}` is `n_g` in-stratum labels.

---

## 3. STEP-BY-STEP PROOF (checked)

**T1(a).** `A={s≥τ}∈σ(X)` ⇒ tower: `R_Q(τ)=E_Q[Y|A]=E_Q[η_Q|A]`. Add/subtract `E_Q[η_P|A]`: `= R_ref + E_Q[η_Q−η_P|A] = R_ref + Δ̄_A`. Then `R_ref = R_P + (R_ref−R_P) = R_P + C_cov`. Exact. ✔ *(both lenses correct)*

**T1(b).** Under A3 the accept event is `{s≥τ}`; A6 pins `τ_c` from `c=Q(s≥τ)`. Realized risk `E_Q[Y|s≥τ_c]=R_Q(τ_c)` is a functional of `(Q,τ_c)` only; `w` enters the *method* solely by choosing which `τ` it reports, and once `c` is fixed, `τ_c` is fixed. ✔ **Use the DRO lens's one-line version.** The **CP lens's Step 2 is muddled** (Error E2 below): it tries to argue "reweighting a conditional expectation over a fixed region … leaves the region's Δ-average unchanged," which is a confused restatement. `Δ̄_A` is *defined* under `Q`; `w` never enters its definition. Drop that paragraph.

**T1(c).** Immediate from (a)+(b): at coverage `c`, realized risk is exactly `R_Q(τ_c)=R_ref+Δ̄_A`; `≤α` iff `Δ̄_A ≤ α−R_ref`. ✔ **No monotonicity of `R_Q(τ)` in `τ` is needed** because the statement is pinned at coverage `c`. (If you instead want "no coverage `≥c` achieves `α`," you *do* need `R_Q` monotone in coverage — an extra, unproven premise. State the coverage-pinned version only. See Scope S1.)

**Lemma 1.** `E_P[w Y 𝟙_A]=E_P[w·η_P·𝟙_A]` by tower; normalize. ✔

**T2.** A certificate that is a function of `(P,Q_X, unlabeled Q)` is constant across `ℰ` (all members share those). Validity on each member ⇒ `≥ sup_{ℰ} R_Q`. `E_Q[Δ|A]` is linear in `Δ`; maximize over `ℬ`. ✔ **Caveat (Error E3):** the CVaR/`f`-divergence closed forms are worst cases over *covariate reweightings of a fixed loss*, which is a **different ambiguity model** than "vary `η_Q` pointwise in `ℬ`." Both give valid label-free lower edges, but they are not the same ball — the paper must pick one and state it, not present them as interchangeable. The `L∞` form (`R_ref+B`) is the clean, assumption-light one and I recommend it as the headline lower edge; present CVaR/χ² as the practically tighter alternative under an explicitly assumed ball.

**T3(a).** In-stratum exchangeability ⇒ test conformity rank uniform on `{1,…,n_g+1}` ⇒ `1/(n_g+1)` miscoverage. ✔

**T3(b).** CRC (Angelopoulos et al.) on the **per-example** bounded monotone loss `ℓ_τ=Y·𝟙{s≥τ}` ⇒ `E[ℓ_{τ̂}]≤α+1/(n_g+1)`. ✔ **But this is the joint error rate `E[Y·𝟙_A]`, not `E[Y|A]`.**

**T3(c) — Error E1 (the important one).** **Both lenses claim the clean `E[selective risk] ≤ α + 1/(n_g+1)` via CRC. This is WRONG for the ratio.** Selective risk `E[Y|s≥τ]=E[Y·𝟙_A]/P(s≥τ)` is a ratio of two expectations with a *random* denominator; it is **not** a per-example bounded monotone loss, so vanilla CRC does not apply. The CP lens papers over this with "…normalized by coverage" and the DRO lens with "loss = accepted-error, bounded in [0,1]" — both hand-wave the denominator. **Correct achievability for the conditional selective risk is the high-probability LTT/QRC form with `O(√(log(1/δ)/n_g))` slack**, or the coverage-pinned concentration `O_p(1/√n_g)`. Keep `1/(n_g+1)` *only* for the coverage guarantee (a) and the joint-error-rate guarantee (b). This correction must propagate into the bracket corollary and into numerical check C3.

**T3(d).** CVaR_m = sup over mass-`≥m` reweightings (variational identity); QRC gives a distribution-free UCB. ✔

---

## 4. DISAGREEMENTS BETWEEN THE TWO LENSES — RESOLVED

**D1 — Is `(s,ν)`-sufficiency (CP-lens A4) an assumption of the impossibility?**
CP lens: yes (bakes it into Theorem 1). DRO lens: no (uses `Δ(x)` at full resolution). **DRO lens is correct.** The decomposition and reweighting-invariance need only `A∈σ(X)`. Demote `(s,ν)`-sufficiency to a stratifier/interpretation premise used solely to (i) call `ν` "the right stratifier" and (ii) tie `Δ̄_A` to E19's fixed-`(s,ν)` number. Even there it is approximate (L4). **Fix: remove it from Theorem 1's hypotheses.**

**D2 — The invariance-of-`Δ̄_A` argument.**
CP lens gives a tangled "reweighting-integrates-to-the-same-mass" justification (partly incorrect). DRO lens gives the clean "realized risk is a functional of `(Q,τ_c)` alone." **DRO lens is correct; use it verbatim.**

**D3 — Pooled vs per-stratum bracket.**
Both lenses' final "bracket" corollary slides between the *pooled* floor `R_Q(τ_c)` and *per-stratum* achievability `R_{Q_g}`, and asserts they "meet." They meet cleanly only **per stratum** (floor `R_{Q_g}(τ_c)` vs ceiling `R_{Q_g}(τ_c)`). The pooled statement requires aggregating strata and is not what the achievability construction controls. **Fix: state the bracket per stratum** (as in §2). Neither lens is "wrong," but the per-stratum framing is the only one that is airtight.

**D4 — CRC gives clean `1/(n_g+1)` for selective risk.**
Both assert it; **both are wrong for the ratio** (Error E1). Resolution: coverage/joint-error → `1/(n_g+1)`; conditional selective risk → LTT/QRC `O(√(log(1/δ)/n_g))`.

**D5 — χ²-DRO constant.** CP lens: `√(ρ·Var)`. DRO lens: `√(2ρ·Var)`. **DRO lens's `√(2ρ·Var)` matches the standard Duchi–Namkoong/Cauchois first-order dual;** the CP lens dropped the factor 2. Minor, but state the exact constant with the χ² convention you adopt.

**D6 — Citations.** CP-lens source list links RCPS/LTT to `arXiv:2407.17358`, which is **not** RCPS. Correct RCPS = **arXiv:2101.02703** (Bates et al., JACM 2021); LTT = **arXiv:2110.01052**. CP lens also omits co-author **Alnur Ali** on Cauchois et al. (see §6). Use the corrected list.

No disagreement is fatal. The result survives all of them.

---

## 5. SCOPE / LIMITATIONS TO STATE IN THE PAPER (what is NOT proven)

- **S1 (Coverage-pinned, not "any coverage").** Theorem 1(c) proves unachievability **at a fixed coverage `c`**. It does **not** prove you cannot reach risk `α` by *lowering* coverage — that stronger claim needs `R_Q(τ)` monotone in coverage (i.e. `s` informative on `Q`), which is unproven and often false near the novel tail. State the coverage-pinned version and, separately, note that dropping coverage is the honest escape *within* 𝒯.
- **S2 (Scope of impossibility = 𝒯).** It binds *scalar thresholds of the fixed `s` with covariate-measurable label-free weights*. It does **not** say "nothing can fix concept shift." Rescoring on `ν`, stacking a second score, or per-stratum thresholds all escape — that is exactly Theorem 3, and it costs in-stratum labels.
- **S3 (Achievability is not label-free).** Theorem 3(a–c) needs `n_g` **exchangeable target-stratum labels**. "Training-free" = free of retraining the base model, **not** free of deployment labels. Truly zero-target-label deployment gets only the conservative DRO/CVaR ceiling (3d) — which is vacuous if the assumed radius under-covers the true drift.
- **S4 (Identifiability of the concept/covariate split).** `C_cov` and `Δ̄_A` require `η_P` evaluated on `Q`'s support. On genuinely novel pockets `supp(Q_X)⊄supp(P_X)`, so `η_P` there is an **extrapolation** and the `0.309 / 0.018` split is **model-relative** (Ben-David representation-relativity). Report `ν` and the `η_P` model explicitly; do not present the split as identifiable.
- **S5 (Weighted CP may be undefined, not just biased).** On novel regions the density ratio `dQ_X/dP_X` can be unbounded/undefined, so the weighted-CP baseline can fail to *exist*, independent of the concept-shift bias. The `+∞` point mass "handles" it only by making coverage trivial.
- **S6 (Thin/coarse strata).** `1/(n_g+1)` and `√(log(1/δ)/n_g)` blow up for small `n_g`; report per-stratum `n_g` and flag vacuous strata (CLAUDE.md rule 5). A mis-specified stratifier leaves within-stratum residual concept drift `Δ̄^resid` that Theorem 3 does **not** control (L4).
- **S7 (Ratio-risk caveat).** The clean `1/(n_g+1)` is for coverage and joint error rate; conditional selective risk carries the `O(√(log(1/δ)/n_g))` high-probability slack.
- **S8 (Atoms/ties).** ipTM ties need randomization; all equalities become `≤/≥` brackets.

---

## 6. NUMERICAL-VALIDATION PROTOCOL

**Generator (closed-form `η`, so every term is exact).**
- `ν_P ~ Beta(2,5)`, `ν_Q ~ Beta(5,2)` (or `ν~Unif[0,1]` with `dQ_X/dP_X ∝ e^{κν}`, `κ` known) — covariate shift toward novel.
- `s | ν ~ N(μ0 − κ_s·ν, σ²)`, `κ_s>0` — confidence degrades with novelty.
- Labels: `η_P(s,ν)=sigmoid(a − b·s + d·ν)`; concept drift `η_Q(s,ν)=sigmoid(a − b·s + d·ν + D·ν)`. Then `Δ(s,ν)` is closed-form and grows with `ν` at fixed `s` (E19 signature). Tune `(κ_s, d, D)` to reproduce an af3-S3-style `0.328 ≈ 0.018 cov + 0.309 concept`.

**Checks (pass criteria):**
- **C1 — Decomposition.** Monte-Carlo `R_Q, R_P, C_cov, Δ̄_A`; verify `R_Q = R_P + C_cov + Δ̄_A` to `< 3×` MC SE.
- **C2 — Reweighting floor (T1 b/c/d).** Sweep a rich family of `w(ν)` (oracle `w*` + deliberately wrong). For each, drive to coverage `c`, record **realized** `R_Q`. Pass: realized `≡ R_Q(τ_c)` across all `w` (variance at MC level); `min_w` realized `= R_Q(τ_c) > α` in the impossibility regime; the weighted-CP **certificate** tracks `R_ref` and under-reports realized by exactly `Δ̄_A`. **Control `D=0`:** weighted CP with `w*` hits `α` (impossibility is concept-specific).
- **C3 — Achievability (T3), CORRECTED.** In the novel stratum, calibrate with **LTT/QRC** (not vanilla CRC) using `n_g` in-stratum labels, over many trials. Pass: exceedance rate `≤ δ` for the *conditional* selective risk; coverage envelope `≤ 1/(n_g+1)`; slack shrinks as `O(1/√n_g)`. Separately verify the *joint*-error CRC bound at `1/(n_g+1)` to demonstrate the two guarantees differ. Mondrian succeeds exactly where C2's reweighting fails.
- **C4 — Unachievability crossing.** Sweep `D` (hence `Δ̄_A`) across `α − R_ref`. Below: Mondrian hits `α`, reweighting fails to *certify* (violates by `Δ̄_A`). Above: even Mondrian must drop coverage to hold `α`. Locates Theorem 1(c)'s threshold empirically.
- **C5 — Stratifier misspecification (L4/S4).** Add a hidden concept coordinate `ζ ⊥ ν`. Confirm Mondrian-on-`ν` controls only `R_{Q_g}+Δ̄^resid`; stratifying on `(ν,ζ)` restores control. Direct test that the *right* stratifier is what buys achievability.
- **C6 — DRO ceiling (T2/T3d).** Compute the `L∞`-ball floor `R_ref+B` and the `χ²`/CVaR ceiling with the **`√(2ρ·Var)`** constant; verify the label-free certificate upper-brackets realized risk and tightens as strata refine.

---

## 7. CITATIONS (corrected)

- **Weighted CP = covariate-shift-only.** Tibshirani, Barber, Candès, Ramdas, *Conformal Prediction Under Covariate Shift*, NeurIPS 2019 — **arXiv:1904.06019**. (Theorem 1 specializes its `Q(Y|X)=P(Y|X)` boundary and names the residual `Δ̄_A`.)
- **Distribution-free conditional impossibility.** Barber, Candès, Ramdas, Tibshirani, *The limits of distribution-free conditional predictive inference*, Information and Inference 10(2):455–482, 2021 — **arXiv:1903.04684**. (Theorem 2 analogue.)
- **Domain-adaptation lower bound.** Ben-David, Blitzer, Crammer, Kulesza, Pereira, Vaughan, *A theory of learning from different domains*, MLJ 79, 2010. (`C_cov` ↔ `d_{HΔH}` correctable; `Δ̄_A` ↔ irreducible `λ`.)
- **`f`-divergence DRO robust validation.** Cauchois, Gupta, **Ali**, Duchi, *Robust Validation: Confident Predictions Even When Distributions Shift*, JASA 2024 — **arXiv:2008.04267**. (DRO ceiling/floor; note the fourth author Alnur Ali, omitted by the CP lens.)
- **Quantile / CVaR risk control.** Snell, Zollo, Deng, Pitassi, Zemel, *Quantile Risk Control*, ICLR 2023 — **arXiv:2212.13629**. (T3c/T3d worst-subpopulation certificate.)
- **RCPS.** Bates, Angelopoulos, Lei, Malik, Jordan, *Distribution-Free, Risk-Controlling Prediction Sets*, JACM 2021 — **arXiv:2101.02703**. **(Fix: the CP-lens link `2407.17358` is a mis-citation.)**
- **Conformal Risk Control.** Angelopoulos, Bates, Fisch, Lei, Schuster, ICLR 2024 — **arXiv:2208.02814**. (Use for coverage/joint-error `1/(n_g+1)` only — **not** for ratio selective risk.)
- **Learn-then-Test.** Angelopoulos, Bates, Candès, Jordan, Lei — **arXiv:2110.01052**. (Ratio selective-risk achievability, T3c.)

---

### One-paragraph headline for the paper (vetted)

For the deployed rule "accept a co-folding pose iff frozen confidence `s ≥ τ`," any training-free covariate reweighting computes its risk certificate from the **source** error-conditional `η_P` (Lemma 1), so at fixed coverage `c` the realized target selective risk splits exactly as `R_Q(τ_c)=R_ref(τ_c)+Δ̄_A(τ_c)` (Thm 1a), and this realized value is **invariant to the weights** (Thm 1b). Level `α` is unachievable at coverage `c` precisely when the accept-region concept gap exceeds the post-covariate slack, `Δ̄_A>α−R_ref` (Thm 1c); when it is not, weighted CP still silently violates its own guarantee by exactly `Δ̄_A` (Thm 1d). No label-free certificate can do better than the DRO worst case `R_ref+sup_ℬ E_Q[Δ|A]` (Thm 2). In-stratum recalibration, given `n_g` exchangeable target-stratum labels, attains that floor — with `1/(n_g+1)` slack for coverage and joint error, and `O(√(log(1/δ)/n_g))` for the conditional selective risk (Thm 3). Per stratum, impossibility floor and achievability ceiling meet at `R_{Q_g}`, the frozen score's own in-stratum accept-region risk: the best any thresholding method can reach, reachable by stratified recalibration and provably not by covariate reweighting. This specializes Ben-David's irreducible-`λ` bound and the Barber–Candès–Ramdas–Tibshirani conditional-coverage impossibility to co-folding pose correctness, and completes the Tibshirani et al. weighted-CP assumption by naming exactly what covariate reweighting cannot fix.