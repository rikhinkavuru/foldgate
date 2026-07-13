# THEORY LENS domain adapt  (Phase 0 grounding, 2026-07-12)

# Training-Free Selective Risk Control Under Concept Shift: An Impossibility + Achievability Bracket

Constructed independently from the domain-adaptation / DRO lens (Ben-David generalization decomposition, importance-weighting invariants, f-divergence DRO duality). I grounded the citations via web search; all five named prior results were verified live (URLs at the end). Every step below is either an exact identity or an inequality I prove; where a claim needs an assumption I state it, and I flag the honest scope explicitly (this is a **scalar-score-threshold** result, not "no method whatsoever").

---

## 1. Setting and assumptions

We observe a deployment (target) law $Q$ and a calibration (source) law $P$ over $(X, Y)$ with derived, **covariate-measurable** coordinates
$$s = s(X)\in\mathbb R \ (\text{frozen base score, higher} = \text{more confident}),\qquad \nu = \nu(X)\ (\text{novelty}).$$
$Y\in\{0,1\}$, $Y=1$ = "error" (RMSD $>2$Å). Write the two label conditionals
$$\eta_P(x)=P(Y{=}1\mid X{=}x),\qquad \eta_Q(x)=Q(Y{=}1\mid X{=}x).$$

**A1 (Frozen score).** $s,\nu$ are fixed measurable functions of $X$; nothing is retrained. (foldgate's training-free constraint.)

**A2 (Selective predictor / risk).** Accept iff $s\ge\tau$. Coverage $c_Q(\tau)=Q(s\ge\tau)$; selective risk $R_Q(\tau)=E_Q[Y\mid s\ge\tau]$.

**A3 (Thresholding class $\mathcal T$ — the honest scope).** The accept region is a super-level set $A_\tau=\{s\ge\tau\}$ of the *fixed scalar* $s$, optionally with boundary randomization to hit an exact coverage. A rule that reorders points using $\nu$ (a different threshold per $\nu$-stratum) is **not** in $\mathcal T$; it is the achievability side.

**A4 (Training-free reweighting = label-free, covariate-measurable).** Weights $w=w(X)\ge0$ are covariate-measurable and estimated *without deployment labels* — e.g. the density-ratio $w=\mathrm dQ_X/\mathrm dP_X$ fit from unlabeled test covariates, exactly as in weighted conformal (Tibshirani–Barber–Candès–Ramdas 2019). Operationally: **the accept set and the certificate depend on the deployment data only through the covariate law $Q_X$**, never through $\eta_Q$.

**A5 (Calibration).** Labeled sample from $P$; possibly unlabeled covariates from $Q$.

**A6 (Continuity, for exact coverage).** $s$ has no atom at the operating threshold under $Q$ (else use boundary randomization).

**A7 (Concept drift — the E19 object).** $\Delta(x):=\eta_Q(x)-\eta_P(x)$. Evaluated at fixed $(s,\nu)$ this is the **reliability drift** $D(\nu)$: movement of $P(Y{=}1\mid s,\nu)$ along $\nu$ at fixed $s$ (E19 measured af3-S3 gap $0.328 = 0.309$ concept $+\,0.018$ covariate, concept CI excluding 0).

**A8 (Achievability only — in-stratum labels).** $n_g$ i.i.d. *labeled* deployment points from the target stratum $g^\star$.

**Definitions.**
$$R_P(\tau)=E_P[Y\mid s\ge\tau]=E_P[\eta_P\mid A_\tau]\quad(\text{naive source risk}),$$
$$R_{\mathrm{ref}}(\tau)=E_Q[\eta_P(X)\mid s\ge\tau]\quad(\text{covariate-corrected reference risk: deployment covariate law, source labels}),$$
$$\bar\Delta_A(\tau)=E_Q[\Delta(X)\mid s\ge\tau]\quad(\text{accept-region-averaged concept gap}),$$
$$C_{\mathrm{cov}}(\tau)=R_{\mathrm{ref}}(\tau)-R_P(\tau)\quad(\text{accept-region covariate gap}).$$

---

## 2. Impossibility / lower bound

### Theorem 1 (Exact E19 decomposition + reweighting-invariance of realized risk).

**(a) Decomposition (exact identity).** For every $\tau$,
$$\boxed{\,R_Q(\tau)=R_P(\tau)+C_{\mathrm{cov}}(\tau)+\bar\Delta_A(\tau)\,}=R_{\mathrm{ref}}(\tau)+\bar\Delta_A(\tau).$$

**(b) Reweighting cannot move realized risk.** Any rule in $\mathcal T$ (A3) that attains deployment coverage $c$ has realized selective risk exactly $R_Q(\tau_c)$, **independent of the weights $w$** (A4).

**(c) Exact unachievability condition.** Target selective risk $\alpha$ at coverage $c$ (on a novel region $T$) is achievable by *some* training-free reweighting rule in $\mathcal T$ **iff** $R_Q(\tau_c)\le\alpha$, i.e.
$$\boxed{\ \bar\Delta_A(\tau_c)\ \le\ \alpha - R_{\mathrm{ref}}(\tau_c)\ }.$$
When the accept-region concept gap exceeds $\alpha-R_{\mathrm{ref}}$, **no covariate-measurable reweighting of the fixed score $s$** achieves $\alpha$.

*Proof.*
(a) Since $A_\tau=\{s\ge\tau\}\in\sigma(X)$, the tower property gives $R_Q(\tau)=E_Q[Y\mid A]=E_Q[\eta_Q\mid A]$. Add and subtract $E_Q[\eta_P\mid A]$:
$$R_Q=\underbrace{E_Q[\eta_P\mid A]}_{R_{\mathrm{ref}}}+\underbrace{E_Q[\eta_Q-\eta_P\mid A]}_{\bar\Delta_A},\qquad R_{\mathrm{ref}}=R_P+(R_{\mathrm{ref}}-R_P)=R_P+C_{\mathrm{cov}}.$$
All three terms are finite conditional expectations; the identity is exact. $C_{\mathrm{cov}}$ holds $\eta_P$ fixed and moves the covariate law $P\!\to\!Q$ within $A$ (the correctable part); $\bar\Delta_A$ holds the covariate law $Q$ fixed and moves $\eta_P\!\to\!\eta_Q$ (the label-conditional part).

(b) Under A3 the accept event is $\{s\ge\tau\}$ (mod boundary randomization). By A6 the deployment coverage $c=Q(s\ge\tau)$ pins $\tau=\tau_c$. Realized selective risk $=E_Q[Y\mid s\ge\tau_c]=R_Q(\tau_c)$ is a functional of $(Q,\tau_c)$ alone. The weights $w$ enter the *method* only by selecting which $\tau$ it reports; once $c$ is fixed, $\tau_c$ is fixed, so realized risk does not depend on $w$. (With atoms, randomization yields a convex combination of the two adjacent super-level risks — still $w$-independent.)

(c) Immediate from (a)+(b): achievable iff $R_Q(\tau_c)=R_{\mathrm{ref}}(\tau_c)+\bar\Delta_A(\tau_c)\le\alpha$. $\qquad\blacksquare$

**Why reweighting is blind to $\bar\Delta_A$ (the mechanism).** A covariate weight acts as $E_P[w\,Y\,\mathbf 1_A]=E_P[w\,\eta_P\,\mathbf 1_A]$: it *multiplies* the source conditional $\eta_P$ and can at best reproduce the deployment covariate mixture, giving $R_{\mathrm{ref}}$. It can never *convert* $\eta_P$ into $\eta_Q$, because supplying $\Delta=\eta_Q-\eta_P$ requires information no covariate-measurable function of $X$ carries. This is exactly why weighted conformal is a **covariate-shift-only** device (Tibshirani et al. 2019, arXiv:1904.06019): its validity theorem assumes $Q(Y\mid X)=P(Y\mid X)$, i.e. $\Delta\equiv0$. Theorem 1 is the specialization of that boundary to *selective risk*, and it names the residual: the accept-region-averaged concept gap.

**Bridge to achievability.** The escape hatch in Theorem 1 is A3. A rule that uses a *different threshold per $\nu$-stratum* re-orders points by $\nu$ and leaves $\mathcal T$; it is no longer a fixed-scalar-score threshold. That is precisely Mondrian — and Theorem 3 shows it must pay for the escape in **in-stratum labels**.

### Theorem 2 (Label-free indistinguishability: no valid certificate under concept-drift ambiguity — the DRO lower edge).

Fix $P$, the target covariate law $Q_X$, and the unlabeled deployment sample. Let the concept-drift ambiguity set be
$$\mathcal E=\{\,Q:\ Q_X\ \text{fixed},\ \eta_Q=\eta_P+\Delta,\ \Delta\in\mathcal B\,\}$$
for an ambiguity ball $\mathcal B$. Any certificate produced by a training-free (A4) procedure is a function of $(P,Q_X,\text{unlabeled }Q\text{-data})$ only, hence is **constant over $\mathcal E$** (all members induce identical observable, label-free data). For the certificate to be **valid** (certified $\ge$ true) on every member,
$$\text{certified risk}(\tau)\ \ge\ \sup_{Q\in\mathcal E}R_Q(\tau)\ =\ R_{\mathrm{ref}}(\tau)+\sup_{\Delta\in\mathcal B}E_Q[\Delta\mid A_\tau].$$
Consequences by ball:
- **Pointwise $L^\infty$ ball** $\{|\Delta(x)|\le B\}$: sup $=B$, so no label-free method can certify $\alpha<R_{\mathrm{ref}}(\tau)+B$.
- **Worst-subpopulation-of-mass-$m$ ball** ($\{\mathrm dQ'/\mathrm dQ\le 1/m\}$): sup $=\mathrm{CVaR}_m^{A}(\eta_Q)-R_{\mathrm{ref}}$ up to the ball's radius, i.e. the certified floor is a **CVaR / $\chi^2$-DRO** worst-subpopulation risk.

*Proof.* Measurability of the certificate wrt the label-free $\sigma$-algebra forces it constant on $\mathcal E$. Validity on each member forces it $\ge$ the pointwise sup. $E_Q[\Delta\mid A]$ is linear in $\Delta$; maximize over $\mathcal B$. For $L^\infty$, $\Delta\equiv B$ attains $B$. For the mass-$m$ reweighting ball, $\sup_{\mathrm dQ'/\mathrm dQ\le1/m}E_{Q'}[\ell\mid A]=\mathrm{CVaR}_m^A(\ell)$ is the standard CVaR variational identity, and the general $f$-divergence ball gives the Cauchois-type dual ($\chi^2$: mean $+\sqrt{2\rho\,\mathrm{Var}}$). $\qquad\blacksquare$

Theorem 2 is the selective-risk analogue of **Barber–Candès–Ramdas–Tibshirani 2021** ("limits of distribution-free conditional predictive inference"): distribution-free *conditional* (per-$\nu$) validity is impossible without in-region labels, so a label-free certificate must inflate to the DRO worst case. It is also the Ben-David (2010) picture: $C_{\mathrm{cov}}$ is the correctable $d_{\mathcal H\Delta\mathcal H}$-type divergence term, and $\bar\Delta_A$ is the irreducible $\lambda$ (joint-optimal / concept) term that no covariate device removes.

---

## 3. Achievability / matching upper bound

### Theorem 3 (Mondrian in-stratum recalibration + DRO certificate).

**(a) Coverage, exact.** Under A8, restrict to stratum $g^\star$. In-stratum points are exchangeable with the in-stratum test point, so the split-conformal threshold $\hat\tau_{g^\star}$ (the appropriate order statistic of the $n_g$ in-stratum scores) has miscoverage inflation $\le 1/(n_g+1)$ — the standard exact distribution-free split-conformal slack, holding **regardless of concept drift**, because calibration is on the deployment stratum itself.

**(b) Selective risk, finite-sample.** With loss = in-stratum accepted-error (bounded in $[0,1]$, monotone in $\tau$):
- **Conformal Risk Control** (Angelopoulos et al.): the CRC threshold gives $E_Q[\text{loss}\mid g^\star]\le\alpha$ using the $\tfrac{n_g}{n_g+1}\hat R+\tfrac{1}{n_g+1}$ rule, i.e. **slack $\le 1/(n_g+1)$** (loss bound $B=1$).
- **Learn-then-Test** (Angelopoulos–Bates): valid super-uniform p-values over a threshold grid give $P_Q\!\big(R_{\mathrm{sel}}(\hat\tau_{g^\star})\le\alpha\big)\ge1-\delta$, slack $O(\sqrt{\log(1/\delta)/n_g})$.

The point: calibrating on the deployment stratum **samples $\eta_Q$ directly**, so it buys back exactly the $\bar\Delta_A$ term Theorem 1 says covariate reweighting cannot. You cannot dodge the concept gap; you pay $n_g$ in-stratum labels for it.

**(c) Worst-subpopulation / DRO, label-lean.** If stratum identity is unavailable but a worst-subpopulation-of-mass-$m$ guarantee suffices, **Quantile Risk Control** (Snell et al. 2022) / CVaR control on the pooled deployment-labeled sample certifies $\mathrm{CVaR}_m$ of accepted error $\le\alpha$, upper-bracketing every subpopulation of mass $\ge m$ (the novel stratum included, if its mass $\ge m$). This is the achievable version of the **Cauchois et al. 2024** $f$-divergence-ball robust-validation certificate and meets the DRO lower edge of Theorem 2.

*Proof sketch.* (a) exchangeability within $g^\star$ ⇒ the test conformity rank is uniform on $\{1,\dots,n_g+1\}$ ⇒ $1/(n_g+1)$ slack. (b) CRC/LTT applied to the monotone bounded in-stratum loss; the guarantees are distribution-free given in-stratum exchangeability, so concept drift is irrelevant once you calibrate in-distribution. (c) CVaR$_m$ = sup over mass-$\ge m$ reweightings (variational identity), and QRC gives a distribution-free upper confidence bound on that quantile-based risk. $\qquad\blacksquare$

### Corollary (the bracket).

For coverage $c$ and target stratum $g^\star$:
$$\underbrace{R_Q(\tau_c)}_{\substack{\text{realized floor (Thm 1),}\\ \text{label-free certifiable floor }R_{\mathrm{ref}}+\text{worst-concept (Thm 2)}}}\ \le\ \text{true selective risk}\ \le\ \underbrace{R_Q^{g^\star}+\tfrac{1}{n_g+1}}_{\text{Mondrian certificate (Thm 3)}}.$$
As strata refine so that $\eta_Q$ is $\approx$ constant within $g^\star$ (within-stratum $\Delta$ homogeneous) and $n_g\to\infty$, the upper edge $\to$ the in-stratum true risk $=$ the lower edge. **Impossibility and achievability therefore characterize the achievable region up to $O(1/n_g)$ finite-sample slack + within-stratum concept heterogeneity.**

---

## 4. Where it is tight vs loose

- **Theorem 1 is an equality** — always tight; the floor $R_Q(\tau_c)$ and the decomposition are exact, not bounds.
- **Theorem 2 lower edge** is tight when $\mathcal B$ is the genuine uncertainty; loose if side information constrains $\Delta$ (e.g. $\Delta$ monotone in $\nu$ shrinks the sup). Pointwise $L^\infty$ is the loosest; the $f$-divergence/CVaR ball is tighter and matches deployed DRO practice.
- **Theorem 3** is tight when $n_g$ is large and within-stratum $\eta_Q$ is homogeneous; loose (conservative) for small $n_g$ (the $1/(n_g+1)$ slack dominates when $\alpha$ is small — thin-stratum vacuity) or coarse strata (residual within-stratum concept variance re-creates a miniature of the same problem, controlled only by the CVaR term).
- **The whole bracket is tight (edges meet)** iff stratification resolves the concept drift *and* $n_g\gg1/\alpha$; it is loose exactly in the thin/coarse-stratum regime the paper should flag as where the guarantee is vacuous (CLAUDE.md rule 5).

---

## 5. Honest limitations

1. **Scope.** The impossibility is for the *scalar-score-threshold* class (A3). It says covariate reweighting of a fixed $s$ cannot fix concept shift — **not** "nothing can." Mondrian escapes by using $\nu$ beyond reweighting, but only by consuming in-stratum labels. State the headline exactly this way.
2. **Labels in the shifted region.** Theorem 3 needs deployment labels where the shift lives. If the novel stratum is label-barren (Mac1 crystal coords release-delayed), Theorem 3 is vacuous and you must fall back to the conservative DRO/worst-subpop certificate (3c). This is the practically important, honest regime.
3. **Representation-relativity of "concept."** $\Delta$ at fixed $(s,\nu)$ depends on what $\nu$ contains; enriching $\nu$ can convert "concept" into "covariate" (Ben-David representation dependence). E19's $0.309/0.018$ split is conditional on the chosen $\nu$ — report $\nu$ explicitly.
4. **Finite-sample on the reweighting side.** Theorem 1 uses the oracle likelihood ratio; estimated $w$ adds its own error, which can *accidentally* cancel or amplify $\bar\Delta_A$ (Theorem 1(b) is about *realized* risk, which is $w$-invariant; the *certificate* a mis-estimated $w$ reports is not, and is not guaranteed valid). CRC controls expectation, LTT/QRC control high-probability — pick per claim; the selective-risk denominator (coverage) must be handled by fixing coverage or using QRC.
5. **Atoms/ties in $s$** (ipTM ties) need randomization for exact coverage.

---

## 6. Numerical-validation protocol (known $P(Y\mid s,\nu)$)

**Generator (closed-form $\eta$, so every term is exact).**
- $\nu\sim\mathrm{Unif}[0,1]$; source covariate law $P_X$; target covariate shift $\mathrm dQ_X/\mathrm dP_X=w^\star(\nu)\propto e^{\kappa\nu}$ (up-weights novel), $\kappa$ known.
- $s\mid\nu\sim\mathcal N(\mu_0-\mu_1\nu,\sigma^2)$ (confidence degrades with novelty), known.
- Labels: $\eta_P(s,\nu)=\sigma_{\!\log}(a-bs+d\nu)$; **concept drift** $\eta_Q(s,\nu)=\sigma_{\!\log}(a-bs+d\nu+\underbrace{D\,\nu}_{\text{drift}})$, so $\Delta(s,\nu)$ is known in closed form and grows with $\nu$ at fixed $s$ (the E19 signature). Set $D$ so that at a novel stratum $\bar\Delta_A>\alpha-R_{\mathrm{ref}}$ (impossibility regime) and a control run with $D=0$ (pure covariate shift, weighted CP should succeed).

**Checks and pass criteria.**
1. **Decomposition identity.** Estimate $R_Q,R_P,C_{\mathrm{cov}},\bar\Delta_A$ by large-sample Monte Carlo; verify $R_Q=R_P+C_{\mathrm{cov}}+\bar\Delta_A$ to within MC error. *Pass:* residual $<3\times$ MC SE. (Also reproduces a $0.31/0.02$-style split when $D,\kappa$ are set to the af3-S3 numbers.)
2. **Reweighting floor (Thm 1b/c).** Sweep a rich family of covariate weights $w(\nu)$ (including the oracle $w^\star$ and deliberately wrong ones); for each, drive the method to deployment coverage $c$ and record *realized* $R_Q$. *Pass:* realized risk $\equiv R_Q(\tau_c)$ across all $w$ (variance across $w$ at MC level), and $\min_w$ realized risk $=R_Q(\tau_c)>\alpha$ in the impossibility regime — no weighting reaches $\alpha$. The weighted-CP *certificate* tracks $R_{\mathrm{ref}}$ (or $R_P$ unweighted), i.e. under-reports true risk by exactly $\bar\Delta_A$. In the $D=0$ control, weighted CP with $w^\star$ hits $\alpha$ (sanity: the impossibility is concept-specific).
3. **Mondrian achievability (Thm 3).** In the novel stratum, calibrate CRC/LTT with $n_g$ in-stratum labels over many trials. *Pass:* empirical selective risk $\le\alpha$, exceedance rate $\le\delta$ (LTT) / expected risk $\le\alpha$ (CRC); coverage-error envelope $\le1/(n_g+1)$; and Mondrian succeeds precisely where step 2's global reweighting fails. Sweep $n_g$ to show slack $\to0$ as $1/n_g$.
4. **DRO bracket (Thm 2 / 3c).** Compute $\mathrm{CVaR}_m^A(\eta_Q)$; verify it upper-brackets realized risk, that QRC certifies $\le$ it, and that refining strata tightens (upper edge $\downarrow$ toward $R_Q^{g^\star}$, lower edge $\uparrow$), edges meeting when within-stratum $\Delta$ is flat and $n_g$ large.

---

## 7. Prior results this specializes / relates to (cite, do not reinvent)

- **Weighted conformal = covariate-shift-only.** Tibshirani, Barber, Candès, Ramdas, *Conformal Prediction Under Covariate Shift*, NeurIPS 2019, arXiv:1904.06019. Theorem 1 specializes its $Q(Y\mid X)=P(Y\mid X)$ boundary to selective risk and names the residual $\bar\Delta_A$.
- **Impossibility of distribution-free conditional inference.** Barber, Candès, Ramdas, Tibshirani, *The limits of distribution-free conditional predictive inference*, Information and Inference (IMA) 10(2):455–482, 2021; arXiv:1903.04684. Theorem 2 is the selective-risk analogue.
- **Domain-adaptation lower bound.** Ben-David, Blitzer, Crammer, Kulesza, Pereira, Vaughan, *A theory of learning from different domains*, MLJ 2010. $C_{\mathrm{cov}}$ ↔ $d_{\mathcal H\Delta\mathcal H}$ (correctable), $\bar\Delta_A$ ↔ $\lambda$ (irreducible concept term).
- **f-divergence DRO robust validation.** Cauchois, Gupta, Ali, Duchi, *Robust Validation: Confident Predictions Even When Distributions Shift*, JASA 2024; arXiv:2008.04267. Provides the ball and the certified DRO upper edge (3c) / lower edge (Thm 2).
- **Quantile / CVaR risk control.** Snell, Zollo, Deng, Pitassi, Zemel, *Quantile Risk Control*, 2022. The worst-subpopulation/CVaR certificate in Theorem 3c.
- **Finite-sample risk control.** Angelopoulos, Bates et al., *Conformal Risk Control* (ICLR 2024, arXiv:2208.02814) and *Learn-then-Test* (arXiv:2110.01052) — the $1/(n_g+1)$ / $O(\sqrt{\log(1/\delta)/n_g})$ slack in Theorem 3.

**Already-named quantities (use their names):** the accept-region concept gap $\bar\Delta_A$ is the selective-risk restriction of Ben-David's $\lambda$; the DRO floor is the Cauchois $f$-divergence-ball worst case / CVaR (Snell); the label-free indistinguishability is the BCRT-2021 conditional-coverage barrier. The *only new object* is the exact three-way selective-risk decomposition $R_Q=R_P+C_{\mathrm{cov}}+\bar\Delta_A$ with the reweighting-invariance of realized risk (Thm 1b) and the resulting sharp unachievability condition $\bar\Delta_A>\alpha-R_{\mathrm{ref}}$.

---

### Cross-check summary for the theory reconciliation

The DRO/domain-adaptation lens reproduces the same result the conformal lens should: (i) a fixed, $w$-invariant realized-risk floor equal to the deployment risk–coverage curve; (ii) an exact additive concept+covariate split of that floor matching E19; (iii) unachievability precisely when the accept-region concept gap exceeds $\alpha-R_{\mathrm{ref}}$; (iv) achievability via in-stratum labels with $1/(n_g+1)$ slack, bracketed above by a CVaR/f-divergence DRO certificate. The honest scope — scalar-score threshold, covariate-measurable label-free weights — is load-bearing and must be stated in the theorem, not softened.

**Sources:** [arXiv:1904.06019 (weighted CP)](https://arxiv.org/abs/1904.06019) · [arXiv:1903.04684 (limits of conditional inference)](https://arxiv.org/pdf/1903.04684) · [Ben-David 2010 MLJ](https://www.alexkulesza.com/pubs/adapt_mlj10.pdf) · [arXiv:2008.04267 (robust validation)](https://arxiv.org/abs/2008.04267) · [Conformal Risk Control (arXiv:2208.02814)](https://arxiv.org/pdf/2208.02814) · Snell et al. Quantile Risk Control (2022).