# MULTIPLICITY SPEC  (Phase 0 grounding, 2026-07-12)

I have everything I need. Web tools are up and the three requested references verified. Here is the specification.

---

# Multiplicity-control spec for foldgate's 5-model claim families

## 0. What the reviewer got right, and the one distinction they missed

The reviewer is correct that the paper makes several **pooled "all five models" statements** with no stated multiplicity control. But "correct for 5 models everywhere" is wrong, because the claim families split into two statistically opposite types, and the correct control is opposite for each:

- **Certificate / validity claims** ("with prob ≥ 1−δ, realized risk ≤ α") — a *for-all/upper-bound* statement. Violations **accumulate**: the joint bad event "some model's guarantee fails" has probability ≤ Σδ_m. To make a **joint** guarantee you must **split δ** (Bonferroni union bound, δ/K). Families **(i) main gate, (ii) LOTO, (iii) group-conditional repair, (vi) m*/ρ***.
- **Significance / discovery claims** ("effect > 0, CI excludes zero") — a *there-exists/rejection* statement. Here the asymmetry flips:
  - a **conjunction** claim "the effect holds for **every** model" is an **intersection-union test** (Berger 1982): requiring each per-model CI to individually exclude zero at level δ **already controls the joint Type-I error at δ — no penalty is needed**. Families **(v) AURC native>combined** and the **E6b downstream gap** are phrased exactly this way ("for every model", "for all five models") and are therefore **already valid as written**.
  - a **disjunction / exploratory grid** ("drift is present *somewhere* / in *these* cells") does inflate Type-I and **does** need FWER (Holm) or FDR (BH). This is family **(iv) E12 reliability drift**, the one family that genuinely needs a correction it does not currently have.

So the honest fix is small and surgical, not a blanket δ/5.

**Constants:** δ = 0.10, K = 5 (`methods_with_enough` → af3, boltz1, boltz1x, chai, protenix). Bonferroni δ/K = **0.02**. The E4/E12 "90% CI excludes zero" is a **one-sided test at 0.05** (percentile [5,95], lower bound > 0) — state this level explicitly.

---

## 1. Per-family verdict table

| Family | Claim type | Correct control | Adjusted level | Recompute? |
|---|---|---|---|---|
| **(i) main gate ≤ α (E1)** | certificate | E1 is per-model *verification*: IUT (require all 5 to pass), no δ-split. The *deployed joint* certificate lives in E13/E22 at δ/K. | δ = 0.10 per model; joint statement → δ/K = 0.02 (routed to E13) | E1: add joint line only |
| **(ii) LOTO validity (E13)** | certificate | Bonferroni union bound on δ for the joint "all models hold" | **δ/K = 0.02** in `ltt_threshold`; keep per-model at 0.10 | **yes** |
| **(iii) group-conditional repair (E3/E3c)** | certificate, **× strata × models** | Per-stratum *local* validity at δ (no umbrella) is the honest primary. Umbrella "all strata × models jointly" → Bonferroni δ/(S·K) | local δ = 0.10; umbrella **δ/(S·K)** (e.g. 4 novel strata × 5 = **0.005**) | add umbrella line |
| **(iv) reliability drift, CI≠0 (E12)** | discovery, large grid | **Romano-Wolf step-down max-t bootstrap** (dependence-aware, most powerful) primary; **BH-FDR** acceptable alternative. Bonferroni/Holm valid but needlessly conservative | FWER 0.10 simultaneous band, or BH q=0.10 | **yes (main change)** |
| **(v) AURC native>combined (E4/E13)** | discovery, **conjunction** "all 5" | **Intersection-union test** — each per-model 90% CI excludes 0 → joint valid, **no penalty**. Add Holm only if a per-model "which models" discovery table is claimed. | per-model 0.05 one-sided; **no correction** for the "all 5" claim | add joint IUT flag only |
| **(vi) m*/ρ* certificates (E17/E22)** | certificate | **Already correct** — E22 uses Bonferroni δ/K = 0.02 (Snell QRC Thm 4.6 union bound). E17 per-model curves carry no umbrella. | δ/K = 0.02 (done) | **none** |

---

## 2. Family-by-family reasoning and exact numbers

### (i)/(ii) Main gate and LOTO — Bonferroni δ/K on the *joint* certificate
Within a single model the LTT gate **already controls the family-wise error over the threshold grid** by fixed-sequence testing (see `risk.py:ltt_threshold` docstring — no Bonferroni over τ, and none should be added). The only uncontrolled multiplicity is across the 5 models, and only when a **joint** statement is made ("with prob ≥ 1−δ every model's realized risk ≤ α"). That joint event fails if *any* model fails; union bound gives Σδ_m, so certify each at **δ/K = 0.02**. Holm gives nothing here: the claim is a required conjunction of certificates (all K must hold), and step-down only helps when you are trying to *maximize rejections*, not when every element is mandatory. Bonferroni is both correct and, at K=5, negligibly conservative — the LTT threshold moves a hair; the binomial acceptance p-value target goes 0.10 → 0.02.
**Honesty carry-over:** at δ/K some LOTO certificates go near-vacuous (chai LOTO coverage 0.009, protenix 0.078 in the current JSON). Keep those reported, not hidden — that is the CLAUDE.md rule 5.

### (iii) Group-conditional repair — two nested families (strata × models)
The claim "each novelty stratum's accepted error ≤ α with prob 1−δ" is a conjunction over strata **within** a model (Mondrian multiplicity) and over models. Group-conditional calibration gives each stratum its **own disjoint calibration set and threshold**, so per-stratum guarantees are the natural *local* unit — report them at δ each with **no correction** (this is the whole point of group-conditional validity). Only the umbrella "all strata of all models are simultaneously valid" needs a union bound: **δ/(S·K)**. With S = number of novel strata actually asserted (e.g. 4) and K = 5, that is **δ/20 = 0.005**. Because the strata's calibration sets are disjoint (independent), Bonferroni here is close to tight and Holm again offers no gain for a required conjunction. Recommend: primary reporting = per-stratum local validity (no correction, correct); add **one** secondary "simultaneous repair" line at δ/(S·K), mirroring E22.

### (iv) E12 reliability drift — the family that actually needs correcting
This is the opposite sign: false **positives** (declaring drift where none exists). The family is large — 5 models × 3 axes × up to 4 strata ≈ 40–55 cells — so Bonferroni would demand α_cell = 0.10/55 ≈ 0.0018, throwing away real power. Two better routes:

- **Primary — Romano-Wolf step-down max-t bootstrap (2005).** E12 already runs a 1000-rep bootstrap. Reuse it: form the studentized drift t_cell = D̂_signed/se_boot, build the null by centering each bootstrap draw (D*_cell − D̂_cell)/se, take the **row-wise max |t| across the cell family** per bootstrap rep, and read step-down adjusted p-values / a single simultaneous critical value from that max distribution. This controls **FWER at 0.10 while exploiting the strong positive dependence** — within a model all cells share the same S0 reference sample, and across models the cells share the same underlying complexes, so the statistics are heavily correlated. Romano-Wolf captures that dependence and is uniformly at least as powerful as Holm/Bonferroni (equal only under independence). Bump N_BOOT to ≥ 2000 for stable tails.
- **Acceptable alternative — Benjamini-Hochberg FDR at q = 0.10** on the one-sided bootstrap p-values, if the intended claim is the collective "drift is elevated on structural novelty" rather than "every named cell drifts." The positive dependence (shared S0) is PRDS-plausible, so BH is valid without the Benjamini-Yekutieli log-factor; note this assumption.
- **The "temporal flat / reweighting admissible" verdicts are non-rejections** — you cannot prove the null by failing to reject. Upgrade those to a **TOST equivalence test** against the SMALL_DRIFT = 0.05 margin, per temporal cell; the joint "temporal flat for all models" is then a conjunction of equivalence rejections → **IUT, no penalty**. This makes the "recency does not degrade confidence" claim rigorous rather than an artifact of a wide CI.

### (v) AURC native > combined (E4/E13) — already valid as a conjunction
The paper's wording is uniformly a **conjunction**: "excludes zero for **every** model", "for **all** models", "for **all five** models" (E6b). A conjunction "effect holds for all K" is an **intersection-union test**: the level-δ test is to require **each** component to individually reject at δ (equivalently, each 90% CI to exclude zero). This **already controls the joint Type-I error at δ with no multiplicity penalty** (Berger 1982). So the E4 paired-bootstrap Δ(AURC) CIs and the E6b recall-gap CIs are **correct as written** — the reviewer's blanket flag misfires here. Two additions:
- State the IUT justification explicitly and add a single boolean `all_models_exclude_zero` = (max over models of one-sided p ≤ δ).
- **Only if** you also present a per-model discovery table ("boltz1 shows it, chai shows it" as separate findings) do those individual rejections form a disjunctive family needing FWER — then apply **Holm** on the 5 one-sided p-values with step-down thresholds **0.02, 0.025, 0.0333, 0.05, 0.10**. Also: E13 currently reports native and combined cluster-CIs *separately*; emit the **paired** cluster-bootstrap Δ(AURC) CI per model so the LOTO IUT is clean.

### (vi) m*/ρ* certificates (E17/E22) — done correctly
`robust.py:simultaneous_certificate` already applies the union bound at **δ/K = 0.02** (`e22` JSON: `delta_per_model = 0.02`), citing Snell et al. QRC Thm 4.6 — this is exactly right for a conjunction of certificates. E17 per-model curves carry no umbrella and each point is individually valid at its pre-specified coverage (docstring already says so). **No change needed**, except: if the paper reads a joint "all models certified" off E17, route that sentence to E22's δ/K number, and pin the joint claim to **one** pre-specified coverage (E22 uses top-20%) so the family is K, not K × |coverages|.

---

## 3. Where NO correction is needed (state these to pre-empt the reviewer)

1. **Within-model threshold search** — LTT/RCPS fixed-sequence testing already controls FWER over the τ-grid; do not add Bonferroni over thresholds (would double-penalize).
2. **Any experiment reported strictly per-model** with per-model conclusions and no "all/every/at-least-one" umbrella — each is its own valid claim (E17 per-model curves; E1 as a per-model verification).
3. **Conjunction "holds for all 5 models" tested by IUT** — E4/E13 AURC and E6b recall gap: requiring each per-model CI to exclude zero at level δ is already a level-δ test of the conjunction (Berger IUT). No penalty.
4. **E22's simultaneous certificate** — already correct at δ/K.

---

## 4. Concrete implementation plan

**`experiments/_common.py`** — add shared machinery so every script uses one definition:
- `K = len(methods_with_enough(df))` (=5) and `DELTA_JOINT = DELTA / K` (=0.02).
- `holm(pvals) -> adjusted` and `bh(pvals, q) -> rejected` utilities.
- `romano_wolf_stepdown(boot_matrix, stat_hat, se) -> adj_p` (max-t step-down over a cells × B bootstrap matrix).
- `iut_all(pvals, level) -> bool` = `max(pvals) <= level`.

**`experiments/e13_loto_validity.py`** (recompute) — call `ltt_threshold(..., delta=DELTA_JOINT)` for a **joint** LOTO certificate; keep the per-model `delta=DELTA` result alongside. Add `folds_holding_joint`, joint pooled risk CI, and an `all_models_hold_joint` flag. Emit a **paired** cluster-bootstrap Δ(AURC) CI per model + `all_models_exclude_zero` (IUT).

**`experiments/e12_reliability_drift.py`** (recompute, main change) — retain the per-cell bootstrap arrays; add Romano-Wolf step-down adjusted p-values + a simultaneous 90% band across the model×axis×stratum family; add BH-FDR flags as an alternative; add TOST equivalence verdicts for temporal cells. Add adjusted `verdict` fields and bump `N_BOOT` to ≥ 2000. Update `make_figure` to draw the simultaneous band.

**`experiments/e1_iid_validity.py`** (light) — add one joint line: report per-model verification unchanged, plus `all_models_verified` (IUT) and a note that the deployed joint certificate at δ/K lives in E13/E22.

**`experiments/e4_selective_utility.py`** (light) — add `all_models_exclude_zero` (IUT) for the Δ(AURC) and pose-Δ(AURC) claims; optionally add Holm-adjusted per-model p-values for a discovery table.

**`experiments/e3_shift_repair.py` / `e3c_combined_conditional.py`** (light) — keep per-stratum local certificates at δ; add one Bonferroni **δ/(S·K)** "simultaneous repair" summary line.

**`experiments/e17_worst_subpop.py`** — no recompute; add a docstring/report note routing any joint claim to E22.

**`experiments/e22_robust_certificates.py`** — already correct; optionally document that δ/K is the union bound and that Holm would not improve a required conjunction of certificates.

**Recompute only:** E12, E13 (+ E1/E4/E3/E3c summary lines). E17, E22 unchanged. Nothing touches `src/foldgate/conformal/*.py` except optionally moving `holm`/`bh`/`romano_wolf_stepdown` into a small `foldgate/selective/multiplicity.py` if you want them shippable — but they are analysis-layer, so `_common.py` is fine.

---

## 5. Reporting recipe (add a "Multiplicity control" methods paragraph)

State, once, in this order: (1) δ = 0.10, K = 5, and the **family definition** for each claim; (2) the **certificate-vs-discovery asymmetry** — certificates get Bonferroni δ/K = 0.02 for joint validity, "all-models" significance conjunctions are IUTs needing no penalty, the E12 exploratory grid gets Romano-Wolf/BH; (3) report **both** raw per-model numbers **and** the adjusted joint statement; (4) name where the joint guarantee goes **vacuous** (chai/protenix LOTO coverage). Keep the house style (no em-dashes, no rule-of-three, no negative parallelism). One clean sentence per family, e.g. "The joint statement that every model's LOTO gate controls error holds with probability at least 0.90 under a Bonferroni union bound (per-model δ = 0.02); read per model the certificate holds at δ = 0.10." For E4/E6b: "We claim the AURC gain for every model, an intersection-union test, so each per-model 90% interval excluding zero certifies the conjunction at the 0.10 level without a further penalty."

---

## 6. References (verified via web this session; add to `REFERENCES.bib`)

- **Holm, S. (1979).** A Simple Sequentially Rejective Multiple Test Procedure. *Scand. J. Statist.* 6:65–70. — FWER step-down; use for E4/E12 per-model discovery tables. [scandinavian journal of statistics](https://www.ime.usp.br/~abe/lista/pdf4R8xPVzCnX.pdf)
- **Benjamini, Y. & Hochberg, Y. (1995).** Controlling the False Discovery Rate. *JRSS-B* 57:289–300. — FDR alternative for the E12 grid. [wiley](https://rss.onlinelibrary.wiley.com/doi/10.1111/j.2517-6161.1995.tb02031.x)
- **Romano, J.P. & Wolf, M. (2005).** Stepwise Multiple Testing as Formalized Data Snooping. *Econometrica* 73:1237–1282. — dependence-aware max-t step-down bootstrap; primary for E12. [wharton pdf](http://www-stat.wharton.upenn.edu/~steele/Courses/956/Resource/MultipleComparision/RomanoWolf05.pdf)
- **Berger, R.L. (1982).** Multiparameter hypothesis testing and acceptance sampling (intersection-union test) — justifies no-penalty for "holds for all K models" conjunctions (E4, E6b). *(already in your literature via IUT; add if not present — not re-verified on web this session, flag as needs-cite-check.)*
- **Snell et al., Quantile Risk Control** (arXiv:2212.13629, Thm 4.6) — union-bound simultaneous certificate; already cited by E22.

**Web-grounding note:** WebSearch/WebFetch were available and used; Holm 1979, Benjamini-Hochberg 1995, and Romano-Wolf 2005 are verified against primary sources above. The **Berger 1982 IUT** citation is from my own knowledge and was **not** re-verified on the web this session — confirm the exact reference before it goes in the bib.