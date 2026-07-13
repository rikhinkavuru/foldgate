# Execution ledger — workshop + journal hardening + theory (2026-07-12, session 2)

Goal (user, verbatim intent): take workshop to ~9 and journal to solid-accept at maximum quality;
turn the concept-shift vacuity into a real impossibility+achievability theorem; ship a general
(non-co-folding) benchmark with co-folding as flagship; add multiplicity control; audit everything
from 6 diverse adversarial reviewer perspectives, repeatedly. No rushing, no quality compromise.
GPU task (J1 FoldBench Protenix regen) deferred until user provides GPU tonight.

## Task board
- [ ] W1 Demote affinity-head EF; lead with pose-confidence (ipTM) selective result + per-stratum error money figure
- [ ] W2 Make concept-shift floor the headline contribution (not the coverage collapse)
- [ ] W3 Cut LIT-PCBA cherry-pick (report both directions or drop, honestly)
- [ ] J1 Second real dataset w/ matched novelty features  [BLOCKED on GPU — instructions to user after]
- [ ] J2 Decision = guarantee on the SAME object (ipTM-linked selective screening, honest, with CIs)
- [ ] J3 Fair screening baseline (scaffold-split actives, decoy-quality diagnostics, docking caveat)
- [ ] T1 Impossibility lower bound + matching achievability (CVaR/DRO) -> tight achievable-region theorem
- [ ] T2 General benchmark (synthetic w/ known conditional risk + a real non-co-folding shift set); co-folding flagship
- [ ] T3 Multiplicity control across the 5 models for main gate/LOTO/group-conditional/drift claims
- [ ] PAPER final workshop PDF (4pp) + extended journal draft
- [ ] AUDIT 6-reviewer adversarial panel -> fix loop until clean

## Phase plan
- Phase 0 GROUND+DESIGN (this): theory construction (2 lenses), benchmark spec, data re-verification, multiplicity spec.
- Phase 1 THEORY: prove + formalize + numerically validate on synthetic (module + E23). Math audit.
- Phase 2 EXPERIMENTS: J2, J3, T2 benchmark (E24), T3 multiplicity (E25), W3.
- Phase 3 PAPER: rewrite around per-stratum error + concept floor + theorem + benchmark + multiplicity. Workshop PDF + journal draft.
- Phase 4 AUDIT: 6-lens panel verify-against-JSON -> fix -> re-audit until clean.

## Results log (append the moment produced)

### Phase 0 GROUND+DESIGN — DONE (2026-07-12)
- Theorem vetted by independent referee (docs/theory/THEOREM_RECONCILED.md). Sound + publishable.
  Impossibility (Thm1): at fixed coverage c, realized target selective risk R_Q(tau_c) = R_ref^cov(tau_c) + Delta_bar_A,
  and Delta_bar_A (accept-region-avg concept drift) is INVARIANT across all covariate-measurable reweightings ->
  level alpha unachievable iff Delta_bar_A > alpha - R_ref (scalar-threshold class). Thm2 DRO label-free lower edge.
  Thm3 achievability: in-stratum Mondrian recalibration attains the floor. CRITICAL CORRECTION caught by referee:
  clean 1/(n_g+1) slack is for COVERAGE and JOINT error only; the ratio selective risk needs LTT/QRC O(sqrt(log(1/d)/n_g)).
  Citations fixed (RCPS=2101.02703, Cauchois 4th author Ali, chi2 const sqrt(2 rho Var)). Scope S1-S8 stated.
- J2/J3/W3 numbers verified vs JSON (docs/theory/DATA_VERIFICATION.md). KEY: the headline EF 26.6 GPCR / 31 DEKOIS
  is the Boltz-2 AFFINITY head (NOT guaranteed). Pose-confidence signal the layer certifies = ipTM EF: DEKOIS 20.7,
  GPCR 9.3, LIT 4.0 -- STILL beats docking (10.3 / 2.0 / 3.0). Honest reframe: pose number is the headline; affinity
  head reported separately + labeled un-guaranteed. Scaffold split computable DEKOIS+GPCR (rdkit + shipped SMILES).
  W3: report Construction 2 (disjoint ec_sim bins) -> GPCR monotone 27->42; footnote/drop Construction 1.
- Benchmark spec (docs/theory/BENCHMARK_SPEC.md): synth generator closed-form P(Y|s,nu) w/ knobs D/E/T; real =
  OpenML electricity (primary, offline) + folktables ACSIncome (graceful). Torch-free, sklearn HistGB base clf.
- Multiplicity spec (docs/theory/MULTIPLICITY_SPEC.md): SURGICAL not blanket. Certificates (E13 LOTO, E3 group-cond,
  E22 m*/rho*) -> Bonferroni delta/K=0.02 (E22 already). Discovery conjunctions 'all 5 models' (E4/E13 AURC, E6b) ->
  IUT, NO penalty (already valid, state it). E12 drift grid -> Romano-Wolf step-down (the one real change) + TOST temporal.
- Env: rdkit PRESENT in .venv (J3 ok); lightgbm+folktables absent (use sklearn HistGB / graceful). 35 tests green pre-build.

### Phase 1+2 IMPLEMENTATION — DONE (2026-07-12). 62 tests pass, ruff clean repo-wide.
- Bench modules: src/foldgate/bench/{synth,certificates,realdata}.py + tests/test_bench.py. synth extended
  with Dq (genuine target concept drift) + eps_floor (irreducible error floor) + oracle_concept_gap +
  oracle_coverage_threshold (backward compatible; Dq=eps=0 reproduces prior). bench __all__ = 25 exports.
- THEOREM VALIDATION (all PASS against ground truth):
  - b1 C1: R_Q = R_P + C_cov + Delta_bar within MC SE; impossibility gap monotone 0->0.584 in D, ->0 at D=0.
  - b3 C2: realized risk invariant across 9 reweightings (resid 0.001); weighted-CP cert under-reports worst by Delta_bar.
  - b2 C3: Mondrian LTT exceedance <=0.083<=delta; slack ~n^-0.45; thin-strata abstain; joint 0.199 vs ratio 0.379 (the
    reconciled correction: clean 1/(n+1) is joint/coverage only, ratio needs LTT/QRC).
  - b4 C6: RCPS UCB validity 0.99-1.0; DRO ball covers at true tilt; chi2 uses sqrt(2 rho Var).
  - b7 (NEW, sharp claims w/ genuine Dq>0): T1a decomposition R_ref 0.273 + Delta_bar 0.205 = R_Q 0.478 = realized 0.476;
    T1d silent violation: w* cert 0.280 ~ R_ref, under-reports by 0.198 ~ Delta_bar; T1c crossing at Dq=1.47 (below
    reachable, above no reweighting hits alpha); T3 achievability: SOURCE-label Mondrian exceedance 1.0 (FAILS) vs
    TARGET-label 0.115 (controls) -> needs target-stratum labels; vacuity: eps=0.30>alpha -> tau* NaN, LTT abstains 100%.
- REAL BENCHMARK: b6 electricity (concept drift 0.364): marginal 0.309, weighted MISLED 0.337, Mondrian controls 0.200.
  b5 ACS (covariate shift): marginal 0.228 worst-state, weighted repairs 0.202, Mondrian 0.201. Covariate-vs-concept
  contrast on two real non-co-folding datasets. Torch-free (sklearn HistGB). folktables MIT/ACS public; OpenML elec license TBC.
- J2 (e23): guaranteed pose signal EF@1% DEKOIS 20.7/GPCR 9.3/LIT 4.0 beats docking 10.3/2.0/3.0; affinity head
  31/26.6/5.1 reported SEPARATELY as not-guaranteed. W3 shift Construction 2: GPCR monotone 27->42, bin edges stated.
- J3 (e24): Murcko scaffold-novelty split DEKOIS+GPCR (DEKOIS novel<familiar all 3 signals; GPCR mixed, honest); all
  methods recomputed (not inherited); decoy-property fair-split NOT computable (no decoy SMILES) -> stated as limitation.
- MULTIPLICITY: E12 Romano-Wolf step-down 42/55 cells FWER-survive; temporal TOST does NOT certify equivalence (honest:
  claim the magnitude contrast, temporal 0.03-0.07 vs structural to 0.63). E13 joint LOTO delta/K=0.02: af3/boltz1/boltz1x
  hold w/ coverage, chai/protenix vacuous-by-abstention (surfaced). E4 IUT all_models_exclude_zero=True. E1 IUT flag.

### Phase 3 PAPER (workshop) — DONE. paper/moml2026_foldgate.{tex,pdf}. Main text EXACTLY 4pp (Conclusion
   on p4, refs p5, appendix theorem-proof + benchmark on p5-6, excluded). No unresolved cites, no em-dashes,
   all refs cited. Reframed: theorem as Sec 4 (impossibility R_Q=R_ref+Delta_bar invariant + achievability +
   vacuity, proof in App A), pose-confidence screening headline (9.3 vs 2.0 docking; affinity 26.6 demoted to
   ungoverned comparator), reliability-drift/concept-shift mechanism, multiplicity paragraph (Romano-Wolf/IUT/
   Bonferroni), general-benchmark validation (elec2 0.31/0.34/0.20, ACS 0.23/0.20/0.20), J3 scaffold split,
   honest W3 shift (GPCR 27->42). All numbers verified vs JSONs.

### Phase 4 AUDIT round 1 DONE (6 reviewers + chair, verified vs JSON). Verdict: theorem SOUND + numbers reproduce;
   5 blockers + majors on labeling/honesty. ALL FIXED:
   - Table1 drift cells af3/boltz1 +.51/+.44 -> +.47/+.47 (e12 0.4718/0.4728). Leak 95.6->95.1. m* 25-45->20-45 +
     added vacuity m*=1. AURC 25-40->28-41. Delta 0.21->0.205, R_Q 0.478 exact.
   - BLOCKER: 42->27 "monotone collapse" was the AFFINITY head -> relabeled as affinity head + governed pose signal
     reported separately (novel 11 < familiar 18). Abstract dropped "guaranteed" before screening EF. Temporal
     "intervals covering zero" was FALSE -> reworded to |D|<=0.07, 2/15 cells resolved. House-style negative
     parallelisms + conclusion tricolon removed.
   - MAJORS: 75%/13% disclosed as ranked-by-affinity/gated-by-pose. elec2/ACS reframed as 3-way (marginal
     under-controls / weighted violates its cert / group-cond certifies by abstaining). Decoy analog-bias confound
     stated. Single-real-dataset (RNP) limitation foregrounded. Both dockers (Gnina 2.0, Glide 3.8) + disjoint CIs.
     AF3 25.8>Boltz2 20.7 disclosed. vovk cite for 1/(n+1) coverage; ties/boundary-randomization clause in App A.
   - Still exactly 4pp main text (conclusion p4, refs+appendix p5-6). 0 unresolved cites, 0 house-style residuals.

### Phase 4 AUDIT round 2 (re-audit, 3 lenses) DONE. Verdict: all 5 blockers + 4 majors CONFIRMED resolved,
   every number reconciles vs JSON. Fixes had introduced ONE defect (garbled+false theorem clause "under-states
   by at least in the accept region") -> DELETED (kept only the oracle-weighted claim, matches App A). Minor
   fixes: n~76->~72 (e2 S4 mean 71.8); CVaR "m*=1 vacuous" -> "m*->1 for most models" (e17: af3/boltz1x can
   certify large subpop); residual negative-parallelisms reworded; abstract "2-3x"->"about twice" (S3/S4=1.9-2.1x);
   cut decoupled ChEMBL sentence + affinity2026 cite for space (stays in journal). FINAL: 4pp main text, 0 unresolved
   cites, 0 house-style residuals, 62 tests green. paper/moml2026_foldgate.pdf = FINAL WORKSHOP SUBMISSION.

### WORKSHOP PAPER COMPLETE. NEXT: extended journal draft (superset: full proof, full benchmark, E21 ChEMBL,
###   all experiments). J1 FoldBench Protenix regen awaits user GPU (recommended: rented A100/H100 80GB VRAM ~$30).

### PAUSED 2026-07-12 (user closed computer). RESUME POINTER:
- In-flight build workflow may have been killed by sleep. On resume:
  1. Check completion: read the impl-build journal at
     ~/.claude/projects/-Users-rikhinkavuru-moml/166114e3-096b-4049-a7ca-80fa8668314a/subagents/workflows/wf_c08f3afd-fad/journal.jsonl
  2. Resume from cache: Workflow({scriptPath: ".../workflows/scripts/foldgate-impl-build-wf_c08f3afd-fad.js",
     resumeFromRunId: "wf_c08f3afd-fad"}) -- completed agents replay instantly, only unfinished re-run.
  3. Then verify on disk: ls src/foldgate/bench/, experiments/{_common,b1..b6,e23,e24}.py, results/{bench/,e23_*,e24_*};
     run `.venv/bin/python -m pytest -q` (expect >35). Then proceed: PAPER rewrite -> 6-reviewer AUDIT loop.
- Remaining after build: W1/W2 paper reframe (pose headline + concept-floor + theorem + benchmark + multiplicity),
  final workshop PDF, extended journal draft, then 6-lens adversarial audit -> fix until clean. J1 (FoldBench Protenix
  GPU regen) still awaits user GPU tonight.
