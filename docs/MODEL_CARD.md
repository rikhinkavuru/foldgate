# Model Card — foldgate reliability layer

## What it is

foldgate is a **training-free, model-agnostic reliability layer** for protein-ligand
co-folding. It does not predict structures. It wraps a frozen co-folding model
(AlphaFold3, Boltz, Chai, Protenix) and turns that model's confidence into a
**risk-controlled accept/abstain decision** with a finite-sample conformal guarantee that
is made robust to novel-pocket / novel-chemotype distribution shift.

## Components

1. **Score** (`foldgate.scores.ScoreCombiner`) — a small gradient-boosted model fit
   **only on a calibration/train fold** that maps cheap per-prediction features (native
   confidence, interface ipTM, ensemble spread, PoseBusters validity, cross-model
   agreement, ligand difficulty) to P(correct). It never sees the test fold and never
   retrains the co-folding model. Training-set novelty is deliberately excluded from the
   score — novelty enters through calibration, keeping the two roles separate.
2. **Certifier** (`foldgate.conformal`) — Learn-then-Test with an exact binomial p-value
   (`ltt_threshold`) for the binary 2 Å gate; a WSR betting bound
   (`continuous_risk_threshold`, `wsr_betting_pvalue`) for the continuous-RMSD gate;
   group-conditional (Mondrian) and weighted (`weighted_ltt_threshold`) variants for the
   shift-robust guarantees.
3. **Gate** (`foldgate.selective`) — accept iff score ≥ τ; risk-coverage curves, AURC,
   per-stratum conditional coverage, bootstrap CIs.

## Intended use

- Decide, with a stated error budget α and confidence 1−δ, **which co-folding poses to
  trust** and which to abstain on, especially on novel targets where naive confidence
  thresholds silently under-control error.
- Rank poses for downstream triage (screening, FEP starting structures) by a reliability
  score that beats native confidence on AURC.

## Out-of-scope / cautions

- **Not a structure predictor** and not a replacement for experimental validation.
- **The guarantee is conditional.** The finite-sample coverage holds under exchangeability
  (i.i.d.) or, under shift, for the group-conditional certificate. The weighted-conformal
  variant is exact **only conditional on correct importance weights**; under real
  novel-pocket shift there is residual concept shift (P(correct | confidence) moves), so
  weighted conformal is scoped as a label-free complement, not the rigorous headline.
- **Cross-model consensus is a feature, not independent validation** — co-folding models
  make correlated errors, so agreement can be overconfident on shared blind spots.
- **Coverage can be near-zero** where native confidence is uninformative (e.g. Chai's
  ranking score), in which case the honest output is to abstain.

## Guarantees, precisely

| Setting | Method | Guarantee |
|---------|--------|-----------|
| i.i.d. | LTT (binomial) | P(selective risk ≤ α) ≥ 1 − δ, finite-sample, distribution-free |
| Novelty shift, labelled strata | group-conditional (Mondrian) | per-stratum P(risk ≤ α) ≥ 1 − δ |
| Novelty shift, unlabelled target | weighted LTT (WSR betting) | P(R_target ≤ α) ≥ 1 − δ, conditional on correct weights |
| Continuous RMSD | WSR betting bound | P(mean bounded-RMSD ≤ target) ≥ 1 − δ |

## Ethical / safety notes

foldgate is a defensive reliability tool: it makes co-folding models *more conservative*
by abstaining when confidence is unreliable. Misuse would be reading an abstention as a
negative result; abstention means "not certifiable at this budget," not "no binder."
