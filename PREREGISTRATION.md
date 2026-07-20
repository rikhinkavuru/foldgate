# Pre-registration — foldgate selective-risk protocol

Frozen 2026-07-20, before the post-audit reanalysis, to remove any suspicion that the
feasibility frontier or the gate thresholds were tuned to the data. Anything analysis-defining
is fixed here; only descriptive additions (tables, CIs, figures) were added afterward.

## Risk-control protocol
- **Loss:** `L = 1[symmetry-corrected ligand-RMSD > 2 Å]` against the shipped BiSyRMSD label.
- **Target / confidence:** α = 0.20 (also reported at α = 0.10), δ = 0.10.
- **Certifier:** Learn-then-Test with fixed-sequence testing; each `H0(τ): P(error | s≥τ) ≥ α`
  tested with the exact binomial tail `P(Bin(n_accept, α) ≤ errors)` at level δ.
- **Coverage grid (fixed, data-independent):** `c ∈ {0.05, 0.06, …, 1.00}` (step 0.01),
  ascending; the fixed-sequence walk grows the accept set from the smallest coverage upward and
  stops at the first failure. `min_accept = 20`. This ordering is a sequence of top-k **ranks**,
  fixed before any risk is computed; only the mapped τ = empirical quantile is data-dependent.
  (Implemented in `src/foldgate/conformal/risk.py:ltt_threshold`, default `coverage_grid`.)

## Novelty stratification
- **Ligand axis:** ECFP4 Morgan Tanimoto to the nearest training ligand (`morgan_tanimoto`).
- **Pocket axis:** SuCOS shape × pocket query-coverage (`sucos_shape_pocket_qcov`).
- **Bins:** quartiles S0 (familiar) … S3, plus a fifth no-analog stratum S4 for missing
  similarity. Quartile edges are taken from the similarity marginal a priori (not chosen to
  produce any result). The primary binning is n_bins = 4 + no-analog; sensitivity to
  {2,4,6} bins and fixed edges is reported but does not change the frontier verdict.

## Splits (leakage discipline)
- **Primary certificates** are target-grouped on `system_id` (nested leave-one-target-out:
  outer GroupKFold; inner grouped fit/calibration split; combiner, threshold, and test target
  mutually disjoint). Every headline coverage/risk number uses this protocol.
- Pose-level (non-grouped) splits appear only in SI sensitivity analyses.

## Endpoints
- **Primary:** per-stratum realized selective risk and coverage under the group-conditional gate;
  the per-stratum feasibility frontier `c*_g = max{c : R_Q,g(τ(c)) ≤ α}`.
- **Secondary:** leakage-free LOTO coverage at certified risk; label-cost curve; graded-loss
  betting-bound operating point; interaction-fingerprint recovery (RMSD-conditioned);
  PoseBusters-joint label; decision-curve net benefit; deployment-computable proxy stratifier.
- **Multiplicity:** per the certificate ledger (Table in the paper) — Bonferroni δ/K for joint
  certificates, Romano-Wolf step-down for the drift grid, intersection-union for "every model"
  effect claims.

## What was NOT pre-registered (post-hoc, descriptive only)
Confidence intervals and accepted-n annotations on already-defined quantities, the CONSORT
count reconciliation, the temporal-axis reframe (a correction of interpretation, not a new
endpoint), and the composition tables. None of these change an analysis decision.
