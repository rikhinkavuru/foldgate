"""foldgate.bench -- general (non-co-folding) validation of the selective-risk theorem.

A torch-free, CPU-only benchmark that checks the impossibility/achievability
theorem on data where the label conditional P(Y=1 | s, nu) is either known in
closed form (synth) or estimated on public tabular sets (realdata):

  synth        closed-form logistic-normal generator with isolated concept (D),
               covariate-in-score (E), and nu-marginal tilt (T) knobs, plus oracle
               per-stratum accept rate / selective risk / threshold / R_mix.
  certificates worst-stratum RCPS UCB (Hoeffding-Bentkus, union bound) and an
               f-divergence / CVaR ball certificate over the stratum simplex.
  realdata     ACS Income and Electricity loaders reduced to the (s, y, nu) triple
               with a leakage-free grouped split; SkipDataset when unavailable.
"""

from .certificates import (
    chi2_closed_form_certificate,
    dro_ball_certificate,
    worst_stratum_rcps_ucb,
)
from .realdata import (
    SkipDataset,
    acs_income,
    acs_income_triple,
    build_triple,
    electricity,
    electricity_triple,
    grouped_split,
)
from .synth import (
    SynthParams,
    chi2_divergence,
    kl_divergence,
    oracle_accept_rate,
    oracle_accept_rates,
    oracle_concept_gap,
    oracle_coverage_threshold,
    oracle_impossibility_gap,
    oracle_marginal_threshold,
    oracle_R_mix,
    oracle_selective_risk,
    oracle_selective_risks,
    oracle_tau_star,
    pi_correct,
    sample,
)

__all__ = [
    "SynthParams",
    "sample",
    "pi_correct",
    "kl_divergence",
    "chi2_divergence",
    "oracle_accept_rate",
    "oracle_accept_rates",
    "oracle_selective_risk",
    "oracle_selective_risks",
    "oracle_tau_star",
    "oracle_R_mix",
    "oracle_marginal_threshold",
    "oracle_coverage_threshold",
    "oracle_concept_gap",
    "oracle_impossibility_gap",
    "worst_stratum_rcps_ucb",
    "dro_ball_certificate",
    "chi2_closed_form_certificate",
    "SkipDataset",
    "acs_income",
    "acs_income_triple",
    "electricity",
    "electricity_triple",
    "build_triple",
    "grouped_split",
]
