"""E14 -- deployable repair: key the group-conditional gate on model disagreement,
not similarity-to-train.

The E3 repair conditions on training-set structural/chemical similarity. RNP ships
that, but for a gated (AF3) or proprietary predictor a practitioner cannot compute
similarity-to-training on a novel target. We test a training-free surrogate: the
model's own intra-ensemble pose disagreement (spread of its diffusion samples'
ligand placement), which needs a single model and no training corpus.

Honest evaluation. Conditioning on any partition restores per-*that-partition*
marginal coverage by construction, so reporting per-disagreement-bin risk would be
circular. Instead we calibrate the gate on disagreement bins and then report realized
selective risk PER SIMILARITY STRATUM (S0-S4) -- the axis the break is defined on --
against two references: the global gate (E2, under-controls on novel strata) and the
similarity-keyed oracle (E3, needs training access). The question is how much of the
oracle's per-stratum error control a training-free surrogate recovers, and where it
fails. We also report where disagreement collapses (correlated errors on no-analog
targets), which is exactly the regime the surrogate cannot cover.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from experiments._common import (
    ALPHA,
    CONF,
    DELTA,
    RESDIR,
    load_delivered,
    methods_with_enough,
    rng,
    save_json,
)
from foldgate.conformal import ltt_threshold

SIM_STRAT = "novelty_stratum"       # the axis the break lives on (evaluation grouping)
DISAGREE = "intra_model_pose_std"   # training-free surrogate (single-model deployable)
N_BINS = 5


def _bin(x, edges):
    b = np.digitize(x, edges[1:-1])
    return np.clip(b, 0, len(edges) - 2)


def run(n_repeats: int = 300) -> dict:
    df = load_delivered()
    methods = methods_with_enough(df)
    g = rng()
    out = {}

    for m in methods:
        sub = df[df.method == m].dropna(subset=[CONF, SIM_STRAT, DISAGREE]).reset_index(drop=True)
        s = sub[CONF].to_numpy()
        y = sub["correct"].to_numpy().astype(int)
        sim = sub[SIM_STRAT].to_numpy().astype(int)
        dis = sub[DISAGREE].to_numpy()
        n = len(sub)
        sim_levels = sorted(np.unique(sim).tolist())

        # pre-specified disagreement bin edges (label-independent)
        edges = np.quantile(dis, np.linspace(0, 1, N_BINS + 1))
        edges[0], edges[-1] = -np.inf, np.inf
        dbin = _bin(dis, edges)
        dlevels = sorted(np.unique(dbin).tolist())

        # diagnostic: does disagreement track novelty? mean disagreement per sim stratum.
        rho = float(spearmanr(dis, sim).correlation)
        mean_dis_by_sim = {int(k): float(dis[sim == k].mean()) for k in sim_levels}

        # accumulators keyed on SIMILARITY stratum for each calibration mode
        modes = ("global", "disagree_keyed", "similarity_oracle")
        acc = {mode: {k: [0, 0, 0] for k in sim_levels} for mode in modes}

        for _ in range(n_repeats):
            perm = g.permutation(n)
            cal, test = perm[: n // 2], perm[n // 2:]
            tau_global = ltt_threshold(s[cal], y[cal], alpha=ALPHA, delta=DELTA)
            tau_dis = {b: (ltt_threshold(s[cal[dbin[cal] == b]], y[cal[dbin[cal] == b]],
                                         alpha=ALPHA, delta=DELTA)
                           if (dbin[cal] == b).sum() >= 40 else None) for b in dlevels}
            tau_sim = {k: (ltt_threshold(s[cal[sim[cal] == k]], y[cal[sim[cal] == k]],
                                         alpha=ALPHA, delta=DELTA)
                           if (sim[cal] == k).sum() >= 40 else None) for k in sim_levels}

            for k in sim_levels:
                tk = test[sim[test] == k]
                if len(tk) == 0:
                    continue
                sk, yk, dk = s[tk], y[tk], dbin[tk]
                _tally(acc["global"][k], sk, yk, np.full(len(tk), _num(tau_global)))
                taus_d = np.array([_num(tau_dis[b]) for b in dk])  # per-pose bin tau
                _tally(acc["disagree_keyed"][k], sk, yk, taus_d)
                _tally(acc["similarity_oracle"][k], sk, yk, np.full(len(tk), _num(tau_sim[k])))

        strata_out = {}
        for k in sim_levels:
            row = {}
            for mode in modes:
                na, ne, nt = acc[mode][k]
                row[mode] = {
                    "selective_risk": (ne / na) if na else float("nan"),
                    "coverage": (na / nt) if nt else 0.0,
                }
            strata_out[k] = row
        out[m] = {
            "n": n,
            "spearman_disagreement_vs_novelty": rho,
            "mean_disagreement_by_stratum": mean_dis_by_sim,
            "strata": strata_out,
        }
    return out


def _num(tau):
    """Map a threshold (or None = abstain) to a numeric never-accept sentinel."""
    return np.inf if tau is None else float(tau)


def _tally(bucket, sk, yk, taus):
    """Accumulate [accepted, errors, total] for a per-pose numeric tau array (inf = abstain)."""
    accept = sk >= taus
    bucket[0] += int(accept.sum())
    bucket[1] += int((1 - yk[accept]).sum())
    bucket[2] += len(sk)


def main() -> None:
    res = run()
    save_json(res, RESDIR / "e14_disagreement_strata.json")
    print(f"E14 -- disagreement-keyed vs similarity-keyed repair  (alpha={ALPHA}, delta={DELTA})\n")
    for m, r in res.items():
        print(f"[{m}]  spearman(disagreement, novelty)={r['spearman_disagreement_vs_novelty']:+.3f}")
        print(f"       {'S':>2} | {'global':>14} | {'disagree-keyed':>16} | {'sim-oracle':>14}")
        for k in sorted(r["strata"]):
            gg = r["strata"][k]["global"]
            dd = r["strata"][k]["disagree_keyed"]
            oo = r["strata"][k]["similarity_oracle"]
            print(f"       S{k} | risk {gg['selective_risk']:.3f} cov {gg['coverage']:.2f} "
                  f"| risk {dd['selective_risk']:.3f} cov {dd['coverage']:.2f} "
                  f"| risk {oo['selective_risk']:.3f} cov {oo['coverage']:.2f}")
        md = r["mean_disagreement_by_stratum"]
        print(f"       mean disagreement by stratum: {{{', '.join(f'S{k}:{v:.2f}' for k,v in md.items())}}}\n")


if __name__ == "__main__":
    main()
