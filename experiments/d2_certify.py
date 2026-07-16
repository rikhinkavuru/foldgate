"""D2 step 2: how many target labels certify the deployed rule, on the strata where it is feasible.

The feasibility map (`d2_feasibility_map.py`) says WHERE the deployed rule holds alpha. This says
what it COSTS to prove it there. On the feasible strata we hide the RMSD labels, reveal one
independent target-label at a time, and record the budget at which each certificate first fires.
RNP has every label, so the reveal is simulated and the only cost is bookkeeping.

WHAT IS BEING COMPARED, AND WHAT IS NOT

  * Hoeffding (pure). The textbook passive RCPS bound, with no binomial term.
  * Hoeffding-Bentkus (the project's existing certifier). min(Hoeffding, e * Binom.cdf), so it
    already carries a binomial tail term.
  * Exact binomial. Certify when P(Bin(n, alpha) <= errors) <= delta. This is the test
    `ltt_threshold` already uses, and it is the exact tool for a Bernoulli mean.
  * WSR betting confidence sequence. Certify when the betting p-value for H0: E[L] >= alpha falls
    below delta. Variance-adaptive.

THE PLANNED WIN WAS REAL BUT AIMED AT A BASELINE WE HAD ALREADY BEATEN. The D2 design predicted
that empirical-Bernstein variance-adaptivity would be the robust saving over passive Hoeffding-RCPS
at the small accepted error rates the feasible cells show (p ~ 0.06-0.18). The measurement says
otherwise, and the reason is structural but NOT the one an earlier draft of this docstring gave.
That draft argued there is "nothing for a variance-adaptive bound to adapt to" because a Bernoulli
variance p(1-p) is a deterministic function of the mean. The premise is true and the inference is
wrong: pure Hoeffding does not use the count's information at all, it substitutes the worst-case
variance 1/4, so there is a great deal to adapt to and WSR does capture it (cheaper than pure
Hoeffding in 12/12 cells where both fire, median 49 vs 102).

The correct statement is about sufficiency, not about variance. The count is a sufficient statistic
for a Bernoulli mean and the exact binomial tail is its exact inversion, so every fixed-n
concentration bound is a RELAXATION of that tail and none can be tighter. Variance-adaptivity buys
back the distance from pure Hoeffding to the tail, but so does a Bentkus binomial term, and this
project's `hb_upper_bound` already carried one (60 labels vs WSR's 62). So the planned upgrade was
measuring itself against a baseline the repo had already left behind, and `ltt_threshold` was
already using the exact tool.

WSR is NOT merely a relaxation of the fixed-n tail: it is an anytime-valid betting sequence, so its
extra width at a fixed n is the premium for validity under optional stopping, which the fixed-n
bounds do not have. That is a different guarantee rather than a worse one.

NO SAVING IS EVER CREDITED TO LABEL PLACEMENT. Certifying a fixed rule at a fixed threshold is
fixed-mean estimation, and active label acquisition does not lower the labels needed for an unbiased
estimate of a fixed mean; the naive "active learning saves labels" framing is ill-posed and is not
claimed. Every design here draws uniformly at random within the stratum, so any difference between
certifiers is a property of the bound and not of the order the labels arrive in.

THREE PIECES OF HONESTY THAT CUT AGAINST THE HEADLINE, ALL KEPT

  1. The comparison is biased IN THE FIXED-n BOUNDS' FAVOUR. A WSR confidence sequence is
     anytime-valid, so stopping the first time it fires is legitimate. The fixed-n bounds
     (Hoeffding, HB, exact binomial) are valid only at a budget fixed in advance, so peeking at
     every budget and stopping on the first success inflates their error rate above delta. We let
     them peek anyway. The tilt runs against WSR and WSR loses regardless, so the ORDERING is
     conservative. The BUDGETS are not: every fixed-n entry, including the exact binomial's
     headline number, is deflated by exactly that peeking and is a lower bound on the planned
     budget a caller could stand behind.
  2. The per-cell budget is a median over the reveal orders in which a certifier FIRED, so a cell
     where a bound fails in a large fraction of orders reports only its successful half. A cell
     counts as certified when the bound fires in more than half the orders (RELIABLE below), which
     is a weaker statement than "certifies this cell". Both the never-fired rate and the budget are
     reported so the two are not confused.
  3. The PPI control covariate is reported as a MEASUREMENT, not as a curve. Prediction-powered
     inference reduces variance to (1 - rho^2) of the label-only baseline (Angelopoulos-Zrnic), but
     that factor is asymptotic and the control-variate variable is unbounded, so it cannot go into
     a bounded WSR bet while keeping the half-width. Splicing an asymptotic PPI interval into a
     finite-sample labels-to-certify plot would compare two different guarantees, so we report the
     measured rho_g profile and the implied asymptotic saving separately and let the reader see how
     small it is.

Outputs `results/d2_certify.json`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scipy.stats import binom  # noqa: E402

from experiments._common import CONF, DELTA, RESDIR, load_delivered, methods_with_enough, rng, save_json  # noqa: E402
from experiments.d2_feasibility_map import AXES, COVARIATES, SOURCE_STRATUM, _rho, target_level  # noqa: E402
from foldgate.conformal.risk import hb_upper_bound, wsr_betting_pvalue  # noqa: E402


def _cert_hoeffding(x: np.ndarray, alpha: float, delta: float) -> bool:
    """Pure Hoeffding UCB on a [0,1] mean, the textbook passive RCPS bound with no binomial term."""
    n = len(x)
    return bool(float(x.mean()) + np.sqrt(np.log(1.0 / delta) / (2.0 * n)) <= alpha)


def _cert_hb(x: np.ndarray, alpha: float, delta: float) -> bool:
    """The project's existing Hoeffding-Bentkus certifier: min(Hoeffding, e * Binom.cdf)."""
    return bool(hb_upper_bound(float(x.mean()), len(x), delta) <= alpha)


def _cert_binom(x: np.ndarray, alpha: float, delta: float) -> bool:
    """Exact binomial test of H0: E[L] >= alpha, the tool `ltt_threshold` already uses."""
    return bool(binom.cdf(int(x.sum()), len(x), alpha) <= delta)


def _cert_wsr(x: np.ndarray, alpha: float, delta: float) -> bool:
    """WSR betting confidence sequence: variance-adaptive, anytime-valid."""
    return bool(wsr_betting_pvalue(x, alpha, delta) <= delta)


# Order matters only for reporting. `binom` is the exact tool for a Bernoulli mean and is the one
# the measurement below vindicates.
CERTIFIERS = {"hoeffding": _cert_hoeffding, "hb": _cert_hb,
              "binom": _cert_binom, "wsr": _cert_wsr}

ALPHA_GRID = (0.10, 0.20)
COVERAGE = 0.5          # the deployed operating point these curves are reported at
N_TRIALS = 200          # random reveal orders per cell
GRID_STEP = 2           # evaluate the certificates every GRID_STEP labels
MIN_CELL = 40           # a cell needs at least this many accepted targets to be worth a curve


def _labels_to_certify(loss: np.ndarray, alpha: float, delta: float,
                       g_rng: np.random.Generator, n_trials: int, step: int) -> dict:
    """Distribution of the budget at which each certificate first fires, over random reveal orders.

    Returns the median budget (inf-safe), the certification rate at each budget, and the fraction of
    trials that never certify within the labels available.
    """
    n = len(loss)
    grid = np.arange(step, n + 1, step)
    first = {k: [] for k in CERTIFIERS}
    for _ in range(n_trials):
        L = loss[g_rng.permutation(n)]
        fire = dict.fromkeys(CERTIFIERS, np.inf)
        for m in grid:
            x = L[:m]
            for k, fn in CERTIFIERS.items():
                if not np.isfinite(fire[k]) and fn(x, alpha, delta):
                    fire[k] = float(m)
            if all(np.isfinite(v) for v in fire.values()):
                break
        for k in CERTIFIERS:
            first[k].append(fire[k])

    def _summ(v):
        a = np.array(v, dtype=float)
        fin = a[np.isfinite(a)]
        # "Certified by budget b" is P(first fire <= b), derived from the first-fire budgets rather
        # than accumulated in the loop, so a trial that fires both certificates at the same budget
        # cannot be counted twice.
        return {
            "median_when_certified": float(np.median(fin)) if fin.size else None,
            "never_certified_rate": float(np.mean(~np.isfinite(a))),
            "cert_rate_by_budget": [float(np.mean(a <= b)) for b in grid],
        }

    out = {"n_available": n, "grid": grid.tolist()}
    out.update({k: _summ(v) for k, v in first.items()})
    return out


def run(delta: float = DELTA, n_trials: int = N_TRIALS, seed: int = 20260715) -> dict:
    df = load_delivered()
    methods = methods_with_enough(df)
    tl = target_level(df)
    g_rng = rng(seed)
    out = {"alphas": list(ALPHA_GRID), "delta": delta, "coverage": COVERAGE,
           "n_trials": n_trials, "min_cell": MIN_CELL, "cells": [], "rho_profile": []}

    for axis, col in AXES.items():
        for m in methods:
            d = tl[(tl.method == m) & tl[col].notna()].copy()
            if len(d) == 0:
                continue
            d["loss"] = (1 - d["correct"].astype(int)).astype(float)
            src = d[d[col] == SOURCE_STRATUM]
            if len(src) < 40:
                continue
            # Same deployed rule as the feasibility map: tau fixed on the source, then frozen.
            tau = float(np.quantile(src[CONF].to_numpy(), 1.0 - COVERAGE))
            for g in sorted(d[col].unique()):
                acc = d[(d[col] == g) & (d[CONF] >= tau)]
                loss = acc["loss"].to_numpy()
                if len(loss) < MIN_CELL:
                    continue
                r = float(loss.mean())

                # rho_g profile (Target 3), measured on every candidate label-free covariate.
                rhos = {cv: _rho(loss, acc[cv].to_numpy(dtype=float))
                        for cv in COVARIATES if cv in acc.columns}
                finite = {k: v for k, v in rhos.items() if np.isfinite(v)}
                best = max(finite, key=lambda k: abs(finite[k])) if finite else None
                out["rho_profile"].append({
                    "axis": axis, "model": m, "stratum": int(g), "n": int(len(loss)),
                    "R_Q": r, "rho": rhos, "best_covariate": best,
                    "rho_best": finite.get(best, float("nan")) if best else float("nan"),
                    # The PPI variance-reduction factor this rho implies, asymptotically.
                    "ppi_variance_kept": (1.0 - finite[best] ** 2) if best else float("nan"),
                })

                for alpha in ALPHA_GRID:
                    feasible = r < alpha
                    cell = {"axis": axis, "model": m, "stratum": int(g), "alpha": alpha,
                            "tau": tau, "R_Q": r, "margin": alpha - r, "feasible": bool(feasible)}
                    if feasible:
                        cell.update(_labels_to_certify(loss, alpha, delta, g_rng, n_trials, GRID_STEP))
                    out["cells"].append(cell)

    # --- headline: which certifier actually costs the fewest target-labels --------------------
    # A cell is kept if ANY certifier fires in the majority of reveal orders. Requiring all four to
    # fire would silently drop exactly the cells that matter -- the thin-margin ones where the
    # exact binomial certifies and pure Hoeffding never does -- and would flatter the weak bounds
    # by scoring them only where they already succeed.
    RELIABLE = 0.5   # a certifier "certifies this cell" if it fires in >half the reveal orders
    wins = []
    for c in out["cells"]:
        if not c.get("feasible") or "wsr" not in c:
            continue
        row = {"axis": c["axis"], "model": c["model"], "stratum": c["stratum"],
               "alpha": c["alpha"], "R_Q": c["R_Q"], "margin": c["margin"], "n": c["n_available"]}
        for k in CERTIFIERS:
            fires = c[k]["never_certified_rate"] < RELIABLE
            row[k] = c[k]["median_when_certified"] if fires else None
            row[f"{k}_never_rate"] = c[k]["never_certified_rate"]
        if any(row[k] is not None for k in CERTIFIERS):
            wins.append(row)

    def _med(k):
        v = [w[k] for w in wins if w[k] is not None]
        return float(np.median(v)) if v else float("nan")

    def _both(a, b, cmp):
        pair = [w for w in wins if w[a] is not None and w[b] is not None]
        return sum(1 for w in pair if cmp(w[a], w[b])), len(pair)

    wsr_hb, n_wsr_hb = _both("wsr", "hb", lambda x, y: x <= y)
    bin_wsr, n_bin_wsr = _both("binom", "wsr", lambda x, y: x <= y)
    out["summary"] = {
        "n_feasible_cells_with_curves": len(wins),
        # How many feasible cells each certifier can certify AT ALL within the labels available.
        # This matters more than the median budget: a bound that never fires has no budget.
        "n_cells_certified": {k: sum(1 for w in wins if w[k] is not None) for k in CERTIFIERS},
        "median_labels_to_certify_where_it_fires": {k: _med(k) for k in CERTIFIERS},
        # The planned claim, scored: does variance-adaptive betting beat the project's existing
        # Hoeffding-Bentkus certifier at small p? Reported whichever way it comes out.
        "n_wsr_better_or_equal_than_hb": [wsr_hb, n_wsr_hb],
        "n_binom_better_or_equal_than_wsr": [bin_wsr, n_bin_wsr],
        "note": ("every certifier is allowed to peek at every budget. That is legitimate only for "
                 "the anytime-valid WSR sequence and is anti-conservative for the fixed-n bounds, "
                 "so the comparison is tilted TOWARD the baselines and against WSR; WSR still does "
                 "not win, which makes the negative result stronger rather than weaker"),
        "cells": wins,
    }
    rp = [r for r in out["rho_profile"] if np.isfinite(r["rho_best"])]
    out["rho_summary"] = {
        "median_abs_rho_best": float(np.median([abs(r["rho_best"]) for r in rp])) if rp else float("nan"),
        "median_ppi_variance_kept": float(np.median([r["ppi_variance_kept"] for r in rp])) if rp else float("nan"),
        "note": ("PPI's (1 - rho^2) is asymptotic and the control-variate variable is unbounded, so "
                 "this is reported as a measured add-on and is NOT spliced into the finite-sample "
                 "labels-to-certify curves above"),
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--delta", type=float, default=DELTA)
    ap.add_argument("--trials", type=int, default=N_TRIALS)
    args = ap.parse_args()
    res = run(delta=args.delta, n_trials=args.trials)
    save_json(res, RESDIR / "d2_certify.json")

    s = res["summary"]
    print(f"=== D2 labels-to-certify (coverage={COVERAGE}, delta={args.delta}, "
          f"{args.trials} reveal orders) ===\n")
    n = s["n_feasible_cells_with_curves"]
    print(f"feasible cells with curves: {n}")
    m = s["median_labels_to_certify_where_it_fires"]
    cc = s["n_cells_certified"]
    print(f"\n  {'certifier':>10} {'cells certified':>16} {'median labels':>14}")
    for k in CERTIFIERS:
        print(f"  {k:>10} {f'{cc[k]}/{n}':>16} {m[k]:>14.0f}")
    a, b = s["n_wsr_better_or_equal_than_hb"]
    print(f"\n  WSR <= Hoeffding-Bentkus in {a}/{b} cells where both fire")
    a, b = s["n_binom_better_or_equal_than_wsr"]
    print(f"  exact binomial <= WSR in    {a}/{b} cells where both fire")
    print(f"\n  {'axis':>7} {'model':>9} {'S':>2} {'alpha':>5} {'n':>5} {'R_Q':>6} {'margin':>7} "
          f"{'hoeff':>6} {'HB':>5} {'binom':>6} {'WSR':>5}")

    def _f(v):
        return "  --  " if v is None else f"{v:6.0f}"
    for w in sorted(s["cells"], key=lambda x: (x["alpha"], x["axis"], x["model"], x["stratum"])):
        print(f"  {w['axis']:>7} {w['model']:>9} {w['stratum']:>2} {w['alpha']:>5} "
              f"{w['n']:>5} {w['R_Q']:>6.3f} {w['margin']:>7.3f} {_f(w['hoeffding'])} "
              f"{_f(w['hb'])} {_f(w['binom'])} {_f(w['wsr'])}")
    print("  (-- = the certifier fails to fire in the majority of reveal orders on the labels "
          "available)")
    r = res["rho_summary"]
    print(f"\nPPI covariate (measured, asymptotic): median |rho| = {r['median_abs_rho_best']:.3f} "
          f"-> variance kept {r['median_ppi_variance_kept']:.3f}")
    print(f"  {r['note']}")
    print(f"\nwrote {RESDIR / 'd2_certify.json'}")


if __name__ == "__main__":
    main()
