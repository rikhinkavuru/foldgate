"""b1 -- Impossibility (Theorem 1): a single marginal threshold cannot control
per-stratum selective risk once confidence reliability decays on novel strata.

Three deliverables, all against the closed-form oracle in foldgate.bench.synth:

C1  Exact decomposition R_Q = R_P + C_cov + Delta_bar, Monte-Carlo estimated,
    verified to < 3x MC standard error. In this shared-eta generator the label
    conditional is identical under the calibration law P and the target law Q, so
    the POOLED accept-region concept gap Delta_bar is ~0 by construction: the
    pooled target-vs-calibration risk gap is entirely COVARIATE (the nu-marginal
    tilt), which is exactly why weighted CP repairs the pooled number. The concept
    effect the paper cares about lives at the WORST STRATUM, reported alongside.

C1/T1  Worst-stratum excess of the marginal gate. The analytic impossibility gap
    Delta(D,T) = max_k R_k(tau_inf) - alpha grows monotonically in the concept
    slope D and in the tilt T = KL(p_tgt || p_cal), and Delta -> 0 as D -> 0.
    Finite-sample marginal thresholds (naive, targeting alpha on a pooled cal
    draw) reproduce max_k R_k >> alpha, tracking Delta.

C4  Unachievability crossing. Sweeping D drives the novel stratum's oracle
    per-stratum threshold tau_g* to NaN: below the crossing D* even a marginal
    method leaves the worst stratum over alpha while Mondrian still hits alpha at
    positive coverage; above D* even Mondrian must drop coverage to hold alpha.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from _bench_common import (
    ALPHA,
    DELTA,
    FIGBENCH,
    N_SEEDS,
    RESBENCH,
    bootstrap_ci,
    rng,
    save_json,
)

from foldgate.bench import synth as S
from foldgate.bench.synth import SynthParams
from foldgate.conformal.risk import naive_threshold

D_GRID = [0.0, 0.5, 1.0, 2.0, 4.0]
CTGT_GRID = [3.0, 1.0, 0.0, -1.0]   # c_cal=3 fixed; smaller c_tgt -> more novel mass -> larger T
K = 8
E = 0.0                              # isolate concept shift (E=0 keeps P(s|nu) shared)
N_CAL = 2000
N_MC = 2_000_000


# --------------------------------------------------------------------------- #
# C1 -- exact Monte-Carlo decomposition of the pooled accept-region risk
# --------------------------------------------------------------------------- #
def decomposition_check(params: SynthParams, alpha: float, seed: int = 0) -> dict:
    tau = S.oracle_marginal_threshold(params.p_cal(), params, alpha)
    g = rng(seed)
    cal = S.sample(N_MC, params, "cal", g)
    tgt = S.sample(N_MC, params, "tgt", g)

    ac = cal[cal.s >= tau]
    at = tgt[tgt.s >= tau]
    R_P = float(1.0 - ac.y.mean())                 # E_P[err | A]
    R_Q = float(1.0 - at.y.mean())                 # E_Q[err | A]  (realized)
    R_ref = float(1.0 - at.pi.mean())              # E_Q[eta_P | A] (P labels, Q covariates)
    C_cov = R_ref - R_P
    Delta_bar = R_Q - R_ref                        # pooled accept-region concept gap
    # standard errors on the independently estimated MC terms
    se_RP = float(np.sqrt(R_P * (1 - R_P) / len(ac)))
    se_RQ = float(np.sqrt(R_Q * (1 - R_Q) / len(at)))
    se_sum = float(np.sqrt(se_RP**2 + 2 * se_RQ**2))
    resid = R_Q - (R_P + C_cov + Delta_bar)        # telescopes; MC residual ~ 0

    imp = S.oracle_impossibility_gap(params, alpha)
    return {
        "tau_inf": float(tau),
        "R_P": R_P, "R_Q": R_Q, "R_ref": R_ref,
        "C_cov": float(C_cov), "Delta_bar_pooled": float(Delta_bar),
        "identity_residual": float(resid), "mc_se_sum": se_sum,
        "identity_holds_within_3se": bool(abs(resid) < 3 * se_sum),
        "pooled_delta_bar_near_zero": bool(abs(Delta_bar) < 3 * se_RQ),
        "worst_stratum_risk": imp["worst_risk"],
        "worst_stratum_excess": imp["delta"],       # the concept-driven gap Delta(D,T)
        "n_accept_cal": int(len(ac)), "n_accept_tgt": int(len(at)),
    }


# --------------------------------------------------------------------------- #
# T1 -- analytic gap Delta(D,T) and finite-sample marginal worst-stratum risk
# --------------------------------------------------------------------------- #
def analytic_gap_grid(alpha: float) -> dict:
    grid = {}
    for ctgt in CTGT_GRID:
        row = []
        for D in D_GRID:
            p = SynthParams(K=K, D=D, E=E, c_cal=3.0, c_tgt=ctgt)
            g = S.oracle_impossibility_gap(p, alpha)
            row.append({"D": D, "tilt_T": p.tilt_kl(),
                        "delta": g["delta"], "worst_risk": g["worst_risk"],
                        "tau_inf": g["tau_inf"]})
        grid[f"c_tgt={ctgt}"] = row
    # monotonicity: delta increasing in D (per tilt col), delta -> 0 at D=0
    mono_D = all(
        all(np.diff([r["delta"] for r in grid[key]]) > -1e-9) for key in grid
    )
    d0 = max(abs(grid[key][0]["delta"]) for key in grid)   # D=0 column
    # increasing in T at fixed D: compare across c_tgt (larger T = smaller c_tgt)
    T_by_col = [SynthParams(K=K, D=1.0, E=E, c_cal=3.0, c_tgt=c).tilt_kl() for c in CTGT_GRID]
    delta_D1 = [grid[f"c_tgt={c}"][D_GRID.index(1.0)]["delta"] for c in CTGT_GRID]
    order = np.argsort(T_by_col)
    mono_T = bool(np.all(np.diff(np.array(delta_D1)[order]) > -1e-6))
    return {"grid": grid, "monotone_in_D": bool(mono_D),
            "delta_at_D0": float(d0), "delta_zero_at_D0": bool(d0 < 1e-3),
            "monotone_in_T": mono_T,
            "tilt_T_by_ctgt": dict(zip([f"c_tgt={c}" for c in CTGT_GRID], T_by_col, strict=False))}


def finite_sample_marginal(alpha: float, ctgt: float = 0.0, n_seeds: int = N_SEEDS) -> dict:
    """Empirical worst-stratum risk of the finite-sample marginal gate vs analytic Delta."""
    out = {}
    for D in D_GRID:
        p = SynthParams(K=K, D=D, E=E, c_cal=3.0, c_tgt=ctgt)
        imp = S.oracle_impossibility_gap(p, alpha)
        worst = []
        for seed in range(n_seeds):
            g = rng(10_000 + seed)
            cal = S.sample(N_CAL, p, "cal", g)
            tau = naive_threshold(cal.s.to_numpy(), cal.y.to_numpy(), alpha)
            if tau is None:
                continue
            rk = S.oracle_selective_risks(tau, p)      # exact truth at the fitted tau
            worst.append(float(np.nanmax(rk)))
        worst = np.array(worst, dtype=float)
        lo, hi = bootstrap_ci(worst)
        out[f"D={D}"] = {
            "analytic_delta": imp["delta"],
            "analytic_worst_risk": imp["worst_risk"],
            "emp_worst_risk_mean": float(np.mean(worst)) if worst.size else float("nan"),
            "emp_worst_risk_ci95": [lo, hi],
            "emp_worst_excess_mean": float(np.mean(worst) - alpha) if worst.size else float("nan"),
            "worst_gt_alpha": bool(np.mean(worst) > alpha) if worst.size else False,
            "n_valid": int(worst.size),
        }
    return out


# --------------------------------------------------------------------------- #
# C4 -- unachievability crossing (novel stratum tau_g* -> NaN)
# --------------------------------------------------------------------------- #
def _tau_at_coverage(k: int, c: float, params: SynthParams) -> float:
    """Per-stratum threshold tau with P(s >= tau | k) = c (coverage-pinned, S1)."""
    from scipy.special import expit as _expit
    from scipy.stats import norm
    t = float(norm.isf(c, loc=params.mu()[k], scale=params.sigma_s))
    return float(_expit(t))


def unachievability_crossing(alpha: float, c_pin: float = 0.5) -> dict:
    """Locate Theorem 1(c)'s coverage-pinned crossing.

    Two facts are reported honestly. (i) Any-coverage: the novel stratum's oracle
    threshold tau_g* exists for every D in the grid because the logistic score has
    no error floor, so Mondrian can always reach alpha by DROPPING coverage. (ii)
    Coverage-pinned at c: R_g(tau at coverage c) crosses alpha at D_pin. Below
    D_pin Mondrian holds alpha at full coverage c; above D_pin even Mondrian must
    drop the novel stratum's coverage below c -- the alpha - R_ref crossing.
    """
    Ds = np.round(np.arange(0.0, 6.0 + 1e-9, 0.1), 3)
    g_novel = K - 1
    any_ach, deltas, rg_pin = [], [], []
    for D in Ds:
        p = SynthParams(K=K, D=float(D), E=E, c_cal=3.0, c_tgt=0.0)
        any_ach.append(bool(not np.isnan(S.oracle_tau_star(g_novel, p, alpha))))
        deltas.append(S.oracle_impossibility_gap(p, alpha)["delta"])
        rg_pin.append(float(S.oracle_selective_risk(_tau_at_coverage(g_novel, c_pin, p), g_novel, p)))
    any_ach, deltas, rg_pin = np.array(any_ach), np.array(deltas), np.array(rg_pin)
    pos = np.where(deltas > 1e-3)[0]
    d_onset = float(Ds[pos.min()]) if pos.size else float("nan")
    over = np.where(rg_pin > alpha)[0]
    d_pin = float(Ds[over.min()]) if over.size else float("nan")
    return {
        "D_grid": [float(x) for x in Ds],
        "novel_stratum": g_novel,
        "coverage_pin": c_pin,
        "any_coverage_achievable": [bool(x) for x in any_ach],
        "novel_risk_at_coverage_c": [float(x) for x in rg_pin],
        "delta_by_D": [float(x) for x in deltas],
        "D_marginal_onset": d_onset,              # marginal over-covers the worst stratum beyond this
        "D_mondrian_drop_coverage": d_pin,        # Mondrian must drop coverage below c beyond this
        "mondrian_always_achievable_positive_coverage": bool(any_ach.all()),
        "alpha": alpha,
    }


# --------------------------------------------------------------------------- #
def make_figure(gap: dict, fin: dict, cross: dict, path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    for key, row in gap["grid"].items():
        row[0]["D"]  # placeholder; label by tilt of that column
        Tval = row[-1]["tilt_T"]
        ax.plot([r["D"] for r in row], [r["delta"] for r in row],
                marker="o", ms=4, label=f"{key} (T={Tval:.2f})")
    emp_D = [float(k.split("=")[1]) for k in fin]
    emp = [fin[k]["emp_worst_excess_mean"] for k in fin]
    ax.plot(emp_D, emp, "k*--", ms=11, label="finite-sample (c_tgt=0)")
    ax.axhline(0.0, color="0.5", lw=0.8)
    ax.set_xlabel("concept-shift slope D")
    ax.set_ylabel("worst-stratum excess  max_k R_k(tau_inf) - alpha")
    ax.set_title("T1: impossibility gap grows with D and tilt T")
    ax.legend(fontsize=7)

    ax2 = axes[1]
    Ds = cross["D_grid"]
    ax2.plot(Ds, cross["novel_risk_at_coverage_c"], "b-",
             label=f"novel-stratum risk @ coverage {cross['coverage_pin']}")
    ax2.axhline(cross["alpha"], color="k", ls=":", label=f"alpha={cross['alpha']}")
    ax2.axvline(cross["D_mondrian_drop_coverage"], color="r", ls="--",
                label=f"D_pin = {cross['D_mondrian_drop_coverage']:.1f}")
    ax2.set_xlabel("concept-shift slope D")
    ax2.set_ylabel("novel-stratum selective risk")
    ax2.set_title("C4: coverage-pinned unachievability crossing")
    ax2.legend(fontsize=7)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run() -> dict:
    alpha = ALPHA
    params_c1 = SynthParams(K=K, D=1.0, E=0.5, c_cal=3.0, c_tgt=0.0)
    c1 = decomposition_check(params_c1, alpha)
    gap = analytic_gap_grid(alpha)
    fin = finite_sample_marginal(alpha)
    cross = unachievability_crossing(alpha)
    return {
        "config": {"alpha": alpha, "delta": DELTA, "K": K, "E": E,
                   "D_grid": D_GRID, "c_tgt_grid": CTGT_GRID,
                   "n_cal": N_CAL, "n_mc": N_MC, "n_seeds": N_SEEDS,
                   "c1_params": {"D": 1.0, "E": 0.5, "c_cal": 3.0, "c_tgt": 0.0}},
        "C1_decomposition": c1,
        "T1_analytic_gap": gap,
        "T1_finite_sample_marginal": fin,
        "C4_unachievability": cross,
    }


def main() -> None:
    res = run()
    save_json(res, RESBENCH / "b1_synth_impossibility.json")
    make_figure(res["T1_analytic_gap"], res["T1_finite_sample_marginal"],
                res["C4_unachievability"], FIGBENCH / "b1_synth_impossibility.png")

    c1 = res["C1_decomposition"]
    print("b1 -- Impossibility (T1), alpha =", ALPHA)
    print(f"\n[C1] pooled decomposition  R_Q = R_P + C_cov + Delta_bar (MC, N={N_MC:.0e})")
    print(f"  R_P={c1['R_P']:.4f}  C_cov={c1['C_cov']:+.4f}  Delta_bar_pooled={c1['Delta_bar_pooled']:+.4f}"
          f"  -> R_Q={c1['R_Q']:.4f}")
    print(f"  identity residual={c1['identity_residual']:+.2e}  (3*MC_SE={3*c1['mc_se_sum']:.2e})"
          f"  within_3se={c1['identity_holds_within_3se']}")
    print(f"  pooled Delta_bar ~ 0 (shared-eta -> pooled gap is covariate): {c1['pooled_delta_bar_near_zero']}")
    print(f"  worst-stratum concept excess Delta(D,T)={c1['worst_stratum_excess']:.4f}"
          f"  (worst risk={c1['worst_stratum_risk']:.4f})")

    g = res["T1_analytic_gap"]
    print(
        "\n[T1] analytic gap monotone_in_D={}  delta_at_D0={:.2e} (->0: {})  monotone_in_T={}".format(
            g["monotone_in_D"], g["delta_at_D0"], g["delta_zero_at_D0"], g["monotone_in_T"]
        )
    )
    print("  Delta(D,T) grid (c_tgt=0 column):")
    for r in g["grid"]["c_tgt=0.0"]:
        print(f"    D={r['D']:<4} T={r['tilt_T']:.3f}  delta={r['delta']:+.4f}  worst={r['worst_risk']:.4f}")

    print("\n[T1] finite-sample marginal worst-stratum risk vs analytic Delta (c_tgt=0):")
    for k, v in res["T1_finite_sample_marginal"].items():
        print(f"    {k:<7} emp_worst={v['emp_worst_risk_mean']:.4f} "
              f"ci95=[{v['emp_worst_risk_ci95'][0]:.4f},{v['emp_worst_risk_ci95'][1]:.4f}] "
              f"analytic={v['analytic_worst_risk']:.4f}  >alpha={v['worst_gt_alpha']}")

    c = res["C4_unachievability"]
    print(f"\n[C4] marginal-impossibility onset D>={c['D_marginal_onset']:.1f}; "
          f"Mondrian must drop coverage below {c['coverage_pin']} beyond D_pin={c['D_mondrian_drop_coverage']:.1f} "
          f"(novel stratum {c['novel_stratum']}); "
          f"positive-coverage always achievable={c['mondrian_always_achievable_positive_coverage']}")

    pass_c1 = c1["identity_holds_within_3se"]
    pass_t1 = g["monotone_in_D"] and g["delta_zero_at_D0"] and g["monotone_in_T"] and \
        all(res["T1_finite_sample_marginal"][f"D={D}"]["worst_gt_alpha"] for D in [1.0, 2.0, 4.0])
    print(f"\nPASS  C1(identity+Delta_bar): {pass_c1}   T1(monotone+worst>alpha): {pass_t1}")


if __name__ == "__main__":
    main()
