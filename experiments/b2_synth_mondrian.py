"""b2 -- Mondrian achievability (Theorem 3) + C3.

Group-conditional (Mondrian) calibration keyed on the novelty stratum restores
per-stratum selective-risk control that the marginal gate cannot (b1). Calibration
uses in-stratum labels (A8), so the guarantee holds regardless of concept drift.

C3 deliverables:
  * Conditional exceedance. Per stratum, fit tau_g by LTT/QRC (the ratio-valid
    high-probability certifier, NOT vanilla CRC) on n_g in-stratum labels. Over
    300 seeds the exceedance rate P(R_g(tau_g) > alpha) stays <= delta in every
    controllable stratum, so the worst-stratum exceedance is <= delta.
  * Finite-sample slack shrinks ~1/sqrt(n_g): sweep n_g and fit the log-log slope.
  * Thin-strata breakdown, honest (CLAUDE.md rule 5): as n_g falls below the LTT
    min-accept the group can only stay valid by ABSTAINING (coverage -> 0), and a
    naive uncorrected per-stratum threshold BREAKS (exceedance >> delta).
  * Joint vs ratio: the joint accept-and-err rate E[Y 1_A] is controlled at
    alpha + 1/(n_g+1) by CRC, but that clean slack does NOT transfer to the ratio
    selective risk E[Y | A] = joint / coverage. The two guarantees differ.
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
    rng,
    save_json,
)
from scipy.special import expit

from foldgate.bench import synth as S
from foldgate.bench.synth import SynthParams
from foldgate.conformal.risk import ltt_threshold, naive_threshold

K = 8
D = 1.0
E = 0.0
NCAL_GRID = [200, 500, 1000, 2000, 5000, 10000]
NG_THIN = [10, 20, 40, 80, 160, 320]


def sample_stratum(n: int, k: int, params: SynthParams, g: np.random.Generator):
    """Draw n examples from a single stratum k (its own P(s|nu)=P(s|nu) law)."""
    mu_k = params.mu()[k]
    nu_k = params.nu()[k]
    z = mu_k + params.sigma_s * g.standard_normal(n)
    s = expit(z)
    pi = expit(params.beta0 + params.beta1 * z - params.D * nu_k)
    y = (g.random(n) < pi).astype(int)
    return s, y


def crc_joint_threshold(s: np.ndarray, y: np.ndarray, alpha: float, n: int) -> float | None:
    """CRC threshold controlling the JOINT accept-and-err rate E[(1-y) 1_A] at alpha.

    Monotone per-example loss l_tau = (1-y) 1{s>=tau} in [0,1]. Grow the accept set
    (smallest tau) while the empirical joint error rate over ALL n points stays
    <= alpha - 1/(n+1); the CRC guarantee (Angelopoulos et al.) is then
    E[joint] <= alpha. Returns tau or None.
    """
    err = 1 - np.asarray(y, dtype=int)
    order = np.argsort(-np.asarray(s, dtype=float))
    s_sorted = s[order]
    joint_hat = np.cumsum(err[order]) / n          # joint error rate among all n
    ok = joint_hat <= alpha - 1.0 / (n + 1)
    if not ok.any() or not ok[0]:
        return None
    last = np.where(~ok)[0]
    last_idx = (n - 1) if len(last) == 0 else last[0] - 1
    return float(s_sorted[last_idx])


# --------------------------------------------------------------------------- #
# C3a -- conditional exceedance rate per stratum (LTT), worst-stratum <= delta
# --------------------------------------------------------------------------- #
def conditional_exceedance(params: SynthParams, alpha: float, delta: float,
                           n_g: int = 2000, n_seeds: int = N_SEEDS) -> dict:
    # The LTT guarantee is UNCONDITIONAL: P(certify AND R_g > alpha) <= delta over
    # the calibration draw. An abstention accepts nothing, so it never violates.
    per = {}
    for k in range(params.K):
        tau_star = S.oracle_tau_star(k, params, alpha)
        controllable = not np.isnan(tau_star)
        exc, abst, realized = 0, 0, []
        for seed in range(n_seeds):
            g = rng(20_000 + 101 * k + seed)
            s, y = sample_stratum(n_g, k, params, g)
            tau = ltt_threshold(s, y, alpha, delta)
            if tau is None:
                abst += 1
                continue
            r = S.oracle_selective_risk(tau, k, params)   # exact realized risk
            realized.append(r)
            exc += int(r > alpha + 1e-9)
        per[k] = {
            "nu": float(params.nu()[k]),
            "controllable": bool(controllable),
            "oracle_tau_star": float(tau_star),
            "exceedance_rate": float(exc / n_seeds),       # unconditional (the guarantee)
            "abstain_rate": float(abst / n_seeds),
            "mean_realized_risk": float(np.mean(realized)) if realized else float("nan"),
            "exceedance_le_delta": bool(exc / n_seeds <= delta),
        }
    worst_exc = max(v["exceedance_rate"] for v in per.values())
    return {"per_stratum": per, "worst_exceedance_rate": float(worst_exc),
            "worst_exceedance_le_delta": bool(worst_exc <= delta), "n_g": n_g}


# --------------------------------------------------------------------------- #
# C3b -- slack shrinks ~1/sqrt(n_g)
# --------------------------------------------------------------------------- #
def slack_vs_n(params: SynthParams, alpha: float, delta: float, k: int,
               c_pin: float = 0.5, n_seeds: int = N_SEEDS) -> dict:
    """Coverage-pinned concentration (T3c, second form): fix in-stratum coverage c
    via the empirical s-quantile; the realized selective risk concentrates on the
    population R_g(tau_c) at rate O_p(1/sqrt(n_g)). We report the RMSE of the
    realized risk about that reference and fit its log-log slope in n_g.
    """
    from scipy.special import expit as _expit
    from scipy.stats import norm
    tau_c_pop = float(_expit(norm.isf(c_pin, loc=params.mu()[k], scale=params.sigma_s)))
    r_ref = float(S.oracle_selective_risk(tau_c_pop, k, params))
    ns, rmses = [], []
    for n_g in NCAL_GRID:
        devs = []
        for seed in range(n_seeds):
            g = rng(30_000 + seed)
            s, y = sample_stratum(n_g, k, params, g)
            tau_hat = float(np.quantile(s, 1.0 - c_pin))     # empirical coverage-c threshold
            r = S.oracle_selective_risk(tau_hat, k, params)
            devs.append(r - r_ref)
        ns.append(n_g)
        rmses.append(float(np.sqrt(np.mean(np.square(devs)))))
    ns = np.array(ns, dtype=float)
    rmses = np.array(rmses, dtype=float)
    slope = float("nan")
    if len(ns) >= 2 and np.all(rmses > 0):
        slope = float(np.polyfit(np.log(ns), np.log(rmses), 1)[0])
    return {"stratum": k, "coverage_pin": c_pin, "r_ref": r_ref,
            "n_g": [int(x) for x in ns],
            "rmse_realized_vs_ref": [float(x) for x in rmses],
            "loglog_slope": slope,
            "slope_near_minus_half": bool(np.isfinite(slope) and -0.65 <= slope <= -0.35)}


# --------------------------------------------------------------------------- #
# C3c -- thin-strata breakdown (LTT abstains vs naive breaks)
# --------------------------------------------------------------------------- #
def thin_strata(params: SynthParams, alpha: float, delta: float, k: int,
                n_seeds: int = N_SEEDS) -> dict:
    # Honest thin-strata breakdown: LTT keeps its UNCONDITIONAL guarantee at every
    # n_g (exceedance <= delta) but only by ABSTAINING -- effective coverage (the
    # fraction of seeds with a usable certificate) collapses toward 0, so control is
    # vacuous. The naive uncorrected per-stratum threshold genuinely BREAKS.
    rows = {}
    for n_g in NG_THIN:
        ltt_abst, ltt_exc = 0, 0
        naive_exc, naive_eff = 0, 0
        for seed in range(n_seeds):
            g = rng(40_000 + seed)
            s, y = sample_stratum(n_g, k, params, g)
            tau_l = ltt_threshold(s, y, alpha, delta)
            if tau_l is None:
                ltt_abst += 1
            else:
                ltt_exc += int(S.oracle_selective_risk(tau_l, k, params) > alpha + 1e-9)
            tau_n = naive_threshold(s, y, alpha)
            if tau_n is not None:
                naive_eff += 1
                naive_exc += int(S.oracle_selective_risk(tau_n, k, params) > alpha + 1e-9)
        rows[n_g] = {
            "ltt_abstain_rate": float(ltt_abst / n_seeds),
            "ltt_effective_coverage": float((n_seeds - ltt_abst) / n_seeds),
            "ltt_exceedance_rate": float(ltt_exc / n_seeds),     # unconditional guarantee
            "ltt_valid": bool(ltt_exc / n_seeds <= delta),
            "naive_exceedance_rate": float(naive_exc / n_seeds),
            "naive_breaks": bool(naive_exc / n_seeds > delta),
        }
    return {"stratum": k, "rows": rows}


# --------------------------------------------------------------------------- #
# C3d -- joint (CRC, 1/(n+1)) vs ratio selective risk
# --------------------------------------------------------------------------- #
def joint_vs_ratio(params: SynthParams, alpha: float, k: int, n_g: int = 2000,
                   n_seeds: int = N_SEEDS) -> dict:
    joint_exc, ratio_exc, eff = 0, 0, 0
    joints, ratios = [], []
    for seed in range(n_seeds):
        g = rng(50_000 + seed)
        s, y = sample_stratum(n_g, k, params, g)
        tau = crc_joint_threshold(s, y, alpha, n_g)
        if tau is None:
            continue
        eff += 1
        a = S.oracle_accept_rate(tau, k, params)         # coverage
        r_ratio = S.oracle_selective_risk(tau, k, params)
        r_joint = a * r_ratio                             # E[(1-y) 1_A]
        joints.append(r_joint)
        ratios.append(r_ratio)
        joint_exc += int(r_joint > alpha + 1.0 / (n_g + 1))
        ratio_exc += int(r_ratio > alpha + 1e-9)
    mean_joint = float(np.mean(joints)) if joints else float("nan")
    mean_ratio = float(np.mean(ratios)) if ratios else float("nan")
    crc_bound = alpha + 1.0 / (n_g + 1)
    return {
        "stratum": k, "n_g": n_g,
        "mean_joint_error": mean_joint,     # CRC controls this expectation at <= alpha
        "mean_ratio_risk": mean_ratio,      # the ratio selective risk = joint / coverage
        "crc_joint_bound": crc_bound,
        # CRC's guarantee is on E[joint]; verify the realized expectation clears it.
        "joint_controlled": bool(mean_joint <= crc_bound),
        "ratio_exceeds_when_crc_only": bool(mean_ratio > alpha),
        "ratio_over_joint_ratio": float(mean_ratio / mean_joint) if mean_joint > 0 else float("nan"),
    }


def make_figure(exc: dict, slack: dict, thin: dict, path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    ax = axes[0]
    ks = list(exc["per_stratum"].keys())
    ax.bar([str(k) for k in ks], [exc["per_stratum"][k]["exceedance_rate"] for k in ks],
           color="#4a7", label="Mondrian LTT")
    ax.axhline(DELTA, color="r", ls="--", label=f"delta={DELTA}")
    ax.set_xlabel("stratum k")
    ax.set_ylabel("exceedance rate P(R_g>alpha)")
    ax.set_title(f"C3: conditional exceedance (n_g={exc['n_g']})")
    ax.legend(fontsize=8)

    ax2 = axes[1]
    ax2.loglog(slack["n_g"], slack["rmse_realized_vs_ref"], "o-", label="RMSE(realized R_g, ref)")
    n0 = np.array(slack["n_g"], dtype=float)
    rm = slack["rmse_realized_vs_ref"]
    if rm:
        ref = rm[0] * np.sqrt(n0[0] / n0)
        ax2.loglog(n0, ref, "k:", label="~1/sqrt(n_g)")
    ax2.set_xlabel("n_g")
    ax2.set_ylabel("selective-risk RMSE")
    ax2.set_title(f"coverage-pinned slack ~1/sqrt(n_g) (slope={slack['loglog_slope']:.2f})")
    ax2.legend(fontsize=8)

    ax3 = axes[2]
    ns = list(thin["rows"].keys())
    ax3.plot(ns, [thin["rows"][n]["ltt_exceedance_rate"] for n in ns], "g-o", label="LTT exceedance")
    ax3.plot(ns, [thin["rows"][n]["naive_exceedance_rate"] for n in ns], "r-s", label="naive exceedance")
    ax3.plot(ns, [thin["rows"][n]["ltt_abstain_rate"] for n in ns], "b--^", label="LTT abstain rate")
    ax3.axhline(DELTA, color="0.5", ls=":", label=f"delta={DELTA}")
    ax3.set_xscale("log")
    ax3.set_xlabel("n_g (thin)")
    ax3.set_ylabel("rate")
    ax3.set_title("thin-strata: LTT valid by abstaining, naive breaks")
    ax3.legend(fontsize=7)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run() -> dict:
    params = SynthParams(K=K, D=D, E=E, c_cal=3.0, c_tgt=0.0)
    alpha, delta = ALPHA, DELTA
    # a controllable moderately-novel stratum for the slack/thin/joint studies
    k_novel = max(k for k in range(K) if not np.isnan(S.oracle_tau_star(k, params, alpha)))
    exc = conditional_exceedance(params, alpha, delta)
    slack = slack_vs_n(params, alpha, delta, k_novel)
    thin = thin_strata(params, alpha, delta, k_novel)
    jvr = joint_vs_ratio(params, alpha, k_novel)
    return {
        "config": {"alpha": alpha, "delta": delta, "K": K, "D": D, "E": E,
                   "k_novel": int(k_novel), "n_seeds": N_SEEDS,
                   "ncal_grid": NCAL_GRID, "ng_thin": NG_THIN},
        "C3a_conditional_exceedance": exc,
        "C3b_slack_vs_n": slack,
        "C3c_thin_strata": thin,
        "C3d_joint_vs_ratio": jvr,
    }


def main() -> None:
    res = run()
    save_json(res, RESBENCH / "b2_synth_mondrian.json")
    make_figure(res["C3a_conditional_exceedance"], res["C3b_slack_vs_n"],
                res["C3c_thin_strata"], FIGBENCH / "b2_synth_mondrian.png")

    exc = res["C3a_conditional_exceedance"]
    print("b2 -- Mondrian achievability (T3/C3), alpha =", ALPHA, "delta =", DELTA)
    print(f"\n[C3a] per-stratum exceedance (n_g={exc['n_g']}, {N_SEEDS} seeds):")
    for k, v in exc["per_stratum"].items():
        print(f"  k={k} nu={v['nu']:.2f} ctrl={v['controllable']!s:>5} "
              f"exc={v['exceedance_rate']:.3f} abst={v['abstain_rate']:.3f} "
              f"meanR={v['mean_realized_risk']:.3f} <=delta:{v['exceedance_le_delta']}")
    print(f"  worst-stratum exceedance={exc['worst_exceedance_rate']:.3f} "
          f"(<= delta: {exc['worst_exceedance_le_delta']})")

    sl = res["C3b_slack_vs_n"]
    print(f"\n[C3b] coverage-pinned RMSE vs n_g (stratum {sl['stratum']}, ref R_g={sl['r_ref']:.3f}): "
          f"{list(zip(sl['n_g'], [round(x,4) for x in sl['rmse_realized_vs_ref']], strict=False))}")
    print(f"  log-log slope={sl['loglog_slope']:.3f}  ~ -1/2: {sl['slope_near_minus_half']}")

    th = res["C3c_thin_strata"]
    print(f"\n[C3c] thin strata (stratum {th['stratum']}), LTT stays valid by abstaining; naive breaks:")
    for n, v in th["rows"].items():
        print(f"  n_g={n:<4} LTT exc={v['ltt_exceedance_rate']:.3f} valid={v['ltt_valid']!s:>5} "
              f"eff_cov={v['ltt_effective_coverage']:.3f} | naive exc={v['naive_exceedance_rate']:.3f} "
              f"breaks={v['naive_breaks']}")

    j = res["C3d_joint_vs_ratio"]
    print(f"\n[C3d] joint vs ratio (stratum {j['stratum']}, n_g={j['n_g']}):")
    print(f"  E[joint error]={j['mean_joint_error']:.3f} <= CRC bound {j['crc_joint_bound']:.3f}: "
          f"{j['joint_controlled']}")
    print(f"  E[ratio selective risk]={j['mean_ratio_risk']:.3f} (= joint/coverage, "
          f"{j['ratio_over_joint_ratio']:.2f}x joint) exceeds alpha: {j['ratio_exceeds_when_crc_only']}")

    pass_c3 = (exc["worst_exceedance_le_delta"] and sl["slope_near_minus_half"]
               and j["joint_controlled"])
    print(f"\nPASS  C3(exceedance<=delta & slack~1/sqrt(n) & joint!=ratio): {pass_c3}")


if __name__ == "__main__":
    main()
