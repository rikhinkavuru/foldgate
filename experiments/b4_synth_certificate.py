"""b4 -- Distributionally-robust certificates (Theorem 2/3d, check C6).

Two label-free / finite-sample certificates on the worst-case selective risk:

  C6a  Worst-stratum RCPS UCB (Hoeffding-Bentkus + union bound over K strata).
       This is the finite-sample (1 - delta) certificate: over 300 seeds the
       validity rate P(max_k R_k <= U) is >= 1 - delta, and the slack U - max_k R_k
       shrinks as the per-stratum counts grow.

  C6b  f-divergence (KL) ball over the K-stratum simplex. The certified worst risk
       sup_{q: KL(q||p_cal) <= rho} R_mix(tau; q) upper-brackets the realized target
       risk R_mix(tau; p_tgt) exactly once the radius rho reaches the true tilt
       KL(p_tgt||p_cal); slack grows smoothly with rho.

  C6c  chi-square closed form R_mix + sqrt(2 rho Var) (THEOREM_RECONCILED D5). The
       sqrt(2 rho Var) constant is the Duchi-Namkoong / Cauchois first-order dual;
       reported against the exact chi-square program.
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

from foldgate.bench import certificates as C
from foldgate.bench import synth as S
from foldgate.bench.synth import SynthParams

K = 6
N_CAL = 3000


# --------------------------------------------------------------------------- #
# C6a -- worst-stratum RCPS UCB: finite-sample validity + slack
# --------------------------------------------------------------------------- #
def _tau_at_coverage(k: int, c: float, params: SynthParams) -> float:
    from scipy.special import expit
    from scipy.stats import norm
    return float(expit(norm.isf(c, loc=params.mu()[k], scale=params.sigma_s)))


def rcps_validity(params: SynthParams, alpha: float, delta: float,
                  n_cal: int = N_CAL, n_seeds: int = N_SEEDS, method: str = "hb",
                  c_pin: float = 0.5) -> dict:
    # Certify the worst-stratum risk at a per-stratum coverage-pinned threshold, so
    # every stratum carries a healthy accepted count (~c * n_k) and the union-bound
    # UCB is non-vacuous and tightens with n_cal. Worst true risk is known exactly.
    taus = [_tau_at_coverage(k, c_pin, params) for k in range(params.K)]
    r_true = np.nan_to_num(
        np.array([S.oracle_selective_risk(taus[k], k, params) for k in range(params.K)]), nan=0.0)
    worst_true = float(r_true.max())

    hits, slacks, Us = 0, [], []
    for seed in range(n_seeds):
        g = rng(80_000 + seed)
        df = S.sample(n_cal, params, "cal", g)
        per = []
        for k in range(params.K):
            acc = df[(df.k == k) & (df.s >= taus[k])]
            per.append((int((1 - acc.y).sum()), len(acc)))
        U = C.worst_stratum_rcps_ucb(per, delta, method, True)["U"]
        Us.append(U)
        hits += int(worst_true <= U + 1e-12)
        slacks.append(U - worst_true)
    validity = hits / n_seeds
    return {
        "method": method, "n_cal": n_cal, "worst_true_risk": worst_true,
        "validity_rate": float(validity),
        "validity_ge_1_minus_delta": bool(validity >= 1 - delta),
        "mean_U": float(np.mean(Us)), "mean_slack": float(np.mean(slacks)),
        "target_validity": 1 - delta,
    }


def rcps_slack_vs_n(params: SynthParams, alpha: float, delta: float) -> dict:
    out = {}
    for n_cal in [500, 1000, 2000, 5000, 10000]:
        r = rcps_validity(params, alpha, delta, n_cal=n_cal, n_seeds=120)
        out[n_cal] = {"validity_rate": r["validity_rate"], "mean_slack": r["mean_slack"]}
    return out


# --------------------------------------------------------------------------- #
# C6b/C6c -- DRO ball certificate vs rho (KL exact, chi2 exact, chi2 closed form)
# --------------------------------------------------------------------------- #
def dro_vs_rho(params: SynthParams, alpha: float) -> dict:
    tau = S.oracle_marginal_threshold(params.p_cal(), params, alpha)
    acc = S.oracle_accept_rates(tau, params)
    rk = S.oracle_selective_risks(tau, params)
    p_cal = params.p_cal()
    kl = S.kl_divergence(params.p_tgt(), p_cal)
    chi2 = S.chi2_divergence(params.p_tgt(), p_cal)
    r_tgt = float(S.oracle_R_mix(tau, params.p_tgt(), params))

    rhos = np.round(np.linspace(0.0, 2.0 * kl, 21), 5)
    kl_cert, chi2_cert, chi2_cf = [], [], []
    for rho in rhos:
        kl_cert.append(C.dro_ball_certificate(rk, acc, p_cal, float(rho), "kl")["certified_worst_risk"])
        chi2_cert.append(C.dro_ball_certificate(rk, acc, p_cal, float(rho), "chi2")["certified_worst_risk"])
        chi2_cf.append(C.chi2_closed_form_certificate(rk, acc, p_cal, float(rho)))

    cert_at_kl = C.dro_ball_certificate(rk, acc, p_cal, kl, "kl")["certified_worst_risk"]
    # find smallest rho on the grid where KL certificate covers the target risk
    kl_arr = np.array(kl_cert)
    covers = np.where(kl_arr >= r_tgt - 1e-6)[0]
    rho_cover = float(rhos[covers.min()]) if covers.size else float("nan")
    mono_kl = bool(np.all(np.diff(kl_arr) >= -1e-6))
    return {
        "tau": float(tau), "true_KL_tilt": float(kl), "true_chi2_tilt": float(chi2),
        "R_mix_target": r_tgt, "R_mix_cal": float(S.oracle_R_mix(tau, p_cal, params)),
        "rhos": [float(x) for x in rhos],
        "kl_certificate": [float(x) for x in kl_cert],
        "chi2_certificate": [float(x) for x in chi2_cert],
        "chi2_closed_form": [float(x) for x in chi2_cf],
        "kl_cert_at_true_tilt": float(cert_at_kl),
        "covers_target_at_true_tilt": bool(cert_at_kl >= r_tgt - 1e-4),
        "rho_first_covers_target": rho_cover,
        "kl_monotone_in_rho": mono_kl,
        "chi2_closed_form_uses_sqrt_2_rho_var": True,   # per certificates.chi2_closed_form_certificate
    }


def refinement_slack(alpha: float) -> dict:
    """C6 'tightens as strata refine': finer strata -> smaller DRO slack at fixed tilt."""
    out = {}
    for Kref in [3, 6, 12, 24]:
        p = SynthParams(K=Kref, D=1.0, E=0.5, c_cal=3.0, c_tgt=0.0)
        tau = S.oracle_marginal_threshold(p.p_cal(), p, alpha)
        acc = S.oracle_accept_rates(tau, p)
        rk = S.oracle_selective_risks(tau, p)
        kl = S.kl_divergence(p.p_tgt(), p.p_cal())
        cert = C.dro_ball_certificate(rk, acc, p.p_cal(), kl, "kl")["certified_worst_risk"]
        r_tgt = float(S.oracle_R_mix(tau, p.p_tgt(), p))
        out[Kref] = {"kl_tilt": float(kl), "certificate": float(cert),
                     "target_risk": r_tgt, "slack": float(cert - r_tgt)}
    return out


def make_figure(dro: dict, rcps_n: dict, path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    ax = axes[0]
    rhos = dro["rhos"]
    ax.plot(rhos, dro["kl_certificate"], "b-o", ms=3, label="KL ball (exact)")
    ax.plot(rhos, dro["chi2_certificate"], "m-s", ms=3, label="chi2 ball (exact)")
    ax.plot(rhos, dro["chi2_closed_form"], "c--", label="chi2 sqrt(2 rho Var)")
    ax.axhline(dro["R_mix_target"], color="g", ls="-", label="realized target R_mix")
    ax.axvline(dro["true_KL_tilt"], color="r", ls="--", label=f"true tilt KL={dro['true_KL_tilt']:.2f}")
    ax.set_xlabel("ambiguity radius rho")
    ax.set_ylabel("certified worst risk")
    ax.set_title("C6: DRO certificate covers target at true tilt")
    ax.legend(fontsize=7)

    ax2 = axes[1]
    ns = list(rcps_n.keys())
    ax2.plot(ns, [rcps_n[n]["mean_slack"] for n in ns], "k-o", label="RCPS UCB slack")
    ax2.set_xscale("log")
    ax2.set_xlabel("n_cal")
    ax2.set_ylabel("U - max_k R_k")
    ax2b = ax2.twinx()
    ax2b.plot(ns, [rcps_n[n]["validity_rate"] for n in ns], "g--^", label="validity rate")
    ax2b.axhline(1 - DELTA, color="0.5", ls=":")
    ax2b.set_ylabel("validity rate")
    ax2b.set_ylim(0.8, 1.01)
    ax2.set_title("C6a: RCPS validity + slack vs n_cal")
    ax2.legend(fontsize=7, loc="upper right")
    ax2b.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run() -> dict:
    alpha, delta = ALPHA, DELTA
    params = SynthParams(K=K, D=1.0, E=0.5, c_cal=3.0, c_tgt=0.0)
    rcps_hb = rcps_validity(params, alpha, delta, method="hb")
    rcps_cp = rcps_validity(params, alpha, delta, method="cp")
    rcps_n = rcps_slack_vs_n(params, alpha, delta)
    dro = dro_vs_rho(params, alpha)
    refine = refinement_slack(alpha)
    return {
        "config": {"alpha": alpha, "delta": delta, "K": K, "n_cal": N_CAL,
                   "n_seeds": N_SEEDS, "params": {"D": 1.0, "E": 0.5, "c_cal": 3.0, "c_tgt": 0.0}},
        "C6a_rcps_validity_hb": rcps_hb,
        "C6a_rcps_validity_cp": rcps_cp,
        "C6a_rcps_slack_vs_n": rcps_n,
        "C6b_dro_vs_rho": dro,
        "C6_refinement_slack": refine,
    }


def main() -> None:
    res = run()
    save_json(res, RESBENCH / "b4_synth_certificate.json")
    make_figure(res["C6b_dro_vs_rho"], res["C6a_rcps_slack_vs_n"],
                FIGBENCH / "b4_synth_certificate.png")

    print("b4 -- DRO / RCPS certificates (C6), alpha =", ALPHA, "delta =", DELTA)
    for tag in ["C6a_rcps_validity_hb", "C6a_rcps_validity_cp"]:
        r = res[tag]
        print(f"\n[{tag}] worst_true={r['worst_true_risk']:.4f}  "
              f"validity={r['validity_rate']:.3f} (>= {r['target_validity']}: {r['validity_ge_1_minus_delta']})  "
              f"mean_U={r['mean_U']:.4f}  mean_slack={r['mean_slack']:+.4f}")

    print("\n[C6a] RCPS validity + slack vs n_cal:")
    for n, v in res["C6a_rcps_slack_vs_n"].items():
        print(f"  n_cal={n:<6} validity={v['validity_rate']:.3f}  mean_slack={v['mean_slack']:+.4f}")

    d = res["C6b_dro_vs_rho"]
    print(f"\n[C6b] DRO KL-ball: true tilt KL={d['true_KL_tilt']:.4f}, target R_mix={d['R_mix_target']:.4f}")
    print(f"  certificate at true tilt = {d['kl_cert_at_true_tilt']:.4f} "
          f"(covers target: {d['covers_target_at_true_tilt']}); "
          f"first covers at rho={d['rho_first_covers_target']:.4f}; monotone_in_rho={d['kl_monotone_in_rho']}")
    print("  rho, KL_cert, chi2_cert, chi2_closed_form (sqrt(2 rho Var)):")
    for i in range(0, len(d["rhos"]), 4):
        print(f"    rho={d['rhos'][i]:.3f}  KL={d['kl_certificate'][i]:.4f}  "
              f"chi2={d['chi2_certificate'][i]:.4f}  cf={d['chi2_closed_form'][i]:.4f}")

    print("\n[C6] DRO slack tightens as strata refine (fixed tilt):")
    for Kref, v in res["C6_refinement_slack"].items():
        print(f"  K={Kref:<3} KL={v['kl_tilt']:.3f} cert={v['certificate']:.4f} "
              f"target={v['target_risk']:.4f} slack={v['slack']:+.4f}")

    pass_c6 = (res["C6a_rcps_validity_hb"]["validity_ge_1_minus_delta"]
               and res["C6a_rcps_validity_cp"]["validity_ge_1_minus_delta"]
               and d["covers_target_at_true_tilt"] and d["kl_monotone_in_rho"])
    print(f"\nPASS  C6(RCPS validity>=1-delta & DRO covers target at true tilt & monotone): {pass_c6}")


if __name__ == "__main__":
    main()
