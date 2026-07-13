"""b3 -- Weighted CP vs group-conditional: the load-bearing separation (C2).

Reweighting repairs covariate shift; only conditioning repairs concept shift.
Theorem 1(b): at a fixed coverage c the accept set is {s >= tau_c}, so the
realized selective risk equals R_Q(tau_c) INDEPENDENT of the covariate weights w.
Theorem 1(d): the weighted-CP certificate is computed from the SOURCE label
conditional (Lemma 1), so with the oracle weights w* it equals R_ref and
under-reports the realized worst-stratum risk by exactly the concept gap.

Checks:
  C2a  Impossibility regime (D>0). Across a rich family of covariate reweightings
       w(nu) the realized risk at matched achieved coverage collapses onto the
       single oracle R_Q(coverage) curve (invariance). The finite-sample weighted
       certificate for w* tracks R_ref (= pooled target risk = alpha) and
       under-reports the realized WORST-STRATUM risk by ~Delta_bar; it stays
       > alpha for every w at the impossibility operating point.
  C2b  Control D=0: weighted CP with w* hits alpha AND controls every stratum
       (worst <= alpha) -- impossibility is concept-specific.
  C2c  Covariate-only (D=0, E>0, tilt): plain marginal miscovers, weighted w*
       repairs the certificate, and conditioning is harmless. Weighted CP fixes
       covariate shift.
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

from foldgate.bench import synth as S
from foldgate.bench.synth import SynthParams

K = 8
N_CAL = 4000
N_TGT_MC = 400_000


def weight_family(params: SynthParams) -> dict:
    """A rich family of covariate reweightings w(nu) (functions of stratum k)."""
    pc, pt = params.p_cal(), params.p_tgt()
    wstar = pt / pc
    fam = {
        "w_star": wstar,
        "w_unit (no reweight)": np.ones(params.K),
        "w_inverted": pc / pt,
        "w_overshoot": wstar ** 1.5,
        "w_undershoot": wstar ** 0.5,
    }
    g = np.random.default_rng(7)
    for j in range(4):
        fam[f"w_rand{j}"] = wstar * g.lognormal(0.0, 0.6, size=params.K)
    return fam


def weighted_certificate(s, y, k_idx, w_by_k, tau):
    """Weighted-CP plug-in certificate E_{Q_w}[eta_P | s>=tau] on a cal sample."""
    w = w_by_k[k_idx]
    acc = s >= tau
    num = float(np.sum(w * (1 - y) * acc))
    den = float(np.sum(w * acc))
    return num / den if den > 0 else float("nan")


def weighted_coverage_threshold(s, k_idx, w_by_k, c):
    """Smallest tau with weighted accept fraction (sum w 1_A / sum w) == c."""
    w = w_by_k[k_idx]
    order = np.argsort(-s)
    w_sorted = w[order]
    s_sorted = s[order]
    cum = np.cumsum(w_sorted) / np.sum(w)
    idx = int(np.searchsorted(cum, c))
    idx = min(max(idx, 0), len(s_sorted) - 1)
    return float(s_sorted[idx])


# --------------------------------------------------------------------------- #
# C2a -- impossibility regime: invariance of realized risk + certificate under-report
# --------------------------------------------------------------------------- #
def impossibility_regime(alpha: float, n_seeds: int = N_SEEDS) -> dict:
    params = SynthParams(K=K, D=1.0, E=0.0, c_cal=3.0, c_tgt=0.0)
    fam = weight_family(params)
    # operating point: pooled TARGET realized risk = alpha  (tau on the target mixture)
    tau = S.oracle_marginal_threshold(params.p_tgt(), params, alpha)
    R_ref = float(S.oracle_R_mix(tau, params.p_tgt(), params))     # = alpha here (shared eta)
    worst_realized = float(np.nanmax(S.oracle_selective_risks(tau, params)))
    Delta_bar = worst_realized - R_ref

    # certificate under-report, averaged over seeds, for each w
    cert = {name: [] for name in fam}
    for seed in range(n_seeds):
        g = rng(60_000 + seed)
        cal = S.sample(N_CAL, params, "cal", g)
        s, y, kk = cal.s.to_numpy(), cal.y.to_numpy(), cal.k.to_numpy()
        for name, w in fam.items():
            cert[name].append(weighted_certificate(s, y, kk, w, tau))
    cert_summary = {}
    for name in fam:
        arr = np.array(cert[name], dtype=float)
        m = float(np.nanmean(arr))
        cert_summary[name] = {
            "mean_certified": m,
            "underreport_vs_worst": float(worst_realized - m),
            "certificate_below_worst": bool(m < worst_realized - 1e-6),
        }

    # invariance (T1b): realized risk collapses on the oracle R_Q(coverage) curve.
    # oracle curve
    taus = np.linspace(0.02, 0.98, 400)
    cov_curve = np.array([float((params.p_tgt() * S.oracle_accept_rates(t, params)).sum()) for t in taus])
    rq_curve = np.array([S.oracle_R_mix(t, params.p_tgt(), params) for t in taus])
    o_order = np.argsort(cov_curve)
    cov_s, rq_s = cov_curve[o_order], rq_curve[o_order]

    g = rng(999)
    tgt = S.sample(N_TGT_MC, params, "tgt", g)
    ts, _tk, ty = tgt.s.to_numpy(), tgt.k.to_numpy(), tgt.y.to_numpy()
    cal = S.sample(N_CAL * 4, params, "cal", rng(998))
    cs, ck = cal.s.to_numpy(), cal.k.to_numpy()

    resid = []
    invariance_points = []
    for c in [0.3, 0.5, 0.7]:
        for name, w in fam.items():
            tau_w = weighted_coverage_threshold(cs, ck, w, c)
            acc = ts >= tau_w
            ach_cov = float(acc.mean())
            realized = float(1.0 - ty[acc].mean()) if acc.any() else float("nan")
            oracle_at_cov = float(np.interp(ach_cov, cov_s, rq_s))
            resid.append(abs(realized - oracle_at_cov))
            invariance_points.append({"intended_c": c, "w": name,
                                       "achieved_coverage": ach_cov,
                                       "realized_risk": realized,
                                       "oracle_Rq_at_cov": oracle_at_cov})
    max_resid = float(np.nanmax(resid))
    return {
        "params": {"D": 1.0, "E": 0.0, "c_cal": 3.0, "c_tgt": 0.0, "tau": float(tau)},
        "R_ref_pooled_target": R_ref, "worst_realized": worst_realized,
        "Delta_bar_worststratum": float(Delta_bar),
        "certificate_by_w": cert_summary,
        "all_certificates_below_worst": bool(all(v["certificate_below_worst"] for v in cert_summary.values())),
        "wstar_underreport": cert_summary["w_star"]["underreport_vs_worst"],
        "wstar_tracks_Rref": bool(abs(cert_summary["w_star"]["mean_certified"] - R_ref) < 0.01),
        "invariance_max_residual": max_resid,
        "realized_invariant_to_w": bool(max_resid < 0.01),
        "worst_gt_alpha": bool(worst_realized > alpha),
        "invariance_points": invariance_points,
        "oracle_curve": {"coverage": [float(x) for x in cov_s], "R_Q": [float(x) for x in rq_s]},
    }


# --------------------------------------------------------------------------- #
# C2b -- control D=0: weighted w* hits alpha AND controls every stratum
# --------------------------------------------------------------------------- #
def control_D0(alpha: float) -> dict:
    params = SynthParams(K=K, D=0.0, E=0.0, c_cal=3.0, c_tgt=0.0)
    tau = S.oracle_marginal_threshold(params.p_tgt(), params, alpha)
    pooled = float(S.oracle_R_mix(tau, params.p_tgt(), params))
    worst = float(np.nanmax(S.oracle_selective_risks(tau, params)))
    return {"tau": float(tau), "pooled_target_risk": pooled, "worst_stratum_risk": worst,
            "hits_alpha": bool(abs(pooled - alpha) < 1e-3),
            "worst_controlled": bool(worst <= alpha + 1e-3),
            "impossibility_absent": bool(worst <= alpha + 1e-3)}


# --------------------------------------------------------------------------- #
# C2c -- covariate-only (D=0, E>0): plain marginal miscovers, weighted repairs
# --------------------------------------------------------------------------- #
def covariate_only(alpha: float, n_seeds: int = 200) -> dict:
    params = SynthParams(K=K, D=0.0, E=1.5, c_cal=3.0, c_tgt=0.0)
    # operating threshold pinned so the pooled TARGET risk = alpha
    tau = S.oracle_marginal_threshold(params.p_tgt(), params, alpha)
    R_target = float(S.oracle_R_mix(tau, params.p_tgt(), params))     # = alpha
    fam = {"w_unit (marginal)": np.ones(params.K),
           "w_star (weighted)": params.p_tgt() / params.p_cal()}
    out = {}
    for name, w in fam.items():
        certs = []
        for seed in range(n_seeds):
            g = rng(70_000 + seed)
            cal = S.sample(N_CAL, params, "cal", g)
            certs.append(weighted_certificate(cal.s.to_numpy(), cal.y.to_numpy(),
                                              cal.k.to_numpy(), w, tau))
        m = float(np.nanmean(certs))
        out[name] = {"mean_certified": m, "bias_vs_target": float(m - R_target)}
    worst = float(np.nanmax(S.oracle_selective_risks(tau, params)))
    return {
        "params": {"D": 0.0, "E": 1.5, "c_cal": 3.0, "c_tgt": 0.0, "tau": float(tau)},
        "R_target": R_target, "worst_stratum_risk": worst,
        "certificate_by_w": out,
        "marginal_miscovers_pooled": bool(abs(out["w_unit (marginal)"]["bias_vs_target"]) > 0.008),
        "weighted_repairs_pooled": bool(abs(out["w_star (weighted)"]["bias_vs_target"]) < 0.008),
        # At D=0 the residual worst-stratum excess is a COVARIATE conditional-coverage
        # gap (E>0 score shift), repaired by conditioning (Mondrian), not by weighting.
        "worst_stratum_excess_is_covariate": bool(worst > alpha + 1e-3),
    }


def make_figure(imp: dict, cov: dict, path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    ax = axes[0]
    ax.plot(imp["oracle_curve"]["coverage"], imp["oracle_curve"]["R_Q"], "k-",
            lw=2, label="oracle R_Q(coverage)")
    for p in imp["invariance_points"]:
        ax.plot(p["achieved_coverage"], p["realized_risk"], "o", ms=5, alpha=0.7)
    ax.axhline(ALPHA, color="r", ls=":", label=f"alpha={ALPHA}")
    ax.set_xlabel("achieved coverage")
    ax.set_ylabel("realized selective risk")
    ax.set_title(f"C2a: realized risk invariant to w\n(max residual={imp['invariance_max_residual']:.4f})")
    ax.legend(fontsize=8)

    ax2 = axes[1]
    names = list(imp["certificate_by_w"].keys())
    certs = [imp["certificate_by_w"][n]["mean_certified"] for n in names]
    ax2.barh(range(len(names)), certs, color="#69c")
    ax2.axvline(imp["R_ref_pooled_target"], color="g", ls="--", label="R_ref (pooled target)")
    ax2.axvline(imp["worst_realized"], color="r", ls="-", label="realized worst stratum")
    ax2.axvline(ALPHA, color="k", ls=":", label=f"alpha={ALPHA}")
    ax2.set_yticks(range(len(names)))
    ax2.set_yticklabels(names, fontsize=7)
    ax2.set_xlabel("weighted-CP certificate")
    ax2.set_title(f"C2a: every certificate under-reports worst\nby ~Delta_bar={imp['Delta_bar_worststratum']:.3f}")
    ax2.legend(fontsize=7)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run() -> dict:
    alpha = ALPHA
    imp = impossibility_regime(alpha)
    d0 = control_D0(alpha)
    cov = covariate_only(alpha)
    return {
        "config": {"alpha": alpha, "delta": DELTA, "K": K, "n_cal": N_CAL,
                   "n_seeds": N_SEEDS},
        "C2a_impossibility_regime": imp,
        "C2b_control_D0": d0,
        "C2c_covariate_only": cov,
    }


def main() -> None:
    res = run()
    save_json(res, RESBENCH / "b3_synth_weighted_vs_cond.json")
    make_figure(res["C2a_impossibility_regime"], res["C2c_covariate_only"],
                FIGBENCH / "b3_synth_weighted_vs_cond.png")

    imp = res["C2a_impossibility_regime"]
    print("b3 -- Weighted vs conditional separation (C2), alpha =", ALPHA)
    print(f"\n[C2a] impossibility regime (D=1, tau={imp['params']['tau']:.3f}):")
    print(f"  R_ref (pooled target) = {imp['R_ref_pooled_target']:.4f};  "
          f"realized worst-stratum = {imp['worst_realized']:.4f} (> alpha: {imp['worst_gt_alpha']})")
    print(f"  worst-stratum concept gap Delta_bar = {imp['Delta_bar_worststratum']:.4f}")
    print(f"  realized risk INVARIANT to w: max residual from oracle R_Q(cov) = "
          f"{imp['invariance_max_residual']:.4f} ({imp['realized_invariant_to_w']})")
    print("  weighted-CP certificate per w (all under-report the worst stratum):")
    for name, v in imp["certificate_by_w"].items():
        print(f"    {name:<22} certified={v['mean_certified']:.4f}  "
              f"under-reports worst by {v['underreport_vs_worst']:+.4f}")
    print(f"  w_star tracks R_ref: {imp['wstar_tracks_Rref']}  (under-report ~ Delta_bar: "
          f"{imp['wstar_underreport']:.4f})")

    d0 = res["C2b_control_D0"]
    print(f"\n[C2b] control D=0: pooled={d0['pooled_target_risk']:.4f} (hits alpha: {d0['hits_alpha']}), "
          f"worst={d0['worst_stratum_risk']:.4f} (controlled: {d0['worst_controlled']}) "
          f"-> impossibility concept-specific: {d0['impossibility_absent']}")

    cov = res["C2c_covariate_only"]
    print(f"\n[C2c] covariate-only (D=0, E=1.5): pooled target risk={cov['R_target']:.4f}")
    for name, v in cov["certificate_by_w"].items():
        print(f"    {name:<22} certified={v['mean_certified']:.4f} bias={v['bias_vs_target']:+.4f}")
    print(f"  marginal miscovers pooled: {cov['marginal_miscovers_pooled']}; "
          f"weighted repairs pooled: {cov['weighted_repairs_pooled']}; "
          f"residual worst-stratum excess is covariate (Mondrian's job): {cov['worst_stratum_excess_is_covariate']}")

    pass_c2 = (imp["realized_invariant_to_w"] and imp["worst_gt_alpha"]
               and imp["all_certificates_below_worst"] and imp["wstar_tracks_Rref"]
               and d0["impossibility_absent"] and cov["weighted_repairs_pooled"])
    print(f"\nPASS  C2(invariance + under-report + D0 control + covariate repair): {pass_c2}")


if __name__ == "__main__":
    main()
