"""b7 -- Genuine P-vs-Q concept drift: the sharp impossibility (Theorem 1c/1d),
the achievability precondition (Theorem 3 needs target-stratum labels), and the
vacuity boundary (no-analog tail).

b1-b3 use the shared-label control (Dq = 0), where the pooled accept-region concept
gap Delta_bar_A is zero and the impossibility shows up only as worst-stratum
under-coverage. This experiment turns on genuine drift between the calibration and
target label laws (Dq > 0), so Delta_bar_A > 0 and the theorem's headline claims are
checkable against a KNOWN conditional:

  A (T1a)  decomposition R_Q(tau_c) = R_ref(tau_c) + Delta_bar_A with Delta_bar_A > 0,
           and a finite-sample target draw realizes R_Q, not R_ref.
  B (T1b/d) across a rich family of covariate reweightings the realized TARGET risk at
           matched coverage is invariant and equals R_Q; the weighted-CP certificate is
           computed from the SOURCE labels (Lemma 1), so it reports R_ref and
           under-reports the realized risk by exactly Delta_bar_A (silent violation).
  C (T1c)  sweep the drift so Delta_bar_A crosses alpha - R_ref; above the crossing NO
           covariate reweighting reaches alpha at the target coverage.
  D (T3)   in-stratum recalibration on TARGET-stratum labels controls the novel stratum,
           while the same recalibration on SOURCE-stratum labels does not: achievability
           needs n_g exchangeable target labels, it is not free of deployment labels.
  E (vacuity) with an irreducible error floor eps > alpha on the novel stratum, no
           threshold certifies alpha at positive coverage; LTT abstains. This is the
           RNP no-analog S4 tail where every method is honestly reduced to abstention.
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
    RESBENCH,
    rng,
    save_json,
)

from foldgate.bench import synth as S
from foldgate.bench.synth import SynthParams
from foldgate.conformal.risk import ltt_threshold

K = 8
N_CAL = 4000
N_TGT_MC = 400_000
N_STRATUM_CAL = 1500     # per-stratum calibration count for the achievability checks
N_STRATUM_TEST = 4000    # per-stratum fresh test count
DRIFT = 2.0              # default target-only concept drift for A/B/D


def weight_family(params: SynthParams) -> dict:
    """Covariate reweightings w(nu) (functions of stratum k); w_star is the oracle."""
    pc, pt = params.p_cal(), params.p_tgt()
    wstar = pt / pc
    fam = {
        "w_star": wstar,
        "w_unit": np.ones(params.K),
        "w_inverted": pc / pt,
        "w_overshoot": wstar ** 1.5,
        "w_undershoot": wstar ** 0.5,
    }
    g = np.random.default_rng(7)
    for j in range(4):
        fam[f"w_rand{j}"] = wstar * g.lognormal(0.0, 0.6, size=params.K)
    return fam


def weighted_coverage_threshold(s, k_idx, w_by_k, c) -> float:
    """Smallest tau with weighted accept fraction (sum w 1_A / sum w) == c."""
    w = w_by_k[k_idx]
    order = np.argsort(-s)
    cum = np.cumsum(w[order]) / np.sum(w)
    idx = min(max(int(np.searchsorted(cum, c)), 0), len(s) - 1)
    return float(s[order][idx])


def weighted_source_certificate(s, y, k_idx, w_by_k, tau) -> float:
    """Weighted-CP plug-in certificate E_{Q_w}[error | s>=tau] on a SOURCE sample.

    Uses source labels only (Lemma 1): whatever the weights, this can only recover
    the source error-conditional, never the drifted target one.
    """
    w = w_by_k[k_idx]
    acc = s >= tau
    den = float(np.sum(w * acc))
    return float(np.sum(w * (1 - y) * acc)) / den if den > 0 else float("nan")


def sample_stratum(n, k, params, population, g):
    """Draw n examples from a single stratum k under the requested label law."""
    nu_k = params.nu()[k]
    mu_k = params.mu()[k]
    z = mu_k + params.sigma_s * g.standard_normal(n)
    s = S.expit(z)
    pi = S.pi_correct(z, nu_k, params, population=population)
    y = (g.random(n) < pi).astype(int)
    return s, y


# --------------------------------------------------------------------------- #
# A -- decomposition with Delta_bar_A > 0, matched by a finite-sample target draw
# --------------------------------------------------------------------------- #
def check_decomposition(alpha: float, drift: float = DRIFT) -> dict:
    params = SynthParams(K=K, D=1.0, Dq=drift, c_cal=3.0, c_tgt=0.0)
    c = 0.5
    g = S.oracle_concept_gap(c, params)
    tau_c = g["tau_c"]

    tgt = S.sample(N_TGT_MC, params, "tgt", rng(1001))
    ts, ty = tgt.s.to_numpy(), tgt.y.to_numpy()
    acc = ts >= tau_c
    realized = float(1.0 - ty[acc].mean())
    se = float(np.sqrt(realized * (1 - realized) / max(acc.sum(), 1)))

    return {
        "drift_Dq": drift, "coverage": c, "tau_c": tau_c,
        "R_ref": g["R_ref"], "R_Q": g["R_Q"], "delta_bar_A": g["delta_bar_A"],
        "realized_target": realized, "mc_se": se,
        "delta_bar_positive": bool(g["delta_bar_A"] > 0.02),
        "realized_matches_RQ": bool(abs(realized - g["R_Q"]) < 4 * se + 1e-3),
        "realized_exceeds_Rref": bool(realized - g["R_ref"] > 0.02),
    }


# --------------------------------------------------------------------------- #
# B -- invariance across reweightings + weighted-CP silent violation (T1b/d)
# --------------------------------------------------------------------------- #
def check_invariance_and_violation(alpha: float, drift: float = DRIFT,
                                   n_seeds: int = 200) -> dict:
    params = SynthParams(K=K, D=1.0, Dq=drift, c_cal=3.0, c_tgt=0.0)
    c = 0.5
    fam = weight_family(params)
    gap = S.oracle_concept_gap(c, params)
    R_ref, R_Q, dbar = gap["R_ref"], gap["R_Q"], gap["delta_bar_A"]

    # oracle R_Q(coverage) curve on the target law for the invariance reference
    taus = np.linspace(0.02, 0.98, 400)
    cov_curve = np.array([float((params.p_tgt() * S.oracle_accept_rates(t, params)).sum()) for t in taus])
    rq_curve = np.array([S.oracle_R_mix(t, params.p_tgt(), params, "tgt") for t in taus])
    o = np.argsort(cov_curve)
    cov_s, rq_s = cov_curve[o], rq_curve[o]

    tgt = S.sample(N_TGT_MC, params, "tgt", rng(2002))
    ts = tgt.s.to_numpy()
    ty = tgt.y.to_numpy()
    cal_big = S.sample(N_CAL * 4, params, "cal", rng(2003))
    cs, ck = cal_big.s.to_numpy(), cal_big.k.to_numpy()

    resid, points = [], []
    for name, w in fam.items():
        tau_w = weighted_coverage_threshold(cs, ck, w, c)
        acc = ts >= tau_w
        realized = float(1.0 - ty[acc].mean()) if acc.any() else float("nan")
        ach = float(acc.mean())
        resid.append(abs(realized - float(np.interp(ach, cov_s, rq_s))))
        points.append({"w": name, "achieved_coverage": ach, "realized_target_risk": realized})
    max_resid = float(np.nanmax(resid))

    # weighted-CP certificate from SOURCE labels, per w, averaged over seeds
    cert = {name: [] for name in fam}
    for seed in range(n_seeds):
        cal = S.sample(N_CAL, params, "cal", rng(20_000 + seed))
        s, y, kk = cal.s.to_numpy(), cal.y.to_numpy(), cal.k.to_numpy()
        tau = weighted_coverage_threshold(cs, ck, fam["w_star"], c)  # fixed accept region
        for name, w in fam.items():
            cert[name].append(weighted_source_certificate(s, y, kk, w, tau))
    cert_summary = {}
    for name in fam:
        m = float(np.nanmean(cert[name]))
        cert_summary[name] = {
            "mean_certified": m,
            "underreports_realized_by": float(R_Q - m),
            "below_realized": bool(m < R_Q - 1e-3),
        }

    return {
        "drift_Dq": drift, "coverage": c,
        "R_ref": R_ref, "R_Q": R_Q, "delta_bar_A": dbar,
        "invariance_max_residual": max_resid,
        "realized_invariant_to_w": bool(max_resid < 0.015),
        "wstar_certificate": cert_summary["w_star"]["mean_certified"],
        "wstar_tracks_Rref": bool(abs(cert_summary["w_star"]["mean_certified"] - R_ref) < 0.015),
        "wstar_underreports_by": cert_summary["w_star"]["underreports_realized_by"],
        "underreport_matches_delta_bar": bool(
            abs(cert_summary["w_star"]["underreports_realized_by"] - dbar) < 0.02),
        "all_certificates_below_realized": bool(all(v["below_realized"] for v in cert_summary.values())),
        "certificate_by_w": cert_summary,
        "invariance_points": points,
        "oracle_curve": {"coverage": [float(x) for x in cov_s], "R_Q": [float(x) for x in rq_s]},
    }


# --------------------------------------------------------------------------- #
# C -- unachievability crossing in the drift (T1c)
# --------------------------------------------------------------------------- #
def check_crossing(alpha: float) -> dict:
    # Pick a coverage where the covariate-corrected reference R_ref sits BELOW alpha
    # at zero drift, so that increasing the concept drift pushes the realized floor
    # R_Q up through alpha (a genuine concept crossing, not a pre-existing covariate
    # gap). R_Q is decreasing in coverage, so a more selective c lowers R_ref.
    base = SynthParams(K=K, D=1.0, Dq=0.0, c_cal=3.0, c_tgt=0.0)
    c = 0.5
    for cand in [0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.07, 0.05]:
        if S.oracle_concept_gap(cand, base)["R_ref"] < 0.6 * alpha:
            c = cand
            break
    grid = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
    rows = []
    for dq in grid:
        params = SynthParams(K=K, D=1.0, Dq=dq, c_cal=3.0, c_tgt=0.0)
        gap = S.oracle_concept_gap(c, params)
        # min realized target risk over the reweighting family at coverage c
        fam = weight_family(params)
        tgt = S.sample(200_000, params, "tgt", rng(3000 + int(dq * 10)))
        cal = S.sample(N_CAL * 2, params, "cal", rng(3100 + int(dq * 10)))
        cs, ck = cal.s.to_numpy(), cal.k.to_numpy()
        ts, ty = tgt.s.to_numpy(), tgt.y.to_numpy()
        realized = []
        for w in fam.values():
            tau_w = weighted_coverage_threshold(cs, ck, w, c)
            acc = ts >= tau_w
            realized.append(float(1.0 - ty[acc].mean()) if acc.any() else np.nan)
        rows.append({
            "Dq": dq, "R_ref": gap["R_ref"], "R_Q": gap["R_Q"],
            "delta_bar_A": gap["delta_bar_A"],
            "slack_alpha_minus_Rref": float(alpha - gap["R_ref"]),
            "reachable_by_reweighting": bool(gap["R_Q"] <= alpha + 1e-3),
            "min_realized_over_family": float(np.nanmin(realized)),
            "min_realized_gt_alpha": bool(np.nanmin(realized) > alpha + 1e-3),
        })
    # locate crossing Dq where R_Q(c) == alpha by linear interpolation
    dqs = np.array([r["Dq"] for r in rows])
    rqs = np.array([r["R_Q"] for r in rows])
    cross = None
    for i in range(len(dqs) - 1):
        if (rqs[i] - alpha) * (rqs[i + 1] - alpha) <= 0 and rqs[i] != rqs[i + 1]:
            cross = float(dqs[i] + (alpha - rqs[i]) * (dqs[i + 1] - dqs[i]) / (rqs[i + 1] - rqs[i]))
            break
    return {"coverage": c, "alpha": alpha, "crossing_Dq": cross, "sweep": rows,
            "monotone_RQ_in_Dq": bool(np.all(np.diff(rqs) >= -1e-6))}


# --------------------------------------------------------------------------- #
# D -- achievability needs TARGET-stratum labels (T3 / assumption A8)
# --------------------------------------------------------------------------- #
def check_achievability_needs_target_labels(alpha: float, drift: float = 1.5,
                                            n_seeds: int = 300) -> dict:
    # A MODERATE novel stratum where a controlling threshold exists on the target law
    # at accept mass >= min_accept, yet a source-calibrated threshold (source error is
    # lower at fixed score) deploys too low and under-covers the target. This isolates
    # the achievability precondition: control needs target-stratum labels, not just
    # covariate reweighting or source-stratum labels.
    params = SynthParams(K=K, D=1.0, Dq=drift, c_cal=3.0, c_tgt=0.0)
    k = K // 2  # nu ~ 0.57, moderate novelty (headroom to control on the target law)
    n_cal = 3000
    n_test = 8000
    src_exceed, tgt_exceed = 0, 0
    src_cov, tgt_cov = [], []
    n_used = 0
    for seed in range(n_seeds):
        g = rng(40_000 + seed)
        s_src, y_src = sample_stratum(n_cal, k, params, "cal", g)   # source labels
        s_tgt, y_tgt = sample_stratum(n_cal, k, params, "tgt", g)   # target labels
        s_te, y_te = sample_stratum(n_test, k, params, "tgt", g)    # fresh target test

        tau_src = ltt_threshold(s_src, y_src, alpha, DELTA)  # Mondrian on SOURCE labels
        tau_tgt = ltt_threshold(s_tgt, y_tgt, alpha, DELTA)  # Mondrian on TARGET labels
        if tau_src is None or tau_tgt is None:
            continue
        n_used += 1
        a_src = s_te >= tau_src
        a_tgt = s_te >= tau_tgt
        r_src = float(1.0 - y_te[a_src].mean()) if a_src.any() else 0.0
        r_tgt = float(1.0 - y_te[a_tgt].mean()) if a_tgt.any() else 0.0
        src_exceed += int(r_src > alpha)
        tgt_exceed += int(r_tgt > alpha)
        src_cov.append(float(a_src.mean()))
        tgt_cov.append(float(a_tgt.mean()))
    return {
        "drift_Dq": drift, "stratum": k, "n_seeds_used": n_used,
        "source_label_exceedance_rate": src_exceed / max(n_used, 1),
        "target_label_exceedance_rate": tgt_exceed / max(n_used, 1),
        "source_mean_coverage": float(np.mean(src_cov)) if src_cov else 0.0,
        "target_mean_coverage": float(np.mean(tgt_cov)) if tgt_cov else 0.0,
        "source_label_calibration_fails": bool(src_exceed / max(n_used, 1) > DELTA + 0.05),
        "target_label_calibration_controls": bool(tgt_exceed / max(n_used, 1) <= DELTA + 0.03),
    }


# --------------------------------------------------------------------------- #
# E -- vacuity boundary: irreducible floor eps > alpha forces abstention
# --------------------------------------------------------------------------- #
def check_vacuity(alpha: float, eps: float = 0.30, n_seeds: int = 200) -> dict:
    params = SynthParams(K=K, D=1.0, Dq=1.0, eps_floor=eps, c_cal=3.0, c_tgt=0.0)
    k = K - 1
    tau_star = S.oracle_tau_star(k, params, alpha, population="tgt")
    min_risk = S.oracle_selective_risk(0.999, k, params, "tgt")
    abstain = 0
    for seed in range(n_seeds):
        g = rng(50_000 + seed)
        s_tgt, y_tgt = sample_stratum(N_STRATUM_CAL, k, params, "tgt", g)
        tau = ltt_threshold(s_tgt, y_tgt, alpha, DELTA)
        abstain += int(tau is None)
    return {
        "eps_floor": eps, "alpha": alpha, "stratum": k,
        "oracle_tau_star_is_nan": bool(np.isnan(tau_star)),
        "min_achievable_risk": float(min_risk),
        "floor_exceeds_alpha": bool(min_risk > alpha),
        "ltt_abstention_rate": abstain / n_seeds,
        "ltt_abstains_as_it_should": bool(abstain / n_seeds > 0.9),
    }


def make_figure(inv: dict, crossing: dict, path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    ax = axes[0]
    ax.plot(inv["oracle_curve"]["coverage"], inv["oracle_curve"]["R_Q"], "k-", lw=2,
            label="oracle R_Q(coverage)")
    for p in inv["invariance_points"]:
        ax.plot(p["achieved_coverage"], p["realized_target_risk"], "o", ms=5, alpha=0.7)
    ax.axhline(inv["R_ref"], color="g", ls="--", label=f"R_ref={inv['R_ref']:.3f}")
    ax.axhline(ALPHA, color="r", ls=":", label=f"alpha={ALPHA}")
    ax.set_xlabel("achieved coverage")
    ax.set_ylabel("realized target risk")
    ax.set_title(f"T1b/d: realized = R_Q invariant to w\ncertificate reports R_ref, "
                 f"short by Delta_bar={inv['delta_bar_A']:.3f}")
    ax.legend(fontsize=8)

    ax2 = axes[1]
    rows = crossing["sweep"]
    dq = [r["Dq"] for r in rows]
    ax2.plot(dq, [r["R_Q"] for r in rows], "b-o", label="R_Q(c) realized floor")
    ax2.plot(dq, [r["R_ref"] for r in rows], "g--s", label="R_ref (covariate-corrected)")
    ax2.plot(dq, [r["min_realized_over_family"] for r in rows], "kx", label="min over reweightings")
    ax2.axhline(crossing["alpha"], color="r", ls=":", label=f"alpha={crossing['alpha']}")
    if crossing["crossing_Dq"] is not None:
        ax2.axvline(crossing["crossing_Dq"], color="0.5", ls="-.",
                    label=f"crossing Dq={crossing['crossing_Dq']:.2f}")
    ax2.set_xlabel("concept-drift magnitude Dq")
    ax2.set_ylabel("selective risk at coverage c")
    ax2.set_title("T1c: above the crossing no reweighting reaches alpha")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run() -> dict:
    alpha = ALPHA
    a = check_decomposition(alpha)
    b = check_invariance_and_violation(alpha)
    c = check_crossing(alpha)
    d = check_achievability_needs_target_labels(alpha)
    e = check_vacuity(alpha)
    return {
        "config": {"alpha": alpha, "delta": DELTA, "K": K,
                   "n_cal": N_CAL, "drift_default": DRIFT},
        "A_decomposition": a,
        "B_invariance_and_silent_violation": b,
        "C_unachievability_crossing": c,
        "D_achievability_needs_target_labels": d,
        "E_vacuity_boundary": e,
    }


def main() -> None:
    res = run()
    save_json(res, RESBENCH / "b7_synth_concept_drift.json")
    make_figure(res["B_invariance_and_silent_violation"],
                res["C_unachievability_crossing"],
                FIGBENCH / "b7_synth_concept_drift.png")

    a = res["A_decomposition"]
    print("b7 -- genuine concept drift, alpha =", ALPHA)
    print(f"\n[A/T1a] Dq={a['drift_Dq']}, coverage={a['coverage']}: "
          f"R_ref={a['R_ref']:.4f} + Delta_bar={a['delta_bar_A']:.4f} = R_Q={a['R_Q']:.4f}; "
          f"finite-sample realized={a['realized_target']:.4f} (+-{a['mc_se']:.4f})")
    print(f"  Delta_bar>0: {a['delta_bar_positive']}; realized==R_Q: {a['realized_matches_RQ']}; "
          f"realized>R_ref: {a['realized_exceeds_Rref']}")

    b = res["B_invariance_and_silent_violation"]
    print(f"\n[B/T1b,d] realized risk invariant to w (max residual={b['invariance_max_residual']:.4f}: "
          f"{b['realized_invariant_to_w']})")
    print(f"  weighted-CP w* certificate={b['wstar_certificate']:.4f} tracks R_ref={b['R_ref']:.4f} "
          f"({b['wstar_tracks_Rref']}); under-reports realized by {b['wstar_underreports_by']:.4f} "
          f"~ Delta_bar={b['delta_bar_A']:.4f} ({b['underreport_matches_delta_bar']})")
    print(f"  every certificate below realized: {b['all_certificates_below_realized']}")

    c = res["C_unachievability_crossing"]
    print(f"\n[C/T1c] R_Q(c) crosses alpha at Dq={c['crossing_Dq']} (monotone: {c['monotone_RQ_in_Dq']}):")
    for r in c["sweep"]:
        print(f"    Dq={r['Dq']:.1f}  R_Q={r['R_Q']:.3f}  reachable_by_reweighting={r['reachable_by_reweighting']}  "
              f"min_realized={r['min_realized_over_family']:.3f} (>alpha: {r['min_realized_gt_alpha']})")

    d = res["D_achievability_needs_target_labels"]
    print(f"\n[D/T3] novel stratum {d['stratum']}, Dq={d['drift_Dq']} ({d['n_seeds_used']} seeds):")
    print(f"  Mondrian on SOURCE labels: exceedance={d['source_label_exceedance_rate']:.3f} "
          f"(fails: {d['source_label_calibration_fails']}), coverage={d['source_mean_coverage']:.3f}")
    print(f"  Mondrian on TARGET labels: exceedance={d['target_label_exceedance_rate']:.3f} "
          f"(controls: {d['target_label_calibration_controls']}), coverage={d['target_mean_coverage']:.3f}")

    e = res["E_vacuity_boundary"]
    print(f"\n[E/vacuity] eps_floor={e['eps_floor']} > alpha: min achievable risk={e['min_achievable_risk']:.3f} "
          f"({e['floor_exceeds_alpha']}); oracle tau* NaN: {e['oracle_tau_star_is_nan']}; "
          f"LTT abstention rate={e['ltt_abstention_rate']:.3f} ({e['ltt_abstains_as_it_should']})")

    passed = (a["delta_bar_positive"] and a["realized_matches_RQ"]
              and b["realized_invariant_to_w"] and b["wstar_tracks_Rref"]
              and b["underreport_matches_delta_bar"] and b["all_certificates_below_realized"]
              and c["crossing_Dq"] is not None and c["monotone_RQ_in_Dq"]
              and d["source_label_calibration_fails"] and d["target_label_calibration_controls"]
              and e["oracle_tau_star_is_nan"] and e["ltt_abstains_as_it_should"])
    print(f"\nPASS  b7 (T1a decomposition + T1b/d invariance/violation + T1c crossing + "
          f"T3 target-labels + vacuity): {passed}")


if __name__ == "__main__":
    main()
