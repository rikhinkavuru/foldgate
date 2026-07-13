"""E4 -- selective-prediction utility: combined score vs native confidence.

Headline is the risk-coverage curve and its area (AURC, lower = better), plus
certified operating points at several target error levels. The combined score
(P(correct) from native confidence + PoseBusters validity + ensemble spread +
ligand difficulty) is the reliability layer; native ranking_score is the
baseline it must beat.

Rigour: a 3-way split keeps the conformal guarantee intact -- the combiner is
fit on a TRAIN fold, the LTT threshold is calibrated on a separate CAL fold
(out-of-sample combiner scores, exchangeable with test), and everything is
scored on TEST.
"""

from __future__ import annotations

import numpy as np

from experiments._common import (
    DELTA,
    FIGDIR,
    RESDIR,
    iut_all,
    load_delivered,
    methods_with_enough,
    rng,
    save_json,
)
from foldgate.conformal import ltt_threshold
from foldgate.scores import ScoreCombiner
from foldgate.scores.combiner import DEFAULT_FEATURES, POSE_FEATURES
from foldgate.selective import aurc, evaluate_gate

ALPHAS = [0.10, 0.20]
NATIVE = "ranking_score"
POSE_SET = DEFAULT_FEATURES + POSE_FEATURES   # W1 structure-based upgrade


def three_way(idx, g):
    perm = g.permutation(idx)
    n = len(perm)
    a, b = int(0.4 * n), int(0.7 * n)
    return perm[:a], perm[a:b], perm[b:]


def run(n_repeats: int = 120) -> dict:
    df = load_delivered()
    methods = methods_with_enough(df)
    g = rng()
    out = {}

    for m in methods:
        sub = df[df.method == m].reset_index(drop=True)
        y = sub["correct"].to_numpy()
        s_nat = sub[NATIVE].to_numpy()
        idx_all = np.arange(len(sub))

        has_pose = "intra_model_pose_std" in sub.columns and sub["intra_model_pose_std"].notna().any()
        aurc_nat, aurc_comb, aurc_pose = [], [], []
        cov = {a: {"native": [], "combined": []} for a in ALPHAS}
        risk = {a: {"native": [], "combined": []} for a in ALPHAS}

        for _ in range(n_repeats):
            tr, cal, te = three_way(idx_all, g)
            comb = ScoreCombiner().fit(sub.iloc[tr], y[tr])
            sc_cal = comb.predict(sub.iloc[cal])
            sc_te = comb.predict(sub.iloc[te])

            aurc_nat.append(aurc(s_nat[te], y[te]))
            aurc_comb.append(aurc(sc_te, y[te]))
            if has_pose:   # W1 upgrade: combined score + structural pose-agreement features
                cp = ScoreCombiner(features=POSE_SET).fit(sub.iloc[tr], y[tr])
                aurc_pose.append(aurc(cp.predict(sub.iloc[te]), y[te]))

            for a in ALPHAS:
                t_nat = ltt_threshold(s_nat[cal], y[cal], alpha=a, delta=DELTA)
                t_comb = ltt_threshold(sc_cal, y[cal], alpha=a, delta=DELTA)
                rn = evaluate_gate(s_nat[te], y[te], t_nat)
                rc = evaluate_gate(sc_te, y[te], t_comb)
                cov[a]["native"].append(rn["coverage"])
                cov[a]["combined"].append(rc["coverage"])
                if rn["n_accept"]:
                    risk[a]["native"].append(rn["selective_risk"])
                if rc["n_accept"]:
                    risk[a]["combined"].append(rc["selective_risk"])

        # Significance via a paired bootstrap over TEST POSES (data-level), not over
        # Monte-Carlo repeats: one held-out split, resample its test rows, and take
        # the paired AURC difference (native - combined). This reflects finite-data
        # sampling error, unlike bootstrapping the correlated per-repeat means.
        tr, cal, te = three_way(idx_all, g)
        comb = ScoreCombiner().fit(sub.iloc[tr], y[tr])
        s_te, sc_te, y_te = s_nat[te], comb.predict(sub.iloc[te]), y[te]
        d_native = aurc(s_te, y_te)
        d_combined = aurc(sc_te, y_te)
        p_te = ScoreCombiner(features=POSE_SET).fit(sub.iloc[tr], y[tr]).predict(sub.iloc[te]) if has_pose else None
        deltas, deltas_pose = [], []
        for _ in range(2000):
            bi = g.integers(0, len(te), len(te))
            deltas.append(aurc(s_te[bi], y_te[bi]) - aurc(sc_te[bi], y_te[bi]))
            if has_pose:   # combined -> combined+pose gain (the W1 upgrade delta)
                deltas_pose.append(aurc(sc_te[bi], y_te[bi]) - aurc(p_te[bi], y_te[bi]))
        d_lo, d_hi = float(np.percentile(deltas, 5)), float(np.percentile(deltas, 95))
        # one-sided bootstrap p-value for H0: delta(AURC) <= 0, for the "for every model"
        # intersection-union test (max over models of this p <= level certifies the joint).
        d_p_one = float((1.0 + np.count_nonzero(np.asarray(deltas) <= 0.0)) / (1.0 + len(deltas)))
        pose_block = {}
        if has_pose:
            p_lo, p_hi = float(np.percentile(deltas_pose, 5)), float(np.percentile(deltas_pose, 95))
            pose_p_one = float((1.0 + np.count_nonzero(np.asarray(deltas_pose) <= 0.0)) / (1.0 + len(deltas_pose)))
            pose_block = {
                "aurc_combined_pose": float(np.mean(aurc_pose)),
                "delta_pose_vs_combined_ci90": [p_lo, p_hi],
                "pose_delta_excludes_zero": bool(p_lo > 0),
                "pose_delta_p_one_sided": pose_p_one,
            }
        out[m] = {
            "n": len(sub),
            "aurc_native": float(np.mean(aurc_nat)),
            "aurc_combined": float(np.mean(aurc_comb)),
            **pose_block,
            "delta_aurc_heldout": float(d_native - d_combined),
            "delta_aurc_ci_data_bootstrap": [d_lo, d_hi],
            "delta_excludes_zero": bool(d_lo > 0),
            "delta_p_one_sided": d_p_one,
            "operating_points": {
                str(a): {
                    "native_coverage": float(np.mean(cov[a]["native"])),
                    "combined_coverage": float(np.mean(cov[a]["combined"])),
                    "native_realized_risk": float(np.mean(risk[a]["native"])) if risk[a]["native"] else float("nan"),
                    "combined_realized_risk":
                        float(np.mean(risk[a]["combined"])) if risk[a]["combined"] else float("nan"),
                }
                for a in ALPHAS
            },
        }

    # "The combined score beats native for EVERY model" is a conjunction, so it is an
    # intersection-union test: requiring each per-model one-sided p <= level certifies the
    # joint claim at that level with NO multiplicity penalty (Berger 1982). No Bonferroni
    # is applied to the within-model threshold grid or to this conjunction.
    pvals = [r["delta_p_one_sided"] for r in out.values()]
    pose_p = [r["pose_delta_p_one_sided"] for r in out.values() if "pose_delta_p_one_sided" in r]
    out["_joint"] = {
        "all_models_exclude_zero": bool(iut_all(pvals, DELTA)),
        "delta_p_one_sided_max": float(max(pvals)) if pvals else float("nan"),
        "all_models_pose_exclude_zero": bool(iut_all(pose_p, DELTA)) if pose_p else None,
        "pose_delta_p_one_sided_max": float(max(pose_p)) if pose_p else None,
        "note": "IUT conjunction over models; no penalty (docs/theory/MULTIPLICITY_SPEC.md).",
    }
    return out


def make_figure(res: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from foldgate.selective import risk_coverage_curve

    df = load_delivered()
    g = rng()
    m = "af3" if "af3" in res else next(k for k in res if not k.startswith("_"))
    sub = df[df.method == m].reset_index(drop=True)
    y = sub["correct"].to_numpy()
    tr, cal, te = three_way(np.arange(len(sub)), g)
    comb = ScoreCombiner().fit(sub.iloc[tr], y[tr])
    cov_n, risk_n = risk_coverage_curve(sub[NATIVE].to_numpy()[te], y[te])
    cov_c, risk_c = risk_coverage_curve(comb.predict(sub.iloc[te]), y[te])

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(cov_n, risk_n, label=f"native ranking_score (AURC={res[m]['aurc_native']:.3f})", color="#c44")
    ax.plot(cov_c, risk_c, label=f"combined score (AURC={res[m]['aurc_combined']:.3f})", color="#48c")
    for a in ALPHAS:
        ax.axhline(a, ls=":", color="k", lw=0.8)
        ax.text(0.01, a + 0.005, f"alpha={a}", fontsize=8)
    ax.set_xlabel("coverage (fraction of poses accepted)")
    ax.set_ylabel("selective risk (error among accepted)")
    ax.set_title(f"E4 ({m}): the combined reliability score dominates native confidence")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_ylim(0, max(0.4, float(np.nanmax(risk_n)) * 1.05))
    fig.tight_layout()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGDIR / "e4_selective_utility.png", dpi=150)
    print(f"saved {FIGDIR / 'e4_selective_utility.png'}")


def main() -> None:
    res = run()
    save_json(res, RESDIR / "e4_selective_utility.json")
    print(f"E4 -- selective utility (combined vs native), delta={DELTA}\n")
    print(f"{'method':10} {'AURC nat':>9} {'AURC comb':>10} {'improve':>8} {'+pose(W1)':>10} {'poseCI>0':>9}")
    print("-" * 62)
    for m, r in res.items():
        if m == "_joint":
            continue
        imp = 100 * (r["aurc_native"] - r["aurc_combined"]) / r["aurc_native"]
        pose = r.get("aurc_combined_pose")
        pimp = 100 * (r["aurc_native"] - pose) / r["aurc_native"] if pose else float("nan")
        pflag = str(r.get("pose_delta_excludes_zero", "")) if pose else ""
        print(f"{m:10} {r['aurc_native']:>9.3f} {r['aurc_combined']:>10.3f} {imp:>7.1f}% "
              f"{pose if pose else float('nan'):>7.3f}({pimp:.0f}%) {pflag:>9}")
    j = res["_joint"]
    print(f"\njoint (IUT, no penalty): all_models_exclude_zero={j['all_models_exclude_zero']} "
          f"(max one-sided p={j['delta_p_one_sided_max']:.4f})"
          + (f"; all_models_pose_exclude_zero={j['all_models_pose_exclude_zero']} "
             f"(max p={j['pose_delta_p_one_sided_max']:.4f})" if j['all_models_pose_exclude_zero'] is not None else ""))
    print("\ncoverage at certified error levels (higher coverage = more usable):")
    for m, r in res.items():
        if m == "_joint":
            continue
        print(f"[{m}]")
        for a, op in r["operating_points"].items():
            print(f"    alpha={a}: native cov {op['native_coverage']:.3f} (risk {op['native_realized_risk']:.3f})"
                  f"  ->  combined cov {op['combined_coverage']:.3f} (risk {op['combined_realized_risk']:.3f})")
    make_figure(res)


if __name__ == "__main__":
    main()
