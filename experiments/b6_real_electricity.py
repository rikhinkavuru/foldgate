"""B6 -- real-data generality of the worst-stratum selective-risk claim (electricity).

Public tabular stress test on OpenML ``electricity`` (elec2, data id 151), the
canonical temporal concept-drift benchmark. We reduce it to the same (s, y, nu)
triple the co-folding pipeline uses: a base HistGradientBoosting classifier is
trained on the earliest temporal block only, then

  s  = max-softmax confidence of the base classifier,
  y  = 1[base prediction correct],
  nu = temporal block index (the shift coordinate).

The temporal block (nu) is the protected time coordinate: block 0 trains f, and
every later block supplies its own calibration and test rows; no later block ever
calibrates another and time is never shuffled across blocks. Three ways to spend
the SAME pooled calibration labels are compared at one operating point:

  MARGINAL   ignore nu, one global threshold from the pooled calibration set
             (plus a "single reference stratum" = calibrate-at-launch variant).
  MONDRIAN   condition on nu, a separate threshold per block from that block's
             calibration half (needs in-stratum labels).
  WEIGHTED   reweight the pooled calibration set toward each target block's score
             distribution (covariate-shift likelihood ratio on s), per block.

Claim under test: MARGINAL under-controls the worst novel block while MONDRIAN
controls it, and WEIGHTED repairs covariate shift but not the concept shift that
elec2 actually contains. The concept-shift diagnostic (does P(correct | s) move?)
is reported so the weighted result is mechanistically explained, not asserted.

Honesty: OpenML ``electricity`` redistribution license is TO BE CONFIRMED (OpenML
lists it public; the original NSW/elec2 terms are not restated on the page).
Realized numbers are reported as they come out, whatever the direction.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foldgate.bench.certificates import worst_stratum_rcps_ucb  # noqa: E402
from foldgate.bench.realdata import SkipDataset, electricity_triple  # noqa: E402
from foldgate.conformal import (  # noqa: E402
    concept_shift_diagnostic,
    estimate_weights_cv,
    ltt_threshold,
    naive_threshold,
    weighted_ltt_threshold,
    weighted_threshold,
)
from foldgate.selective import aurc, evaluate_gate  # noqa: E402

RESDIR = ROOT / "results" / "bench"
ALPHAS = [0.10, 0.20]
DELTA = 0.10
N_BLOCKS = 6
CAL_FRAC = 0.5          # each later block's rows split into a calibration and a test half
N_BOOT = 1000
SEED = 20260712


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=os.path.dirname(__file__)
        ).decode().strip()
    except Exception:
        return "unknown"


def save_json(obj, path: Path) -> None:
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float))


def blocks_from_triple(tri):
    """Reshape the (s, y, nu, split) triple into per-group cal/test score arrays.

    Group keys are stringified so the same analysis serves an integer temporal block
    (electricity) or a string state code (ACS). The grouping coordinate nu is the
    protected shift axis: no group ever calibrates another. Within a group the
    loader's split assigns rows to a calibration half and a test half, so the two
    halves are exchangeable inside that fixed stratum (the Mondrian assumption).
    """
    out = {}
    for b in sorted(tri["nu"].unique()):
        rows = tri[tri["nu"] == b]
        cal = rows[rows["split"] == "cal"]
        test = rows[rows["split"] == "test"]
        out[str(b)] = {
            "cal_s": cal["s"].to_numpy(), "cal_y": cal["y"].to_numpy(),
            "test_s": test["s"].to_numpy(), "test_y": test["y"].to_numpy(),
        }
    return out


def _per_block_eval(blocks, tau_of_block) -> dict:
    """Evaluate a gate on every group's TEST half; tau_of_block(b) gives the threshold.

    Returns per-group coverage/risk/(errors,n_accept) plus the worst (max) realized
    selective risk over groups that accepted anything, and the union-bound RCPS UCB
    on that worst-group risk over the ACCEPTING groups (an abstained group carries no
    selective risk, so it does not force the union bound to a vacuous 1.0).
    """
    per_block, counts = {}, []
    worst_risk, worst_block = -1.0, None
    for b, d in blocks.items():
        tau = tau_of_block(b)
        g = evaluate_gate(d["test_s"], d["test_y"], tau)
        n_acc = g["n_accept"]
        errors = int(round(g["selective_risk"] * n_acc)) if n_acc else 0
        counts.append((errors, n_acc))
        per_block[b] = {
            "tau": g["tau"], "coverage": g["coverage"], "selective_risk": g["selective_risk"],
            "n_accept": n_acc, "errors": errors, "n_test": g["n"],
        }
        if n_acc > 0 and np.isfinite(g["selective_risk"]) and g["selective_risk"] > worst_risk:
            worst_risk, worst_block = g["selective_risk"], b
    accepting = [(e, n) for (e, n) in counts if n > 0]
    cert = worst_stratum_rcps_ucb(accepting, delta=DELTA, method="hb")
    total_acc = sum(n for _, n in counts)
    total_test = sum(v["n_test"] for v in per_block.values())
    return {
        "per_block": per_block,
        "worst_block": worst_block,
        "worst_block_risk": float(worst_risk) if worst_block is not None else float("nan"),
        "certified_worst_risk_ucb": cert["U"],
        "overall_coverage": float(total_acc / total_test) if total_test else 0.0,
    }


def _bootstrap_worst_risk(blocks, tau_of_block, g) -> list[float]:
    """Group-stratified bootstrap of the worst-group realized selective risk.

    Thresholds are held fixed (conditioning on calibration, the split-CP convention);
    only the test rows are resampled, within each group, so the group marginals are
    preserved. Each replicate recomputes every group's error-among-accepted and takes
    the max over groups that accepted anything.
    """
    prepared = [(d["test_s"], d["test_y"], tau_of_block(b)) for b, d in blocks.items()]
    vals = []
    for _ in range(N_BOOT):
        worst = -1.0
        for s, y, tau in prepared:
            if tau is None:
                continue
            n = len(s)
            idx = g.integers(0, n, n)
            acc = s[idx] >= tau
            if acc.sum() == 0:
                continue
            worst = max(worst, float(1.0 - y[idx][acc].mean()))
        if worst >= 0.0:
            vals.append(worst)
    return vals


def _ci(vals, lo=5, hi=95):
    if not vals:
        return [float("nan"), float("nan")]
    return [float(np.percentile(vals, lo)), float(np.percentile(vals, hi))]


def run_alpha(blocks, pooled_cal_s, pooled_cal_y, alpha, g) -> dict:
    """One operating point: build MARGINAL / MONDRIAN / WEIGHTED gates and score them."""
    # ----- MARGINAL (pooled): ignore nu, one plug-in threshold from the union of all
    # calibration halves (the exact same labels Mondrian slices per group). -----
    tau_marg = naive_threshold(pooled_cal_s, pooled_cal_y, alpha)
    tau_marg_cert = ltt_threshold(pooled_cal_s, pooled_cal_y, alpha, DELTA)

    # ----- MARGINAL (single reference stratum): calibrate once on one reference group
    # and deploy that frozen threshold to every group. For a temporal benchmark this
    # is "calibrate at launch on the first block, never recalibrate". -----
    ref_b = next(iter(blocks))
    tau_single = naive_threshold(blocks[ref_b]["cal_s"], blocks[ref_b]["cal_y"], alpha)

    # ----- MONDRIAN: per-group plug-in threshold from that group's calibration half.
    tau_mond, tau_mond_cert = {}, {}
    for b, d in blocks.items():
        tau_mond[b] = naive_threshold(d["cal_s"], d["cal_y"], alpha)
        tau_mond_cert[b] = ltt_threshold(d["cal_s"], d["cal_y"], alpha, DELTA)

    # ----- WEIGHTED: reweight the pooled calibration set toward each target group's
    # score distribution (covariate-shift LR on s), then a weighted threshold. The
    # plug-in variant is the fair analogue of the plug-in marginal/mondrian; the LTT
    # variant is the finite-sample certified analogue of mondrian_certified.
    tau_wt, tau_wt_cert, n_eff = {}, {}, {}
    for b, d in blocks.items():
        w = estimate_weights_cv(pooled_cal_s, d["test_s"], seed=SEED)
        tau_wt[b] = weighted_threshold(pooled_cal_s, pooled_cal_y, w, alpha, DELTA)
        tau_wt_cert[b] = weighted_ltt_threshold(pooled_cal_s, pooled_cal_y, w, alpha, DELTA)
        s2 = float(np.sum(w * w))
        n_eff[b] = float(w.sum() ** 2 / s2) if s2 > 0 else 0.0

    methods = {
        "marginal_pooled": (lambda b, t=tau_marg: t),
        "marginal_single": (lambda b, t=tau_single: t),
        "mondrian": (lambda b: tau_mond[b]),
        "weighted": (lambda b: tau_wt[b]),
        "marginal_pooled_certified": (lambda b, t=tau_marg_cert: t),
        "weighted_certified": (lambda b: tau_wt_cert[b]),
        "mondrian_certified": (lambda b: tau_mond_cert[b]),
    }
    res = {}
    for name, fn in methods.items():
        ev = _per_block_eval(blocks, fn)
        ev["worst_block_risk_ci90"] = _ci(_bootstrap_worst_risk(blocks, fn, g))
        ev["worst_block_risk_exceeds_alpha"] = (
            bool(ev["worst_block_risk"] > alpha) if np.isfinite(ev["worst_block_risk"]) else None
        )
        res[name] = ev
    res["weighted_n_eff_per_block"] = {b: n_eff[b] for b in n_eff}
    res["reference_stratum"] = ref_b
    res["alpha"] = alpha
    return res


def score_aurc_diagnostic(blocks, g) -> dict:
    """AURC of the shared max-softmax score, per group and worst-group (score-level).

    AURC is a threshold-free property of the confidence RANKING, so it is identical
    across the three gates (they share s); it is reported once as the score's
    ranking-quality diagnostic. A worst-group AURC well above the pooled AURC means
    the confidence ordering itself degrades on the drifted group, the root cause the
    gates must cope with.
    """
    per_block, worst, worst_b = {}, -1.0, None
    all_s, all_y = [], []
    for b, d in blocks.items():
        a = aurc(d["test_s"], d["test_y"])
        per_block[b] = float(a)
        all_s.append(d["test_s"])
        all_y.append(d["test_y"])
        if a > worst:
            worst, worst_b = a, b
    pooled_aurc = aurc(np.concatenate(all_s), np.concatenate(all_y))
    prepared = [(d["test_s"], d["test_y"]) for d in blocks.values()]
    boot = []
    for _ in range(N_BOOT):
        w = -1.0
        for s, y in prepared:
            idx = g.integers(0, len(s), len(s))
            w = max(w, aurc(s[idx], y[idx]))
        boot.append(w)
    return {
        "per_block_aurc": per_block,
        "worst_block_aurc": float(worst),
        "worst_block_aurc_ci90": _ci(boot),
        "worst_block": worst_b,
        "pooled_test_aurc": float(pooled_aurc),
    }


def concept_diagnostics(blocks, pooled_cal_s, pooled_cal_y) -> dict:
    """Does P(correct | confidence) move source->target? (concept vs covariate shift)."""
    worst_b = min(blocks, key=lambda b: float(blocks[b]["test_y"].mean()))
    d = blocks[worst_b]
    per_worst = concept_shift_diagnostic(pooled_cal_s, pooled_cal_y, d["test_s"], d["test_y"])
    all_s = np.concatenate([bd["test_s"] for bd in blocks.values()])
    all_y = np.concatenate([bd["test_y"] for bd in blocks.values()])
    overall = concept_shift_diagnostic(pooled_cal_s, pooled_cal_y, all_s, all_y)
    return {
        "worst_block_by_base_accuracy": worst_b,
        "worst_block_base_correct_rate": float(d["test_y"].mean()),
        "concept_gap_worst_block": {
            "max_abs_gap": per_worst["max_abs_gap"],
            "mean_abs_gap_target_weighted": per_worst["mean_abs_gap_target_weighted"],
        },
        "concept_gap_pooled": {
            "max_abs_gap": overall["max_abs_gap"],
            "mean_abs_gap_target_weighted": overall["mean_abs_gap_target_weighted"],
        },
    }


def analyze(blocks, meta_extra) -> dict:
    """Shared analysis over a per-group {cal_s,cal_y,test_s,test_y} dict (b5 reuses this)."""
    pooled_cal_s = np.concatenate([d["cal_s"] for d in blocks.values()])
    pooled_cal_y = np.concatenate([d["cal_y"] for d in blocks.values()])
    g = np.random.default_rng(SEED)
    per_alpha = {str(a): run_alpha(blocks, pooled_cal_s, pooled_cal_y, a, g) for a in ALPHAS}
    block_summary = {
        b: {
            "n_cal": int(len(d["cal_s"])), "n_test": int(len(d["test_s"])),
            "cal_correct_rate": float(d["cal_y"].mean()),
            "test_correct_rate": float(d["test_y"].mean()),
        }
        for b, d in blocks.items()
    }
    return {
        "meta": meta_extra,
        "block_summary": block_summary,
        "score_aurc_diagnostic": score_aurc_diagnostic(blocks, np.random.default_rng(SEED + 1)),
        "concept_shift_diagnostic": concept_diagnostics(blocks, pooled_cal_s, pooled_cal_y),
        "operating_points": per_alpha,
    }


def run() -> dict:
    t0 = time.time()
    tri = electricity_triple(N_BLOCKS, seed=SEED, cal_frac=CAL_FRAC)   # s, y, nu, split
    blocks = blocks_from_triple(tri)
    meta = {
        "dataset": "OpenML electricity (elec2, data_id 151)",
        "loader": "sklearn.datasets.fetch_openml(name='electricity', version=1)",
        "license": "TO BE CONFIRMED (OpenML lists it public; original NSW/elec2 terms not restated)",
        "shift_type": "temporal concept drift (canonical elec2)",
        "base_classifier": "HistGradientBoostingClassifier (torch-free)",
        "nu": "temporal block index; train block 0, calibrate/test later blocks; time never shuffled across blocks",
        "reference_stratum_meaning": "earliest later block = calibrate-at-launch baseline",
        "n_blocks": N_BLOCKS, "cal_frac": CAL_FRAC, "delta": DELTA, "alphas": ALPHAS,
        "n_boot": N_BOOT, "seed": SEED, "git_sha": git_sha(),
        "n_eval_later_blocks": int(len(tri)),
        "runtime_sec": round(time.time() - t0, 1),
    }
    out = analyze(blocks, meta)
    out["meta"]["runtime_sec"] = round(time.time() - t0, 1)
    return out


def _fmt(x, nd=3):
    return "None " if x is None else (f"{x:.{nd}f}" if isinstance(x, float) and np.isfinite(x) else str(x))


def print_report(res, title, ref_label) -> None:
    m = res["meta"]
    print(f"{title}  [git {m['git_sha'][:10]}, {m.get('runtime_sec', '?')}s]")
    ad = res["score_aurc_diagnostic"]
    cd = res["concept_shift_diagnostic"]
    aci = ad["worst_block_aurc_ci90"]
    print(f"score AURC: pooled {ad['pooled_test_aurc']:.3f}  worst-group {ad['worst_block_aurc']:.3f} "
          f"(group {ad['worst_block']}) CI90 [{aci[0]:.3f},{aci[1]:.3f}]")
    print(f"concept shift P(correct|s) gap: worst-group max {cd['concept_gap_worst_block']['max_abs_gap']:.3f}, "
          f"pooled max {cd['concept_gap_pooled']['max_abs_gap']:.3f}  "
          f"(worst group {cd['worst_block_by_base_accuracy']}, base acc {cd['worst_block_base_correct_rate']:.3f})")
    labels = {"marginal_single": f"marginal_single({ref_label})"}
    order = ["marginal_pooled", "marginal_single", "weighted", "mondrian",
             "marginal_pooled_certified", "weighted_certified", "mondrian_certified"]
    for a in ALPHAS:
        block = res["operating_points"][str(a)]
        print(f"\nalpha={a}  (worst-group selective risk should be <= alpha)")
        print(f"  {'method':30} {'worstR':>7} {'CI90':>17} {'>a?':>5} {'certU':>6} {'cov':>6}")
        for name in order:
            r = block[name]
            ci = f"[{_fmt(r['worst_block_risk_ci90'][0])},{_fmt(r['worst_block_risk_ci90'][1])}]"
            print(f"  {labels.get(name, name):30} {_fmt(r['worst_block_risk']):>7} {ci:>17} "
                  f"{str(r['worst_block_risk_exceeds_alpha']):>5} "
                  f"{_fmt(r['certified_worst_risk_ucb']):>6} {_fmt(r['overall_coverage']):>6}")


def main() -> None:
    try:
        res = run()
    except SkipDataset as e:
        res = {"skipped": True, "reason": str(e),
               "meta": {"dataset": "OpenML electricity", "git_sha": git_sha()}}
        save_json(res, RESDIR / "b6_real_electricity.json")
        print(f"B6 SKIPPED: {e}")
        return
    save_json(res, RESDIR / "b6_real_electricity.json")
    print_report(res, "B6 -- real-data (electricity/elec2) worst-group selective risk", "launch")
    print(f"\nsaved {RESDIR / 'b6_real_electricity.json'}")


if __name__ == "__main__":
    main()
