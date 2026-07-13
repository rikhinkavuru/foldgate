"""E21 -- the abstention story on a decoupled task: selective Boltz-2 affinity prediction.

A second, independent dataset (Zenodo 10.5281/zenodo.18669539: a ChEMBL-derived Boltz-2 affinity
benchmark, 356 targets) tests whether the reliability layer generalizes past pose correctness to a
different endpoint, binding-affinity regression. The label is the measured pChEMBL value; the
prediction is the Boltz-2 predicted affinity; the reliability signal is the Boltz-2 confidence
score. We ask the selective-prediction question directly: does abstaining on low-confidence
predictions lower the error of what remains, and does the error rise on novel targets and novel
chemotypes, the same shift the pose study documents.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from experiments._common import RESDIR, rng, save_json

AFF_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "external", "screening_affinity")
S6 = "Data_S6_benchmark_dataset_for_predictive_performance.csv"
S8 = "Data_S8_predictive_performance_metrics_for_each_target.csv"


def _selective_mae(err, reliability, coverages=(1.0, 0.75, 0.5, 0.25)):
    """MAE of the top-coverage fraction ranked by reliability (abstain low-reliability)."""
    order = np.argsort(-reliability)
    e = err[order]
    n = len(e)
    out = []
    for c in coverages:
        k = max(1, int(round(c * n)))
        out.append({"coverage": float(c), "mae": float(np.mean(e[:k])), "n": k})
    return out


def _random_mae(err, coverage, g, n_boot=500):
    n = len(err)
    k = max(1, int(round(coverage * n)))
    vals = [float(np.mean(err[g.integers(0, n, k)])) for _ in range(n_boot)]
    return float(np.mean(vals)), float(np.percentile(vals, 5)), float(np.percentile(vals, 95))


def _bin_mae(err, novelty, edges):
    out = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=False):
        m = (novelty > lo) & (novelty <= hi)
        if m.sum() >= 20:
            out.append({"lo": float(lo), "hi": float(hi), "mae": float(np.mean(err[m])), "n": int(m.sum())})
    return out


def run() -> dict:
    p6 = os.path.join(AFF_DIR, S6)
    if not os.path.exists(p6):
        return {"error": "affinity dataset not downloaded", "path": p6}
    d = pd.read_csv(p6)
    d = d.dropna(subset=["pChEMBL_value_median", "Predicted_affinity_value", "Confidence_score"])
    err = np.abs(d["Predicted_affinity_value"].to_numpy() - d["pChEMBL_value_median"].to_numpy())
    conf = d["Confidence_score"].to_numpy()
    g = rng()

    # does confidence rank error? Spearman(confidence, -error)
    from scipy.stats import spearmanr
    rho = float(spearmanr(conf, -err).correlation)

    sel = _selective_mae(err, conf)
    rmean, rlo, rhi = _random_mae(err, 0.5, g)

    # novelty axes: compound similarity to train (per compound) + target sequence identity (per target)
    comp_sim = d["Compound_structural_similarity"].to_numpy() if "Compound_structural_similarity" in d.columns else None
    out = {
        "n": int(len(d)),
        "n_targets": int(d["UniProt_ID"].nunique()),
        "spearman_confidence_vs_neg_error": rho,
        "selective_mae_curve": sel,
        "mae_full": sel[0]["mae"],
        "mae_conf50": sel[2]["mae"],
        "mae_rand50_mean": rmean,
        "mae_rand50_ci90": [rlo, rhi],
        "conf50_beats_random": bool(sel[2]["mae"] < rlo),
    }
    if comp_sim is not None:
        out["mae_by_compound_similarity"] = _bin_mae(err, comp_sim, [-0.01, 0.3, 0.5, 0.7, 1.01])

    p8 = os.path.join(AFF_DIR, S8)
    if os.path.exists(p8):
        t = pd.read_csv(p8)
        if "Sequence_identity" in t.columns and "MAE_median" in t.columns:
            tt = t.dropna(subset=["Sequence_identity", "MAE_median"])
            seq = tt["Sequence_identity"].to_numpy()
            mae = tt["MAE_median"].to_numpy()
            out["target_mae_by_seq_identity"] = _bin_mae(mae, seq, [0, 40, 70, 95, 101])
            out["spearman_seqid_vs_target_mae"] = float(spearmanr(seq, mae).correlation)
    return out


def main() -> None:
    res = run()
    save_json(res, RESDIR / "e21_affinity_selective.json")
    print("E21 -- selective Boltz-2 affinity prediction (Zenodo 18669539)\n")
    if "error" in res:
        print("  SKIPPED:", res["error"])
        return
    print(f"  n={res['n']} compounds, {res['n_targets']} targets")
    print(f"  Spearman(confidence, -|error|) = {res['spearman_confidence_vs_neg_error']:+.3f}")
    print(f"  MAE full={res['mae_full']:.3f} -> confidence-abstain@50%={res['mae_conf50']:.3f} "
          f"vs random@50%={res['mae_rand50_mean']:.3f} (beats random: {res['conf50_beats_random']})")
    if "mae_by_compound_similarity" in res:
        print("  MAE by compound-train similarity: " +
              ", ".join(f"[{b['lo']:.1f},{b['hi']:.1f}]:{b['mae']:.2f}(n{b['n']})"
                        for b in res["mae_by_compound_similarity"]))
    if "target_mae_by_seq_identity" in res:
        print("  target MAE by sequence identity: " +
              ", ".join(f"[{b['lo']:.0f},{b['hi']:.0f}]:{b['mae']:.2f}(n{b['n']})"
                        for b in res["target_mae_by_seq_identity"]))


if __name__ == "__main__":
    main()
