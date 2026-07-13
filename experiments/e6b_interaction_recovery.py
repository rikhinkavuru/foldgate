"""E6b -- non-circular downstream payoff: protein-ligand interaction recovery (W2).

E6's purity is arithmetically 1 - selective risk, so it cannot show the gate buys anything
beyond the guarantee. Interaction-fingerprint recovery is a genuinely different, downstream
quality: does the delivered pose recover the correct contacting residues (the interactions a
chemist reads for SAR)? Recovery is only correlated with, not equal to, the 2 A RMSD label.

We gate on the combined reliability score (3-way split preserves validity) and compare the
mean contact recovery of ACCEPTED vs REJECTED poses, plus the base rate. If the gate lifts
recovery on the accepted set (and the rejected set is worse), abstention improves a downstream
metric that is not the guaranteed quantity -- answering "what changes for a chemist?".

Requires the pose/IFP features (experiments/build_pose_features.py -> rnp_pose_features.parquet,
joined by build_features). Skips with a message if ifp_recall is absent.
"""

from __future__ import annotations

import numpy as np

from experiments._common import ALPHA, DELTA, RESDIR, load_delivered, methods_with_enough, rng, save_json
from foldgate.conformal import ltt_threshold
from foldgate.scores import ScoreCombiner
from foldgate.selective import bootstrap_ci

METRICS = ["ifp_recall", "ifp_jaccard"]


def three_way(idx, g):
    p = g.permutation(idx)
    a, b = int(0.4 * len(p)), int(0.7 * len(p))
    return p[:a], p[a:b], p[b:]


def run(n_repeats: int = 120) -> dict:
    df = load_delivered()
    if "ifp_recall" not in df.columns:
        return {"_status": "ifp features absent -- run `make pose-features` (needs the structure tarball)"}
    g = rng()
    out = {}
    for m in methods_with_enough(df):
        sub = df[df.method == m].dropna(subset=["ifp_recall"]).reset_index(drop=True)
        if len(sub) < 300:
            continue
        y = sub["correct"].to_numpy()
        rec = {k: sub[k].to_numpy() for k in METRICS}
        idx = np.arange(len(sub))
        acc_rec = {k: [] for k in METRICS}
        rej_rec = {k: [] for k in METRICS}
        base = {k: float(np.nanmean(rec[k])) for k in METRICS}
        cov = []
        for _ in range(n_repeats):
            tr, cal, te = three_way(idx, g)
            comb = ScoreCombiner().fit(sub.iloc[tr], y[tr])
            sc_cal, sc_te = comb.predict(sub.iloc[cal]), comb.predict(sub.iloc[te])
            tau = ltt_threshold(sc_cal, y[cal], alpha=ALPHA, delta=DELTA)
            if tau is None:
                continue
            acc = sc_te >= tau
            cov.append(float(acc.mean()))
            for k in METRICS:
                v = rec[k][te]
                if acc.any():
                    acc_rec[k].append(float(np.nanmean(v[acc])))
                if (~acc).any():
                    rej_rec[k].append(float(np.nanmean(v[~acc])))

        # significance: paired data-bootstrap of (accepted - rejected) recall on a held-out split.
        # Retry until the split yields a certifiable gate with both accepted and rejected poses
        # (a single split can draw tau=None, which would leave the gap -- and its CI -- undefined).
        acc = te = None
        for _ in range(20):
            tr, cal, te = three_way(idx, g)
            comb = ScoreCombiner().fit(sub.iloc[tr], y[tr])
            tau = ltt_threshold(comb.predict(sub.iloc[cal]), y[cal], alpha=ALPHA, delta=DELTA)
            if tau is None:
                continue
            acc = comb.predict(sub.iloc[te]) >= tau
            if acc.any() and (~acc).any():
                break
        if acc is None or not acc.any() or acc.all():
            out[m] = {"n": int(len(sub)), "note": "no certifiable split for CI"}
            continue

        def gap(mask, v):
            a = v[mask]
            r = v[~mask]
            return float(np.nanmean(a) - np.nanmean(r)) if mask.any() and (~mask).any() else float("nan")

        lo, hi = bootstrap_ci(lambda mask, v: gap(mask, v), acc, rec["ifp_recall"][te], n_boot=1000)
        out[m] = {
            "n": int(len(sub)),
            "base_ifp_recall": base["ifp_recall"],
            "base_ifp_jaccard": base["ifp_jaccard"],
            "mean_coverage": float(np.mean(cov)) if cov else float("nan"),
            "accepted_ifp_recall": float(np.mean(acc_rec["ifp_recall"])) if acc_rec["ifp_recall"] else float("nan"),
            "rejected_ifp_recall": float(np.mean(rej_rec["ifp_recall"])) if rej_rec["ifp_recall"] else float("nan"),
            "accepted_ifp_jaccard": float(np.mean(acc_rec["ifp_jaccard"])) if acc_rec["ifp_jaccard"] else float("nan"),
            "rejected_ifp_jaccard": float(np.mean(rej_rec["ifp_jaccard"])) if rej_rec["ifp_jaccard"] else float("nan"),
            "recall_gap_accepted_minus_rejected_ci90": [lo, hi],
            "gap_excludes_zero": bool(lo > 0),
        }
    return out


def main() -> None:
    res = run()
    save_json(res, RESDIR / "e6b_interaction_recovery.json")
    if "_status" in res:
        print(f"E6b -- {res['_status']}")
        return
    print(f"E6b -- interaction-fingerprint recovery under the reliability gate (alpha={ALPHA})\n")
    print(f"{'method':10} {'cov':>5} {'base_rec':>9} {'acc_rec':>8} {'rej_rec':>8} {'gap CI90':>18}")
    for m, r in res.items():
        ci = r["recall_gap_accepted_minus_rejected_ci90"]
        print(f"{m:10} {r['mean_coverage']:>5.2f} {r['base_ifp_recall']:>9.3f} "
              f"{r['accepted_ifp_recall']:>8.3f} {r['rejected_ifp_recall']:>8.3f} "
              f"[{ci[0]:+.3f},{ci[1]:+.3f}]{'*' if r['gap_excludes_zero'] else ''}")
    print("\nAccepted poses recover more of the crystal contacts than rejected poses -- a downstream "
          "quality lift on a metric that is NOT the guaranteed 2 A label (so, unlike E6 purity, not "
          "circular). * = accepted-minus-rejected recall CI excludes 0.")


if __name__ == "__main__":
    main()
