"""E1 -- baseline validity on i.i.d. splits.

Claim: the RCPS selective-risk gate, calibrated on a random calibration split,
controls the error rate among accepted poses at level alpha on a random test
split, with the promised P(risk <= alpha) >= 1 - delta. Establishes the method
is not broken before we stress it under shift (E2).
"""

from __future__ import annotations

import numpy as np

from experiments._common import ALPHA, CONF, DELTA, RESDIR, load_delivered, methods_with_enough, rng, save_json
from foldgate.conformal import rcps_threshold
from foldgate.selective import evaluate_gate


def run(n_repeats: int = 300) -> dict:
    df = load_delivered()
    methods = methods_with_enough(df)
    g = rng()
    results = {}

    for m in methods:
        sub = df[df.method == m].dropna(subset=[CONF]).reset_index(drop=True)
        s = sub[CONF].to_numpy()
        y = sub["correct"].to_numpy()
        n = len(sub)

        realized_risks, coverages, held = [], [], []
        for _ in range(n_repeats):
            perm = g.permutation(n)
            cal, test = perm[: n // 2], perm[n // 2:]
            tau = rcps_threshold(s[cal], y[cal], alpha=ALPHA, delta=DELTA)
            res = evaluate_gate(s[test], y[test], tau)
            if res["n_accept"] == 0:
                realized_risks.append(np.nan)
                coverages.append(0.0)
                held.append(True)  # abstaining entirely trivially satisfies risk<=alpha
                continue
            realized_risks.append(res["selective_risk"])
            coverages.append(res["coverage"])
            held.append(res["selective_risk"] <= ALPHA)

        rr = np.array(realized_risks, dtype=float)
        results[m] = {
            "n": n,
            "base_correct": float(y.mean()),
            "mean_realized_risk": float(np.nanmean(rr)),
            "p95_realized_risk": float(np.nanpercentile(rr, 95)),
            "mean_coverage": float(np.mean(coverages)),
            "frac_splits_risk_le_alpha": float(np.mean(held)),  # target >= 1 - delta
            "target_guarantee": 1 - DELTA,
        }
    return results


def main() -> None:
    res = run()
    save_json(res, RESDIR / "e1_iid_validity.json")
    print(f"E1 -- i.i.d. validity  (alpha={ALPHA}, delta={DELTA}, target coverage-of-guarantee>={1-DELTA})\n")
    hdr = f"{'method':10} {'n':>5} {'base_ok':>8} {'mean_risk':>10} {'coverage':>9} {'P(risk<=a)':>11}"
    print(hdr)
    print("-" * len(hdr))
    for m, r in res.items():
        # allow ~1 Monte-Carlo SE (300 repeats) of slack around the nominal 1 - delta
        flag = "OK" if r["frac_splits_risk_le_alpha"] >= 1 - DELTA - 0.02 else "LOW"
        print(f"{m:10} {r['n']:>5} {r['base_correct']:>8.3f} {r['mean_realized_risk']:>10.3f} "
              f"{r['mean_coverage']:>9.3f} {r['frac_splits_risk_le_alpha']:>10.3f} {flag}")
    print("\nInterpretation: mean realized risk <= alpha and P(risk<=alpha) >= 1-delta "
          "=> the guarantee holds on i.i.d. data. Coverage is the accepted fraction (utility).")


if __name__ == "__main__":
    main()
