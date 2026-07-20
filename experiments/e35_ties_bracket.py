"""E35 -- how wide is the "exact identity" bracket under score ties?

Reviewer R1.2: the impossibility theorem's equalities assume the score has no atom at
the operating threshold. Real ipTM/ranking scores carry ties, handled by boundary
randomization, under which the equalities become <=/>= brackets. The abstract and
conclusion say "exact identity". This script measures the bracket width so the claim
is honest: at each deployed threshold, the ties mass (fraction of poses sharing the
threshold value) is the maximum coverage the randomization can move, i.e. the width of
the bracket in coverage points. If it is < 0.5% coverage the "exact" language survives
with a footnote; otherwise it must be softened.

For each model and each score (ranking_score, iface_iptm) we report:
  - max_atom_mass: the largest fraction of poses sharing any single score value (the
    worst-case atom anywhere on the grid),
  - ties_at_tau: the ties mass exactly at the deployed alpha=0.20 gate threshold tau
    (the operationally relevant bracket width),
  - the threshold tau and the accepted coverage there.

Output: results/e35_ties_bracket.json
"""

from __future__ import annotations

import numpy as np

from experiments._common import (
    ALPHA,
    CONF,
    DELTA,
    RESDIR,
    load_delivered,
    methods_with_enough,
    save_json,
)
from foldgate.conformal import ltt_threshold

SCORES = ["ranking_score", "iface_iptm"]


def _atom_stats(vals: np.ndarray) -> dict:
    v = vals[np.isfinite(vals)]
    n = len(v)
    if n == 0:
        return {"n": 0, "max_atom_mass": None, "n_distinct": 0}
    uniq, counts = np.unique(v, return_counts=True)
    return {
        "n": int(n),
        "n_distinct": int(len(uniq)),
        "max_atom_mass": round(float(counts.max()) / n, 5),
        "frac_distinct": round(len(uniq) / n, 4),
    }


def run() -> dict:
    df = load_delivered()
    methods = methods_with_enough(df)
    out = {"alpha": ALPHA, "delta": DELTA, "per_model": {}}
    worst = 0.0
    worst_tau = 0.0
    for m in methods:
        sub = df[df.method == m]
        model_out = {}
        for score in SCORES:
            if score not in sub.columns:
                continue
            d = sub.dropna(subset=[score, "correct"])
            vals = d[score].to_numpy(dtype=float)
            y = d["correct"].to_numpy().astype(int)
            stats = _atom_stats(vals)
            # deployed alpha-gate threshold on the full model data (native protocol)
            tau = ltt_threshold(vals, y, alpha=ALPHA, delta=DELTA)
            ties_at_tau = None
            cov = None
            if tau is not None and stats["n"]:
                ties_at_tau = round(float(np.mean(vals == tau)), 5)
                cov = round(float(np.mean(vals >= tau)), 4)
                worst = max(worst, ties_at_tau)
                worst_tau = max(worst_tau, stats["max_atom_mass"] or 0.0)
            stats["tau"] = None if tau is None else round(float(tau), 6)
            stats["coverage_at_tau"] = cov
            stats["ties_at_tau"] = ties_at_tau
            model_out[score] = stats
        out["per_model"][m] = model_out

    out["_summary"] = {
        "worst_ties_at_tau_over_models": round(worst, 5),
        "worst_atom_mass_anywhere": round(worst_tau, 5),
        "exact_language_survives": bool(worst < 0.005),
        "note": (
            "ties_at_tau is the bracket width in coverage points at the deployed "
            "threshold; worst over models is the number to quote. If < 0.005 (0.5% "
            "coverage) the exact-identity language survives with a footnote."
        ),
    }
    return out


def main() -> None:
    res = run()
    save_json(res, RESDIR / "e35_ties_bracket.json")
    print("E35 -- ties / atom mass at the deployed threshold (bracket width)\n")
    for m, mo in res["per_model"].items():
        for score, s in mo.items():
            if s["n"] == 0:
                continue
            print(f"  [{m:>9}] {score:>14}: distinct {s['frac_distinct']:.3f} "
                  f"max_atom {s['max_atom_mass']:.4f}  tau={s['tau']} "
                  f"cov={s['coverage_at_tau']} ties@tau={s['ties_at_tau']}")
    sm = res["_summary"]
    print(f"\nworst ties@tau over models = {sm['worst_ties_at_tau_over_models']:.4f} "
          f"({'exact survives' if sm['exact_language_survives'] else 'soften to <=/>= bracket'})")


if __name__ == "__main__":
    main()
