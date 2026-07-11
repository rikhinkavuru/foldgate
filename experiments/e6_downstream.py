"""E6a -- downstream payoff inside RNP: does abstaining clean the pose set?

A practitioner carrying delivered poses into downstream work (structure-based
design, pocket analysis, FEP starting structures) wants a high-purity set. We
compare using ALL top-1 poses vs the combined-score gate's accepted set:

  purity      = fraction of kept poses that are correct (RMSD <= 2 A)
  enrichment  = purity(accepted) / purity(all)
  retained    = how many correct poses are kept
  interface   = mean LDDT-PLI of kept poses (structural quality, not just pass/fail)

This turns the statistics into a practice-relevant number; the Mac1 virtual-screen
enrichment arm follows when its coordinates release.
"""

from __future__ import annotations

import numpy as np

from experiments._common import DELTA, RESDIR, load_delivered, methods_with_enough, rng, save_json
from foldgate.conformal import ltt_threshold
from foldgate.scores import ScoreCombiner

ALPHAS = [0.10, 0.20]


def three_way(idx, g):
    perm = g.permutation(idx)
    n = len(perm)
    return perm[: int(0.4 * n)], perm[int(0.4 * n): int(0.7 * n)], perm[int(0.7 * n):]


def run(n_repeats: int = 120) -> dict:
    df = load_delivered()
    methods = methods_with_enough(df)
    g = rng()
    out = {}
    for m in methods:
        sub = df[df.method == m].reset_index(drop=True)
        y = sub["correct"].to_numpy()
        lddt = sub["lddt_pli"].to_numpy() if "lddt_pli" in sub else np.full(len(sub), np.nan)
        idx = np.arange(len(sub))
        rows = {a: {"purity": [], "enrich": [], "retain_frac": [], "kept_frac": [],
                    "iface_all": [], "iface_kept": []} for a in ALPHAS}
        for _ in range(n_repeats):
            tr, cal, te = three_way(idx, g)
            comb = ScoreCombiner().fit(sub.iloc[tr], y[tr])
            sc_cal, sc_te = comb.predict(sub.iloc[cal]), comb.predict(sub.iloc[te])
            base_purity = y[te].mean()
            n_correct_total = int(y[te].sum())
            for a in ALPHAS:
                tau = ltt_threshold(sc_cal, y[cal], alpha=a, delta=DELTA)
                acc = sc_te >= tau if tau is not None else np.zeros(len(te), bool)
                n_acc = int(acc.sum())
                if n_acc == 0:
                    continue
                purity = y[te][acc].mean()
                rows[a]["purity"].append(purity)
                rows[a]["enrich"].append(purity / base_purity)
                rows[a]["retain_frac"].append(int(y[te][acc].sum()) / max(n_correct_total, 1))
                rows[a]["kept_frac"].append(n_acc / len(te))
                rows[a]["iface_all"].append(float(np.nanmean(lddt[te])))
                rows[a]["iface_kept"].append(float(np.nanmean(lddt[te][acc])))
        out[m] = {
            "base_purity": float(y.mean()),
            "by_alpha": {
                str(a): {
                    "accepted_purity": float(np.mean(r["purity"])) if r["purity"] else float("nan"),
                    "enrichment": float(np.mean(r["enrich"])) if r["enrich"] else float("nan"),
                    "correct_retained_frac": float(np.mean(r["retain_frac"])) if r["retain_frac"] else 0.0,
                    "kept_frac": float(np.mean(r["kept_frac"])) if r["kept_frac"] else 0.0,
                    "iface_lddt_all": float(np.nanmean(r["iface_all"])) if r["iface_all"] else float("nan"),
                    "iface_lddt_kept": float(np.nanmean(r["iface_kept"])) if r["iface_kept"] else float("nan"),
                }
                for a, r in rows.items()
            },
        }
    return out


def main() -> None:
    res = run()
    save_json(res, RESDIR / "e6_downstream.json")
    print(f"E6a -- downstream pose-set purity (combined-score gate), delta={DELTA}\n")
    for m, r in res.items():
        print(f"[{m}]  base purity {r['base_purity']:.3f}")
        for a, v in r["by_alpha"].items():
            print(f"    alpha={a}: kept {v['kept_frac']:.2f} of poses, purity "
                  f"{v['base_purity'] if False else v['accepted_purity']:.3f} "
                  f"(enrichment {v['enrichment']:.2f}x), keeps {v['correct_retained_frac']:.2f} of all correct poses; "
                  f"iface LDDT-PLI {v['iface_lddt_all']:.3f} -> {v['iface_lddt_kept']:.3f}")
        print()


if __name__ == "__main__":
    main()
