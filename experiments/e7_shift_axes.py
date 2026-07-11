"""E7 -- the break generalizes across shift axes.

E2 showed the exchangeability break along ligand-novelty. Here we confirm it is
not specific to that axis: a global iid-calibrated gate under-controls error on
the novel end of POCKET-novelty and TEMPORAL (release-date) strata too.
"""

from __future__ import annotations

import numpy as np

from experiments._common import ALPHA, CONF, DELTA, RESDIR, load_delivered, methods_with_enough, rng, save_json
from foldgate.conformal import ltt_threshold
from foldgate.selective import conditional_coverage

AXES = {
    "ligand_novelty": "novelty_stratum",
    "pocket_novelty": "pocket_novelty_stratum",
    "temporal": "temporal_stratum",
}


def break_for_axis(s, y, strat, g, n_repeats):
    n = len(s)
    levels = sorted(np.unique(strat).tolist())
    err = {k: 0 for k in levels}
    acc = {k: 0 for k in levels}
    for _ in range(n_repeats):
        perm = g.permutation(n)
        cal, te = perm[: n // 2], perm[n // 2:]
        tau = ltt_threshold(s[cal], y[cal], alpha=ALPHA, delta=DELTA)
        if tau is None:
            continue
        cc = conditional_coverage(s[te], y[te], strat[te], tau)
        for k, v in cc.items():
            if v["n_accept"]:
                err[k] += v["selective_risk"] * v["n_accept"]
                acc[k] += v["n_accept"]
    return {int(k): {"risk": (err[k] / acc[k]) if acc[k] else float("nan")} for k in levels}


def run(method="af3", n_repeats=300) -> dict:
    df = load_delivered()
    sub = df[df.method == method].reset_index(drop=True)
    s, y = sub[CONF].to_numpy(), sub["correct"].to_numpy()
    out = {}
    for name, col in AXES.items():
        if col not in sub:
            continue
        strat = sub[col].to_numpy().astype(int)
        out[name] = break_for_axis(s, y, strat, rng(), n_repeats)
    return out


def main() -> None:
    res = {}
    df = load_delivered()
    for m in methods_with_enough(df):
        res[m] = run(m)
    save_json(res, RESDIR / "e7_shift_axes.json")
    print(f"E7 -- break across shift axes (alpha={ALPHA}). Per-stratum realized risk under a global gate.\n")
    for m, axes in res.items():
        print(f"[{m}]")
        for name, strata in axes.items():
            row = "  ".join(f"S{k}:{v['risk']:.2f}" for k, v in sorted(strata.items()))
            print(f"    {name:16} {row}")
        print()
    print(f"Break confirmed on an axis when the novel (high-S) strata exceed alpha={ALPHA} "
          "while low-S strata sit below it.")


if __name__ == "__main__":
    main()
