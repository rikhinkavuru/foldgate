"""E54 -- native-gate stability across diffusion seeds (reviewer D25).

The delivered pose is the top-1 by ranking_score over the diffusion samples a model
draws per target. AF3 in RNP draws 5 samples for each of 5 random seeds (25 poses per
system/ligand). Reviewer D25 asks: is the certified native gate an artefact of which
seed you happened to run?

We answer two questions on AF3:

  1. Seed stability. Rank the 5 seeds within each (system, ligand) and, for each seed
     slot, deliver that seed's own top-1-by-ranking_score pose. This yields five
     independent single-seed deliveries that differ only by the RNG. We fit the
     alpha=0.20 native LTT gate on each and report how far the certified threshold tau,
     the certified coverage, and the realized selective risk move across seeds
     (range / std).

  2. Selection sensitivity. Compare gating the standard top-1-by-score pose (over all 25)
     against gating a uniformly random sample per target (many draws). If ranking_score
     is doing real work, the top-1 delivery should calibrate to a higher coverage at the
     same certified risk than a random-pose delivery.

LTT is fit on the full delivery here (an in-calibration certificate), so tau and its
coverage are the certificate's own operating point -- exactly the quantity whose seed
sensitivity we want to expose.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from experiments._common import ALPHA, CONF, DELTA, RESDIR, rng, save_json
from foldgate.conformal import ltt_threshold
from foldgate.io.rnp import RMSD_THRESHOLD_A
from foldgate.selective import evaluate_gate
from foldgate.selective.metrics import clopper_pearson

AF3_CSV = Path("data/raw/predictions/predictions/af3.csv")
KEY = ["system_id", "ligand_instance_chain"]


def load_af3_samples() -> pd.DataFrame:
    """All proper-ligand AF3 poses with a label, keyed by (system, ligand, seed, sample)."""
    df = pd.read_csv(AF3_CSV, low_memory=False).rename(columns={"target": "system_id"})
    proper = df["ligand_is_proper"].astype(str).str.lower().isin({"true", "1", "1.0"})
    df = df[proper].dropna(subset=["rmsd", CONF]).copy()
    df["correct"] = (df["rmsd"] <= RMSD_THRESHOLD_A).astype(int)
    return df


def deliver_top1(df: pd.DataFrame) -> pd.DataFrame:
    """Top-1 by ranking_score per (system, ligand) -- the standard delivered pose."""
    return (df.sort_values(CONF, ascending=False)
              .drop_duplicates(KEY)
              .reset_index(drop=True))


def fit_gate(sub: pd.DataFrame) -> dict:
    """Certified native LTT gate on a delivery: tau, certified coverage, realized risk."""
    s = sub[CONF].to_numpy()
    y = sub["correct"].to_numpy().astype(int)
    tau = ltt_threshold(s, y, alpha=ALPHA, delta=DELTA)
    ev = evaluate_gate(s, y, tau)
    lo, hi = clopper_pearson(ev["n_accept"], ev["n"]) if ev["n"] else (float("nan"), float("nan"))
    return {
        "tau": ev["tau"],
        "coverage": ev["coverage"],
        "coverage_ci90": [lo, hi],
        "selective_risk": ev["selective_risk"],
        "n_accept": ev["n_accept"],
        "n": ev["n"],
        "base_correct": float(y.mean()),
    }


def _spread(vals: list[float]) -> dict:
    a = np.array([v for v in vals if v is not None and np.isfinite(v)], dtype=float)
    if a.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"),
                "max": float("nan"), "range": float("nan"), "n": 0}
    return {"mean": float(a.mean()), "std": float(a.std(ddof=0)),
            "min": float(a.min()), "max": float(a.max()),
            "range": float(a.max() - a.min()), "n": int(a.size)}


def seed_stability(df: pd.DataFrame) -> dict:
    """Fit the gate on each within-system seed slot's own top-1 delivery; report spread."""
    # Rank seeds 0..4 within each (system, ligand) by seed value (stable, RNG-agnostic).
    df = df.copy()
    df["seed_rank"] = (
        df.groupby(KEY)["seed"].transform(lambda s: s.rank(method="dense").astype(int) - 1)
    )
    n_slots = int(df["seed_rank"].max()) + 1
    per_slot = {}
    taus, covs, risks, ns = [], [], [], []
    for r in range(n_slots):
        slot = df[df["seed_rank"] == r]
        deliv = deliver_top1(slot)
        g = fit_gate(deliv)
        per_slot[str(r)] = g
        taus.append(g["tau"])
        covs.append(g["coverage"])
        risks.append(g["selective_risk"])
        ns.append(g["n"])
    return {
        "n_seed_slots": n_slots,
        "per_seed_slot": per_slot,
        "tau_spread": _spread(taus),
        "coverage_spread": _spread(covs),
        "selective_risk_spread": _spread(risks),
        "delivery_size_spread": _spread(ns),
    }


def selection_sensitivity(df: pd.DataFrame, n_random: int = 200) -> dict:
    """Top-1-by-score delivery vs uniformly-random-sample-per-target delivery."""
    g = rng()
    top1 = fit_gate(deliver_top1(df))

    taus, covs, risks = [], [], []
    idx_by_key = {k: sub.index.to_numpy() for k, sub in df.groupby(KEY)}
    for _ in range(n_random):
        pick = [g.choice(ix) for ix in idx_by_key.values()]
        deliv = df.loc[pick]
        gate = fit_gate(deliv)
        taus.append(gate["tau"])
        covs.append(gate["coverage"])
        risks.append(gate["selective_risk"])

    return {
        "top1_by_score": top1,
        "random_sample_per_target": {
            "n_draws": n_random,
            "tau": _spread(taus),
            "coverage": _spread(covs),
            "selective_risk": _spread(risks),
        },
        "coverage_gain_top1_vs_random_mean": float(top1["coverage"] - np.nanmean(covs)),
    }


def run() -> dict:
    df = load_af3_samples()
    stab = seed_stability(df)
    sens = selection_sensitivity(df)
    return {
        "_meta": {
            "model": "af3",
            "alpha": ALPHA,
            "delta": DELTA,
            "conf": CONF,
            "n_poses": int(len(df)),
            "n_systems_ligands": int(df.groupby(KEY).ngroups),
            "note": "Certified gate fit on the full delivery (in-calibration operating "
                    "point). Seed slots rank the within-system seeds; each slot delivers "
                    "its own top-1-by-ranking_score pose.",
        },
        "seed_stability": stab,
        "selection_sensitivity": sens,
    }


def main() -> None:
    res = run()
    save_json(res, RESDIR / "e54_seed_stability.json")
    m = res["_meta"]
    print(f"E54 -- AF3 native-gate seed stability  (alpha={ALPHA}, delta={DELTA})")
    print(f"{m['n_poses']} poses over {m['n_systems_ligands']} (system,ligand) units\n")

    stab = res["seed_stability"]
    print(f"per-seed-slot certified gate ({stab['n_seed_slots']} slots):")
    print(f"  {'slot':>4} {'n':>5} {'tau':>7} {'coverage':>9} {'sel_risk':>9}")
    for r, g in stab["per_seed_slot"].items():
        tau = g["tau"] if g["tau"] is not None else float("nan")
        print(f"  {r:>4} {g['n']:>5} {tau:>7.3f} {g['coverage']:>9.3f} "
              f"{g['selective_risk']:>9.3f}")
    ts, cs, rs = stab["tau_spread"], stab["coverage_spread"], stab["selective_risk_spread"]
    print(f"\n  tau      across seeds: range={ts['range']:.3f}  std={ts['std']:.3f}")
    print(f"  coverage across seeds: range={cs['range']:.3f}  std={cs['std']:.3f}")
    print(f"  sel_risk across seeds: range={rs['range']:.3f}  std={rs['std']:.3f}")

    sens = res["selection_sensitivity"]
    t1 = sens["top1_by_score"]
    rnd = sens["random_sample_per_target"]
    print("\ntop-1-by-score vs random-sample-per-target delivery:")
    tau1 = t1["tau"] if t1["tau"] is not None else float("nan")
    print(f"  top-1 : tau={tau1:.3f}  coverage={t1['coverage']:.3f}  "
          f"sel_risk={t1['selective_risk']:.3f}")
    print(f"  random: tau={rnd['tau']['mean']:.3f}  coverage={rnd['coverage']['mean']:.3f}"
          f" (+/-{rnd['coverage']['std']:.3f})  sel_risk={rnd['selective_risk']['mean']:.3f}")
    print(f"  coverage gain (top-1 - random): "
          f"{sens['coverage_gain_top1_vs_random_mean']:+.3f}")


if __name__ == "__main__":
    main()
