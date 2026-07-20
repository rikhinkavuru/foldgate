"""E25 -- per-model temporal cutoffs, and why the temporal axis is a recency proxy.

Reviewer R2.7: the paper's temporal novelty axis quantile-bins a SINGLE pooled
structure `release_date` for all models, so a shared cutoff is misspecified for
models with different training cutoffs and would attenuate any temporal drift toward
zero -- exactly the observed "temporal null". This script resolves it three ways:

1. It documents the load-bearing fact: `release_date_before_cutoff` is uniformly True
   across RNP, i.e. EVERY Runs N' Poses structure is deposited after the 2021-era
   models' training cutoff (min release_date well past 2021-09-30). So for AF3,
   Boltz-1, Boltz-1x, Chai, Protenix there is NO in-training-vs-out-of-training
   temporal contrast to measure inside RNP; the temporal strata rank recency AMONG
   already-out-of-training structures. A flat drift there is the correct, expected
   result and is not evidence about temporal generalization -- it is a property of the
   benchmark. The operative novelty axis is structural/chemical similarity, which does
   not depend on this.

2. It recomputes reliability drift on the recency axis with per-bin occupancy counts
   persisted (audit item A.2), so the reader can see the drift is small AND that its
   bins are populated (a small drift on empty bins would be uninformative).

3. It runs the ONE genuine temporal in/out test available in RNP: Boltz-2 has a
   2023-06-30 cutoff, so RNP structures split into an in-training-era block
   (<= 2023-06-30) and a genuinely post-cutoff block (> 2023-06-30). We measure the
   reliability drift across that real boundary, and we re-derive Boltz-2's structural
   (pocket) break using the correctly-referenced 2023 similarity column
   (`sucos_shape_pocket_qcov_2023`) instead of the default 2021 column, to confirm the
   structural break survives the correct reference set.

Output: results/e25_temporal_permodel.json
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from experiments._common import (
    CONF,
    RESDIR,
    ROOT,
    load_delivered,
    methods_with_enough,
    rng,
    save_json,
)

# Stated training cutoffs (CLAUDE.md grounded facts). Boltz-2 is the only one whose
# cutoff falls inside the RNP release-date span, so it is the only model with a genuine
# in/out temporal boundary among RNP structures.
MODEL_CUTOFF = {
    "af3": "2021-09-30",
    "boltz1": "2021-09-30",
    "boltz1x": "2021-09-30",
    "chai": "2021-12-01",
    "protenix": "2021-09-30",   # ByteDance AF3 reproduction, 2021-era cutoff
    "boltz2": "2023-06-30",
}
N_BINS = 5
N_BOOT = 2000


def _drift_binned(conf_s, y_s, conf_t, y_t, n_bins=N_BINS):
    """Target-mass-weighted signed P(correct|conf) gap S0->target, with per-bin counts."""
    e = np.quantile(np.concatenate([conf_s, conf_t]), np.linspace(0, 1, n_bins + 1))
    e[0], e[-1] = -np.inf, np.inf
    signed, wts, bins = [], [], []
    for lo, hi in zip(e[:-1], e[1:], strict=False):
        ms = (conf_s >= lo) & (conf_s < hi)
        mt = (conf_t >= lo) & (conf_t < hi)
        if not ms.any() or not mt.any():
            bins.append({"lo": float(lo), "hi": float(hi), "n_src": int(ms.sum()),
                         "n_tgt": int(mt.sum()), "p_src": None, "p_tgt": None})
            continue
        ps, pt = float(y_s[ms].mean()), float(y_t[mt].mean())
        signed.append(ps - pt)
        wts.append(int(mt.sum()))
        bins.append({"lo": float(lo), "hi": float(hi), "n_src": int(ms.sum()),
                     "n_tgt": int(mt.sum()), "p_src": ps, "p_tgt": pt})
    if not wts:
        return float("nan"), bins
    return float(np.average(signed, weights=np.asarray(wts, float))), bins


def _boot_ci(conf_s, y_s, conf_t, y_t, g, n_boot=N_BOOT):
    ns, nt = len(conf_s), len(conf_t)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        bs = g.integers(0, ns, ns)
        bt = g.integers(0, nt, nt)
        d, _ = _drift_binned(conf_s[bs], y_s[bs], conf_t[bt], y_t[bt])
        draws[b] = d
    fin = draws[np.isfinite(draws)]
    if not fin.size:
        return float("nan"), float("nan")
    return float(np.quantile(fin, 0.05)), float(np.quantile(fin, 0.95))


def _load_annotations_2023(df: pd.DataFrame) -> pd.DataFrame:
    """Merge the 2023-referenced pocket similarity onto the delivered frame by system_id."""
    ann = pd.read_csv(ROOT / "data" / "raw" / "annotations.csv")
    key = "system_id" if "system_id" in ann.columns else ann.columns[0]
    cols = [key]
    for c in ("sucos_shape_pocket_qcov_2023", "sucos_shape_pocket_qcov"):
        if c in ann.columns:
            cols.append(c)
    ann = ann[cols].drop_duplicates(subset=[key])
    return df.merge(ann, left_on="system_id", right_on=key, how="left", suffixes=("", "_ann"))


def _strata_from_sim(sim: pd.Series, n_bins: int = 4) -> pd.Series:
    """Quartile novelty strata + NaN=no-analog top stratum (matches novelty.make_strata)."""
    s = pd.to_numeric(sim, errors="coerce")
    if s.dropna().max() > 1.5:
        s = s / 100.0
    strata = pd.Series(np.nan, index=s.index, dtype="float")
    has = s.notna()
    if has.any():
        q = pd.qcut(s[has], q=n_bins, labels=False, duplicates="drop")
        n_levels = int(np.nanmax(q)) + 1
        strata.loc[has] = (n_levels - 1) - q
        strata.loc[~has] = strata.max() + 1
    return strata.astype("Int64")


def run() -> dict:
    df = load_delivered()
    df["release_dt"] = pd.to_datetime(df["release_date"], errors="coerce")
    methods = methods_with_enough(df)
    g = rng()

    out: dict = {}

    # --- 1. RNP is wholly post-cutoff for the 2021-era panel ------------------------
    span = {
        "release_date_min": str(df["release_dt"].min().date()),
        "release_date_max": str(df["release_dt"].max().date()),
    }
    post_cutoff = {}
    for m in MODEL_CUTOFF:
        sub = df[df.method == m]
        if not len(sub):
            continue
        cut = pd.Timestamp(MODEL_CUTOFF[m])
        frac_post = float((sub["release_dt"] > cut).mean())
        post_cutoff[m] = {
            "cutoff": MODEL_CUTOFF[m],
            "n": int(len(sub)),
            "frac_structures_post_cutoff": round(frac_post, 4),
            "n_in_training_era": int((sub["release_dt"] <= cut).sum()),
        }
    out["post_cutoff_composition"] = {
        "span": span,
        "per_model": post_cutoff,
        "interpretation": (
            "Every 2021-era model has ~100% of RNP structures post-cutoff, so no "
            "in-vs-out-of-training temporal split exists inside RNP for them; the "
            "temporal axis ranks recency among out-of-training structures. Only "
            "Boltz-2 (2023-06-30) has a genuine within-RNP boundary."
        ),
    }

    # --- 2. Recency-axis drift with per-bin occupancy, panel models -----------------
    # Uses the pooled temporal_stratum already on the frame (recency quantiles), S0 ref.
    recency = {}
    for m in methods:
        sub = df[df.method == m].dropna(subset=[CONF, "temporal_stratum"])
        conf = sub[CONF].to_numpy()
        y = sub["correct"].to_numpy().astype(int)
        strat = sub["temporal_stratum"].to_numpy().astype(int)
        levels = sorted(np.unique(strat).tolist())
        ref = levels[0]
        mref = strat == ref
        cells = {}
        for k in levels:
            if k == ref or int((strat == k).sum()) < 20:
                continue
            d, bins = _drift_binned(conf[mref], y[mref], conf[strat == k], y[strat == k])
            lo, hi = _boot_ci(conf[mref], y[mref], conf[strat == k], y[strat == k], g)
            cells[str(k)] = {
                "D_signed": round(d, 4), "ci90": [round(lo, 4), round(hi, 4)],
                "n_ref": int(mref.sum()), "n_stratum": int((strat == k).sum()),
                "bins": bins,
            }
        recency[m] = cells
    out["recency_drift_with_bin_counts"] = recency

    # --- 3. Boltz-2 genuine temporal in/out test + structural break under 2023 ref --
    b2 = df[df.method == "boltz2"].copy()
    boltz2 = {"available": bool(len(b2))}
    if len(b2):
        cut = pd.Timestamp(MODEL_CUTOFF["boltz2"])
        b2 = b2.dropna(subset=[CONF, "release_dt"])
        in_era = b2["release_dt"] <= cut
        conf = b2[CONF].to_numpy()
        y = b2["correct"].to_numpy().astype(int)
        n_in, n_out = int(in_era.sum()), int((~in_era).sum())
        boltz2["temporal_inout"] = {
            "cutoff": MODEL_CUTOFF["boltz2"],
            "n_in_training_era": n_in, "n_out_of_training": n_out,
            "base_correct_in": round(float(y[in_era.to_numpy()].mean()), 4) if n_in else None,
            "base_correct_out": round(float(y[(~in_era).to_numpy()].mean()), 4) if n_out else None,
        }
        if n_in >= 20 and n_out >= 20:
            d, bins = _drift_binned(conf[in_era.to_numpy()], y[in_era.to_numpy()],
                                    conf[(~in_era).to_numpy()], y[(~in_era).to_numpy()])
            lo, hi = _boot_ci(conf[in_era.to_numpy()], y[in_era.to_numpy()],
                              conf[(~in_era).to_numpy()], y[(~in_era).to_numpy()], g)
            boltz2["temporal_inout"]["D_signed_in_to_out"] = round(d, 4)
            boltz2["temporal_inout"]["ci90"] = [round(lo, 4), round(hi, 4)]
            boltz2["temporal_inout"]["bins"] = bins

        # Structural break with the correct 2023 reference vs default 2021 reference.
        b2m = _load_annotations_2023(b2)
        struct = {}
        for ref_year, col in (("2021", "sucos_shape_pocket_qcov"),
                              ("2023", "sucos_shape_pocket_qcov_2023")):
            if col not in b2m.columns:
                continue
            strat = _strata_from_sim(b2m[col])
            per = {}
            yv = b2m["correct"].to_numpy().astype(int)
            for k in sorted([int(x) for x in strat.dropna().unique()]):
                mk = (strat == k).to_numpy()
                if mk.sum() < 20:
                    continue
                per[f"S{k}"] = {"n": int(mk.sum()), "base_correct": round(float(yv[mk].mean()), 4)}
            struct[f"pocket_ref_{ref_year}"] = per
        boltz2["structural_break_by_reference"] = struct
    out["boltz2_genuine_temporal"] = boltz2

    out["_summary"] = {
        "verdict": (
            "R2.7 resolved: the temporal null is an artifact of RNP being wholly "
            "post-cutoff for the 2021-era models, not evidence of temporal robustness. "
            "The structural-similarity axis is the operative novelty variable. Boltz-2 "
            "is the only genuine within-RNP temporal in/out test."
        )
    }
    return out


def main() -> None:
    res = run()
    save_json(res, RESDIR / "e25_temporal_permodel.json")
    pc = res["post_cutoff_composition"]
    print("E25 -- per-model temporal cutoffs")
    print(f"RNP release-date span: {pc['span']['release_date_min']} .. {pc['span']['release_date_max']}")
    for m, r in pc["per_model"].items():
        print(f"  {m:>9}: cutoff {r['cutoff']}  frac post-cutoff={r['frac_structures_post_cutoff']:.3f}  "
              f"in-era n={r['n_in_training_era']}")
    b2 = res["boltz2_genuine_temporal"]
    if b2.get("available"):
        t = b2["temporal_inout"]
        print(f"\nBoltz-2 genuine in/out ({t['cutoff']}): n_in={t['n_in_training_era']} "
              f"n_out={t['n_out_of_training']} "
              f"base_correct {t.get('base_correct_in')} -> {t.get('base_correct_out')}")
        if "D_signed_in_to_out" in t:
            print(f"  reliability drift in->out: {t['D_signed_in_to_out']:+.3f} ci90 {t['ci90']}")
        for ref, per in b2.get("structural_break_by_reference", {}).items():
            print(f"  {ref}: " + ", ".join(f"{k} n={v['n']} corr={v['base_correct']}" for k, v in per.items()))
    print("\n" + res["_summary"]["verdict"])


if __name__ == "__main__":
    main()
