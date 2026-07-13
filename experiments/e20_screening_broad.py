"""E20 -- broadened, reviewer-proof selective virtual screening.

Hardens the E16 flagship for the journal version:
* three co-folding models where the released screen provides them (Boltz-2, Protenix, AF3),
  each with its own screening score and pose-reliability signal;
* EF@1 percent and BEDROC aggregated over targets with a target-level bootstrap CI, beside a
  Gnina docking baseline (the native threshold to beat);
* a PRE-REGISTERED abstention gate: the pose-reliability threshold is calibrated on Runs N'
  Poses pose correctness by LTT (accepted RNP poses have error <= alpha), frozen per model, and
  applied to the screen without ever seeing the screen labels, so no one can claim the gate was
  tuned on the enrichment;
* a self-computed chemical-novelty shift curve: actives are binned by their Tanimoto similarity
  to the training set (shipped per-active), and EF is recomputed within each bin against all
  decoys, so the direction of the shift is under our control.

The gate is a heuristic transfer (calibrated on pose correctness, applied to activity), which we
state rather than dress up as a guarantee.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

from experiments._common import RESDIR, load_delivered, rng, save_json
from foldgate.conformal import ltt_threshold
from foldgate.selective.enrichment import bedroc, enrichment_factor, random_abstention_ef, roc_auc

SCREEN_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "external", "screening")
DATASETS = {"dekois": "dekois_scores", "lipcba": "lipcba_scores", "gpcr": "gpcr_scores"}
FRAC = 0.01
# per screen model: (score file, per-compound ranker column, reliability column, RNP method key)
MODELS = {
    "boltz2": ("boltz_scores.csv", "affinity_probability_binary", "iptm", "boltz2"),
    "protenix": ("pix_scores.csv", "ranking_score", "iptm", "protenix"),
    "af3": ("af3_scores.csv", "ranking_score", "iptm", "af3"),
}


def rnp_iptm_tau(alpha=0.20, delta=0.10) -> dict:
    """Pre-registered per-model pose-reliability thresholds: LTT on RNP iface_iptm vs correctness."""
    d = load_delivered()
    out = {}
    for m in ("boltz2", "protenix", "af3", "boltz1", "boltz1x", "chai"):
        s = d[d.method == m].dropna(subset=["iface_iptm", "correct"])
        if len(s) >= 100:
            out[m] = ltt_threshold(s["iface_iptm"].to_numpy(), s["correct"].to_numpy().astype(int), alpha, delta)
    return out


def _per_compound(path, ranker, reliab):
    """Load a score file and reduce to one row per compound (best pose by ranker)."""
    if not os.path.exists(path):
        return None
    d = pd.read_csv(path)
    idc = "lid" if "lid" in d.columns else ("ID" if "ID" in d.columns else None)
    if idc is None or "label" not in d.columns or ranker not in d.columns:
        return None
    keep = [idc, "label", ranker] + ([reliab] if reliab in d.columns else [])
    d = d[keep].rename(columns={idc: "lid"}).dropna(subset=[ranker])
    agg = {"label": ("label", "max"), ranker: (ranker, "max")}
    if reliab in d.columns:
        agg[reliab] = (reliab, "max")
    d = d.groupby("lid", as_index=False).agg(**agg)
    if d["label"].sum() < 3 or (d["label"] == 0).sum() < 20:
        return None
    return d


def _dock(tdir):
    p = os.path.join(tdir, "gnina_scores.csv")
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p)
    idc = "ID" if "ID" in d.columns else ("lid" if "lid" in d.columns else None)
    if idc and "cnnscore" in d.columns:
        return d[[idc, "cnnscore"]].rename(columns={idc: "lid"}).groupby("lid", as_index=False).max()
    return None


def _target_row(df, ranker, reliab, tau, dock, g):
    y = df["label"].to_numpy().astype(int)
    sc = df[ranker].to_numpy()
    n, n_act = len(y), int(y.sum())
    row = {"ef": enrichment_factor(sc, y, FRAC), "bedroc": bedroc(sc, y), "auc": roc_auc(sc, y)}
    if dock is not None:
        dj = df.merge(dock, on="lid", how="left")
        dv = dj["cnnscore"].to_numpy()
        row["ef_dock"] = enrichment_factor(dv, y, FRAC) if np.isfinite(dv).any() else float("nan")
    else:
        row["ef_dock"] = float("nan")
    # pre-registered RNP-calibrated ipTM gate applied to the screen (fixed top-k denominator)
    if reliab in df.columns and tau is not None:
        rel = df[reliab].to_numpy()
        keep = rel >= tau
        k = max(1, int(round(FRAC * n)))
        if keep.sum() >= k and n_act:
            sk, yk = sc[keep], y[keep]
            top = np.argsort(-sk)[:k]
            row["ef_reg_gate"] = float((yk[top].sum() / k) / (n_act / n))
            row["cov_reg_gate"] = float(keep.mean())
        else:
            row["ef_reg_gate"], row["cov_reg_gate"] = float("nan"), float(keep.mean())
        # selective at 50% by reliability vs random control
        rorder = np.argsort(-rel)
        keep50 = rorder[: max(1, n // 2)]
        sk, yk = sc[keep50], y[keep50]
        k = max(1, int(round(FRAC * n)))
        top = np.argsort(-sk)[:k]
        row["ef_sel50"] = float((yk[top].sum() / k) / (n_act / n)) if n_act else float("nan")
        rmean, _, rhi = random_abstention_ef(sc, y, 0.5, FRAC)
        row["ef_rand50"] = rmean
        row["beats_rand"] = 1.0 if np.isfinite(row["ef_sel50"]) and row["ef_sel50"] > rhi else 0.0
    return row


def _boot(vals, g, n=2000):
    v = np.asarray([x for x in vals if np.isfinite(x)], float)
    if not len(v):
        return {"median": float("nan"), "ci90": [float("nan"), float("nan")], "n": 0}
    meds = [np.median(v[g.integers(0, len(v), len(v))]) for _ in range(n)]
    return {"median": float(np.median(v)),
            "ci90": [float(np.quantile(meds, 0.05)), float(np.quantile(meds, 0.95))], "n": int(len(v))}


def _shift_curve(dset_dir, targets, ranker, reliab):
    """EF@1% within active-to-train similarity bins (actives binned, all decoys kept)."""
    bins = [(-0.01, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.01)]
    acc = {f"{lo}-{hi}": [] for lo, hi in bins}
    for t in targets:
        tdir = os.path.join(dset_dir, t)
        sim_p = os.path.join(tdir, "actives_similarity.csv")
        mf = os.path.join(tdir, "boltz_scores.csv")
        if not (os.path.exists(sim_p) and os.path.exists(mf)):
            continue
        sim = pd.read_csv(sim_p)
        simcol = "ec_sim" if "ec_sim" in sim.columns else None
        idc = "ID" if "ID" in sim.columns else ("lid" if "lid" in sim.columns else None)
        if not simcol or not idc:
            continue
        df = _per_compound(mf, ranker, reliab)
        if df is None:
            continue
        y = df["label"].to_numpy().astype(int)
        sc = df[ranker].to_numpy()
        simmap = dict(zip(sim[idc].astype(str), sim[simcol], strict=False))
        act_sim = df["lid"].astype(str).map(simmap)
        for lo, hi in bins:
            in_bin = (df["label"] == 1) & (act_sim > lo) & (act_sim <= hi)
            if in_bin.sum() < 2:
                continue
            mask = (df["label"] == 0) | in_bin.to_numpy()
            yb, sb = y[mask], sc[mask]
            nb, nab = len(yb), int(yb.sum())
            k = max(1, int(round(FRAC * nb)))
            top = np.argsort(-sb)[:k]
            acc[f"{lo}-{hi}"].append(float((yb[top].sum() / k) / (nab / nb)) if nab else float("nan"))
    return {b: (float(np.nanmedian(v)) if v else float("nan"), len([x for x in v if np.isfinite(x)]))
            for b, v in acc.items()}


def run() -> dict:
    g = rng()
    taus = rnp_iptm_tau()
    out = {"rnp_iptm_tau": taus, "datasets": {}}
    for name, sub in DATASETS.items():
        dset_dir = os.path.join(SCREEN_DIR, sub)
        if not os.path.isdir(dset_dir):
            continue
        targets = sorted(os.path.basename(t) for t in glob.glob(os.path.join(dset_dir, "*")) if os.path.isdir(t))
        out["datasets"][name] = {"n_targets": len(targets), "models": {}}
        for mk, (fname, ranker, reliab, rnp_key) in MODELS.items():
            tau = taus.get(rnp_key)
            rows = []
            for t in targets:
                tdir = os.path.join(dset_dir, t)
                df = _per_compound(os.path.join(tdir, fname), ranker, reliab)
                if df is not None:
                    rows.append(_target_row(df, ranker, reliab, tau, _dock(tdir), g))
            if not rows:
                continue
            keys = ["ef", "bedroc", "auc", "ef_dock", "ef_reg_gate", "cov_reg_gate", "ef_sel50", "ef_rand50"]
            agg = {k: _boot([r.get(k, float("nan")) for r in rows], g) for k in keys}
            agg["frac_beats_random"] = float(np.mean([r.get("beats_rand", 0.0) for r in rows]))
            agg["rnp_iptm_tau"] = tau
            out["datasets"][name]["models"][mk] = {"n_targets_scored": len(rows), "agg": agg}
        # self-computed shift curve (Boltz-2 affinity ranker), where actives_similarity ships
        out["datasets"][name]["shift_boltz2_affinity"] = _shift_curve(
            dset_dir, targets, "affinity_probability_binary", "iptm")
    return out


def main() -> None:
    res = run()
    save_json(res, RESDIR / "e20_screening_broad.json")
    print("E20 -- broadened selective screening (multi-model, CIs, pre-registered gate)\n")
    print("RNP-calibrated ipTM reliability tau:", {k: round(v, 3) for k, v in res["rnp_iptm_tau"].items() if v})
    for name, r in res["datasets"].items():
        print(f"\n[{name}] targets={r['n_targets']}")
        for mk, mr in r["models"].items():
            a = mr["agg"]
            print(f"  {mk:8s}: EF@1% {a['ef']['median']:.2f} {a['ef']['ci90']} | BEDROC {a['bedroc']['median']:.3f} "
                  f"| dock {a['ef_dock']['median']:.2f} | reg-gate EF {a['ef_reg_gate']['median']:.2f} "
                  f"(cov {a['cov_reg_gate']['median']:.2f}) | sel50 {a['ef_sel50']['median']:.2f} vs rand "
                  f"{a['ef_rand50']['median']:.2f} beats {a['frac_beats_random']:.0%}")
        sc = r["shift_boltz2_affinity"]
        if any(np.isfinite(v[0]) for v in sc.values()):
            print("  shift (active-train-sim bin -> median EF@1%): " +
                  ", ".join(f"{b}:{v[0]:.1f}(n{v[1]})" for b, v in sc.items()))


if __name__ == "__main__":
    main()
