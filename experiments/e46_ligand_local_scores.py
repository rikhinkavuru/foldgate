"""E46 (reviewer D1): gate on ligand-LOCAL confidence, not global ranking_score / interface ipTM.

Extracts per-pose ligand-local confidence from the released co-folding CIFs (per-atom ligand
pLDDT lives in the B-factor column, 0-100) and tests whether the feasibility-frontier impossibility
(d2) holds on these scores rather than on ranking_score.

Ligand-local scores per DELIVERED (top-1) pose:
  * mean ligand-atom pLDDT
  * min  ligand-atom pLDDT
  * worst-quartile ligand pLDDT (mean of the lowest-25% ligand atoms)
Higher = more confident, same direction as ranking_score, so the deployed rule stays "accept iff s >= tau".

PL-PAE (min/mean receptor-residue x ligand-atom PAE) is NOT extractable: the RNP structure dump
(prediction_files.tar.gz) ships only .cif + ranking_scores.csv per (model, system) -- no NPZ/PAE
arrays. This is verified against the tarball member listing; reported as an extraction gap.

Streaming: reads data/processed/delivered_poses.tar.gz (1.3 GB, 12,597 delivered CIFs = the top-1
argmax-ranking_score pose per (model, system), same selection rule as target_level), one member at
a time. Ligand-chain selection mirrors foldgate.features.pose_agreement.select_ligand exactly
(pick the non-amino-acid chain whose heavy-atom count == delivered ligand_num_heavy_atoms).

Then, per (model, ligand/pocket-novelty stratum), reuses d2_feasibility_map's frontier logic to
report c*_feasible and count zero-frontier non-reference cells, comparing each ligand-local score
to ranking_score. Finally recomputes reliability drift D(nu) (e12) on the best ligand-local score.
"""

from __future__ import annotations

import sys
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from experiments._common import DELTA, RESDIR, load_delivered, methods_with_enough, rng, save_json  # noqa: E402
from experiments.d2_feasibility_map import (  # noqa: E402
    COVERAGE_GRID,
    MIN_ACCEPT,
    SOURCE_STRATUM,
    _cell,
    _frontier,
    target_level,
)

DELIVERED_TAR = ROOT / "data" / "processed" / "delivered_poses.tar.gz"
EXTRACT_CACHE = ROOT / "data" / "processed" / "ligand_local_plddt.parquet"
ALPHAS = (0.10, 0.20)
LOCAL_SCORES = ["ligand_plddt_mean", "ligand_plddt_min", "ligand_plddt_wq"]
AXES = {"ligand": "novelty_stratum", "pocket": "pocket_novelty_stratum"}

# ---------------------------------------------------------------------------
# 1. extraction: per-atom ligand pLDDT from the delivered CIF B-factor column
# ---------------------------------------------------------------------------


def _ligand_plddt(cif_text: str, expected_heavy: int | None):
    """Return per-heavy-atom ligand pLDDT array for the delivered ligand chain, or None.

    Mirrors pose_agreement.select_ligand: heavy-atom count of the chosen non-amino-acid chain
    must equal ``expected_heavy`` (predicted CIFs carry no crystallographic H; a few models write
    explicit H which we drop). B-factor column holds per-atom pLDDT (0-100).
    """
    import gemmi

    doc = gemmi.cif.read_string(cif_text)
    st = gemmi.make_structure_from_block(doc.sole_block())
    m = st[0]
    ligs: dict[str, tuple[np.ndarray, np.ndarray]] = {}  # chain -> (elements, plddt) heavy only
    for ch in m:
        is_aa = any(
            (gemmi.find_tabulated_residue(r.name) is not None
             and gemmi.find_tabulated_residue(r.name).is_amino_acid())
            for r in ch
        )
        if is_aa:
            continue
        el, b = [], []
        for r in ch:
            for a in r:
                el.append(a.element.atomic_number)
                b.append(a.b_iso)
        if not el:
            continue
        el = np.asarray(el, int)
        b = np.asarray(b, float)
        keep = el != 1
        ligs[ch.name] = (el[keep], b[keep])
    if not ligs:
        return None
    if expected_heavy is None:
        if len(ligs) != 1:
            return None
        return next(iter(ligs.values()))[1]
    for name in sorted(ligs):
        el_h, b_h = ligs[name]
        if int(len(el_h)) == int(expected_heavy):
            return b_h
    return None


def _summ(plddt: np.ndarray) -> dict:
    """mean / min / worst-quartile (mean of atoms at or below the 25th pct) ligand pLDDT."""
    p = np.asarray(plddt, float)
    p = p[np.isfinite(p)]
    if p.size == 0:
        return {"ligand_plddt_mean": np.nan, "ligand_plddt_min": np.nan,
                "ligand_plddt_wq": np.nan, "n_ligand_atoms": 0}
    q25 = np.quantile(p, 0.25)
    wq = p[p <= q25]
    return {
        "ligand_plddt_mean": float(p.mean()),
        "ligand_plddt_min": float(p.min()),
        "ligand_plddt_wq": float(wq.mean()) if wq.size else float(p.min()),
        "n_ligand_atoms": int(p.size),
    }


def extract(heavy_lookup: dict, log_every: int = 2000) -> pd.DataFrame:
    if EXTRACT_CACHE.exists():
        print(f"loading cached extraction {EXTRACT_CACHE}", flush=True)
        return pd.read_parquet(EXTRACT_CACHE)
    if not DELIVERED_TAR.exists():
        raise SystemExit(f"missing {DELIVERED_TAR} (run experiments/d1_extract_delivered.py)")
    rows = []
    n = n_ok = 0
    with tarfile.open(DELIVERED_TAR, "r:gz") as t:
        for m in t:                                        # one member at a time (memory-safe)
            if not m.isfile() or not m.name.endswith(".cif"):
                continue
            method = m.name.split("/")[0]
            system = m.name.split("/")[-1][:-4]
            n += 1
            exp = heavy_lookup.get((system, method))
            try:
                plddt = _ligand_plddt(t.extractfile(m).read().decode(), exp)
            except Exception:                              # noqa: BLE001 - a malformed CIF must not kill the run
                plddt = None
            row = {"system_id": system, "method": method}
            if plddt is not None:
                row.update(_summ(plddt))
                n_ok += 1
            else:
                row.update({"ligand_plddt_mean": np.nan, "ligand_plddt_min": np.nan,
                            "ligand_plddt_wq": np.nan, "n_ligand_atoms": 0})
            rows.append(row)
            if n % log_every == 0:
                print(f"  {n} CIFs, {n_ok} ligand-pLDDT extracted", flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(EXTRACT_CACHE, index=False)
    print(f"extracted {n_ok}/{n} CIFs -> {EXTRACT_CACHE}", flush=True)
    return df


# ---------------------------------------------------------------------------
# 2. feasibility frontier per score (mirrors d2_feasibility_map, parametrized by score column)
# ---------------------------------------------------------------------------


def frontier_for_score(tl: pd.DataFrame, score_col: str, methods, alphas=ALPHAS,
                       delta: float = DELTA, seed: int = 20260715) -> dict:
    """Per (axis, model, stratum) c*_feasible using ``score_col`` as the gate. Same conventions
    as d2: tau set on the source (S0) calibration half, deployed frozen to every stratum."""
    g_rng = rng(seed)
    out = {"score": score_col, "axes": {}}
    for axis, col in AXES.items():
        axis_out = {"models": {}}
        for m in methods:
            d = tl[(tl.method == m) & tl[col].notna() & tl[score_col].notna()].copy()
            if len(d) == 0:
                continue
            d["loss"] = (1 - d["correct"].astype(int)).astype(float)
            src_all = d[d[col] == SOURCE_STRATUM]
            sys_ids = np.array(sorted(src_all.system_id.unique()))
            g_rng.shuffle(sys_ids)
            cal_ids = set(sys_ids[: len(sys_ids) // 2])
            src_cal = src_all[src_all.system_id.isin(cal_ids)]
            src_eval = src_all[~src_all.system_id.isin(cal_ids)]
            if len(src_cal) < MIN_ACCEPT:
                continue
            frames = {int(gg): (src_eval if gg == SOURCE_STRATUM else d[d[col] == gg])
                      for gg in sorted(d[col].unique())}
            taus = {float(c): float(np.quantile(src_cal[score_col].to_numpy(), 1.0 - c))
                    for c in COVERAGE_GRID}
            model_out = {"n_source_cal": int(len(src_cal)), "alpha": {}}
            for alpha in alphas:
                strata = {}
                for gg, dg in frames.items():
                    if len(dg) == 0:
                        continue
                    cells = {}
                    for c, tau in taus.items():
                        acc = dg[dg[score_col] >= tau]
                        cells[c] = _cell(acc["loss"].to_numpy(), len(dg), alpha, delta)
                    strata[gg] = {
                        "n_targets_stratum": int(len(dg)),
                        "c_star_feasible": _frontier(cells, "feasible"),
                        "c_star_certified": _frontier(cells, "certified"),
                    }
                model_out["alpha"][str(alpha)] = {"strata": strata}
            axis_out["models"][m] = model_out
        out["axes"][axis] = axis_out
    return out


def count_zero_frontier(front: dict, alpha: float) -> dict:
    """Count zero c*_feasible non-reference cells (both axes pooled), matching d2's 40-cell family."""
    zero = tot = 0
    per_axis = {}
    for axis, ao in front["axes"].items():
        az = at = 0
        for m, mo in ao["models"].items():
            strata = mo["alpha"][str(alpha)]["strata"]
            for gg, s in strata.items():
                if int(gg) == SOURCE_STRATUM:
                    continue
                at += 1
                if s["c_star_feasible"] == 0.0:
                    az += 1
        per_axis[axis] = {"n_nonref_cells": at, "n_zero_frontier": az}
        zero += az
        tot += at
    return {"n_nonref_cells": tot, "n_zero_frontier": zero, "per_axis": per_axis}


def ligand_frontier_table(front: dict, alpha: float) -> dict:
    """Per (model, ligand-novelty stratum) c*_feasible, the reviewer's requested table."""
    ao = front["axes"]["ligand"]
    tbl = {}
    for m, mo in ao["models"].items():
        strata = mo["alpha"][str(alpha)]["strata"]
        tbl_m = {}
        for gg, s in strata.items():
            tbl_m[f"S{int(gg)}"] = {
                "c_star_feasible": s["c_star_feasible"],
                "n": s["n_targets_stratum"],
            }
        tbl[m] = tbl_m
    return tbl


# ---------------------------------------------------------------------------
# 3. reliability drift D(nu) on the best ligand-local score (mirrors e12)
# ---------------------------------------------------------------------------


def _edges(a, b, n_bins=5):
    e = np.quantile(np.concatenate([a, b]), np.linspace(0, 1, n_bins + 1))
    e[0], e[-1] = -np.inf, np.inf
    return e


def _drift(conf_s, y_s, conf_t, y_t, edges):
    signed, absg, wts = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:], strict=False):
        ms = (conf_s >= lo) & (conf_s < hi)
        mt = (conf_t >= lo) & (conf_t < hi)
        if not ms.any() or not mt.any():
            continue
        ps, pt = float(y_s[ms].mean()), float(y_t[mt].mean())
        signed.append(ps - pt)
        absg.append(abs(ps - pt))
        wts.append(int(mt.sum()))
    if not wts:
        return float("nan"), float("nan")
    w = np.asarray(wts, float)
    return float(np.average(signed, weights=w)), float(np.average(absg, weights=w))


def drift_on_score(tl: pd.DataFrame, score_col: str, methods, axis_col="novelty_stratum",
                   n_boot=2000, seed=20260710) -> dict:
    g = rng(seed)
    out = {}
    for m in methods:
        s = tl[(tl.method == m)].dropna(subset=[score_col, axis_col])
        conf = s[score_col].to_numpy()
        y = s["correct"].to_numpy().astype(int)
        strat = s[axis_col].to_numpy().astype(int)
        levels = sorted(np.unique(strat).tolist())
        ref = levels[0]
        m_ref = strat == ref
        ks = [k for k in levels if k != ref and int((strat == k).sum()) >= 20]
        res = {}
        edges_by_k = {k: _edges(conf[m_ref], conf[strat == k]) for k in ks}
        for k in ks:
            d_s, d_a = _drift(conf[m_ref], y[m_ref], conf[strat == k], y[strat == k], edges_by_k[k])
            res[k] = {"D_signed": d_s, "D_abs": d_a,
                      "n_ref": int(m_ref.sum()), "n_stratum": int((strat == k).sum())}
        # shared-resample bootstrap CI (one index per rep drives all k, as in e12)
        n = len(conf)
        boot = {k: np.empty(n_boot) for k in ks}
        for b in range(n_boot):
            bi = g.integers(0, n, n)
            cb, yb, stb = conf[bi], y[bi], strat[bi]
            rb = stb == ref
            cref, yref = cb[rb], yb[rb]
            for k in ks:
                km = stb == k
                ss, _ = _drift(cref, yref, cb[km], yb[km], edges_by_k[k])
                boot[k][b] = ss
        for k in ks:
            fin = boot[k][np.isfinite(boot[k])]
            res[k]["D_signed_ci90"] = [
                float(np.quantile(fin, 0.05)) if fin.size else float("nan"),
                float(np.quantile(fin, 0.95)) if fin.size else float("nan"),
            ]
        out[m] = res
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def run() -> dict:
    df = load_delivered()
    methods = methods_with_enough(df)
    tl = target_level(df)                                  # one delivered row per (system, method)
    heavy_lookup = {(r.system_id, r.method): (int(r.ligand_num_heavy_atoms)
                    if pd.notna(r.ligand_num_heavy_atoms) else None)
                    for r in tl.itertuples()}

    ext = extract(heavy_lookup)
    tl = tl.merge(ext[["system_id", "method"] + LOCAL_SCORES + ["n_ligand_atoms"]],
                  on=["system_id", "method"], how="left")

    # extraction coverage per model (fraction of delivered target rows with a ligand pLDDT)
    cov = {}
    for m in methods:
        sub = tl[tl.method == m]
        cov[m] = {
            "n_target_rows": int(len(sub)),
            "n_ligand_plddt": int(sub["ligand_plddt_mean"].notna().sum()),
            "frac": float(sub["ligand_plddt_mean"].notna().mean()),
            "median_ligand_atoms": float(sub["n_ligand_atoms"].replace(0, np.nan).median()),
            # per-atom resolution: some models (Boltz-1) write one constant per ligand residue
            "frac_min_eq_mean": float((np.isclose(sub["ligand_plddt_min"], sub["ligand_plddt_mean"])
                                       & sub["ligand_plddt_mean"].notna()).sum()
                                      / max(sub["ligand_plddt_mean"].notna().sum(), 1)),
        }

    # discrimination AUC (why feasibility moves): does the score rank correct>incorrect within stratum?
    from sklearn.metrics import roc_auc_score
    scores = ["ranking_score"] + LOCAL_SCORES

    def _auc(sub, sc):
        d = sub.dropna(subset=[sc, "correct"])
        if d["correct"].nunique() < 2 or len(d) < 20:
            return float("nan")
        return float(roc_auc_score(d["correct"], d[sc]))

    auc = {}
    for m in methods:
        sub = tl[tl.method == m]
        novel = sub[sub.novelty_stratum.isin([3, 4])]
        auc[m] = {sc: {"all": _auc(sub, sc), "ligand_novel_S3S4": _auc(novel, sc)} for sc in scores}

    # frontier per score, incl. ranking_score baseline recomputed with identical settings
    frontiers = {sc: frontier_for_score(tl, sc, methods) for sc in scores}

    zero_counts = {}
    ligand_tables = {}
    for sc in scores:
        zero_counts[sc] = {str(a): count_zero_frontier(frontiers[sc], a) for a in ALPHAS}
        ligand_tables[sc] = {str(a): ligand_frontier_table(frontiers[sc], a) for a in ALPHAS}

    # best ligand-local score: the one whose non-reference zero-frontier count (both axes, alpha=0.20)
    # is largest, i.e. strengthens the impossibility most.
    best = max(LOCAL_SCORES, key=lambda s: zero_counts[s]["0.2"]["n_zero_frontier"])
    drift_best = drift_on_score(tl, best, methods, axis_col="novelty_stratum")
    drift_ranking = drift_on_score(tl, "ranking_score", methods, axis_col="novelty_stratum")

    return {
        "note": ("Ligand-local confidence = per-atom ligand pLDDT from CIF B-factors. PL-PAE not "
                 "extractable: RNP prediction_files.tar.gz ships only .cif + ranking_scores.csv "
                 "(no NPZ/PAE arrays), verified from the tarball member listing."),
        "alphas": list(ALPHAS),
        "min_accept": MIN_ACCEPT,
        "source_stratum": SOURCE_STRATUM,
        "n_target_rows": int(len(tl)),
        "methods": methods,
        "extraction_coverage": cov,
        "discrimination_auc": auc,
        "zero_frontier_counts": zero_counts,
        "ligand_novelty_frontier": ligand_tables,
        "best_ligand_local_score": best,
        "drift_best_ligand_local": drift_best,
        "drift_ranking_score": drift_ranking,
        "pl_pae": {"available": False,
                   "reason": "RNP structure dump has no NPZ/PAE arrays (only .cif + ranking_scores.csv)"},
    }


def main() -> None:
    res = run()
    save_json(res, RESDIR / "e46_ligand_local_scores.json")
    print("\n=== E46 ligand-local scores ===")
    print(res["note"], "\n")
    print("extraction coverage per model (frac delivered rows with ligand pLDDT | median atoms | frac min==mean):")
    for m, c in res["extraction_coverage"].items():
        print(f"  {m:>9}  {c['n_ligand_plddt']:>5}/{c['n_target_rows']:<5} = {c['frac']:.2f}  "
              f"| atoms~{c['median_ligand_atoms']:.0f} | min==mean {c['frac_min_eq_mean']:.2f}")
    print("\nzero-frontier NON-REFERENCE cells (both axes pooled; d2's 40-cell family):")
    for a in [str(x) for x in res["alphas"]]:
        print(f"  alpha={a}")
        for sc in ["ranking_score"] + LOCAL_SCORES:
            z = res["zero_frontier_counts"][sc][a]
            lig = z["per_axis"]["ligand"]; poc = z["per_axis"]["pocket"]
            print(f"    {sc:>20}: {z['n_zero_frontier']:>2}/{z['n_nonref_cells']:<2}  "
                  f"(ligand {lig['n_zero_frontier']}/{lig['n_nonref_cells']}, "
                  f"pocket {poc['n_zero_frontier']}/{poc['n_nonref_cells']})")
    print(f"\nbest ligand-local score = {res['best_ligand_local_score']}")
    print("reliability drift D_signed(nu) on ligand-novelty axis (S0 reference):")
    for m, r in res["drift_best_ligand_local"].items():
        parts = []
        for k in sorted(r):
            lo, hi = r[k]["D_signed_ci90"]
            parts.append(f"S{k}={r[k]['D_signed']:+.3f}[{lo:+.2f},{hi:+.2f}]")
        print(f"  {m:>9}  " + "  ".join(parts))
    print(f"\nwrote {RESDIR / 'e46_ligand_local_scores.json'}")


if __name__ == "__main__":
    main()
