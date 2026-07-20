"""E40 -- dataset composition (reviewer R2.6) + drift-bin occupancy (appendix A.2).

Two reviewer-facing sanity artifacts, both descriptive, both CPU-fast, no torch.

Part A (R2.6): what is actually IN the benchmark. Over the 2,425 unique systems
(one pose per system_id), report the ligand physicochemical distributions
(molecular weight, rotatable bonds, heavy atoms), a fragment/drug-like/large
decomposition by MW, the target-class composition via the PDB header keyword
(`entry_keywords` in annotations.csv, the only class-like column shipped), a
distinct-receptor proxy for diversity, and a rough cofactor/ion/peptide vs
drug-like split from a stated heavy-atom / rotatable-bond heuristic.

Part B (A.2): the reliability-drift figure quotes up to +0.63 (Protenix pocket S3).
Drift bins on the confidence score and compares P(correct|conf) across strata; on
novel strata the high-confidence bins can be nearly empty, making the headline
number fragile. For Protenix and AF3 on the POCKET axis, using 5 quantile bins of
ranking_score SHARED between S0 (familiar) and S3 (novel), we report the per-bin
count and P(correct) in each stratum and flag any bin under 10 targets in either.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from experiments._common import CONF, RESDIR, ROOT, load_delivered

# Part B knobs
POCKET_COL = "pocket_novelty_stratum"
S_FAMILIAR = 0
S_NOVEL = 3
N_BINS = 5
THIN_BIN = 10          # flag bins under this count in either stratum
DRIFT_METHODS = ["protenix", "af3"]

# Part A ligand-size buckets (MW, Daltons)
MW_FRAGMENT = 300.0    # < this reads as a small fragment
MW_DRUGLIKE = 500.0    # 300-500 is the classic drug-like window; > this is "large"

# Part A cofactor/ion/peptide heuristic (stated, rough)
ION_MAX_HEAVY = 3      # <= this many heavy atoms reads as an ion / tiny cofactor fragment
PEPTIDE_MIN_ROT = 15   # >= this many rotatable bonds reads as a peptide / very flexible chain


def _num_summary(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0}
    return {
        "n": int(x.size),
        "min": float(np.min(x)),
        "q25": float(np.quantile(x, 0.25)),
        "median": float(np.median(x)),
        "mean": float(np.mean(x)),
        "q75": float(np.quantile(x, 0.75)),
        "max": float(np.max(x)),
    }


def part_a(df: pd.DataFrame) -> dict:
    # One pose per system_id (the union: physicochemistry is a system-level property).
    uni = df.drop_duplicates(subset="system_id").reset_index(drop=True)
    n_sys = int(len(uni))

    mw = uni["ligand_molecular_weight"].to_numpy(dtype=float)
    rot = uni["ligand_num_rot_bonds"].to_numpy(dtype=float)
    heavy = uni["ligand_num_heavy_atoms"].to_numpy(dtype=float)

    # MW size decomposition
    mw_fin = mw[np.isfinite(mw)]
    n_fragment = int(np.count_nonzero(mw_fin < MW_FRAGMENT))
    n_druglike = int(np.count_nonzero((mw_fin >= MW_FRAGMENT) & (mw_fin <= MW_DRUGLIKE)))
    n_large = int(np.count_nonzero(mw_fin > MW_DRUGLIKE))

    # Cofactor/ion vs peptide vs drug-like (mutually exclusive, ion takes precedence)
    is_ion = np.isfinite(heavy) & (heavy <= ION_MAX_HEAVY)
    is_peptide = np.isfinite(rot) & (rot >= PEPTIDE_MIN_ROT) & (~is_ion)
    is_druglike = ~(is_ion | is_peptide)

    # Target-class proxy: PDB header keyword. Primary token = text before the first '/'.
    ann = pd.read_csv(ROOT / "data" / "raw" / "annotations.csv", low_memory=False)
    ann_u = ann.drop_duplicates(subset="system_id")
    kw = ann_u.set_index("system_id")["entry_keywords"]
    pdb = ann_u.set_index("system_id")["entry_pdb_id"]
    merged_kw = uni["system_id"].map(kw)
    merged_pdb = uni["system_id"].map(pdb)
    n_kw_missing = int(merged_kw.isna().sum())

    primary = (
        merged_kw.dropna()
        .astype(str)
        .str.split("/").str[0]
        .str.strip()
        .str.upper()
    )
    class_counts = primary.value_counts()
    top_classes = [
        {"class": c, "n": int(n), "frac": round(float(n) / n_sys, 4)}
        for c, n in class_counts.head(15).items()
    ]
    top_class_name = str(class_counts.index[0])
    top_class_frac = float(class_counts.iloc[0]) / n_sys
    dominated = bool(top_class_frac >= 0.5)

    n_distinct_pdb = int(merged_pdb.dropna().nunique())
    n_distinct_ccd = None
    if "ligand_ccd_code" in ann_u.columns:
        ccd = ann_u.set_index("system_id")["ligand_ccd_code"]
        n_distinct_ccd = int(uni["system_id"].map(ccd).dropna().nunique())

    return {
        "n_systems": n_sys,
        "ligand_properties": {
            "molecular_weight": _num_summary(mw),
            "num_rot_bonds": _num_summary(rot),
            "num_heavy_atoms": _num_summary(heavy),
        },
        "mw_size_decomposition": {
            "definition": {
                "fragment": f"MW < {MW_FRAGMENT:g}",
                "drug_like": f"{MW_FRAGMENT:g} <= MW <= {MW_DRUGLIKE:g}",
                "large": f"MW > {MW_DRUGLIKE:g}",
            },
            "n_with_mw": int(mw_fin.size),
            "fragment": {"n": n_fragment, "frac": round(n_fragment / max(mw_fin.size, 1), 4)},
            "drug_like": {"n": n_druglike, "frac": round(n_druglike / max(mw_fin.size, 1), 4)},
            "large": {"n": n_large, "frac": round(n_large / max(mw_fin.size, 1), 4)},
        },
        "ligand_type_decomposition": {
            "heuristic": (
                f"ion/tiny-cofactor = num_heavy_atoms <= {ION_MAX_HEAVY}; "
                f"peptide/flexible = num_rot_bonds >= {PEPTIDE_MIN_ROT} (and not an ion); "
                "drug-like small molecule = everything else. Rough, size-only; "
                "no explicit ion/cofactor/peptide column exists in the annotations."
            ),
            "ion_or_tiny_cofactor": {"n": int(is_ion.sum()), "frac": round(int(is_ion.sum()) / n_sys, 4)},
            "peptide_or_flexible": {"n": int(is_peptide.sum()), "frac": round(int(is_peptide.sum()) / n_sys, 4)},
            "drug_like_small_molecule": {"n": int(is_druglike.sum()), "frac": round(int(is_druglike.sum()) / n_sys, 4)},
        },
        "target_class": {
            "source_column": "entry_keywords (PDB header classification, from annotations.csv)",
            "note": (
                "This is the only class-like column shipped; primary token taken as the "
                "text before the first '/'. It is a coarse PDB header keyword, not a "
                "curated enzyme/receptor family."
            ),
            "n_class_missing": n_kw_missing,
            "n_distinct_classes": int(class_counts.size),
            "top_classes": top_classes,
            "most_common_class": top_class_name,
            "most_common_class_frac": round(top_class_frac, 4),
            "dominated_by_single_class": dominated,
        },
        "diversity_proxy": {
            "n_distinct_receptors_pdb_entry": n_distinct_pdb,
            "n_distinct_ligand_ccd_codes": n_distinct_ccd,
            "note": (
                "distinct PDB entries and distinct ligand CCD codes as diversity proxies; "
                "no UniProt column is shipped in annotations.csv."
            ),
        },
    }


def _pocket_drift_bins(sub: pd.DataFrame) -> dict:
    s0 = sub[sub[POCKET_COL] == S_FAMILIAR].dropna(subset=[CONF, "correct"])
    s3 = sub[sub[POCKET_COL] == S_NOVEL].dropna(subset=[CONF, "correct"])
    c0 = s0[CONF].to_numpy(dtype=float)
    c3 = s3[CONF].to_numpy(dtype=float)
    y0 = s0["correct"].to_numpy().astype(int)
    y3 = s3["correct"].to_numpy().astype(int)

    if c0.size == 0 or c3.size == 0:
        return {"error": "empty S0 or S3", "n_S0": int(c0.size), "n_S3": int(c3.size)}

    # Shared quantile edges over the pooled S0+S3 ranking_score, open at the ends.
    edges = np.quantile(np.concatenate([c0, c3]), np.linspace(0, 1, N_BINS + 1))
    edges[0], edges[-1] = -np.inf, np.inf

    bins = []
    any_thin = False
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:], strict=False)):
        m0 = (c0 >= lo) & (c0 < hi)
        m3 = (c3 >= lo) & (c3 < hi)
        n0, n3 = int(m0.sum()), int(m3.sum())
        thin = (n0 < THIN_BIN) or (n3 < THIN_BIN)
        any_thin = any_thin or thin
        bins.append({
            "bin": i,
            "conf_lo": (None if not np.isfinite(lo) else round(float(lo), 4)),
            "conf_hi": (None if not np.isfinite(hi) else round(float(hi), 4)),
            "n_S0": n0,
            "n_S3": n3,
            "p_correct_S0": (round(float(y0[m0].mean()), 4) if n0 else None),
            "p_correct_S3": (round(float(y3[m3].mean()), 4) if n3 else None),
            "thin_bin": bool(thin),
        })
    return {
        "n_S0": int(c0.size),
        "n_S3": int(c3.size),
        "p_correct_S0_overall": round(float(y0.mean()), 4),
        "p_correct_S3_overall": round(float(y3.mean()), 4),
        "shared_edges": [None if not np.isfinite(e) else round(float(e), 4) for e in edges],
        "bins": bins,
        "any_thin_bin": bool(any_thin),
        "n_thin_bins": int(sum(b["thin_bin"] for b in bins)),
    }


def part_b(df: pd.DataFrame) -> dict:
    out = {
        "config": {
            "axis": "pocket",
            "familiar_stratum": S_FAMILIAR,
            "novel_stratum": S_NOVEL,
            "n_bins": N_BINS,
            "binning": f"{N_BINS} quantile bins of {CONF} shared (pooled) between S0 and S3",
            "thin_bin_flag": f"< {THIN_BIN} targets in either stratum",
        },
        "methods": {},
    }
    for m in DRIFT_METHODS:
        out["methods"][m] = _pocket_drift_bins(df[df.method == m])
    return out


def run() -> dict:
    df = load_delivered()
    return {
        "part_a_composition": part_a(df),
        "part_b_pocket_drift_bins": part_b(df),
    }


def main() -> None:
    res = run()
    from experiments._common import save_json  # local import to keep top clean
    save_json(res, RESDIR / "e40_composition.json")

    a = res["part_a_composition"]
    print(f"E40 Part A -- dataset composition over {a['n_systems']} unique systems")
    for name, s in a["ligand_properties"].items():
        print(f"   {name:>18}: min={s['min']:.1f} q25={s['q25']:.1f} med={s['median']:.1f} "
              f"mean={s['mean']:.1f} q75={s['q75']:.1f} max={s['max']:.1f}")
    d = a["mw_size_decomposition"]
    print(f"   MW buckets: fragment(<300)={d['fragment']['n']} ({d['fragment']['frac']:.0%})  "
          f"drug-like(300-500)={d['drug_like']['n']} ({d['drug_like']['frac']:.0%})  "
          f"large(>500)={d['large']['n']} ({d['large']['frac']:.0%})")
    t = a["ligand_type_decomposition"]
    print(f"   type: ion/tiny={t['ion_or_tiny_cofactor']['n']}  "
          f"peptide/flex={t['peptide_or_flexible']['n']}  "
          f"drug-like={t['drug_like_small_molecule']['n']}")
    tc = a["target_class"]
    print(f"   target-class proxy = {tc['source_column']}")
    print(f"   top class: {tc['most_common_class']} ({tc['most_common_class_frac']:.1%}), "
          f"dominated={tc['dominated_by_single_class']}, {tc['n_distinct_classes']} classes")
    for c in tc["top_classes"][:8]:
        print(f"      {c['class']:>28}: {c['n']:4d} ({c['frac']:.1%})")
    dp = a["diversity_proxy"]
    print(f"   diversity: {dp['n_distinct_receptors_pdb_entry']} distinct PDB receptors, "
          f"{dp['n_distinct_ligand_ccd_codes']} distinct ligand CCDs")

    print("\nE40 Part B -- pocket S3 drift-bin occupancy (5 shared quantile bins of ranking_score)")
    for m, r in res["part_b_pocket_drift_bins"]["methods"].items():
        if "error" in r:
            print(f"[{m}] {r['error']}")
            continue
        print(f"[{m}]  n_S0={r['n_S0']} n_S3={r['n_S3']}  "
              f"P(correct) S0={r['p_correct_S0_overall']:.3f} S3={r['p_correct_S3_overall']:.3f}  "
              f"thin_bins={r['n_thin_bins']}")
        print(f"   {'bin':>3} {'conf_lo':>9} {'conf_hi':>9} {'nS0':>5} {'nS3':>5} "
              f"{'pC_S0':>7} {'pC_S3':>7}  flag")
        for b in r["bins"]:
            lo = "-inf" if b["conf_lo"] is None else f"{b['conf_lo']:.3f}"
            hi = "+inf" if b["conf_hi"] is None else f"{b['conf_hi']:.3f}"
            p0 = "  n/a" if b["p_correct_S0"] is None else f"{b['p_correct_S0']:.3f}"
            p3 = "  n/a" if b["p_correct_S3"] is None else f"{b['p_correct_S3']:.3f}"
            flag = "THIN" if b["thin_bin"] else ""
            print(f"   {b['bin']:>3} {lo:>9} {hi:>9} {b['n_S0']:>5} {b['n_S3']:>5} "
                  f"{p0:>7} {p3:>7}  {flag}")


if __name__ == "__main__":
    main()
