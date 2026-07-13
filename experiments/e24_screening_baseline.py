"""E24 -- fair, self-computed screening baseline with a scaffold-novelty split (J3).

This is the honest "does virtual-screening enrichment survive on novel scaffolds" test for the
two datasets that ship per-active chemistry (DEKOIS 2.0 and the GPCR set both ship
``actives_similarity.csv`` with SMILES, ``ec_sim`` = ECFP Tanimoto-to-train, ``ec_mk_sim`` =
Murcko-scaffold Tanimoto-to-train). Nothing here is inherited from the source paper's stat
tables; every EF and BEDROC is recomputed from the raw per-compound score files.

Three parts:

(1) Scaffold-novelty split of enrichment. We compute the Murcko scaffold of every active with
    RDKit, group actives by scaffold, and split them into novel-scaffold (low ``ec_mk_sim``) vs
    familiar-scaffold (high ``ec_mk_sim``). Within each split we recompute EF@1 percent against
    the full decoy background for three signals: the Boltz-2 pose-confidence head (ipTM), the
    Boltz-2 affinity head (binder probability), and Gnina docking (CNNscore). We report a
    target-level bootstrap CI, a scaffold-collapsed variant (one best-scoring active per Murcko
    scaffold, which removes analog-series inflation), and the unique-scaffold counts per split.

(2) Side-by-side recompute of EF@1 percent and BEDROC for every available method (pose, affinity,
    Gnina CNNscore, Glide, plus Protenix and AF3 pose confidences where shipped) so the ranking of
    methods is under our control, not the source paper's.

(3) A decoy-quality diagnostic to the extent the shipped data allows: active-side physicochemical
    property distributions (MW / logP / TPSA / HBD / HBA / rotatable bonds via RDKit) and the
    active-to-train similarity distributions (``ec_sim`` and ``ec_mk_sim``). Decoy SMILES are NOT
    shipped anywhere in the tree, so a full decoy property-match / analog-bias check is not
    computable here. We list that as a limitation rather than fake it.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Lipinski
from rdkit.Chem.Scaffolds import MurckoScaffold

from experiments._common import RESDIR, rng, save_json
from foldgate.selective.enrichment import bedroc, enrichment_factor

RDLogger.DisableLog("rdApp.*")

SCREEN_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "external", "screening")
# only these two ship actives_similarity.csv (SMILES + ec_sim + ec_mk_sim); LIT-PCBA does not
DATASETS = {"dekois": "dekois_scores", "gpcr": "gpcr_scores"}
FRAC = 0.01
FIXED_CUT = 0.5          # secondary novel/familiar split at a fixed scaffold-Tanimoto threshold
MIN_GROUP_ACTIVES = 2    # need at least this many actives in a split to score its EF
MIN_ACT, MIN_DEC = 3, 20  # per-target minima for a method to be scored

# three signals for the scaffold split: (label, boltz column, higher_is_better)
SPLIT_SIGNALS = {
    "pose_iptm": ("boltz", "iptm", True),
    "affinity_prob": ("boltz", "affinity_probability_binary", True),
    "gnina_cnnscore": ("gnina", "cnnscore", True),
}

# every method we recompute side by side: name -> (category, file, column, higher_is_better)
# glide is dataset-specific (docked into AF3 poses on DEKOIS, native Glide on GPCR) and handled below.
METHODS = {
    "boltz2_pose_iptm": ("pose", "boltz_scores.csv", "iptm", True),
    "boltz2_pose_ptm": ("pose", "boltz_scores.csv", "ptm", True),
    "boltz2_pose_confidence": ("pose", "boltz_scores.csv", "confidence_score", True),
    "boltz2_affinity_prob": ("affinity", "boltz_scores.csv", "affinity_probability_binary", True),
    "protenix_pose_ranking": ("pose", "pix_scores.csv", "ranking_score", True),
    "protenix_pose_iptm": ("pose", "pix_scores.csv", "iptm", True),
    "af3_pose_ranking": ("pose", "af3_scores.csv", "ranking_score", True),      # DEKOIS only
    "af3_pose_iptm": ("pose", "af3_scores.csv", "iptm", True),                  # DEKOIS only
    "gnina_cnnscore": ("dock", "gnina_scores.csv", "cnnscore", True),
    "gnina_vina": ("dock", "gnina_scores.csv", "vina", False),
    "gnina_cnnaffinity": ("dock", "gnina_scores.csv", "cnnaffinity", True),
}
GLIDE = {  # dataset -> (file, column); Glide docking, lower-is-better, 10000 = failed dock
    "dekois": ("af3_glide_scores.csv", "glide_min_SP_top1"),
    "gpcr": ("glide_scores.csv", "score"),
}


# ---------------------------------------------------------------------------- loaders

def _scored(path, col, higher):
    """Load a score file -> DataFrame[lid, label, s] one row per compound (best s = most active)."""
    if not os.path.exists(path):
        return None
    d = pd.read_csv(path)
    idc = "lid" if "lid" in d.columns else ("ID" if "ID" in d.columns else None)
    if idc is None or col not in d.columns:
        return None
    lab = "label" if "label" in d.columns else None
    if lab is None:
        return None
    d = d[[idc, lab, col]].copy()
    d.columns = ["lid", "label", "raw"]
    d["lid"] = d["lid"].astype(str)
    d = d.dropna(subset=["raw"])
    d["s"] = d["raw"] if higher else -d["raw"]         # higher s = more likely active
    d = d.groupby("lid", as_index=False).agg(label=("label", "max"), s=("s", "max"))
    return d


def _murcko(smiles: str) -> str | None:
    """Bemis-Murcko scaffold SMILES; falls back to canonical SMILES for acyclic molecules."""
    try:
        scaf = MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles)
    except Exception:
        return None
    if scaf:
        return scaf
    mol = Chem.MolFromSmiles(smiles)          # acyclic -> keep the molecule as its own scaffold
    return Chem.MolToSmiles(mol) if mol is not None else None


def _target_active_frame(tdir):
    """Per active: score for each signal + Murcko scaffold + ec_mk_sim/ec_sim; plus decoy scores.

    Returns (actives_df, decoys_by_signal) where actives_df has one row per active compound with
    columns [lid, scaffold, ec_mk_sim, ec_sim, <signal score columns>] and decoys_by_signal maps
    each signal to the numpy array of decoy scores for that signal.
    """
    sim_p = os.path.join(tdir, "actives_similarity.csv")
    if not os.path.exists(sim_p):
        return None
    sim = pd.read_csv(sim_p)
    if not {"ID", "smiles", "ec_mk_sim"}.issubset(sim.columns):
        return None
    sim = sim[["ID", "smiles", "ec_sim", "ec_mk_sim"]].copy()
    sim["lid"] = sim["ID"].astype(str)
    sim["scaffold"] = sim["smiles"].map(_murcko)
    sim = sim.dropna(subset=["scaffold", "ec_mk_sim"])
    if len(sim) < MIN_GROUP_ACTIVES:
        return None

    act = sim[["lid", "scaffold", "ec_sim", "ec_mk_sim"]].copy()
    decoys = {}
    for sig, (src, col, higher) in SPLIT_SIGNALS.items():
        fname = "boltz_scores.csv" if src == "boltz" else "gnina_scores.csv"
        sc = _scored(os.path.join(tdir, fname), col, higher)
        if sc is None:
            return None
        amap = dict(zip(sc.lid, sc.s, strict=False))
        act[sig] = act["lid"].map(amap)
        decoys[sig] = sc[sc.label == 0]["s"].to_numpy()
    return act, decoys


# ---------------------------------------------------------------------------- EF helpers

def _ef_group(pos_scores, decoy_scores):
    pos_scores = np.asarray([x for x in pos_scores if np.isfinite(x)], float)
    dec = np.asarray([x for x in decoy_scores if np.isfinite(x)], float)
    if len(pos_scores) < MIN_GROUP_ACTIVES or len(dec) < MIN_DEC:
        return float("nan")
    s = np.concatenate([pos_scores, dec])
    y = np.concatenate([np.ones(len(pos_scores), int), np.zeros(len(dec), int)])
    return enrichment_factor(s, y, FRAC)


def _boot(vals, g, n=2000):
    v = np.asarray([x for x in vals if np.isfinite(x)], float)
    if not len(v):
        return {"median": float("nan"), "ci90": [float("nan"), float("nan")], "n": 0}
    meds = [np.median(v[g.integers(0, len(v), len(v))]) for _ in range(n)]
    return {"median": float(np.median(v)),
            "ci90": [float(np.quantile(meds, 0.05)), float(np.quantile(meds, 0.95))],
            "n": int(len(v))}


# ---------------------------------------------------------------------------- part 1

def _split_result(per_target, g):
    """Aggregate the per-target novel/familiar EF records into medians + target-bootstrap CIs."""
    out = {}
    for grp in ("novel", "familiar"):
        out[grp] = {
            "ef": _boot([r[grp]["ef"] for r in per_target], g),
            "ef_scaffold_collapsed": _boot([r[grp]["ef_collapsed"] for r in per_target], g),
            "n_actives_total": int(sum(r[grp]["n_act"] for r in per_target)),
            "n_scaffolds_total": int(sum(r[grp]["n_scaff"] for r in per_target)),
            "n_targets_scored": int(sum(np.isfinite(r[grp]["ef"]) for r in per_target)),
        }
    return out


def scaffold_split(targets_frames, cut, g):
    """For each signal, EF@1% on novel vs familiar scaffolds (novel = ec_mk_sim < cut)."""
    result = {}
    for sig in SPLIT_SIGNALS:
        per_target = []
        for act, decoys in targets_frames:
            dec = decoys[sig]
            row = {}
            for grp, mask in (("novel", act["ec_mk_sim"] < cut), ("familiar", act["ec_mk_sim"] >= cut)):
                sub = act[mask.to_numpy()]
                pos = sub[sig].to_numpy()
                pos = pos[np.isfinite(pos)]
                ef = _ef_group(pos, dec)
                # scaffold-collapsed: one best-scoring active per Murcko scaffold
                sub2 = sub.dropna(subset=[sig])
                if len(sub2):
                    best = sub2.sort_values(sig, ascending=False).drop_duplicates("scaffold")
                    ef_c = _ef_group(best[sig].to_numpy(), dec)
                    n_scaff = int(sub2["scaffold"].nunique())
                    n_act = int(len(sub2))
                else:
                    ef_c, n_scaff, n_act = float("nan"), 0, 0
                row[grp] = {"ef": ef, "ef_collapsed": ef_c, "n_act": n_act, "n_scaff": n_scaff}
            per_target.append(row)
        result[sig] = _split_result(per_target, g)
    return result


# ---------------------------------------------------------------------------- part 2

def recompute_methods(dset, dset_dir, targets, g):
    out = {}
    for name, (cat, fname, col, higher) in METHODS.items():
        efs, beds, ns, nacts = [], [], [], []
        for t in targets:
            d = _scored(os.path.join(dset_dir, t, fname), col, higher)
            if d is None:
                continue
            y = d["label"].to_numpy().astype(int)
            if int(y.sum()) < MIN_ACT or int((y == 0).sum()) < MIN_DEC:
                continue
            s = d["s"].to_numpy()
            efs.append(enrichment_factor(s, y, FRAC))
            beds.append(bedroc(s, y))
            ns.append(len(y))
            nacts.append(int(y.sum()))
        if not efs:
            continue
        out[name] = {
            "category": cat,
            "ef01": _boot(efs, g),
            "bedroc": _boot(beds, g),
            "median_n_compounds": float(np.median(ns)),
            "median_n_actives": float(np.median(nacts)),
        }
    # glide (dataset-specific file/column)
    gfile, gcol = GLIDE[dset]
    efs, beds, ns, nacts = [], [], [], []
    for t in targets:
        d = _scored(os.path.join(dset_dir, t, gfile), gcol, higher=False)
        if d is None:
            continue
        y = d["label"].to_numpy().astype(int)
        if int(y.sum()) < MIN_ACT or int((y == 0).sum()) < MIN_DEC:
            continue
        s = d["s"].to_numpy()
        efs.append(enrichment_factor(s, y, FRAC))
        beds.append(bedroc(s, y))
        ns.append(len(y))
        nacts.append(int(y.sum()))
    if efs:
        out["glide"] = {"category": "dock", "ef01": _boot(efs, g), "bedroc": _boot(beds, g),
                        "median_n_compounds": float(np.median(ns)), "median_n_actives": float(np.median(nacts)),
                        "note": f"{gfile}:{gcol} (Glide docking, lower-is-better; 10000=failed dock)"}
    return out


# ---------------------------------------------------------------------------- part 3

def active_property_diagnostic(all_smiles, all_ecsim, all_ecmk):
    descs = {"MW": Descriptors.MolWt, "logP": Descriptors.MolLogP, "TPSA": Descriptors.TPSA,
             "HBD": Lipinski.NumHDonors, "HBA": Lipinski.NumHAcceptors,
             "RotB": Descriptors.NumRotatableBonds}
    vals = {k: [] for k in descs}
    n_parsed = 0
    for smi in all_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        n_parsed += 1
        for k, fn in descs.items():
            try:
                vals[k].append(float(fn(mol)))
            except Exception:
                pass

    def _dist(x):
        a = np.asarray([v for v in x if np.isfinite(v)], float)
        if not len(a):
            return {"n": 0}
        return {"n": int(len(a)), "mean": float(a.mean()), "std": float(a.std()),
                "min": float(a.min()), "q25": float(np.quantile(a, 0.25)),
                "median": float(np.median(a)), "q75": float(np.quantile(a, 0.75)),
                "max": float(a.max())}

    return {
        "n_actives_with_smiles": len(all_smiles),
        "n_actives_parsed": n_parsed,
        "active_physchem": {k: _dist(v) for k, v in vals.items()},
        "active_to_train_similarity": {"ec_sim": _dist(all_ecsim), "ec_mk_sim": _dist(all_ecmk)},
        "limitation": (
            "Decoy SMILES are not shipped anywhere in this data tree (only actives_similarity.csv "
            "carries a smiles column), so decoy physicochemical distributions and a full "
            "decoy-to-active property-match / analog-bias check (the standard DEKOIS/DUD-E "
            "diagnostic) are NOT computable here. Only active-side properties and active-to-train "
            "similarity are reported. LIT-PCBA ships no actives_similarity.csv and no SMILES, so "
            "it is excluded from every part of this experiment."
        ),
    }


# ---------------------------------------------------------------------------- driver

def run():
    g = rng()
    out = {"frac": FRAC, "datasets": {}}
    for name, sub in DATASETS.items():
        dset_dir = os.path.join(SCREEN_DIR, sub)
        if not os.path.isdir(dset_dir):
            continue
        targets = sorted(os.path.basename(t) for t in glob.glob(os.path.join(dset_dir, "*")) if os.path.isdir(t))

        # part 1: build per-target active/decoy frames, then split
        frames, all_smiles, all_ecsim, all_ecmk, all_mk = [], [], [], [], []
        for t in targets:
            tf = _target_active_frame(os.path.join(dset_dir, t))
            if tf is None:
                continue
            act, decoys = tf
            frames.append((act, decoys))
            all_ecsim.extend(act["ec_sim"].tolist())
            all_ecmk.extend(act["ec_mk_sim"].tolist())
            all_mk.extend(act["ec_mk_sim"].tolist())
            sim_p = os.path.join(dset_dir, t, "actives_similarity.csv")
            all_smiles.extend(pd.read_csv(sim_p)["smiles"].astype(str).tolist())

        median_cut = float(np.median(all_mk)) if all_mk else float("nan")
        ds = {
            "n_targets": len(targets),
            "n_targets_with_actives_similarity": len(frames),
            "scaffold_split_median": {"cut_ec_mk_sim": median_cut,
                                      "signals": scaffold_split(frames, median_cut, g)},
            "scaffold_split_fixed_0.5": {"cut_ec_mk_sim": FIXED_CUT,
                                         "signals": scaffold_split(frames, FIXED_CUT, g)},
            "methods_recomputed": recompute_methods(name, dset_dir, targets, g),
            "decoy_quality_diagnostic": active_property_diagnostic(all_smiles, all_ecsim, all_ecmk),
        }
        out["datasets"][name] = ds
    return out


def _fmt(b):
    return f"{b['median']:6.2f} [{b['ci90'][0]:5.2f},{b['ci90'][1]:6.2f}] (nt={b['n']})"


def main():
    res = run()
    save_json(res, RESDIR / "e24_screening_baseline.json")

    print("E24 -- fair self-computed screening baseline + scaffold-novelty split (DEKOIS, GPCR)\n")
    for name, ds in res["datasets"].items():
        cut = ds["scaffold_split_median"]["cut_ec_mk_sim"]
        print(f"===== {name.upper()}  (targets={ds['n_targets']}, "
              f"with actives_similarity={ds['n_targets_with_actives_similarity']}) =====")
        print(f"scaffold-novelty split, median ec_mk_sim cut = {cut:.3f} "
              f"(novel = scaffold-to-train Tanimoto < cut)\n")
        print(f"  {'signal':16s} {'novel EF@1% [ci90]':30s} {'familiar EF@1% [ci90]':30s}  "
              f"n_scaf novel/fam")
        sigs = ds["scaffold_split_median"]["signals"]
        for sig, r in sigs.items():
            nv, fm = r["novel"], r["familiar"]
            print(f"  {sig:16s} {_fmt(nv['ef']):30s} {_fmt(fm['ef']):30s}  "
                  f"{nv['n_scaffolds_total']:4d}/{fm['n_scaffolds_total']:<4d}")
        print("  scaffold-collapsed (one active per Murcko scaffold):")
        for sig, r in sigs.items():
            nv, fm = r["novel"], r["familiar"]
            print(f"  {sig:16s} {_fmt(nv['ef_scaffold_collapsed']):30s} "
                  f"{_fmt(fm['ef_scaffold_collapsed']):30s}")
        print("\n  recomputed EF@1% / BEDROC per method (target-median, ci90):")
        for m, r in ds["methods_recomputed"].items():
            print(f"    {m:24s} [{r['category']:8s}] EF {_fmt(r['ef01'])}  BEDROC "
                  f"{r['bedroc']['median']:.3f} [{r['bedroc']['ci90'][0]:.3f},{r['bedroc']['ci90'][1]:.3f}]")
        dq = ds["decoy_quality_diagnostic"]
        pc = dq["active_physchem"]
        print(f"\n  active physchem (n={dq['n_actives_parsed']}): "
              f"MW {pc['MW']['median']:.0f}, logP {pc['logP']['median']:.2f}, "
              f"TPSA {pc['TPSA']['median']:.0f}, HBD {pc['HBD']['median']:.0f}, "
              f"HBA {pc['HBA']['median']:.0f}, RotB {pc['RotB']['median']:.0f}")
        sim = dq["active_to_train_similarity"]
        print(f"  active->train sim: ec_sim median {sim['ec_sim']['median']:.3f}, "
              f"ec_mk_sim median {sim['ec_mk_sim']['median']:.3f}")
        print(f"  LIMIT: {dq['limitation'][:88]}...\n")


if __name__ == "__main__":
    main()
