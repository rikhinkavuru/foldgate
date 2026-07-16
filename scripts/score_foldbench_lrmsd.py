#!/usr/bin/env python3
"""Self-consistent ligand-RMSD scoring for regenerated FoldBench Protenix predictions.

For each target we take the model's own top-ranking sample, superpose the predicted
protein onto the deposited assembly using pocket Calpha atoms (residues within 10 A of the
deposited target ligand), apply that transform to the predicted ligand, and compute a
symmetry-corrected heavy-atom RMSD (spyrmsd) against the deposited ligand. success = the
pocket-aligned ligand RMSD <= 2 A.

Both the confidence feature (interface-ipTM) and this label come from the SAME regenerated
run, so feature and label are self-consistent -- we do not borrow FoldBench's released
poses or lrmsd (those were produced by a different Protenix version / MSA snapshot). The
deposited assembly CIF (fetched from RCSB by the input builder) is the ground truth.
"""
import argparse
import csv
import glob
import json
from pathlib import Path

import gemmi
import numpy as np
from spyrmsd import rmsd as srmsd

COVR = {  # covalent radii (A) for bond inference
    "H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "S": 1.05, "P": 1.07,
    "F": 0.57, "CL": 1.02, "BR": 1.20, "I": 1.39, "B": 0.84, "SE": 1.20,
    "NA": 1.66, "MG": 1.41, "K": 2.03, "CA": 1.76, "ZN": 1.22, "FE": 1.32,
    "MN": 1.39, "CU": 1.32, "NI": 1.24, "CO": 1.26,
}


def one_letter(res_name):
    info = gemmi.find_tabulated_residue(res_name)
    return info.one_letter_code.upper() if info else "X"


def protein_ca(structure):
    """Return list of (chain_name, [(one_letter, seqid, np.array(xyz))]) for peptide chains."""
    out = []
    model = structure[0]
    for chain in model:
        poly = chain.get_polymer()
        if len(poly) == 0:
            continue
        if poly.check_polymer_type() not in (
            gemmi.PolymerType.PeptideL,
            gemmi.PolymerType.PeptideD,
        ):
            continue
        residues = []
        for res in poly:
            ca = res.find_atom("CA", "*")
            if ca is None:
                continue
            residues.append(
                (one_letter(res.name), res.seqid.num, np.array([ca.pos.x, ca.pos.y, ca.pos.z]))
            )
        if residues:
            out.append((chain.name, residues))
    return out


def ligand_atoms(structure, ccd):
    """All heavy-atom copies of a CCD ligand: list of dict{atom_name: (element, xyz)}."""
    copies = []
    model = structure[0]
    for chain in model:
        for res in chain:
            if res.name != ccd:
                continue
            atoms = {}
            for at in res:
                el = at.element.name.upper()
                if el == "H":
                    continue
                atoms[at.name] = (el, np.array([at.pos.x, at.pos.y, at.pos.z]))
            if atoms:
                copies.append(atoms)
    return copies


def match_ca(gt_chain_res, pred_all):
    """Greedy sequence match: pick the predicted chain whose sequence best contains gt's,
    return aligned CA pairs (gt_xyz, pred_xyz) by longest common subsequence on one-letter."""
    gt_seq = "".join(r[0] for r in gt_chain_res)
    best = None
    for pname, pres in pred_all:
        pseq = "".join(r[0] for r in pres)
        # simple global-ish: find pred substring alignment via difflib
        import difflib

        sm = difflib.SequenceMatcher(None, gt_seq, pseq, autojunk=False)
        blocks = sm.get_matching_blocks()
        matched = sum(b.size for b in blocks)
        if best is None or matched > best[0]:
            best = (matched, pname, pres, blocks)
    if best is None or best[0] < 4:
        return [], []
    _, _, pres, blocks = best
    gt_pts, pred_pts = [], []
    for b in blocks:
        for k in range(b.size):
            gt_pts.append(gt_chain_res[b.a + k][2])
            pred_pts.append(pres[b.b + k][2])
    return gt_pts, pred_pts


def infer_adj(elements, coords):
    n = len(elements)
    adj = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(coords[i] - coords[j])
            ri = COVR.get(elements[i], 0.77)
            rj = COVR.get(elements[j], 0.77)
            if d < ri + rj + 0.45:
                adj[i, j] = adj[j, i] = 1
    return adj


ELNUM = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9, "P": 15, "S": 16, "CL": 17,
         "BR": 35, "I": 53, "B": 5, "SE": 34, "NA": 11, "MG": 12, "K": 19,
         "CA": 20, "ZN": 30, "FE": 26, "MN": 25, "CU": 29, "NI": 28, "CO": 27}


def score_target(pred_cif, gt_cif, ccd, pocket_r=10.0):
    pred = gemmi.read_structure(str(pred_cif))
    pred.setup_entities()
    gt = gemmi.read_structure(str(gt_cif))
    gt.setup_entities()

    gt_lig = ligand_atoms(gt, ccd)
    pred_lig = ligand_atoms(pred, ccd)
    if not gt_lig or not pred_lig:
        return None, "no-ligand"

    gt_prot = protein_ca(gt)
    pred_prot = protein_ca(pred)
    if not gt_prot or not pred_prot:
        return None, "no-protein"

    # pocket residues: GT CA within pocket_r of any GT target-ligand atom (first copy set)
    lig_xyz = np.array([xyz for cp in gt_lig for (_, xyz) in cp.values()])

    gt_pts_all, pred_pts_all = [], []
    for cname, cres in gt_prot:
        gp, pp = match_ca(cres, pred_prot)
        # restrict to pocket
        for g, p in zip(gp, pp):
            if np.min(np.linalg.norm(lig_xyz - g, axis=1)) <= pocket_r:
                gt_pts_all.append(g)
                pred_pts_all.append(p)
    if len(gt_pts_all) < 4:
        # fall back to global (all matched CA)
        gt_pts_all, pred_pts_all = [], []
        for cname, cres in gt_prot:
            gp, pp = match_ca(cres, pred_prot)
            gt_pts_all += gp
            pred_pts_all += pp
    if len(gt_pts_all) < 4:
        return None, "too-few-ca"

    sup = gemmi.superpose_positions(
        [gemmi.Position(*p) for p in gt_pts_all],
        [gemmi.Position(*p) for p in pred_pts_all],
    )
    tr = sup.transform  # maps pred -> gt frame

    def apply(xyz):
        p = tr.apply(gemmi.Position(*xyz))
        return np.array([p.x, p.y, p.z])

    # best RMSD over (pred copy, gt copy) pairs, symmetry-corrected
    best_rmsd = None
    for gcopy in gt_lig:
        gnames = list(gcopy.keys())
        gel = [gcopy[n][0] for n in gnames]
        gxyz = np.array([gcopy[n][1] for n in gnames])
        try:
            adj = infer_adj(gel, gxyz)
            anum = np.array([ELNUM.get(e, 6) for e in gel])
        except Exception:
            continue
        for pcopy in pred_lig:
            # pair predicted atoms to gt by atom name
            if not set(gnames).issubset(pcopy.keys()):
                # atom-name mismatch: skip (different naming) -> try intersection
                common = [n for n in gnames if n in pcopy]
                if len(common) < max(4, int(0.6 * len(gnames))):
                    continue
                gn = common
            else:
                gn = gnames
            gel2 = [gcopy[n][0] for n in gn]
            gxyz2 = np.array([gcopy[n][1] for n in gn])
            pxyz2 = np.array([apply(pcopy[n][1]) for n in gn])
            adj2 = infer_adj(gel2, gxyz2)
            anum2 = np.array([ELNUM.get(e, 6) for e in gel2])
            try:
                r = srmsd.symmrmsd(gxyz2, pxyz2, anum2, anum2, adj2, adj2, minimize=False)
            except Exception:
                r = float(np.sqrt(np.mean(np.sum((gxyz2 - pxyz2) ** 2, axis=1))))
            if best_rmsd is None or r < best_rmsd:
                best_rmsd = float(r)
    if best_rmsd is None:
        return None, "no-atom-match"
    return best_rmsd, "ok"


def best_sample(target_dir):
    """Global top-ranking_score pose across ALL seeds x samples (delivered pose).

    Returns (seed, sample_idx, ranking_score, summary, cif_path) or None. Mirrors the
    RNP top-1-by-ranking_score delivered convention across the full seed x sample set.
    """
    best = None
    for seed_dir in glob.glob(str(target_dir) + "/seed_*/predictions"):
        seed = Path(seed_dir).parent.name.replace("seed_", "")
        for f in glob.glob(seed_dir + "/*summary_confidence_sample_*.json"):
            s = json.load(open(f))
            idx = f.split("sample_")[-1].split(".")[0]
            if best is None or s["ranking_score"] > best[2]:
                cif = Path(seed_dir).parent / "predictions" / (
                    Path(f).name.replace("summary_confidence_", "").replace(".json", ".cif")
                )
                best = (seed, idx, s["ranking_score"], s, str(cif))
    return best


def interface_iptm(summary, comp, target_ccd):
    """Average chain_pair_iptm over protein x target-ligand chain pairs.

    Matches RNP's iface_iptm (the rmsd-mapped average protein-ligand chain-pair ipTM),
    so the calibrate-on-RNP / deploy-on-FoldBench transfer is apples-to-apples.
    """
    cpi = np.array(summary["chain_pair_iptm"])
    prot_idx = [i for i, (t, c) in enumerate(comp) if t == "protein"]
    lig_idx = [i for i, (t, c) in enumerate(comp) if t == "ligand" and c == target_ccd]
    if not prot_idx or not lig_idx or cpi.shape[0] != len(comp):
        return None
    vals = [cpi[p, l] for p in prot_idx for l in lig_idx]
    return float(np.mean(vals)) if vals else None


def load_comp(af3_json):
    """pdb -> ordered list of (type, ccd_or_None) matching chain order in chain_pair_iptm."""
    out = {}
    for t in json.load(open(af3_json)):
        comp = []
        for s in t["sequences"]:
            k = next(iter(s))
            n = len(s[k]["id"]) if isinstance(s[k].get("id"), list) else 1
            if k == "ligand":
                for _ in range(n):
                    comp.append(("ligand", s[k]["ccdCodes"][0]))
            else:
                for _ in range(n):
                    comp.append(("protein" if k == "protein" else k, None))
        out[t["name"]] = comp
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_root", required=True, help="fb_pred dir (per-target subdirs)")
    ap.add_argument("--gt_cache", required=True, help="deposited assembly CIFs")
    ap.add_argument("--af3_json", required=True)
    ap.add_argument("--target_ligands_csv", required=True, help="pdb_id,target_ccd")
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="")
    args = ap.parse_args()

    comps = load_comp(args.af3_json)
    tlig = {}
    with open(args.target_ligands_csv) as f:
        for row in csv.DictReader(f):
            tlig[row["pdb_id"]] = row["target_ccd"]

    only = set(args.only.split(",")) if args.only else None
    rows = []
    names = sorted(comps)
    for i, name in enumerate(names):
        if only and name not in only:
            continue
        pdb = name.replace("-assembly1", "")
        target_dir = Path(args.pred_root) / name
        gt_cif = Path(args.gt_cache) / f"{pdb}-assembly1.cif"
        ccd = tlig.get(name, "").strip("()")
        rec = {"pdb_id": name, "target_ccd": ccd, "seed": None, "sample": None,
               "iptm_iface": None, "ranking_score": None, "lrmsd": None, "status": ""}
        if not target_dir.exists() or not gt_cif.exists() or not ccd:
            rec["status"] = "missing-input"
            rows.append(rec)
            continue
        bs = best_sample(target_dir)
        if bs is None:
            rec["status"] = "no-sample"
            rows.append(rec)
            continue
        seed, idx, rank, summary, pred_cif = bs
        rec["seed"] = seed
        rec["sample"] = idx
        rec["ranking_score"] = float(rank)
        rec["iptm_iface"] = interface_iptm(summary, comps[name], ccd)
        pred_cif = Path(pred_cif)
        try:
            lr, st = score_target(pred_cif, gt_cif, ccd)
            rec["lrmsd"] = lr
            rec["status"] = st
        except Exception as e:  # noqa: BLE001
            rec["status"] = f"err:{e}"
        rows.append(rec)
        if (i + 1) % 25 == 0 or only:
            print(f"[{i+1}/{len(names)}] {name} ccd={ccd} iptm={rec['iptm_iface']} "
                  f"lrmsd={rec['lrmsd']} {rec['status']}", flush=True)

    import pandas as pd

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    ok = df[df.lrmsd.notna()]
    print(f"\nscored {len(ok)}/{len(df)} targets; "
          f"success<=2A: {(ok.lrmsd <= 2).mean():.3f}" if len(ok) else "none scored")


if __name__ == "__main__":
    main()
