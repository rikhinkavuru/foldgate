#!/usr/bin/env python3
"""Self-score regenerated Boltz-2 / Chai-1 poses for the D3 external datasets.

For every target we take the model's OWN top-ranked pose, extract its native confidence
fields, pocket-superpose the predicted protein onto the deposited RCSB entry (CA atoms
within 10 A of the deposited CCD ligand), apply that transform to the predicted ligand,
and compute a symmetry-corrected heavy-atom ligand-RMSD (spyrmsd) against the deposited
ligand. correct = pocket-aligned RMSD <= 2 A. Feature and label come from the SAME
regenerated run, so they are self-consistent (identical convention to the FoldBench
regen scorer, src/foldgate/features/pose.py / scripts/score_foldbench_lrmsd.py).

The predicted ligand is matched to the crystal ligand by graph isomorphism (elements +
inferred connectivity), NOT by atom name -- Boltz/Chai emit a generic ligand residue name
because the input is a SMILES, so name-based matching (used in the FoldBench regen where
both sides shared a CCD) does not apply here.

Runs on the GPU box after inference. Output results/gpu_d3/scored.csv is a ~MB file:
  dataset, model, target_id, pdb_id, ligand_ccd, <confidence fields>, ligand_rmsd, correct, status
"""
from __future__ import annotations

import argparse
import glob
import json
import time
import urllib.request
from pathlib import Path

import gemmi
import numpy as np
import pandas as pd
from spyrmsd import rmsd as srmsd

RCSB_CIF = "https://files.rcsb.org/download/{}.cif"

COVR = {  # covalent radii (A) for bond inference (from score_foldbench_lrmsd.py)
    "H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "S": 1.05, "P": 1.07,
    "F": 0.57, "CL": 1.02, "BR": 1.20, "I": 1.39, "B": 0.84, "SE": 1.20,
    "NA": 1.66, "MG": 1.41, "K": 2.03, "CA": 1.76, "ZN": 1.22, "FE": 1.32,
    "MN": 1.39, "CU": 1.32, "NI": 1.24, "CO": 1.26,
}
ELNUM = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9, "P": 15, "S": 16, "CL": 17,
         "BR": 35, "I": 53, "B": 5, "SE": 34, "NA": 11, "MG": 12, "K": 19,
         "CA": 20, "ZN": 30, "FE": 26, "MN": 25, "CU": 29, "NI": 28, "CO": 27}

# residues that are never the ligand of interest (solvent / cryo / buffer / ions handled
# separately by the >= min-heavy-atom filter)
_NONLIGAND = {"HOH", "DOD", "WAT"}


# ----------------------------------------------------------------- ground truth
def fetch_gt(pdb: str, cache: Path) -> Path | None:
    dst = cache / f"{pdb}.cif"
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    cache.mkdir(parents=True, exist_ok=True)
    for attempt in range(4):
        try:
            with urllib.request.urlopen(RCSB_CIF.format(pdb), timeout=60) as r:
                dst.write_bytes(r.read())
            return dst
        except Exception:  # noqa: BLE001
            time.sleep(2 * (attempt + 1))
    return None


# ------------------------------------------------------------------ geometry io
def one_letter(res_name: str) -> str:
    info = gemmi.find_tabulated_residue(res_name)
    return info.one_letter_code.upper() if info else "X"


def protein_ca(structure) -> list:
    out = []
    model = structure[0]
    for chain in model:
        poly = chain.get_polymer()
        if len(poly) == 0:
            continue
        if poly.check_polymer_type() not in (gemmi.PolymerType.PeptideL, gemmi.PolymerType.PeptideD):
            continue
        residues = []
        for res in poly:
            ca = res.find_atom("CA", "*")
            if ca is None:
                continue
            residues.append((one_letter(res.name), res.seqid.num, np.array([ca.pos.x, ca.pos.y, ca.pos.z])))
        if residues:
            out.append((chain.name, residues))
    return out


def _heavy(res) -> dict:
    atoms = {}
    for at in res:
        el = at.element.name.upper()
        if el == "H":
            continue
        atoms[at.name] = (el, np.array([at.pos.x, at.pos.y, at.pos.z]))
    return atoms


def ligand_by_ccd(structure, ccd: str) -> list:
    """All heavy-atom copies of a CCD ligand: list of dict{name:(el,xyz)}."""
    copies = []
    for chain in structure[0]:
        for res in chain:
            if res.name != ccd:
                continue
            a = _heavy(res)
            if a:
                copies.append(a)
    return copies


def predicted_ligand(structure, min_heavy: int = 5) -> dict | None:
    """The folded ligand in a Boltz/Chai prediction: the largest non-polymer heavy-atom
    residue (structure has one protein complex + one small-molecule ligand)."""
    best = None
    model = structure[0]
    poly_res_ids = set()
    for chain in model:
        poly = chain.get_polymer()
        for res in poly:
            poly_res_ids.add(id(res))
    for chain in model:
        for res in chain:
            if id(res) in poly_res_ids or res.name in _NONLIGAND:
                continue
            a = _heavy(res)
            if len(a) >= min_heavy and (best is None or len(a) > len(best)):
                best = a
    return best


def infer_adj(elements, coords) -> np.ndarray:
    n = len(elements)
    adj = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(coords[i] - coords[j])
            if d < COVR.get(elements[i], 0.77) + COVR.get(elements[j], 0.77) + 0.45:
                adj[i, j] = adj[j, i] = 1
    return adj


def _mol(atoms: dict):
    names = list(atoms.keys())
    el = [atoms[n][0] for n in names]
    xyz = np.array([atoms[n][1] for n in names])
    anum = np.array([ELNUM.get(e, 6) for e in el])
    adj = infer_adj(el, xyz)
    return anum, adj, xyz


def kabsch(gt_pts, pred_pts):
    sup = gemmi.superpose_positions(
        [gemmi.Position(*p) for p in gt_pts],
        [gemmi.Position(*p) for p in pred_pts],
    )
    return sup.transform


def match_ca(gt_chain_res, pred_all):
    import difflib

    gt_seq = "".join(r[0] for r in gt_chain_res)
    best = None
    for _pname, pres in pred_all:
        pseq = "".join(r[0] for r in pres)
        sm = difflib.SequenceMatcher(None, gt_seq, pseq, autojunk=False)
        blocks = sm.get_matching_blocks()
        matched = sum(b.size for b in blocks)
        if best is None or matched > best[0]:
            best = (matched, pres, blocks)
    if best is None or best[0] < 4:
        return [], []
    _, pres, blocks = best
    gt_pts, pred_pts = [], []
    for b in blocks:
        for k in range(b.size):
            gt_pts.append(gt_chain_res[b.a + k][2])
            pred_pts.append(pres[b.b + k][2])
    return gt_pts, pred_pts


def score_pose(pred_cif: Path, gt_cif: Path, ccd: str, pocket_r: float = 10.0):
    pred = gemmi.read_structure(str(pred_cif))
    pred.setup_entities()
    gt = gemmi.read_structure(str(gt_cif))
    gt.setup_entities()

    gt_lig = ligand_by_ccd(gt, ccd)
    if not gt_lig:
        return None, "gt-no-ligand"
    pred_lig = predicted_ligand(pred)
    if not pred_lig:
        return None, "pred-no-ligand"

    gt_prot, pred_prot = protein_ca(gt), protein_ca(pred)
    if not gt_prot or not pred_prot:
        return None, "no-protein"

    # Precompute the CA correspondence once (sequence-matched, chain by chain).
    ca_pairs = []
    for _c, cres in gt_prot:
        gp, pp = match_ca(cres, pred_prot)
        ca_pairs.extend(zip(gp, pp))
    if len(ca_pairs) < 4:
        return None, "too-few-ca"
    ca_gt = np.array([g for g, _ in ca_pairs])
    ca_pred = [p for _, p in ca_pairs]

    p_anum, p_adj, p_xyz0 = _mol(pred_lig)

    # Superpose per ligand COPY on that copy's own pocket, then take the best-matching
    # site. RCSB deposits often carry several copies of a ligand in different sites (63%
    # of the PLINDER subset); pooling their pockets into one frame scatters the alignment
    # and inflates every RMSD, so each copy must define and be scored in its own frame.
    best = None
    matched_any = False
    for gcopy in gt_lig:
        g_anum, g_adj, g_xyz = _mol(gcopy)
        if len(g_anum) != len(p_anum) or sorted(g_anum) != sorted(p_anum):
            continue  # element multiset must match for graph isomorphism
        matched_any = True
        cxyz = np.vstack([np.asarray(xyz) for (_el, xyz) in gcopy.values()])
        # pocket = CA within pocket_r of THIS copy; fall back to global CA if too few
        near = np.min(np.linalg.norm(ca_gt[:, None, :] - cxyz[None, :, :], axis=2), axis=1) <= pocket_r
        if near.sum() >= 4:
            gt_pts = ca_gt[near]
            pred_pts = [ca_pred[i] for i in np.where(near)[0]]
        else:
            gt_pts, pred_pts = ca_gt, ca_pred
        tr = kabsch(list(gt_pts), pred_pts)  # maps pred -> gt frame for this copy
        p_xyz = np.array([[(q := tr.apply(gemmi.Position(*x))).x, q.y, q.z] for x in p_xyz0])
        try:
            r = srmsd.symmrmsd(g_xyz, p_xyz, g_anum, p_anum, g_adj, p_adj, minimize=False)
        except Exception:  # noqa: BLE001
            continue
        if best is None or r < best:
            best = float(r)
    if not matched_any:
        return None, "no-graph-match"
    if best is None:
        return None, "no-graph-match"
    return best, "ok"


# ---------------------------------------------------------------- confidence io
def boltz_delivered(target_dir: Path):
    """Boltz writes predictions/<name>/<name>_model_0.cif (rank 0 = best) + confidence json."""
    preds = list(target_dir.glob("predictions/*"))
    if not preds:
        return None
    d = preds[0]
    name = d.name
    cif = d / f"{name}_model_0.cif"
    conf = d / f"confidence_{name}_model_0.json"
    if not cif.exists():
        cifs = sorted(d.glob(f"{name}_model_*.cif"))
        cif = cifs[0] if cifs else None
    fields = {}
    if conf.exists():
        c = json.load(open(conf))
        for k in ("confidence_score", "ptm", "iptm", "ligand_iptm", "protein_iptm",
                  "complex_plddt", "complex_iplddt"):
            fields[k] = c.get(k)
    return (cif, fields) if cif and cif.exists() else None


def chai_delivered(target_dir: Path):
    """Chai writes pred.model_idx_{i}.cif + scores.model_idx_{i}.npz; pick max aggregate_score."""
    scores = sorted(target_dir.glob("scores.model_idx_*.npz"))
    if not scores:
        return None
    best_i, best_agg, best_npz = None, -np.inf, None
    for sp in scores:
        i = int(sp.stem.split("_")[-1])
        z = np.load(sp, allow_pickle=True)
        agg = float(np.ravel(z["aggregate_score"])[0]) if "aggregate_score" in z else -np.inf
        if agg > best_agg:
            best_i, best_agg, best_npz = i, agg, z
    cif = target_dir / f"pred.model_idx_{best_i}.cif"
    if not cif.exists():
        return None
    fields = {"aggregate_score": best_agg}
    for k in ("ptm", "iptm"):
        if k in best_npz:
            fields[k] = float(np.ravel(best_npz[k])[0])
    if "per_chain_pair_iptm" in best_npz:  # ligand = last chain; proxy ligand_iptm
        m = np.array(best_npz["per_chain_pair_iptm"])
        m = m[0] if m.ndim == 3 else m
        if m.ndim == 2 and m.shape[0] >= 2:
            fields["ligand_iptm"] = float(np.max(m[-1, :-1]))
    return cif, fields


# ------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifests", nargs="+", required=True)
    ap.add_argument("--boltz_root", default="results/gpu_d3/boltz2")
    ap.add_argument("--chai_root", default="results/gpu_d3/chai")
    ap.add_argument("--gt_cache", default="results/gpu_d3/ground_truth")
    ap.add_argument("--out", default="results/gpu_d3/scored.csv")
    ap.add_argument("--only", default="", help="comma-sep target_ids for a dry validation run")
    args = ap.parse_args()

    only = set(args.only.split(",")) if args.only else None
    gt_cache = Path(args.gt_cache)
    rows = []
    for mpath in args.manifests:
        man = pd.read_csv(mpath)
        dataset = man["dataset"].iloc[0]
        for _, r in man.iterrows():
            tid = r["target_id"]
            if only and tid not in only:
                continue
            safe = str(tid).replace("/", "_").replace(" ", "_")
            pdb, ccd = r["pdb_id"], str(r.get("ligand_ccd") or "").strip()
            label_ok = bool(r.get("label_available")) and ccd and ccd.lower() != "nan"
            gt = fetch_gt(pdb, gt_cache) if label_ok else None
            for model, root, deliver in (
                ("boltz2", Path(args.boltz_root) / dataset, boltz_delivered),
                ("chai", Path(args.chai_root) / dataset, chai_delivered),
            ):
                # Boltz nests under boltz_results_<name>/; Chai writes <name>/ directly
                cand = list(root.glob(f"boltz_results_{safe}")) + [root / safe]
                tdir = next((c for c in cand if c.exists()), None)
                rec = {"dataset": dataset, "model": model, "target_id": tid,
                       "pdb_id": pdb, "ligand_ccd": ccd, "ligand_rmsd": None,
                       "correct": None, "status": ""}
                if tdir is None:
                    rec["status"] = "no-prediction"
                    rows.append(rec)
                    continue
                dv = deliver(tdir)
                if dv is None:
                    rec["status"] = "no-delivered-pose"
                    rows.append(rec)
                    continue
                cif, fields = dv
                rec.update(fields)
                if label_ok and gt is not None:
                    try:
                        lr, st = score_pose(Path(cif), gt, ccd)
                        rec["ligand_rmsd"] = lr
                        rec["correct"] = int(lr <= 2.0) if lr is not None else None
                        rec["status"] = st
                    except Exception as e:  # noqa: BLE001
                        rec["status"] = f"err:{e}"
                else:
                    rec["status"] = "no-label" if not label_ok else "no-gt"
                rows.append(rec)

    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    ok = df[df.ligand_rmsd.notna()]
    print(f"wrote {args.out}: {len(df)} rows, scored {len(ok)} with RMSD")
    if len(ok):
        for m, g in ok.groupby("model"):
            print(f"  {m}: success<=2A = {(g.ligand_rmsd <= 2).mean():.3f}  (n={len(g)})")


if __name__ == "__main__":
    main()
