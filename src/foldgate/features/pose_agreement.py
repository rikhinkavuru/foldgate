"""Cross-model and intra-model POSE-agreement features (needs the structure tarball).

`agreement.py` gives cross-model *confidence* agreement (leave-one-out ipTM). This module
adds the stronger, orthogonal *structural* signal: do the predicted binding modes agree?

Two signals, both model-agnostic and training-free:

* **intra-model pose diversity** -- across a model's 25 diffusion predictions (5 seeds x 5
  samples) for one target, how much does the ligand placement move? A model that samples a
  single tight mode is more trustworthy than one that flips between modes. A model's own
  samples share ligand atom order, so this needs no cross-model matching (robust primary).

* **cross-model pose agreement** -- does the delivered pose of model M land in the same
  binding mode as the other models' delivered poses? A consensus mode is corroborating,
  with the honest caveat that co-folding models make correlated errors (consensus is a
  feature, not independent validation).

Method (grounded in the RNP layout `prediction_files/{model}/{system}/seed-*_sample-*.cif`):
parse coordinates with gemmi; the ligand is a non-amino-acid chain, pLDDT sits in the
B-factor column. To compare binding *mode* (not conformer), first superpose the two
structures on their shared receptor pocket (Kabsch on pocket Calpha), then take the
symmetry-corrected ligand RMSD with spyrmsd `symmrmsd(minimize=False)` -- `minimize=True`
would Kabsch-fit the ligand onto itself and discard the placement signal we want.

Scope: features are computed for the 88% of systems with a single ligand chain in the CIF;
multi-ligand-chain systems return NaN (the combiner handles missing features natively).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_COV = {1: 0.31, 5: 0.84, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57, 14: 1.11, 15: 1.07,
        16: 1.05, 17: 1.02, 34: 1.20, 35: 1.20, 53: 1.39}
POCKET_CUTOFF = 8.0   # Angstrom: receptor Calpha within this of any ligand atom
MODE_CUTOFF = 2.0     # Angstrom: "same binding mode" ligand-RMSD threshold
_MIN_POCKET = 4       # need at least this many shared pocket Calpha to superpose
XMODEL_CAP = 10.0     # cap cross-model ligand-RMSD: disagreement saturates, and receptor internal
                      # symmetry (homodimers with equivalent sites) can otherwise inflate a pair to
                      # tens of Angstrom without adding information beyond "the models disagree"


def parse_pose(cif_path: str | Path) -> dict:
    """Parse a predicted complex CIF from a file path (see ``_pose_from_structure``)."""
    import gemmi
    return _pose_from_structure(gemmi.read_structure(str(cif_path)))


def parse_pose_str(cif_text: str) -> dict:
    """Parse a predicted complex CIF from an in-memory string (for streaming a tarball)."""
    import gemmi
    doc = gemmi.cif.read_string(cif_text)
    return _pose_from_structure(gemmi.make_structure_from_block(doc.sole_block()))


def _pose_from_structure(st) -> dict:
    """Return {'lig': {chain -> {'el','xyz','names','plddt'}}, 'ca': (N,3), 'cak': [(chain,seqid)]}.

    Ligand chains are the non-amino-acid chains; receptor Calpha carry the pocket frame.
    """
    import gemmi
    m = st[0]
    lig: dict[str, dict] = {}
    ca_xyz, ca_key = [], []
    aa_idx = 0                       # ordinal of the protein chain (chain NAMES differ across models)
    for ch in m:
        is_aa = any(
            (gemmi.find_tabulated_residue(r.name) is not None
             and gemmi.find_tabulated_residue(r.name).is_amino_acid())
            for r in ch
        )
        if is_aa:
            for r in ch:
                for a in r:
                    if a.name == "CA":
                        ca_xyz.append([a.pos.x, a.pos.y, a.pos.z])
                        ca_key.append((aa_idx, r.seqid.num))   # (protein-chain ordinal, seqid)
            aa_idx += 1
        else:
            el, xyz, names, b = [], [], [], []
            for r in ch:
                for a in r:
                    el.append(a.element.atomic_number)
                    xyz.append([a.pos.x, a.pos.y, a.pos.z])
                    names.append(a.name)
                    b.append(a.b_iso)
            if el:
                lig[ch.name] = {"el": np.array(el, int), "xyz": np.array(xyz, float),
                                "names": names, "plddt": float(np.mean(b))}
    return {"lig": lig, "ca": np.array(ca_xyz, float), "cak": ca_key}


def select_ligand(pose: dict, expected_heavy: int | None = None):
    """Return (el, xyz, names, plddt) for RNP's delivered ligand chain, or None.

    A predicted CIF holds every non-amino-acid chain (the drug ligand plus any cofactors,
    ions, or crystallographic copies). We pick the chain whose heavy-atom count matches the
    delivered ligand's ``ligand_num_heavy_atoms`` (predicted CIFs carry no hydrogens, so heavy
    count == atom count). If ``expected_heavy`` is None, fall back to the sole ligand chain.
    Ties (identical-size copies) resolve to the first chain by name -- valid for intra-model
    diversity (a model labels its copies consistently across samples).
    """
    lig = pose["lig"]
    if not lig:
        return None

    def _heavy(d):
        """Heavy-atom subset (some models, e.g. Protenix, emit explicit hydrogens; ligand RMSD is
        a heavy-atom quantity, and dropping H keeps the element multiset consistent across models)."""
        keep = d["el"] != 1
        names = [n for n, k in zip(d["names"], keep, strict=False) if k]
        return d["el"][keep], d["xyz"][keep], names, d["plddt"]

    if expected_heavy is None:
        if len(lig) != 1:
            return None
        return _heavy(next(iter(lig.values())))
    for name in sorted(lig):
        d = lig[name]
        if int((d["el"] != 1).sum()) == int(expected_heavy):
            return _heavy(d)
    return None


def ligand_adjacency(elements: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    """Distance-based bond graph on the ligand (consistent across a molecule's poses)."""
    n = len(elements)
    if n == 0:
        return np.zeros((0, 0), int)
    d = np.linalg.norm(xyz[:, None] - xyz[None, :], axis=-1)
    radii = np.array([_COV.get(int(e), 0.77) for e in elements])
    thresh = radii[:, None] + radii[None, :] + 0.45
    adj = ((d < thresh) & (d > 0.1)).astype(int)
    np.fill_diagonal(adj, 0)
    return adj


def _kabsch(P: np.ndarray, Q: np.ndarray):
    """Rigid transform (R, t) best mapping P onto Q, with a reflection guard."""
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    U, _, Vt = np.linalg.svd(Pc.T @ Qc)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, Q.mean(0) - R @ P.mean(0)


def _pocket_keys(pose: dict, lig_xyz: np.ndarray, cutoff: float = POCKET_CUTOFF):
    ca = pose["ca"]
    if len(ca) == 0 or len(lig_xyz) == 0:
        return []
    dmin = np.linalg.norm(ca[:, None] - lig_xyz[None, :], axis=-1).min(1)
    return [pose["cak"][i] for i in np.where(dmin < cutoff)[0]]


def _superpose_ligand(ref_pose, mob_pose, mob_lig_xyz, pocket_keys):
    """Kabsch-align mob onto ref on the shared pocket Calpha; return transformed ligand xyz or None."""
    idx_m = {k: i for i, k in enumerate(mob_pose["cak"])}
    idx_r = {k: i for i, k in enumerate(ref_pose["cak"])}
    shared = [k for k in pocket_keys if k in idx_m and k in idx_r]
    if len(shared) < _MIN_POCKET:
        return None
    P = np.array([mob_pose["ca"][idx_m[k]] for k in shared])
    Q = np.array([ref_pose["ca"][idx_r[k]] for k in shared])
    R, t = _kabsch(P, Q)
    return (R @ mob_lig_xyz.T).T + t


def intra_model_pose_features(poses: list, ranking_scores: list, expected_heavy: int | None = None) -> dict:
    """Pose-diversity features across one model's samples for one target.

    poses: parse_pose dicts (one per sample). ranking_scores: matching scalars; the delivered
    pose is the argmax. Ligand-RMSD of every other sample to the delivered pose is measured
    after pocket superposition; we report its spread and the fraction in the delivered mode.
    expected_heavy selects the delivered ligand chain in multi-ligand CIFs.
    """
    empty = {"intra_model_pose_std": np.nan, "intra_model_pose_median": np.nan,
             "pose_consensus_frac": np.nan, "n_samples_pose": 0}
    sel = [select_ligand(p, expected_heavy) for p in poses]
    keep = [i for i, s in enumerate(sel) if s is not None]
    if len(keep) < 2:
        return empty
    from spyrmsd import rmsd as srmsd
    best = keep[int(np.argmax([ranking_scores[i] for i in keep]))]
    el_r, xyz_r, _, _ = sel[best]
    adj = ligand_adjacency(el_r, xyz_r)
    pk = _pocket_keys(poses[best], xyz_r)
    rmsds = []
    for i in keep:
        if i == best:
            continue
        el_m, xyz_m, _, _ = sel[i]
        if len(el_m) != len(el_r):
            continue
        aligned = _superpose_ligand(poses[best], poses[i], xyz_m, pk)
        if aligned is None:
            continue
        rmsds.append(float(srmsd.symmrmsd(xyz_r, aligned, el_r, el_m, adj, adj, minimize=False)))
    if not rmsds:
        return empty
    rmsds = np.array(rmsds)
    return {"intra_model_pose_std": float(rmsds.std()),
            "intra_model_pose_median": float(np.median(rmsds)),
            "pose_consensus_frac": float(np.mean(rmsds <= MODE_CUTOFF)),
            "n_samples_pose": len(keep)}


def cross_model_pose_features(delivered: dict, expected_heavy: int | None = None) -> dict:
    """Cross-model pose agreement per model for ONE system.

    delivered: {model -> parse_pose dict of that model's delivered pose}. For each model M,
    superpose the others onto M's pocket and take symmetry-corrected ligand RMSD (matching
    ligand atoms across models by name; a pair is skipped if atom sets do not correspond).
    expected_heavy selects the delivered ligand chain in multi-ligand CIFs.
    """
    from spyrmsd import rmsd as srmsd
    sel = {m: select_ligand(p, expected_heavy) for m, p in delivered.items()}
    adjs = {m: (ligand_adjacency(s[0], s[1]) if s is not None else None) for m, s in sel.items()}
    out = {}
    for mi, si in sel.items():
        if si is None:
            out[mi] = _empty_xmodel()
            continue
        el_r, xyz_r, _, _ = si
        adj_r = adjs[mi]
        sorted_r = np.sort(el_r)
        pk = _pocket_keys(delivered[mi], xyz_r)
        rmsds = []
        for mj, sj in sel.items():
            if mj == mi or sj is None:
                continue
            el_m, xyz_m, _, _ = sj
            # same molecule (element multiset) required; atom ORDER may differ across models --
            # symmrmsd finds the graph isomorphism, so no name matching is needed.
            if len(el_m) != len(el_r) or not np.array_equal(np.sort(el_m), sorted_r):
                continue
            aligned = _superpose_ligand(delivered[mi], delivered[mj], xyz_m, pk)
            if aligned is None:
                continue
            try:
                r = float(srmsd.symmrmsd(xyz_r, aligned, el_r, el_m, adj_r, adjs[mj], minimize=False))
            except Exception:  # noqa: BLE001 - graph mismatch on a pathological ligand
                continue
            rmsds.append(min(r, XMODEL_CAP))
        if not rmsds:
            out[mi] = _empty_xmodel()
            continue
        rmsds = np.array(rmsds)
        out[mi] = {"xmodel_pose_rmsd_median": float(np.median(rmsds)),
                   "xmodel_pose_rmsd_min": float(rmsds.min()),
                   "pose_consensus_cluster_size": int(np.sum(rmsds <= MODE_CUTOFF)) + 1,
                   "xmodel_n_pose": len(rmsds)}
    return out



def _empty_xmodel() -> dict:
    return {"xmodel_pose_rmsd_median": np.nan, "xmodel_pose_rmsd_min": np.nan,
            "pose_consensus_cluster_size": np.nan, "xmodel_n_pose": 0}
