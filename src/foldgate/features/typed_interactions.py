"""Typed protein-ligand interaction fingerprints (E51 / reviewer D2).

E6b/E32 used an UNTYPED 4.5 A heavy-atom contact fingerprint, which is dominated by
hydrophobic bulk (most receptor residues sit within 4.5 A of the ligand through nonpolar
carbon contacts). Reviewer D2 asks whether the gate's interaction-recovery lift survives on
the POLAR / DIRECTIONAL interaction classes a medicinal chemist actually reads (H-bonds,
salt bridges, pi-stacking), not just hydrophobic-contact count.

This module computes a TYPED interaction fingerprint per structure: for each interaction
class it returns the set of receptor residues (keyed on (seqid, resname), matching
interactions.py so predicted and crystal CIFs -- which rename chains but share residue
numbering -- line up). Per-TYPE recovery is then recall of the crystal interactions of that
type by the delivered pose.

TOOLING / PROTONATION ASSUMPTIONS (stated for the paper):
  * ProLIF 2.2.0 is installed, but AF3/Boltz/Chai predicted CIFs AND the deposited crystal
    coordinates carry NO explicit hydrogens and no bond-order records. ProLIF's HBond /
    donor-acceptor detectors need explicit hydrogens on the donor, so they return ~0 H-bonds
    on hydrogen-free input. Adding hydrogens to ~5k complexes with correct tautomer /
    protonation states is a large, non-reproducible error source (crystal coords are as
    deposited), so we use a hydrogen-free heavy-atom typed scheme instead (PLIP / LigPlot
    style: heavy-atom donor-acceptor distances for H-bonds, charged-group distances for salt
    bridges, ring-centroid distance + plane angle for pi-stacking).
  * Protein protonation is assigned by standard residue rules at physiological pH: Asp/Glu
    carboxylates anionic; Arg/Lys cationic; His treated as a (protonatable) cationic +
    aromatic residue. Backbone and all sidechain N/O are treated as potential H-bond partners
    (element-based), which is the conventional heavy-atom H-bond definition when H is absent.
  * Ligand bond orders, aromaticity and formal charges are perceived from 3D coordinates with
    RDKit rdDetermineBonds (net charge assumed 0 -- unknown per ligand). This drives the
    ligand side of salt-bridge (formal charge sign) and pi-stacking (aromatic rings). When
    perception fails, those two TYPES are marked unavailable (NaN) for that structure and
    excluded from recovery; H-bond and hydrophobic never depend on perception.

Cutoffs (heavy-atom, hydrogen-free):
  H-bond      protein N/O  <-> ligand N/O            <= 3.5 A
  salt bridge protein charged atom <-> oppositely-charged ligand atom  <= 5.0 A
  pi-stacking aromatic ring centroids  <= 5.5 A  AND  plane angle in [0,30] U [50,90] deg
  hydrophobic protein sidechain/CA carbon <-> ligand carbon            <= 4.5 A
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

TYPES = ("hbond", "salt_bridge", "pi_stacking", "hydrophobic")

HBOND_CUT = 3.5
SALT_CUT = 5.0
PISTACK_CUT = 5.5
HYDRO_CUT = 4.5

# protein charged-group atoms (physiological pH)
CATION = {"ARG": {"NE", "NH1", "NH2"}, "LYS": {"NZ"}, "HIS": {"ND1", "NE2"}}
ANION = {"ASP": {"OD1", "OD2"}, "GLU": {"OE1", "OE2"}}
# protein aromatic rings (residue -> tuple of ring atom-name sets)
AROMATIC = {
    "PHE": [{"CG", "CD1", "CD2", "CE1", "CE2", "CZ"}],
    "TYR": [{"CG", "CD1", "CD2", "CE1", "CE2", "CZ"}],
    "HIS": [{"CG", "ND1", "CD2", "CE1", "NE2"}],
    "TRP": [{"CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2"}, {"CG", "CD1", "NE1", "CE2", "CD2"}],
}


def _read(cif_source):
    import gemmi
    if isinstance(cif_source, (str, Path)):
        return gemmi.read_structure(str(cif_source))
    if isinstance(cif_source, (bytes, bytearray)):
        doc = gemmi.cif.read_string(bytes(cif_source).decode())
        return gemmi.make_structure_from_block(doc.sole_block())
    doc = gemmi.cif.read_string(cif_source)
    return gemmi.make_structure_from_block(doc.sole_block())


def _is_aa(ch):
    import gemmi
    return any(
        (gemmi.find_tabulated_residue(r.name) is not None
         and gemmi.find_tabulated_residue(r.name).is_amino_acid())
        for r in ch
    )


def _collect_protein(model):
    """Gather typed protein atom arrays keyed for interaction detection."""
    keys, xyz, elem, name, resn = [], [], [], [], []
    for ch in model:
        if not _is_aa(ch):
            continue
        for r in ch:
            for a in r:
                if a.element.atomic_number <= 1:
                    continue
                keys.append((r.seqid.num, r.name))
                xyz.append([a.pos.x, a.pos.y, a.pos.z])
                elem.append(a.element.name)
                name.append(a.name)
                resn.append(r.name)
    return (np.asarray(keys, dtype=object), np.asarray(xyz, float),
            np.asarray(elem), np.asarray(name), np.asarray(resn))


def _ring_centroids(model):
    """List of (residue_key, centroid[3], normal[3]) for protein aromatic rings."""
    out = []
    for ch in model:
        if not _is_aa(ch):
            continue
        for r in ch:
            rings = AROMATIC.get(r.name)
            if not rings:
                continue
            coords = {a.name: [a.pos.x, a.pos.y, a.pos.z] for a in r}
            for ringset in rings:
                pts = [coords[n] for n in ringset if n in coords]
                if len(pts) >= 4:
                    c, nrm = _plane(np.asarray(pts, float))
                    out.append(((r.seqid.num, r.name), c, nrm))
    return out


def _plane(pts):
    c = pts.mean(0)
    _, _, vh = np.linalg.svd(pts - c)
    return c, vh[2]


def _lig_arrays(model, expected_heavy):
    """Return (xyz, elem) of the selected ligand's heavy atoms, or (None, None)."""
    cands = {}
    for ch in model:
        if _is_aa(ch):
            continue
        atoms = [(a.element.name, [a.pos.x, a.pos.y, a.pos.z]) for r in ch for a in r
                 if a.element.atomic_number > 1]
        if atoms:
            cands[ch.name] = atoms
    if not cands:
        return None, None
    chosen = None
    if expected_heavy is None:
        chosen = next(iter(cands.values())) if len(cands) == 1 else None
    else:
        for nm in sorted(cands):
            if len(cands[nm]) == int(expected_heavy):
                chosen = cands[nm]
                break
    if chosen is None:
        return None, None
    elem = np.asarray([e for e, _ in chosen])
    xyz = np.asarray([p for _, p in chosen], float)
    return xyz, elem


_PT = None
PLANAR_TOL = 0.55   # A: max out-of-plane deviation for a ring to count as aromatic/planar


def _perceive_ligand(xyz, elem):
    """Connectivity-only ligand perception from 3D coords (no bond-order/charge guess).

    RDKit's full bond-order perception (rdDetermineBonds.DetermineBonds) is fragile on real
    ligands (it over-bonds crowded geometries -> valence errors) and needs the unknown net
    charge, so instead we perceive only connectivity (covalent radii) and derive:
      * aromatic rings  -> 5-/6-membered rings that are planar (max out-of-plane dev < tol),
      * anionic atoms   -> terminal O of a carboxylate / phosphate / sulfonate
                           (C/P/S bonded to >=2 O with >=1 terminal O),
      * cationic atoms  -> guanidinium / amidine N (C bonded to >=2 N) and aliphatic-amine N
                           (N bonded only to C, not in a ring, no carbonyl neighbor).
    Returns (rings, anion_mask, cation_mask); rings is a list of (centroid[3], normal[3]);
    the masks are bool arrays aligned to xyz. Returns (None, None, None) only if connectivity
    itself fails.
    """
    global _PT
    try:
        from rdkit import Chem
        from rdkit.Chem import rdDetermineBonds
        from rdkit.Geometry import Point3D
        if _PT is None:
            _PT = Chem.GetPeriodicTable()
        rw = Chem.RWMol()
        for e in elem:
            rw.AddAtom(Chem.Atom(int(_PT.GetAtomicNumber(str(e)))))
        conf = Chem.Conformer(len(elem))
        for i, (x, y, z) in enumerate(xyz):
            conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
        mol = rw.GetMol()
        mol.AddConformer(conf, assignId=True)
        rdDetermineBonds.DetermineConnectivity(mol)
        Chem.FastFindRings(mol)   # DetermineConnectivity leaves RingInfo uninitialized
    except Exception:  # noqa: BLE001 - connectivity is the only hard requirement
        return None, None, None

    n = len(elem)
    nbr = [[] for _ in range(n)]
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        nbr[i].append(j)
        nbr[j].append(i)
    sym = [str(e) for e in elem]

    # aromatic/planar rings
    rings = []
    ri = mol.GetRingInfo()
    for ring in ri.AtomRings():
        if len(ring) in (5, 6):
            pts = xyz[list(ring)]
            c, nrm = _plane(pts)
            dev = np.abs((pts - c) @ nrm).max()
            if dev < PLANAR_TOL:
                rings.append((c, nrm))

    anion = np.zeros(n, bool)
    cation = np.zeros(n, bool)
    for i in range(n):
        if sym[i] in ("C", "S", "P"):
            oxy = [j for j in nbr[i] if sym[j] == "O"]
            if len(oxy) >= 2 and any(len(nbr[j]) == 1 for j in oxy):
                for j in oxy:
                    if len(nbr[j]) == 1:
                        anion[j] = True
        if sym[i] == "C":  # guanidinium / amidine carbon
            namine = [j for j in nbr[i] if sym[j] == "N"]
            if len(namine) >= 2:
                for j in namine:
                    cation[j] = True
    for i in range(n):
        if sym[i] == "N" and not cation[i]:
            heavy_nbr = [j for j in nbr[i] if sym[j] != "H"]
            if heavy_nbr and all(sym[j] == "C" for j in heavy_nbr) and ri.NumAtomRings(i) == 0:
                # exclude amide N: a C neighbor bonded to a terminal O
                amide = any(any(sym[k] == "O" and len(nbr[k]) == 1 for k in nbr[j])
                            for j in heavy_nbr)
                if not amide:
                    cation[i] = True
    return rings, anion, cation


def typed_fingerprint(cif_source, expected_heavy):
    """Return (fps, lig_perceived) where fps = {type: frozenset((seqid, resname))}.

    lig_perceived is False when RDKit ligand perception failed, in which case salt_bridge and
    pi_stacking are returned empty and callers should treat them as NaN.
    """
    st = _read(cif_source)
    model = st[0]
    lig_xyz, lig_elem = _lig_arrays(model, expected_heavy)
    empty = {t: frozenset() for t in TYPES}
    if lig_xyz is None or len(lig_xyz) == 0:
        return empty, False
    pkey, pxyz, pelem, pname, presn = _collect_protein(model)
    if len(pxyz) == 0:
        return empty, False

    lig_polar = np.isin(lig_elem, ["N", "O"])
    lig_carbon = lig_elem == "C"

    fps = {t: set() for t in TYPES}

    # ---- H-bond: protein N/O <-> ligand N/O <= 3.5 A ----
    p_polar = np.isin(pelem, ["N", "O"])
    _add_contacts(fps["hbond"], pkey, pxyz[p_polar], p_polar, lig_xyz[lig_polar], HBOND_CUT)

    # ---- hydrophobic: protein carbon (excl. backbone carbonyl C) <-> ligand carbon <= 4.5 ----
    p_hyd = (pelem == "C") & (pname != "C")
    _add_contacts(fps["hydrophobic"], pkey, pxyz[p_hyd], p_hyd, lig_xyz[lig_carbon], HYDRO_CUT)

    # ---- ligand perception for salt bridge + pi-stacking ----
    rings, anion, cation = _perceive_ligand(lig_xyz, lig_elem)
    perceived = rings is not None

    if perceived:
        # salt bridge: protein cation <-> ligand anion, protein anion <-> ligand cation
        lig_pos = lig_xyz[cation]
        lig_neg = lig_xyz[anion]
        p_cat = np.array([nm in CATION.get(rn, set()) for nm, rn in zip(pname, presn)])
        p_ani = np.array([nm in ANION.get(rn, set()) for nm, rn in zip(pname, presn)])
        if len(lig_neg):
            _add_contacts(fps["salt_bridge"], pkey, pxyz[p_cat], p_cat, lig_neg, SALT_CUT)
        if len(lig_pos):
            _add_contacts(fps["salt_bridge"], pkey, pxyz[p_ani], p_ani, lig_pos, SALT_CUT)
        # pi-stacking: aromatic ring centroids <=5.5 and plane angle parallel or T
        prot_rings = _ring_centroids(model)
        for rkey, c, nrm in prot_rings:
            for lc, lnrm in rings:
                if np.linalg.norm(c - lc) <= PISTACK_CUT:
                    ang = np.degrees(np.arccos(min(1.0, abs(float(np.dot(nrm, lnrm))))))
                    if ang <= 30.0 or ang >= 50.0:
                        fps["pi_stacking"].add(rkey)

    out = {t: frozenset(fps[t]) for t in TYPES}
    return out, perceived


def _add_contacts(target_set, pkey, pxyz_sub, pmask, lig_sub, cutoff):
    """Add residue keys of protein atoms (subset pxyz_sub with mask pmask over pkey) that
    sit within cutoff of any ligand atom in lig_sub."""
    if len(pxyz_sub) == 0 or len(lig_sub) == 0:
        return
    dmin = np.linalg.norm(pxyz_sub[:, None, :] - lig_sub[None, :, :], axis=-1).min(1)
    hit = np.where(dmin < cutoff)[0]
    if len(hit) == 0:
        return
    sub_keys = pkey[pmask]
    for i in hit:
        target_set.add(tuple(sub_keys[i]))


def typed_recovery(pred_fps, true_fps, pred_perc, true_perc):
    """Per-type recall of crystal interactions by the predicted pose.

    Returns {f'{type}_recall': float|nan, f'n_true_{type}': int}. salt_bridge / pi_stacking
    are NaN when either ligand failed RDKit perception (ambiguous typing)."""
    out = {}
    for t in TYPES:
        needs_perc = t in ("salt_bridge", "pi_stacking")
        n_true = len(true_fps[t])
        out[f"n_true_{t}"] = n_true
        if needs_perc and not (pred_perc and true_perc):
            out[f"{t}_recall"] = np.nan
            out[f"n_true_{t}"] = np.nan
            continue
        if n_true == 0:
            out[f"{t}_recall"] = np.nan
        else:
            out[f"{t}_recall"] = len(pred_fps[t] & true_fps[t]) / n_true
    return out


def load_true_typed(ground_truth_tar, limit_systems=None):
    """Stream ground_truth.tar.gz -> {system_id: {heavy_count: (fps, perceived)}}."""
    import tarfile
    out: dict[str, dict[int, tuple]] = {}
    with tarfile.open(str(ground_truth_tar), "r|gz") as t:
        for m in t:
            if not m.isfile() or not m.name.endswith("system.cif"):
                continue
            parts = m.name.split("/")
            if "ground_truth" not in parts:
                continue
            system = parts[parts.index("ground_truth") + 1]
            if limit_systems is not None and system not in limit_systems:
                continue
            raw = t.extractfile(m).read()
            try:
                st = _read(raw)
            except Exception:  # noqa: BLE001
                continue
            model = st[0]
            per_heavy = {}
            # enumerate every non-AA ligand chain by heavy-atom count
            for ch in model:
                if _is_aa(ch):
                    continue
                heavy = sum(1 for r in ch for a in r if a.element.atomic_number > 1)
                if heavy < 4:  # skip waters/ions/tiny
                    continue
                fps, perc = typed_fingerprint(raw, heavy)
                per_heavy[heavy] = (fps, perc)
            if per_heavy:
                out[system] = per_heavy
    return out
