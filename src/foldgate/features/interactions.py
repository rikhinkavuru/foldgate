"""Protein-ligand interaction-fingerprint (contact) recovery vs the crystal structure (E6b / W2).

A genuinely downstream, non-circular quality metric. E6's purity is arithmetically 1 - selective
risk, so it cannot show the gate buys anything beyond the guarantee. Contact recovery instead asks
a different question -- one a medicinal chemist actually reads: does the predicted pose recover the
*correct protein-ligand interactions* (the contacting residues)? A pose can clear 2 A RMSD yet miss
key contacts, or sit just over 2 A yet recover them, so recovery is not a relabelling of the RMSD
gate.

Contacts are intramolecular (which receptor residues sit within a cutoff of the ligand), so no
predicted-to-true superposition is needed. Predicted and ground-truth CIFs share residue numbering
but rename chains, so we key contacts on (seqid, resname), not (chain, seqid).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

CONTACT_CUTOFF = 4.0   # Angstrom heavy-atom contact distance
IFP_CUTOFF = 4.5       # slightly looser for the fingerprint (captures H-bond/vdW shell)


def _read(cif_source):
    import gemmi
    if isinstance(cif_source, (str, Path)):
        return gemmi.read_structure(str(cif_source))
    if isinstance(cif_source, (bytes, bytearray)):
        doc = gemmi.cif.read_string(bytes(cif_source).decode())
        return gemmi.make_structure_from_block(doc.sole_block())
    doc = gemmi.cif.read_string(cif_source)  # assume text
    return gemmi.make_structure_from_block(doc.sole_block())


def contact_fingerprint(cif_source, expected_heavy: int | None, cutoff: float = IFP_CUTOFF) -> frozenset:
    """Return the set of (seqid, resname) receptor residues contacting the delivered ligand.

    ``expected_heavy`` selects the ligand chain by heavy-atom count in multi-ligand CIFs
    (predicted CIFs carry no hydrogens); None uses the sole non-amino-acid chain.
    """
    import gemmi
    st = _read(cif_source)
    m = st[0]
    prot_key, prot_xyz = [], []
    lig_chains: dict[str, list] = {}
    for ch in m:
        is_aa = any(
            (gemmi.find_tabulated_residue(r.name) is not None
             and gemmi.find_tabulated_residue(r.name).is_amino_acid())
            for r in ch
        )
        if is_aa:
            for r in ch:
                for a in r:
                    if a.element.atomic_number > 1:
                        prot_key.append((r.seqid.num, r.name))
                        prot_xyz.append([a.pos.x, a.pos.y, a.pos.z])
        else:
            atoms = [(a.element.atomic_number, [a.pos.x, a.pos.y, a.pos.z]) for r in ch for a in r]
            heavy = [xyz for el, xyz in atoms if el > 1]
            if heavy:
                lig_chains[ch.name] = heavy
    lig = _select(lig_chains, expected_heavy)
    if lig is None or not prot_xyz:
        return frozenset()
    lig = np.asarray(lig, float)
    pxyz = np.asarray(prot_xyz, float)
    dmin = np.linalg.norm(pxyz[:, None] - lig[None, :], axis=-1).min(1)
    idx = np.where(dmin < cutoff)[0]
    return frozenset(prot_key[i] for i in idx)


def _select(lig_chains: dict, expected_heavy: int | None):
    if not lig_chains:
        return None
    if expected_heavy is None:
        return next(iter(lig_chains.values())) if len(lig_chains) == 1 else None
    for name in sorted(lig_chains):
        if len(lig_chains[name]) == int(expected_heavy):
            return lig_chains[name]
    return None


def ifp_metrics(pred: frozenset, true: frozenset) -> dict:
    """Recall / precision / Jaccard of predicted contacts against the crystal contacts."""
    if not true:
        return {"ifp_recall": np.nan, "ifp_precision": np.nan, "ifp_jaccard": np.nan,
                "n_true_contacts": 0, "n_pred_contacts": len(pred)}
    inter = len(pred & true)
    union = len(pred | true)
    return {"ifp_recall": inter / len(true),
            "ifp_precision": inter / len(pred) if pred else np.nan,
            "ifp_jaccard": inter / union if union else np.nan,
            "n_true_contacts": len(true), "n_pred_contacts": len(pred)}


def load_true_contacts(ground_truth_tar: str | Path, cutoff: float = IFP_CUTOFF) -> dict:
    """Stream ground_truth.tar.gz -> {system_id: {heavy_atom_count: contact_fingerprint}}.

    Each system ships ``system.cif`` (the crystal complex). We index the true contact set by the
    ligand's heavy-atom count so the driver can match it to each model's delivered ligand.
    """
    import tarfile
    out: dict[str, dict[int, frozenset]] = {}
    with tarfile.open(str(ground_truth_tar), "r|gz") as t:
        for m in t:
            if not m.isfile() or not m.name.endswith("system.cif"):
                continue
            parts = m.name.split("/")
            if "ground_truth" not in parts:
                continue
            system = parts[parts.index("ground_truth") + 1]
            raw = t.extractfile(m).read()
            try:
                st = _read(raw)
            except Exception:  # noqa: BLE001
                continue
            # index every ligand chain's contacts by its heavy-atom count
            per_heavy = {}
            mm = st[0]
            import gemmi
            prot_key, prot_xyz, lig_chains = [], [], {}
            for ch in mm:
                is_aa = any(
                    (gemmi.find_tabulated_residue(r.name) is not None
                     and gemmi.find_tabulated_residue(r.name).is_amino_acid())
                    for r in ch
                )
                if is_aa:
                    for r in ch:
                        for a in r:
                            if a.element.atomic_number > 1:
                                prot_key.append((r.seqid.num, r.name))
                                prot_xyz.append([a.pos.x, a.pos.y, a.pos.z])
                else:
                    heavy = [[a.pos.x, a.pos.y, a.pos.z] for r in ch for a in r
                             if a.element.atomic_number > 1]
                    if heavy:
                        lig_chains[ch.name] = heavy
            if not prot_xyz:
                continue
            pxyz = np.asarray(prot_xyz, float)
            for heavy in lig_chains.values():
                lig = np.asarray(heavy, float)
                dmin = np.linalg.norm(pxyz[:, None] - lig[None, :], axis=-1).min(1)
                fp = frozenset(prot_key[i] for i in np.where(dmin < cutoff)[0])
                per_heavy[len(heavy)] = fp
            out[system] = per_heavy
    return out
