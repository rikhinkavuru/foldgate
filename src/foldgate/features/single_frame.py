"""D1: pose geometry in a SINGLE common frame containing the crystal pose.

This module exists because of one sentence in D1's Theorem T1. The floor

    1{ RMSD(x_a, x_b) > 2*rho }  <=  e_a + e_b ,   e_m = 1{ RMSD(x_m, y*) > rho },

is the triangle inequality applied to three points -- model a's pose, model b's pose, and the
crystal pose y* -- and the triangle inequality is a statement about three points in ONE metric
space. If the pairwise distance and the label distance are measured after different superpositions,
they are distances in different frames, the three points never coexist, and the inequality does not
connect them. The floor would then be a number with no theorem behind it.

`pose_agreement.cross_model_pose_features` superposes the other models onto EACH reference model's
pocket in turn, so its pair distances live in as many frames as there are reference models, and the
label it would be compared against (RNP's shipped BiSyRMSD) is computed under a third convention.
Those features are fine for their original purpose -- a monotone "do the models agree" covariate --
and are wrong for T1. This module replaces them for D1:

  1. The pocket is defined ONCE, by the crystal ligand, as the crystal receptor's C-alpha within
     `POCKET_CUTOFF` of it. Every model superposes onto that same target, so there is exactly one
     frame per system rather than one per reference model.
  2. Every model's receptor is Kabsch-superposed onto the crystal receptor on those shared pocket
     C-alpha, and its ligand is carried along by the same rigid transform.
  3. Both quantities are then measured in that frozen crystal frame with identical settings:
     the label RMSD(x_m, y*) and the pairwise RMSD(x_a, x_b). The label is RECOMPUTED here rather
     than taken from RNP's shipped `rmsd`, because a label from another alignment convention would
     not be the third point of this triangle. (The recomputed label is checked against the shipped
     one in `experiments/d1_single_frame.py` -- agreement is evidence the frame is right, and the
     two are not required to be identical since RNP's convention differs in detail.)

Why the triangle inequality survives symmetry correction. Symmetry-corrected RMSD with
`minimize=False` is d([a],[b]) = min_{g in G} ||a - g.b||, the quotient of the Euclidean metric on
R^{3n} by the ligand's automorphism group G acting by permutation. G acts by isometries, so for the
minimisers g1 of (a,b) and h1 of (b,c),
    d([a],[c]) <= ||a - g1.h1.c|| <= ||a - g1.b|| + ||g1.b - g1.h1.c|| = d([a],[b]) + d([b],[c]).
It is a genuine metric on the quotient, so T1 goes through. `minimize=True` would Kabsch-fit each
ligand onto the other, discarding the placement that the whole argument is about, and would not be
a metric between poses at all.

The superposition residual `eps` (the pocket C-alpha RMSD after Kabsch) is returned per model, not
assumed small: it is the empirical quantity that instantiates T1's deployment-time frame-transfer
slack. In the crystal frame used for validation it is a diagnostic; at deployment, where y* is
absent, it is what forces the trigger up from 2*rho to 2*rho + 2*eps.
"""

from __future__ import annotations

import numpy as np

from .pose_agreement import (
    _MIN_POCKET,
    POCKET_CUTOFF,
    _kabsch,
    ligand_adjacency,
    select_ligand,
)

RHO = 2.0              # Angstrom: the ligand-RMSD label radius (RMSD <= RHO is "correct")
DISAGREE = 2.0 * RHO   # Angstrom: the T1 geometric trigger; > 2*rho forces at least one error


def parse_sdf(text: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Parse a V2000 SDF's atom block into (elements, xyz), heavy atoms only.

    RNP ships each crystal ligand as its own `ligand_files/{chain}.sdf`. We read the coordinates
    and element symbols directly rather than going through RDKit: the automorphism group is
    recovered downstream from a distance-based bond graph (the same construction used on the
    predicted CIFs, which carry no bond block), so nothing here needs sanitised chemistry, and a
    strict sanitiser would reject exactly the unusual ligands we most want to keep.
    """
    lines = text.splitlines()
    if len(lines) < 4:
        return None
    try:
        n_atoms = int(lines[3][:3])
    except (ValueError, IndexError):
        return None
    el, xyz = [], []
    for ln in lines[4:4 + n_atoms]:
        try:
            x, y, z = float(ln[0:10]), float(ln[10:20]), float(ln[20:30])
            sym = ln[31:34].strip()
        except (ValueError, IndexError):
            return None
        z_num = _SYMBOL_TO_Z.get(sym)
        if z_num is None or z_num == 1:      # skip hydrogens: ligand RMSD is a heavy-atom quantity
            continue
        el.append(z_num)
        xyz.append([x, y, z])
    if not el:
        return None
    return np.array(el, int), np.array(xyz, float)


_SYMBOL_TO_Z = {
    "H": 1, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Na": 11, "Mg": 12, "Al": 13, "Si": 14,
    "P": 15, "S": 16, "Cl": 17, "K": 19, "Ca": 20, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28,
    "Cu": 29, "Zn": 30, "As": 33, "Se": 34, "Br": 35, "Ru": 44, "I": 53, "Pt": 78, "Au": 79,
}


def n_protein_chains(pose: dict) -> int:
    """Number of distinct protein chains in a parsed pose (C-alpha keys are (chain_ordinal, seqid))."""
    return len({k[0] for k in pose["cak"]})


def n_ligand_candidates(pose: dict, expected_heavy: int | None) -> int:
    """How many ligand chains in this CIF have exactly `expected_heavy` heavy atoms.

    The count, not the choice, is what matters for the frame. `select_ligand` resolves ties to the
    first chain by name, which is fine for intra-model diversity (a model names its copies
    consistently) but is a coin flip when a prediction contains several interchangeable copies of
    the same ligand in different sites. More than one candidate means the instance's identity is
    ambiguous and it must be excluded from the single-frame set rather than silently guessed.
    """
    if expected_heavy is None:
        return len(pose["lig"])
    return sum(1 for d in pose["lig"].values()
               if int((d["el"] != 1).sum()) == int(expected_heavy))


def frame_is_bijective(pose: dict, crystal: dict, expected_heavy: int | None) -> bool:
    """Whether this (prediction, crystal) pair admits an UNAMBIGUOUS single common frame.

    Requires a SINGLE protein chain on both sides and exactly one candidate ligand copy. Both
    clauses were forced by measurement, not caution, and the numbers are worth stating because the
    weaker criterion looks adequate and is not (agreement with RNP's shipped label, in Spearman /
    correctness-call agreement / fraction off by more than 5 A):

        no filter                                13433   0.620 / 0.854 / 0.212
        chain COUNTS match + unique ligand copy   9545   0.700 / 0.891 / 0.144
        ... of those, multi-chain receptors       3382   0.373 / 0.702 / 0.406   <- broken
        single-chain receptor + unique copy       6163   0.995 / 0.994 / 0.001   <- clean

    Why the multi-chain cases break. C-alpha are keyed on (chain ordinal, seqid). RNP's
    `receptor.cif` ships only the system's receptor while a co-folding model predicts the full
    assembly, so for 8ttz the crystal has one chain and AF3 predicts the homodimer; and even when
    the counts agree, nothing pins predicted chain A to crystal chain 1.A rather than 2.A. For a
    homodimer the two protomers are identical, so the WRONG assignment superposes the backbone just
    as well -- eps stays small -- while carrying the ligand into the other protomer's site tens of
    Angstrom away. Neither eps nor the triangle-inequality check can see this: eps is small by
    construction, and a pose displaced into the wrong site is far from everything, so its trigger
    fires and the inequality holds VACUOUSLY. Only comparison against an independently computed
    label exposes it.

    Why we exclude rather than repair. The standard evaluation convention (and RNP's own
    BiSyRMSD) resolves the assignment by minimising over symmetry-equivalent chain mappings. That
    is physically right -- two equivalent sites are indistinguishable, so occupying either is
    equally correct -- but it is a per-instance choice made with reference to the crystal ligand,
    which would put a label inside a statistic that has to stay label-free, and it would let each
    model pick its own mapping, i.e. its own frame, which is exactly what T1 forbids. Doing this
    properly means quotienting by the receptor symmetry group and is left as future work; here the
    ambiguous instances are excluded and counted.
    """
    return (n_protein_chains(pose) == 1
            and n_protein_chains(crystal) == 1
            and n_ligand_candidates(pose, expected_heavy) == 1)


def pocket_keys_from_crystal(crystal: dict, lig_xyz: np.ndarray,
                             cutoff: float = POCKET_CUTOFF) -> list:
    """C-alpha keys of the crystal receptor within `cutoff` of the crystal ligand.

    Defined once per system from the CRYSTAL ligand, so every model is superposed onto the same
    target and the resulting frame is shared. This is the single line that makes the frame common.
    """
    ca = crystal["ca"]
    if len(ca) == 0 or len(lig_xyz) == 0:
        return []
    dmin = np.linalg.norm(ca[:, None] - lig_xyz[None, :], axis=-1).min(1)
    return [crystal["cak"][i] for i in np.where(dmin < cutoff)[0]]


def to_crystal_frame(pred: dict, crystal: dict, pocket_keys: list,
                     lig_xyz: np.ndarray) -> tuple[np.ndarray, float, int] | None:
    """Map a predicted ligand into the crystal frame; return (xyz_in_frame, eps, n_shared).

    Kabsch-superposes the predicted receptor onto the crystal receptor on the shared pocket
    C-alpha and applies the SAME rigid transform to the ligand. `eps` is the post-superposition
    pocket C-alpha RMSD: how well this model reproduced the pocket backbone, and the empirical
    slack term of T1's deployment form. Returns None when fewer than `_MIN_POCKET` pocket C-alpha
    are shared (the frame would be under-determined).
    """
    idx_p = {k: i for i, k in enumerate(pred["cak"])}
    idx_c = {k: i for i, k in enumerate(crystal["cak"])}
    shared = [k for k in pocket_keys if k in idx_p and k in idx_c]
    if len(shared) < _MIN_POCKET:
        return None
    P = np.array([pred["ca"][idx_p[k]] for k in shared])
    Q = np.array([crystal["ca"][idx_c[k]] for k in shared])
    R, t = _kabsch(P, Q)
    eps = float(np.sqrt(((((R @ P.T).T + t) - Q) ** 2).sum(1).mean()))
    return (R @ lig_xyz.T).T + t, eps, len(shared)


def sym_rmsd(el_a: np.ndarray, xyz_a: np.ndarray,
             el_b: np.ndarray, xyz_b: np.ndarray) -> float:
    """Symmetry-corrected heavy-atom ligand RMSD between two poses already in a common frame.

    `minimize=False` keeps this a distance between placements (see the module docstring); each side
    contributes its own distance-based bond graph so spyrmsd can recover the isomorphism when atom
    ORDER differs between the crystal SDF and a predicted CIF. Returns NaN when the two atom sets
    are not the same molecule (a mismatch is reported and skipped, never silently coerced).
    """
    if len(el_a) != len(el_b) or not np.array_equal(np.sort(el_a), np.sort(el_b)):
        return float("nan")
    from spyrmsd import rmsd as srmsd
    try:
        return float(srmsd.symmrmsd(
            xyz_a, xyz_b, el_a, el_b,
            ligand_adjacency(el_a, xyz_a), ligand_adjacency(el_b, xyz_b),
            minimize=False))
    except Exception:  # noqa: BLE001 - non-isomorphic graphs on a pathological ligand
        return float("nan")


def system_single_frame(crystal_lig: tuple[np.ndarray, np.ndarray],
                        crystal: dict,
                        delivered: dict[str, dict],
                        expected_heavy: int | None) -> dict:
    """All single-frame geometry for ONE system's ligand instance.

    crystal_lig : (elements, xyz) of the crystal ligand, from its SDF -- the latent target y*.
    crystal     : parsed crystal receptor (`_pose_from_structure` of receptor.cif).
    delivered   : {method -> parsed delivered predicted complex}.

    Returns {'per_model': {m: {rmsd_true, eps, n_pocket}}, 'pairs': {(a,b): rmsd}}, every distance
    measured in the one crystal frame this system defines.
    """
    el_true, xyz_true = crystal_lig
    pk = pocket_keys_from_crystal(crystal, xyz_true)
    per_model: dict[str, dict] = {}
    framed: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for m, pose in delivered.items():
        sel = select_ligand(pose, expected_heavy)
        if sel is None:
            continue
        el_m, xyz_m, _, _ = sel
        got = to_crystal_frame(pose, crystal, pk, xyz_m)
        if got is None:
            continue
        xyz_f, eps, n_shared = got
        framed[m] = (el_m, xyz_f)
        per_model[m] = {
            "rmsd_true": sym_rmsd(el_true, xyz_true, el_m, xyz_f),
            "eps": eps,
            "n_pocket": n_shared,
        }
    pairs: dict[tuple[str, str], float] = {}
    ms = sorted(framed)
    for i, a in enumerate(ms):
        for b in ms[i + 1:]:
            pairs[(a, b)] = sym_rmsd(framed[a][0], framed[a][1], framed[b][0], framed[b][1])
    return {"per_model": per_model, "pairs": pairs, "n_pocket_keys": len(pk)}
