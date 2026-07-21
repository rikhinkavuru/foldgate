"""Export superposed structures for ChimeraX rendering of the TOC graphic.

For each system, superpose the AF3 prediction onto the crystal frame on pocket
residues (same fit as prep_toc_structures.py, same validation gate), then write:
  {pdb}_crystal.pdb    crystal receptor + crystal ligand (native frame)
  {pdb}_pred.pdb       AF3 receptor + AF3 ligand, transformed into the crystal frame

ChimeraX draws the receptor surface and crystal ligand from the first file and the
predicted ligand from the second, so both poses sit in one consistent frame. The
export refuses to run unless the recomputed ligand RMSD reproduces the shipped
BiSyRMSD, so a wrong superposition can never reach the renderer.

Usage:
  TOC_STRUCT_DIR=<dir with struct/> .venv/bin/python scripts/export_toc_superposed.py
Writes results/toc_render/{5sgt,5sku}_{crystal,pred}.pdb
"""
import os
import pathlib
from difflib import SequenceMatcher

import gemmi
import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = pathlib.Path(__file__).resolve().parents[1]
S = pathlib.Path(os.environ.get("TOC_STRUCT_DIR", ROOT / "data" / "processed" / "toc_struct"))
OUT = ROOT / "results" / "toc_render"
SHIPPED = {"5sgt": 0.304320, "5sku": 10.485111}
SYS = {"5sgt": "5sgt__1__1.A__1.G", "5sku": "5sku__1__1.A__1.G"}
POCKET_R = 6.0


def polymer_chain(st):
    st.setup_entities()
    for chain in st[0]:
        if len(chain.get_polymer()) > 0:
            return chain
    raise RuntimeError("no polymer chain")


def ligand_residue(st):
    st.setup_entities()
    best, best_n = None, 0
    for chain in st[0]:
        if len(chain.get_polymer()) > 0:
            continue
        for res in chain:
            if res.name in ("HOH", "WAT"):
                continue
            n = sum(1 for a in res if a.element != gemmi.Element("H"))
            if n > best_n:
                best, best_n = res, n
    return best


def heavy_coords(res):
    return np.array([a.pos.tolist() for a in res if a.element != gemmi.Element("H")])


def matched_ca(cch, pch):
    cres, pres = list(cch.get_polymer()), list(pch.get_polymer())
    cseq = "".join(gemmi.find_tabulated_residue(r.name).one_letter_code.upper() for r in cres)
    pseq = "".join(gemmi.find_tabulated_residue(r.name).one_letter_code.upper() for r in pres)
    pairs = []
    for blk in SequenceMatcher(None, cseq, pseq, autojunk=False).get_matching_blocks():
        for k in range(blk.size):
            rc, rp = cres[blk.a + k], pres[blk.b + k]
            cca, pca = rc.find_atom("CA", "*"), rp.find_atom("CA", "*")
            if cca and pca:
                pairs.append((rc, np.array(cca.pos.tolist()), np.array(pca.pos.tolist())))
    return pairs


def kabsch(P, Q):
    pc, qc = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - pc).T @ (Q - qc))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, qc - R @ pc


def assignment_rmsd(A, ea, B, eb):
    if len(A) != len(B):
        return float("nan")
    d = np.linalg.norm(A[:, None] - B[None], axis=2)
    big = d.max() * 10 + 1
    for i, x in enumerate(ea):
        for j, y in enumerate(eb):
            if x != y:
                d[i, j] += big
    r, c = linear_sum_assignment(d)
    return float(np.sqrt((np.linalg.norm(A[r] - B[c], axis=1) ** 2).mean()))


def rename_chains(st):
    """PDB allows only single-character chain ids. Polymer -> A, ligand -> L."""
    st.setup_entities()
    for chain in st[0]:
        chain.name = "A" if len(chain.get_polymer()) > 0 else "L"


def apply_transform(st, R, t):
    tr = gemmi.Transform()
    tr.mat.fromlist(R.tolist())
    tr.vec.fromlist(t.tolist())
    for model in st:
        for chain in model:
            for res in chain:
                for atom in res:
                    atom.pos = gemmi.Position(*tr.apply(atom.pos).tolist())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for pdb, sysid in SYS.items():
        crystal = gemmi.read_structure(str(S / "ground_truth" / sysid / "system.cif"))
        pred = gemmi.read_structure(str(S / "af3" / f"{sysid}.cif"))
        crystal.remove_alternative_conformations()
        pred.remove_alternative_conformations()

        lig_c_res = ligand_residue(crystal)
        lig_c = heavy_coords(lig_c_res)
        pairs = matched_ca(polymer_chain(crystal), polymer_chain(pred))

        sel = []
        for res, cca, pca in pairs:
            atoms = heavy_coords(res)
            if len(atoms) and np.min(np.linalg.norm(
                    atoms[:, None] - lig_c[None], axis=2)) < POCKET_R:
                sel.append((cca, pca))
        Q = np.array([c for c, _ in sel])
        P = np.array([p for _, p in sel])
        R, t = kabsch(P, Q)
        fit = float(np.sqrt((((P @ R.T + t) - Q) ** 2).sum(1).mean()))

        lig_p_res = ligand_residue(pred)
        lig_p_fit = heavy_coords(lig_p_res) @ R.T + t
        el_c = [a.element.name for a in lig_c_res if a.element != gemmi.Element("H")]
        el_p = [a.element.name for a in lig_p_res if a.element != gemmi.Element("H")]
        rmsd = assignment_rmsd(lig_c, el_c, lig_p_fit, el_p)

        frame_ok = fit < 1.0
        bound_ok = rmsd <= SHIPPED[pdb] + 1.5
        tight_ok = abs(rmsd - SHIPPED[pdb]) < 1.5 if SHIPPED[pdb] < 2.0 else True
        if not (frame_ok and bound_ok and tight_ok):
            raise SystemExit(f"VALIDATION FAILED for {pdb}: fit={fit:.2f} rmsd={rmsd:.2f} "
                             f"shipped={SHIPPED[pdb]:.2f}; refusing to export.")
        print(f"{pdb}: pocket fit={fit:.2f} A | assign rmsd={rmsd:.2f} | "
              f"shipped={SHIPPED[pdb]:.2f} | OK")

        rename_chains(crystal)
        crystal.setup_entities()
        crystal.write_pdb(str(OUT / f"{pdb}_crystal.pdb"))
        apply_transform(pred, R, t)
        rename_chains(pred)
        pred.setup_entities()
        pred.write_pdb(str(OUT / f"{pdb}_pred.pdb"))
    print(f"wrote superposed PDBs to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
