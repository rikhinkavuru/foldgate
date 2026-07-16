#!/usr/bin/env python3
"""Reconstruct FoldBench AlphaFold3-format input JSON for the protein-ligand targets.

FoldBench ships only a 4-target demo `alphafold3_inputs.json`. This rebuilds the full
input list from the deposited biological assemblies (RCSB `-assembly1.cif`) so Protenix
can be re-run to recover interface-ipTM (chain_pair_iptm), which the released FoldBench
tables omit.

Rule (matches the demo composition): for each `<pdb>-assembly1`, take every polymer chain
(protein / RNA / DNA) grouped by identical one-letter sequence -> one entry with `count`,
and every non-polymer, non-water residue as a CCD ligand grouped by code -> one entry with
`count`. Waters dropped. This reproduces the demo (8tuz, 7fwf, 5sbj, 8e3r) exactly.

Parity note: sequences/ligands come from the same deposited assemblies FoldBench used, so
composition matches; MSAs are fetched by Protenix `--use_msa_server` (ColabFold), the same
source FoldBench's make_predictions.sh uses. Nonstandard-residue modifications are not
reconstructed (left empty), a stated parity limitation.
"""
import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import gemmi

RCSB = "https://files.rcsb.org/download/{}-assembly1.cif"


def fetch_assembly(pdb: str, cache: Path) -> Path:
    dst = cache / f"{pdb}-assembly1.cif"
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    url = RCSB.format(pdb)
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                data = r.read()
            dst.write_bytes(data)
            return dst
        except Exception as e:  # noqa: BLE001
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))
    return dst


def polytype(kind) -> str:
    if kind in (gemmi.PolymerType.PeptideL, gemmi.PolymerType.PeptideD):
        return "protein"
    if kind == gemmi.PolymerType.Rna:
        return "rna"
    if kind in (gemmi.PolymerType.Dna, gemmi.PolymerType.DnaRnaHybrid):
        return "dna"
    return ""


def build_target(pdb: str, cif_path: Path) -> dict:
    # canonical polymer sequences + nonpolymer CCDs straight from the mmCIF metadata
    # (this is what FoldBench used: modified residues resolve to their parent letter,
    #  and terminal caps like ACE/NH2 stay part of the polymer entity, not ligands).
    doc = gemmi.cif.read(str(cif_path))
    block = doc.sole_block()
    canon = {}  # entity_id -> canonical one-letter sequence
    for row in block.find("_entity_poly.", ["entity_id", "pdbx_seq_one_letter_code_can"]):
        canon[row.str(0)] = row.str(1).replace("\n", "").replace(" ", "").upper()
    nonpoly = {}  # entity_id -> CCD comp id (water excluded)
    for row in block.find("_pdbx_entity_nonpoly.", ["entity_id", "comp_id"]):
        comp = row.str(1)
        if comp not in ("HOH", "DOD", "WAT"):
            nonpoly[row.str(0)] = comp

    st = gemmi.read_structure(str(cif_path))
    st.setup_entities()

    poly_entries = []  # (ptype, seq, count)
    ligand_entries = []  # (ccd, count)
    for ent in st.entities:
        count = len(ent.subchains)
        if ent.entity_type == gemmi.EntityType.Polymer:
            ptype = polytype(ent.polymer_type)
            seq = canon.get(ent.name, "")
            if not ptype or not seq:
                continue
            poly_entries.append((ptype, seq, count))
        elif ent.entity_type == gemmi.EntityType.NonPolymer:
            ccd = nonpoly.get(ent.name)
            if ccd:
                ligand_entries.append((ccd, count))

    sequences = []
    idc = 0
    id_letters = [chr(c) for c in range(ord("A"), ord("Z") + 1)] + [
        a + b for a in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" for b in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    ]

    def next_ids(n):
        nonlocal idc
        out = id_letters[idc : idc + n]
        idc += n
        return out

    for (ptype, seq, n) in poly_entries:
        ids = next_ids(n)
        sequences.append(
            {
                ptype: {
                    "id": ids if n > 1 else ids[0],
                    "sequence": seq,
                    "modifications": [],
                    "unpairedMsa": None,
                    "pairedMsa": None,
                    "templates": None,
                }
            }
        )
    for (ccd, n) in ligand_entries:
        ids = next_ids(n)
        sequences.append(
            {"ligand": {"id": ids if n > 1 else ids[0], "ccdCodes": [ccd]}}
        )

    return {
        "dialect": "alphafold3",
        "version": 2,
        "name": f"{pdb}-assembly1",
        "sequences": sequences,
    }


def composition(target: dict):
    out = []
    for s in target["sequences"]:
        k = next(iter(s))
        if k == "ligand":
            ids = s[k]["id"]
            n = len(ids) if isinstance(ids, list) else 1
            out.append(("ligand", tuple(s[k]["ccdCodes"]), n))
        else:
            ids = s[k]["id"]
            n = len(ids) if isinstance(ids, list) else 1
            out.append((k, s[k]["sequence"][:15], n))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets_csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default="/tmp/fb_cif")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", default="", help="comma-sep pdb ids (no -assembly1) for validation")
    args = ap.parse_args()

    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)

    import csv

    pdbs = []
    with open(args.targets_csv) as f:
        for row in csv.DictReader(f):
            pid = row["pdb_id"].replace("-assembly1", "")
            if pid not in pdbs:
                pdbs.append(pid)
    if args.only:
        want = set(args.only.split(","))
        pdbs = [p for p in pdbs if p in want]
    if args.limit:
        pdbs = pdbs[: args.limit]

    targets = []
    failed = []
    for i, pdb in enumerate(pdbs):
        try:
            cif = fetch_assembly(pdb, cache)
            tgt = build_target(pdb, cif)
            if not tgt["sequences"]:
                raise ValueError("no sequences parsed")
            targets.append(tgt)
            print(f"[{i+1}/{len(pdbs)}] {pdb}: {composition(tgt)}", flush=True)
        except Exception as e:  # noqa: BLE001
            failed.append((pdb, str(e)))
            print(f"[{i+1}/{len(pdbs)}] {pdb}: FAILED {e}", file=sys.stderr, flush=True)

    Path(args.out).write_text(json.dumps(targets, indent=2))
    print(f"\nwrote {len(targets)} targets to {args.out}; {len(failed)} failed")
    if failed:
        Path(args.out + ".failed.json").write_text(json.dumps(failed, indent=2))


if __name__ == "__main__":
    main()
