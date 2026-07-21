#!/usr/bin/env python3
"""Build the two D3 external-dataset manifests (no GPU, network only).

Produces one row per co-folding target for two held-out benchmarks so foldgate's
gate / certification frontier can be run on a 2nd and 3rd dataset beyond RNP:

  * PoseBusters-V2  -- 308 crystal complexes, canonical pose benchmark. The
    PDB/CCD id list is the authoritative maabuu list. Protein sequence(s) and the
    target-ligand SMILES come from RCSB (GraphQL); the ligand CCD is given.

  * PLINDER test    -- the PLINDER-derived MLSB 2024 co-folding challenge inputs
    (346 systems, all in PLINDER's *test* split). receptor_sequence + ligand_smiles
    ship in the lightweight `mlsb/inputs.parquet` (139 KB) -- no multi-TB bucket.
    We additionally resolve each ligand to a PDB CCD by matching its SMILES to the
    parent entry's non-polymer components, so self-scoring can reuse the same
    RCSB-CIF + CCD ligand-RMSD path as PoseBusters/FoldBench. Systems whose ligand
    is not a single CCD (peptides / covalent / branched) get an empty CCD and
    label_available=False (still folded for confidence, just no RMSD label).

Ground truth for BOTH datasets is the RCSB deposited entry CIF
(https://files.rcsb.org/download/<PDB>.cif); self-scoring locates the ligand by CCD.

Manifest schema (identical for both):
  dataset, target_id, pdb_id, ligand_ccd, ligand_smiles,
  protein_sequences (| -joined unique protein-entity sequences),
  protein_counts     (| -joined copy count per sequence),
  gt_cif_url, label_available, note
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

try:
    from rdkit import Chem
    from rdkit import RDLogger

    RDLogger.logger().setLevel(RDLogger.CRITICAL)
    _HAVE_RDKIT = True
except Exception:  # noqa: BLE001
    _HAVE_RDKIT = False

HERE = Path(__file__).resolve().parent
MANIFEST_DIR = HERE / "manifests"
GRAPHQL = "https://data.rcsb.org/graphql"
RCSB_CIF = "https://files.rcsb.org/download/{}.cif"

ENTRY_Q = """
%s: entry(entry_id: "%s") {
  rcsb_id
  polymer_entities {
    entity_poly { pdbx_seq_one_letter_code_can rcsb_entity_polymer_type }
    rcsb_polymer_entity_container_identifiers { auth_asym_ids }
  }
  nonpolymer_entities {
    rcsb_nonpolymer_entity_container_identifiers { auth_asym_ids nonpolymer_comp_id }
    nonpolymer_comp { rcsb_chem_comp_descriptor { SMILES_stereo SMILES } }
  }
}
"""

CHEMCOMP_Q = """
%s: chem_comp(comp_id: "%s") { rcsb_chem_comp_descriptor { SMILES_stereo SMILES } }
"""


def _alias(pdb: str) -> str:
    return "e_" + pdb.lower()


def _post_graphql(query: str, retries: int = 4) -> dict:
    body = json.dumps({"query": "{\n" + query + "\n}"}).encode()
    req = urllib.request.Request(
        GRAPHQL, data=body, headers={"Content-Type": "application/json"}
    )
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                out = json.loads(r.read())
            if "errors" in out and "data" not in out:
                raise RuntimeError(out["errors"][:1])
            return out.get("data", {}) or {}
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"graphql failed: {last}")


def fetch_entries(pdb_ids: list[str], batch: int = 25) -> dict:
    """pdb_id(upper) -> {proteins:[(seq,count)], ligands:{CCD:smiles}}."""
    out: dict[str, dict] = {}
    uniq = sorted({p.upper() for p in pdb_ids})
    for i in range(0, len(uniq), batch):
        chunk = uniq[i : i + batch]
        q = "".join(ENTRY_Q % (_alias(p), p) for p in chunk)
        data = _post_graphql(q)
        for p in chunk:
            ent = data.get(_alias(p))
            rec = {"proteins": [], "ligands": {}}
            if ent:
                for pe in ent.get("polymer_entities") or []:
                    ep = pe.get("entity_poly") or {}
                    if ep.get("rcsb_entity_polymer_type") != "Protein":
                        continue
                    seq = (ep.get("pdbx_seq_one_letter_code_can") or "").replace("\n", "").strip().upper()
                    ids = (pe.get("rcsb_polymer_entity_container_identifiers") or {}).get("auth_asym_ids") or []
                    if seq:
                        rec["proteins"].append((seq, max(1, len(ids))))
                for ne in ent.get("nonpolymer_entities") or []:
                    ci = ne.get("rcsb_nonpolymer_entity_container_identifiers") or {}
                    ccd = ci.get("nonpolymer_comp_id")
                    desc = (ne.get("nonpolymer_comp") or {}).get("rcsb_chem_comp_descriptor") or {}
                    smi = desc.get("SMILES_stereo") or desc.get("SMILES")
                    if ccd and smi:
                        rec["ligands"][ccd] = smi
            out[p] = rec
        print(f"  entries {min(i + batch, len(uniq))}/{len(uniq)}", flush=True)
    return out


def fetch_ccd_smiles(ccds: list[str], batch: int = 40) -> dict:
    out: dict[str, str] = {}
    uniq = sorted({c for c in ccds if c})
    for i in range(0, len(uniq), batch):
        chunk = uniq[i : i + batch]
        q = "".join(CHEMCOMP_Q % ("c_" + c.lower(), c) for c in chunk)
        data = _post_graphql(q)
        for c in chunk:
            d = data.get("c_" + c.lower())
            if d:
                desc = d.get("rcsb_chem_comp_descriptor") or {}
                smi = desc.get("SMILES_stereo") or desc.get("SMILES")
                if smi:
                    out[c] = smi
    return out


def canon(smiles: str) -> str | None:
    if not _HAVE_RDKIT or not smiles:
        return None
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    try:
        return Chem.MolToSmiles(m)
    except Exception:  # noqa: BLE001
        return None


def canon_nostereo(smiles: str) -> str | None:
    c = canon(smiles)
    if c is None:
        return None
    m = Chem.MolFromSmiles(c)
    if m is None:
        return None
    Chem.RemoveStereochemistry(m)
    return Chem.MolToSmiles(m)


# --------------------------------------------------------------------------- PB
# Obsolete/duplicate PDB entries in the published 308 list -> current replacement
# (7D6O was superseded by 8J79; documented by degrado-lab/PoseBusters-Benchmark).
PB_REPLACE = {"7D6O": "8J79"}


def build_posebusters(ids_file: Path) -> pd.DataFrame:
    pairs = []
    for line in ids_file.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        pdb, _, ccd = line.partition("_")
        pdb = PB_REPLACE.get(pdb.upper(), pdb.upper())
        pairs.append((pdb, ccd.upper()))
    pdbs = [p for p, _ in pairs]
    entries = fetch_entries(pdbs)

    # CCDs whose SMILES was not in the entry non-polymer list -> chem_comp fallback
    missing = sorted({ccd for pdb, ccd in pairs if ccd not in entries.get(pdb, {}).get("ligands", {})})
    fallback = fetch_ccd_smiles(missing) if missing else {}

    rows = []
    for pdb, ccd in pairs:
        rec = entries.get(pdb, {"proteins": [], "ligands": {}})
        smi = rec["ligands"].get(ccd) or fallback.get(ccd, "")
        seqs = [s for s, _ in rec["proteins"]]
        counts = [c for _, c in rec["proteins"]]
        note = []
        if not seqs:
            note.append("no-protein")
        if not smi:
            note.append("no-ligand-smiles")
        rows.append(
            {
                "dataset": "posebusters_v2",
                "target_id": f"{pdb}_{ccd}",
                "pdb_id": pdb,
                "ligand_ccd": ccd,
                "ligand_smiles": smi,
                "protein_sequences": "|".join(seqs),
                "protein_counts": "|".join(str(c) for c in counts),
                "gt_cif_url": RCSB_CIF.format(pdb),
                "label_available": bool(seqs and smi),
                "note": ";".join(note),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------- PLINDER
def build_plinder(inputs_parquet: Path) -> pd.DataFrame:
    inp = pd.read_parquet(inputs_parquet)
    inp["pdb_id"] = inp["system_id"].str.split("__").str[0].str.upper()
    entries = fetch_entries(inp["pdb_id"].tolist())

    rows = []
    n_ccd = 0
    for _, r in inp.iterrows():
        pdb = r["pdb_id"]
        smi = r["ligand_smiles"]
        rec = entries.get(pdb, {"proteins": [], "ligands": {}})
        # resolve CCD by SMILES match against the parent entry's non-polymer comps
        ccd = ""
        tgt_iso = canon(smi)
        tgt_flat = canon_nostereo(smi)
        for cand_ccd, cand_smi in rec["ligands"].items():
            if tgt_iso and canon(cand_smi) == tgt_iso:
                ccd = cand_ccd
                break
        if not ccd:  # relax to connectivity-only (ignore stereo) match
            for cand_ccd, cand_smi in rec["ligands"].items():
                if tgt_flat and canon_nostereo(cand_smi) == tgt_flat:
                    ccd = cand_ccd
                    break
        if ccd:
            n_ccd += 1
        note = "" if ccd else "ccd-unresolved(no-RMSD-label)"
        rows.append(
            {
                "dataset": "plinder_test",
                "target_id": r["system_id"],
                "pdb_id": pdb,
                "ligand_ccd": ccd,
                "ligand_smiles": smi,
                "protein_sequences": r["receptor_sequence"],
                "protein_counts": "1",
                "gt_cif_url": RCSB_CIF.format(pdb),
                "label_available": bool(ccd),
                "note": note,
            }
        )
    print(f"  PLINDER CCD resolved: {n_ccd}/{len(rows)}", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pb_ids", required=True, help="posebusters_pdb_ccd_ids.txt (PDB_CCD per line)")
    ap.add_argument("--plinder_inputs", required=True, help="mlsb inputs.parquet")
    ap.add_argument("--out_dir", default=str(MANIFEST_DIR))
    args = ap.parse_args()

    if not _HAVE_RDKIT:
        raise SystemExit("rdkit required for PLINDER CCD resolution; use .venv/bin/python")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("building PoseBusters-V2 manifest ...", flush=True)
    pb = build_posebusters(Path(args.pb_ids))
    pb_path = out_dir / "posebusters_v2.csv"
    pb.to_csv(pb_path, index=False)
    print(
        f"  wrote {pb_path}: {len(pb)} targets, "
        f"{int(pb.label_available.sum())} label-ready\n", flush=True
    )

    print("building PLINDER-test manifest ...", flush=True)
    pl = build_plinder(Path(args.plinder_inputs))
    pl_path = out_dir / "plinder_subset.csv"
    pl.to_csv(pl_path, index=False)
    print(
        f"  wrote {pl_path}: {len(pl)} systems, "
        f"{int(pl.label_available.sum())} label-ready", flush=True
    )


if __name__ == "__main__":
    main()
