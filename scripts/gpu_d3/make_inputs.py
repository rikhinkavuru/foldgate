#!/usr/bin/env python3
"""Generate per-target Boltz-2 YAML and Chai-1 FASTA inputs from a D3 manifest.

No GPU, no network. Reads manifests/*.csv and writes:
  boltz_inputs/<dataset>/<target>.yaml   (Boltz-2: version 1, sequences[protein.., ligand.smiles])
  chai_inputs/<dataset>/<target>.fasta   (Chai-1: >protein|name=.. / >ligand|name=.. SMILES)

By default each unique protein-entity sequence is folded as ONE copy plus one ligand
copy. This keeps runtime bounded and yields a clean single-chain superposition frame for
self-scoring (see the D1 single-frame protomer note). Pass --respect-counts to instead
emit the deposited stoichiometry from the manifest's protein_counts column.

Targets with an empty ligand SMILES or empty protein sequence are skipped (they cannot be
folded); their ids are printed so the count reconciles with the manifest.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent

# chain-id letters for Boltz (A, B, ... then AA, AB, ...)
_LETTERS = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
_IDS = _LETTERS + [a + b for a in _LETTERS for b in _LETTERS]


def safe_name(target_id: str) -> str:
    return target_id.replace("/", "_").replace(" ", "_")


def parse_seqs(row: dict, respect_counts: bool) -> list[tuple[str, int]]:
    seqs = [s for s in (row.get("protein_sequences") or "").split("|") if s]
    counts_raw = [c for c in (row.get("protein_counts") or "").split("|") if c]
    out = []
    for i, s in enumerate(seqs):
        c = 1
        if respect_counts and i < len(counts_raw):
            try:
                c = max(1, int(counts_raw[i]))
            except ValueError:
                c = 1
        out.append((s, c))
    return out


def boltz_yaml(proteins: list[tuple[str, int]], smiles: str) -> str:
    lines = ["version: 1", "sequences:"]
    idx = 0
    for seq, count in proteins:
        ids = _IDS[idx : idx + count]
        idx += count
        id_field = ids[0] if count == 1 else "[" + ", ".join(ids) + "]"
        lines += [
            "  - protein:",
            f"      id: {id_field}",
            f"      sequence: {seq}",
        ]
    lig_id = _IDS[idx]
    # single-quote SMILES; escape embedded single quotes for YAML
    esc = smiles.replace("'", "''")
    lines += [
        "  - ligand:",
        f"      id: {lig_id}",
        f"      smiles: '{esc}'",
        "",
    ]
    return "\n".join(lines)


def chai_fasta(proteins: list[tuple[str, int]], smiles: str) -> str:
    blocks = []
    idx = 0
    for seq, count in proteins:
        for _ in range(count):
            cid = _IDS[idx]
            idx += 1
            blocks.append(f">protein|name={cid}\n{seq}")
    blocks.append(f">ligand|name={_IDS[idx]}\n{smiles}")
    return "\n".join(blocks) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="manifests/<dataset>.csv")
    ap.add_argument("--boltz_dir", default=str(HERE / "boltz_inputs"))
    ap.add_argument("--chai_dir", default=str(HERE / "chai_inputs"))
    ap.add_argument("--respect-counts", action="store_true", help="use deposited stoichiometry")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    with open(args.manifest) as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[: args.limit]
    dataset = rows[0]["dataset"] if rows else Path(args.manifest).stem

    bdir = Path(args.boltz_dir) / dataset
    cdir = Path(args.chai_dir) / dataset
    bdir.mkdir(parents=True, exist_ok=True)
    cdir.mkdir(parents=True, exist_ok=True)

    n_ok, skipped = 0, []
    for row in rows:
        tid = safe_name(row["target_id"])
        smiles = (row.get("ligand_smiles") or "").strip()
        proteins = parse_seqs(row, args.respect_counts)
        if not smiles or not proteins:
            skipped.append(row["target_id"])
            continue
        (bdir / f"{tid}.yaml").write_text(boltz_yaml(proteins, smiles))
        (cdir / f"{tid}.fasta").write_text(chai_fasta(proteins, smiles))
        n_ok += 1

    print(f"[{dataset}] wrote {n_ok} Boltz YAML -> {bdir}")
    print(f"[{dataset}] wrote {n_ok} Chai FASTA -> {cdir}")
    if skipped:
        print(f"[{dataset}] skipped {len(skipped)} (empty seq/smiles): {skipped[:8]}...")


if __name__ == "__main__":
    main()
