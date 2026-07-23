"""Build the ligand-novelty reference pool for the D3 external transfer (e60).

Extracts every RNP ground-truth ligand SDF, canonicalizes to SMILES with RDKit, and writes
the unique set to data/external/d3/rnp_ref.smi. e60 measures each external ligand's chemical
novelty as 1 - (max ECFP4 Tanimoto to this pool).

Honest scope: this reference is the RNP delivered ligands (post-2021 PDB), a broad and
diverse ligand set, so it measures chemical novelty relative to a reused benchmark, NOT
similarity to the models' pre-cutoff training set (which RNP ships precomputed for its own
targets but is not available for arbitrary external ligands). A training-cutoff-matched
novelty axis is the proper test and the stated next step; under this proxy the strata are
less cleanly separated by true novelty, which is why the external frontier collapse is
milder than RNP's.

Usage: .venv/bin/python scripts/build_d3_novelty_reference.py [--gt-tar PATH] [--out PATH]
"""
from __future__ import annotations

import argparse
import glob
import pathlib
import subprocess
import tempfile

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
ROOT = pathlib.Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-tar", type=pathlib.Path, default=ROOT / "data" / "raw" / "ground_truth.tar.gz")
    ap.add_argument("--out", type=pathlib.Path, default=ROOT / "data" / "external" / "d3" / "rnp_ref.smi")
    ap.add_argument("--min-heavy", type=int, default=6)
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as td:
        subprocess.run(["tar", "-xzf", str(args.gt_tar), "-C", td, "--include", "*ligand_files*.sdf"],
                       check=True)
        smis = set()
        for f in glob.glob(f"{td}/**/*.sdf", recursive=True):
            try:
                for m in Chem.SDMolSupplier(f, sanitize=True):
                    if m is not None and m.GetNumHeavyAtoms() >= args.min_heavy:
                        smis.add(Chem.MolToSmiles(m))
            except Exception:  # noqa: BLE001
                pass
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(sorted(smis)))
    print(f"wrote {len(smis)} unique reference SMILES -> {args.out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
