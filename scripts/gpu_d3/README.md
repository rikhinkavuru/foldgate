# D3: external-dataset co-folding generation (Boltz-2 + Chai-1)

Self-contained GPU package that regenerates Boltz-2 and Chai-1 predictions (with native
confidence) for two held-out benchmarks beyond RNP, so foldgate's accept/abstain gate and
certification frontier can be evaluated on a 2nd and 3rd dataset. Everything that does not
need a GPU (manifests, tool inputs) is already prepared here; the GPU box only runs
inference and self-scoring.

Addresses reviewer D3 (a single benchmark is not enough to claim generality).

## Datasets

| Dataset | Systems | Label-ready | Source of truth |
|---|---|---|---|
| **PoseBusters-V2** | 308 | 308 | Authoritative maabuu `posebusters_pdb_ccd_ids.txt`; sequences + ligand SMILES from RCSB; ligand CCD given. (Obsolete `7D6O` -> current `8J79`, per degrado-lab.) |
| **PLINDER test** | 346 | 307 | PLINDER-derived MLSB-2024 co-folding challenge `inputs.parquet` (139 KB); all 346 are in PLINDER's **test** split. `receptor_sequence` + `ligand_smiles` ship directly; CCD resolved by matching SMILES to the parent entry's ligands so the RMSD label reuses the same path. |

"Label-ready" = the crystal ligand resolves to a single PDB CCD, so a ligand-RMSD label
can be computed. The 39 PLINDER systems without a CCD (peptide / covalent / branched
ligands) are still folded for confidence; they just carry no RMSD label.

**PLINDER, not PLINDER-bucket.** Only the 139 KB split-inputs parquet is fetched. The
multi-TB systems bucket is never downloaded; ground truth for both datasets comes from the
per-entry RCSB CIF (~1 MB each, fetched lazily at scoring time and cached).

## Files

```
manifests/posebusters_v2.csv   one row/target: target_id, pdb_id, ligand_ccd,
manifests/plinder_subset.csv     ligand_smiles, protein_sequences, gt_cif_url, label_available
boltz_inputs/<dataset>/*.yaml   Boltz-2 inputs (version 1; protein + ligand.smiles)
chai_inputs/<dataset>/*.fasta   Chai-1 inputs (>protein|name / >ligand|name SMILES)
build_manifests.py              (re)build the manifests from the id lists (needs network)
make_inputs.py                  (re)build boltz/chai inputs from a manifest (no network)
setup.sh   run.sh   score.sh   package.sh
selfscore.py                    confidence extraction + pocket-aligned symmetry-corrected RMSD
```

## Runbook (Lambda H100)

```bash
# 1. launch a CUDA Ubuntu H100 instance, then on the box:
git clone <this-repo> && cd <repo>/scripts/gpu_d3

# 2. install Boltz-2, Chai-1, scorer (isolated venvs)
bash setup.sh

# 3. smoke test: first 2 targets of each dataset x model (validates end to end, ~10 min)
bash run.sh --smoke
bash score.sh          # should print per-model success rates for the smoke targets

# 4. full run (resumable; safe to re-run after an interruption)
bash run.sh            # ~40-60 H100-hours, wall-clock longer (MSA server)

# 5. self-score + package the small file to copy back
bash score.sh          # writes results/gpu_d3/scored.csv
bash package.sh        # writes results/gpu_d3/d3_package.tar.gz (a few MB)
```

Copy `d3_package.tar.gz` back; it holds `scored.csv` plus every Boltz confidence JSON and
Chai scores NPZ (structures and PAE/pLDDT tensors are left on the box).

`scored.csv` columns: `dataset, model, target_id, pdb_id, ligand_ccd, <confidence fields>,
ligand_rmsd, correct(=rmsd<=2), status`. Boltz fields: `confidence_score, ptm, iptm,
ligand_iptm, protein_iptm, complex_plddt, complex_iplddt`. Chai fields: `aggregate_score,
ptm, iptm, ligand_iptm` (proxy from `per_chain_pair_iptm`).

## Compute budget

654 targets x 2 models = **1,308 inferences**, 5 diffusion samples each.
- Median receptor ~330 residues -> Boltz-2 ~1-2 min, Chai-1 ~1 min per target on one H100.
- **GPU compute ~40-60 H100-hours.** A long tail of large receptors (>800 residues; a few
  are >1,100) dominates the upper end.
- Wall-clock exceeds GPU time on the first pass because MSAs come from the public ColabFold
  MMseqs2 server (rate-limited); they cache per unique sequence, so re-runs are GPU-bound.
  Set `MSA_SERVER_URL` to a local MMseqs2 server to remove the rate limit.
- One H100 is enough. To halve wall-clock, run `--models boltz2` and `--models chai` on two
  GPUs in parallel.

## Conventions and decisions

- **Single-copy folding.** Each unique protein-entity sequence is folded as one copy plus
  one ligand copy (`make_inputs.py` default). This bounds runtime and yields a clean
  single-chain superposition frame for scoring (cf. the D1 single-frame protomer trap).
  `make_inputs.py --respect-counts` emits deposited stoichiometry instead.
- **Self-consistent label.** Confidence feature and RMSD label come from the *same*
  regenerated pose. The delivered pose is each model's own top rank (Boltz `_model_0`;
  Chai `argmax aggregate_score`), mirroring the RNP top-1 convention.
- **RMSD.** Pocket CA within 10 A of the deposited ligand define the superposition; the
  transform maps the predicted ligand into the crystal frame; RMSD is symmetry-corrected
  (spyrmsd). The predicted ligand is matched to the crystal ligand by graph isomorphism
  (elements + inferred bonds), not atom name, because SMILES input yields a generic ligand
  residue name. Identical machinery to `scripts/score_foldbench_lrmsd.py`.
- **MSAs on** for both models (`--use_msa_server` / `--use-msa-server`), matching how RNP /
  FoldBench evaluate these models.

## Limitations

- 39 PLINDER systems and 0 PoseBusters carry no RMSD label (non-CCD ligands); still useful
  for confidence-only analyses.
- Single-copy folding can miss interface pockets in obligate multimers; use
  `--respect-counts` for those if needed.
- ColabFold MSA-server availability / rate limits gate the first pass.

## Regenerating inputs

```bash
# manifests (network): pass the id lists
.venv/bin/python build_manifests.py --pb_ids posebusters_pdb_ccd_ids.txt \
    --plinder_inputs mlsb_inputs.parquet
# tool inputs (no network):
.venv/bin/python make_inputs.py --manifest manifests/posebusters_v2.csv
.venv/bin/python make_inputs.py --manifest manifests/plinder_subset.csv
```

Source id lists:
- PoseBusters 308: `github.com/maabuu/posebusters` attachment `posebusters_pdb_ccd_ids.txt`.
- PLINDER MLSB inputs: `gs://plinder/2024-06/v2/splits/mlsb/inputs.parquet` (public;
  `https://storage.googleapis.com/download/storage/v1/b/plinder/o/2024-06%2Fv2%2Fsplits%2Fmlsb%2Finputs.parquet?alt=media`).
