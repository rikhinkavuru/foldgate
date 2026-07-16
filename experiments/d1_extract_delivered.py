"""D1 step 0: cache each model's DELIVERED pose CIF out of the 39.5 GB RNP structure tarball.

`build_pose_features.py` streams the same tarball but parses all ~25 diffusion CIFs per
(model, system) and then throws the coordinates away, keeping only scalar features. D1 needs the
raw delivered coordinates again (to redo pose RMSD in a single common frame), so re-streaming the
whole tarball for every experiment would be the dominant cost. This driver streams it ONCE and
writes a small, reusable artifact:

    data/processed/delivered_poses.tar.gz     members: {model}/{system}.cif
    data/processed/delivered_manifest.csv     (model, system, seed, sample, ranking_score)

The delivered pose is the argmax of `ranking_score` in the group's `ranking_scores.csv`, which is
the same selection rule `build_pose_features.py` uses, so the cached pose matches the delivered
row in `rnp_delivered.parquet`. Only the delivered CIF is parsed downstream; here we copy bytes,
so the run is I/O bound rather than gemmi bound.

Torch-free, CPU-only, resumable-by-restart (the output tar is written atomically at the end).
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

TARBALL = ROOT / "data" / "raw" / "prediction_files.tar.gz"
OUT_TAR = ROOT / "data" / "processed" / "delivered_poses.tar.gz"
OUT_MANIFEST = ROOT / "data" / "processed" / "delivered_manifest.csv"

# Tarball dirs -> parquet `method` key (io.rnp.RNP_METHODS inverted). rfaa is not a diffusion
# co-folding model and af3_no_template is an ablation; neither is an analyzed method, so both are
# skipped. boltz2 is kept: it widens K for the packing floor even though its temporal cutoff
# differs (2023-06-30 vs 2021-09-30), which the novelty strata already encode.
DIR_TO_METHOD = {
    "af3": "af3",
    "boltz": "boltz1",
    "boltz1x": "boltz1x",
    "boltz2": "boltz2",
    "chai": "chai",
    "protenix": "protenix",
}


def _group_key(name: str):
    """Return (model_dir, system, basename) for a prediction_files member, else None."""
    parts = name.split("/")
    if "prediction_files" not in parts:
        return None
    i = parts.index("prediction_files")
    if len(parts) < i + 4:
        return None
    return parts[i + 1], parts[i + 2], parts[-1]


def _pick_delivered(files: dict[str, bytes]):
    """Return (cif_bytes, seed, sample, ranking_score) for the top-ranked sample, else None."""
    ranking = files.get("ranking_scores.csv")
    if ranking is None:
        return None
    best = None
    for r in csv.DictReader(io.StringIO(ranking.decode())):
        try:
            score = float(r["ranking_score"])
            seed, sample = int(r["seed"]), int(r["sample"])
        except (KeyError, ValueError, TypeError):
            continue
        if best is None or score > best[0]:
            best = (score, seed, sample)
    if best is None:
        return None
    score, seed, sample = best
    cif = files.get(f"seed-{seed}_sample-{sample}.cif")
    if cif is None:
        return None
    return cif, seed, sample, score


def run(limit: int | None = None, log_every: int = 200) -> int:
    if not TARBALL.exists():
        raise SystemExit(f"missing {TARBALL}")
    OUT_TAR.parent.mkdir(parents=True, exist_ok=True)
    tmp_tar = OUT_TAR.with_suffix(".tmp")
    manifest = []
    cur = None
    files: dict[str, bytes] = {}
    n_groups = n_written = 0
    t0 = time.time()

    out = tarfile.open(tmp_tar, "w:gz", compresslevel=6)

    def flush() -> bool:
        """Write the buffered group's delivered pose; return False to stop the stream."""
        nonlocal files, n_groups, n_written, cur
        if cur is None or not files:
            files = {}
            return True
        model_dir, system = cur
        files_local, files = files, {}
        n_groups += 1
        got = _pick_delivered(files_local)
        if got is not None:
            cif, seed, sample, score = got
            method = DIR_TO_METHOD[model_dir]
            info = tarfile.TarInfo(name=f"{method}/{system}.cif")
            info.size = len(cif)
            info.mtime = 0
            out.addfile(info, io.BytesIO(cif))
            manifest.append({"method": method, "system_id": system, "seed": seed,
                             "sample": sample, "ranking_score": score})
            n_written += 1
        if n_groups % log_every == 0:
            rate = n_groups / max(time.time() - t0, 1e-9)
            print(f"  {n_groups} groups seen, {n_written} delivered written "
                  f"({rate:.1f} groups/s, {time.time()-t0:.0f}s)", flush=True)
        return not (limit and n_groups >= limit)

    with tarfile.open(TARBALL, "r|gz") as t:
        for m in t:
            if not m.isfile():
                continue
            g = _group_key(m.name)
            if g is None:
                continue
            model_dir, system, base = g
            if model_dir not in DIR_TO_METHOD:
                continue
            if cur != (model_dir, system):
                if not flush():
                    break
                cur = (model_dir, system)
            if base.endswith(".cif") or base == "ranking_scores.csv":
                files[base] = t.extractfile(m).read()
    flush()
    out.close()
    tmp_tar.replace(OUT_TAR)

    with open(OUT_MANIFEST, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["method", "system_id", "seed", "sample", "ranking_score"])
        w.writeheader()
        w.writerows(manifest)
    print(f"\nwrote {OUT_TAR} ({n_written} delivered poses over {n_groups} groups) "
          f"in {time.time()-t0:.0f}s", flush=True)
    print(f"wrote {OUT_MANIFEST}", flush=True)
    return n_written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None, help="stop after N (model, system) groups")
    args = ap.parse_args()
    run(limit=args.limit)


if __name__ == "__main__":
    main()
