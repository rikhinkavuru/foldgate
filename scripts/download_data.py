#!/usr/bin/env python3
"""Download the released Runs N' Poses (RNP) tabular artifacts foldgate needs.

Reuse-first: foldgate never runs a co-folding model. It consumes RNP's released
per-prediction confidences, accuracy labels, and pre-computed training-similarity.
This pulls the ~52 MB tabular bundle (no GPU, no structure blobs) into data/raw/
and extracts it, leaving the tree that experiments/build_features.py expects:

    data/raw/predictions/predictions/*.csv     # per-method ranking scores, ipTM, RMSD, LDDT-PLI
    data/raw/annotations.csv                    # training-similarity + temporal metadata
    data/raw/posebusters/posebusters_results/   # PoseBusters validity per pose

Source: Zenodo record 18366081 (RNP), Apache-2.0. Concept DOI 10.5281/zenodo.14794785.
Idempotent: files already present are skipped. Run:  python scripts/download_data.py
"""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
import urllib.request
from pathlib import Path

ZENODO_RECORD = "18366081"
API = f"https://zenodo.org/api/records/{ZENODO_RECORD}"
# Files we need (a subset of the record); tarballs are extracted after download.
WANTED = {
    "annotations.csv": None,
    "predictions.tar.gz": "predictions",       # extract-marker dir under data/raw/
    "posebusters_results.tar.gz": "posebusters",
}
# Structures for the pose-agreement + interaction-recovery features (W1/W2). Large; opt in with
# --structures. prediction_files is 39.5 GB; keep it around ~40 GB of free disk (streamed, not
# extracted, by experiments/build_pose_features.py). ground_truth is small.
STRUCTURES = {
    "ground_truth.tar.gz": None,     # left as a tarball; streamed by the feature builders
    "prediction_files.tar.gz": None,
}
RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
EXTERNAL = Path(__file__).resolve().parents[1] / "data" / "external"

# Released co-folded virtual screens for the selective-screening experiments (E16/E20), no GPU.
# Shen et al., Chemical Science 2026 (DOI 10.1039/D5SC06481C), CC-BY-4.0.
SCREEN_RECORD = "17568813"
SCREEN_FILES = ["dekois_scores.tar.gz", "dekois_stat.tar.gz", "lipcba_scores.tar.gz",
                "lipcba_stat.tar.gz", "gpcr_scores.tar.gz", "gpcr_stat.tar.gz"]
# Decoupled second task: ChEMBL-derived Boltz-2 affinity benchmark (E21), CC-BY-4.0.
AFFINITY_RECORD = "18669539"
AFFINITY_FILES = ["Data_S6_benchmark_dataset_for_predictive_performance.csv",
                  "Data_S5_dataset_AB.csv",
                  "Data_S8_predictive_performance_metrics_for_each_target.csv",
                  "Data_S2_curated_data_records.csv"]


def _fetch_url(record: str, name: str) -> str:
    return f"https://zenodo.org/api/records/{record}/files/{name}/content"


def download_screening(force: bool) -> None:
    """Fetch the released co-folded screens (E16/E20) and the affinity benchmark (E21)."""
    sdir = EXTERNAL / "screening"
    sdir.mkdir(parents=True, exist_ok=True)
    for name in SCREEN_FILES:
        dest = sdir / name
        marker = sdir / name.replace(".tar.gz", "")
        if marker.exists() and not force:
            print(f"[skip] {name} (extracted)")
            continue
        if not dest.exists() or force:
            _fetch(_fetch_url(SCREEN_RECORD, name), dest)
        with tarfile.open(dest) as t:
            t.extractall(sdir, filter="data")
    adir = EXTERNAL / "screening_affinity"
    adir.mkdir(parents=True, exist_ok=True)
    for name in AFFINITY_FILES:
        dest = adir / name
        if dest.exists() and not force:
            print(f"[skip] {name}")
            continue
        _fetch(_fetch_url(AFFINITY_RECORD, name), dest)
    print("Screening data ready under data/external/. Next:  make experiments")


def _fetch(url: str, dest: Path) -> None:
    print(f"  downloading {dest.name} ...", flush=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:  # noqa: S310 (trusted host)
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.rename(dest)


def _file_urls() -> dict[str, str]:
    """Resolve filename -> download URL from the Zenodo record API, with a direct fallback."""
    try:
        with urllib.request.urlopen(API) as r:  # noqa: S310
            entries = json.load(r).get("files", [])
        urls = {}
        for e in entries:
            key = e.get("key") or e.get("filename")
            link = (e.get("links") or {}).get("self") or (e.get("links") or {}).get("download")
            if key and link:
                urls[key] = link
        if urls:
            return urls
    except Exception as exc:  # noqa: BLE001 - fall back to the conventional URL scheme
        print(f"  (Zenodo API unavailable: {exc}; using direct file URLs)")
    return {
        name: f"https://zenodo.org/records/{ZENODO_RECORD}/files/{name}?download=1"
        for name in WANTED
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    ap.add_argument("--structures", action="store_true",
                    help="also fetch the 39.5 GB structure tarballs for pose/interaction features (W1/W2)")
    ap.add_argument("--screening", action="store_true",
                    help="fetch the released co-folded virtual screens + affinity benchmark (E16/E20/E21)")
    args = ap.parse_args()

    if args.screening:
        download_screening(args.force)

    RAW.mkdir(parents=True, exist_ok=True)
    urls = _file_urls()
    wanted = dict(WANTED)
    if args.structures:
        wanted.update(STRUCTURES)
    for name, marker in wanted.items():
        dest = RAW / name
        extracted = RAW / marker if marker else dest
        if extracted.exists() and not args.force:
            print(f"[skip] {name} (already present at {extracted.relative_to(RAW.parent.parent)})")
            continue
        if name not in urls:
            print(f"[warn] {name} not found in Zenodo record {ZENODO_RECORD}; skipping")
            continue
        if not dest.exists() or args.force:
            _fetch(urls[name], dest)
        if marker:
            print(f"  extracting {name} ...", flush=True)
            with tarfile.open(dest) as t:
                t.extractall(RAW, filter="data")
    print("\nRaw RNP artifacts ready. Next:  make features   (-> data/processed/rnp_delivered.parquet)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
