"""Build cross-model + intra-model POSE-agreement features from the RNP structure tarball (W1).

Streams ``data/raw/prediction_files.tar.gz`` (39.5 GB) ONCE without extracting to disk:
files arrive grouped as ``prediction_files/{model}/{system}/{seed-*_sample-*.cif | ranking_scores.csv}``.
For each (model, system) we compute intra-model pose diversity across the 25 diffusion samples
and cache the delivered (top ranking_score) pose; a final pass computes cross-model agreement
per system from the cache. Output: ``data/processed/rnp_pose_features.parquet`` keyed by
(system_id, method), left-joined onto the delivered-pose table in build_features.

Torch-free, CPU-only. ~2-3 h single-core over ~300k CIFs. Progress is logged; use --limit N
to process only the first N (model, system) groups for a smoke test.
"""

from __future__ import annotations

import argparse
import csv
import io
import pickle
import sys
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foldgate.features import interactions as ix  # noqa: E402
from foldgate.features import pose_agreement as pa  # noqa: E402
from foldgate.io.rnp import RNP_METHODS  # noqa: E402

# Structure-tarball model dirs are named by CSV basename (af3, boltz, boltz1x, chai, protenix,
# boltz2, af3_no_template); the delivered table keys on the RNP method key (boltz1 for "boltz").
# Map dir -> method so the join lands; unmapped dirs (e.g. af3_no_template, rfaa) keep their name
# and simply fail to join (dropped), which is correct -- they are not analyzed methods.
DIR_TO_METHOD = {csv: key for key, csv in RNP_METHODS.items()}
# Only the analyzed methods (methods_with_enough) need pose features; skip the tarball dirs
# that never join downstream (af3_no_template, boltz2 below the count threshold, rfaa). This
# roughly a third fewer CIFs to parse. Cross-model consensus is then over the analyzed models.
ANALYZED_DIRS = {"af3", "boltz", "boltz1x", "chai", "protenix"}

TARBALL = ROOT / "data" / "raw" / "prediction_files.tar.gz"
GROUND_TRUTH = ROOT / "data" / "raw" / "ground_truth.tar.gz"
DELIVERED = ROOT / "data" / "processed" / "rnp_delivered.parquet"
OUT = ROOT / "data" / "processed" / "rnp_pose_features.parquet"
CKPT_ROWS = ROOT / "data" / "processed" / "_pose_intra_ckpt.parquet"
CKPT_CACHE = ROOT / "data" / "processed" / "_pose_cache_ckpt.pkl"
CKPT_EVERY = 400   # persist intra rows + delivered-pose cache every N groups (resumable)


def _heavy_lookup() -> dict:
    """{(system_id, method) -> delivered ligand heavy-atom count} to select the ligand in the CIF."""
    if not DELIVERED.exists():
        return {}
    df = pd.read_parquet(DELIVERED, columns=["system_id", "method", "ligand_num_heavy_atoms"])
    df = df.dropna(subset=["ligand_num_heavy_atoms"])
    g = df.groupby(["system_id", "method"])["ligand_num_heavy_atoms"].first()
    return {k: int(v) for k, v in g.items()}


def _group_key(name: str):
    """Return (model, system, basename) for a prediction_files member, or None."""
    parts = name.split("/")
    if "prediction_files" not in parts:
        return None
    i = parts.index("prediction_files")
    if len(parts) < i + 4:
        return None
    return parts[i + 1], parts[i + 2], parts[-1]


def _process_group(files: dict, expected_heavy: int | None, true_fp) -> tuple[dict | None, dict | None]:
    """From one (model, system) group's files, return (intra_row, delivered_pose_or_None).

    intra_row also carries interaction-fingerprint recovery of the delivered pose vs the crystal
    contacts ``true_fp`` (E6b), when ground truth is available.
    """
    ranking = files.get("ranking_scores.csv")
    scores = {}
    if ranking is not None:
        for r in csv.DictReader(io.StringIO(ranking.decode())):
            scores[(int(r["seed"]), int(r["sample"]))] = float(r["ranking_score"])
    poses, sc, raws = [], [], []
    for fn, raw in files.items():
        if not fn.endswith(".cif"):
            continue
        try:
            seed = int(fn.split("seed-")[1].split("_")[0])
            samp = int(fn.split("sample-")[1].split(".")[0])
        except (IndexError, ValueError):
            continue
        try:
            poses.append(pa.parse_pose_str(raw.decode()))
            sc.append(scores.get((seed, samp), np.nan))
            raws.append(raw)
        except Exception:  # noqa: BLE001 - a single malformed CIF should not kill the run
            continue
    if len(poses) < 2:
        return None, None
    intra = pa.intra_model_pose_features(poses, sc, expected_heavy)
    best = int(np.argmax(sc))
    delivered = poses[best] if pa.select_ligand(poses[best], expected_heavy) is not None else None
    # E6b: contact recovery of the delivered pose vs the crystal fingerprint
    if true_fp is not None:
        try:
            pred_fp = ix.contact_fingerprint(raws[best], expected_heavy)
            intra.update(ix.ifp_metrics(pred_fp, true_fp))
        except Exception:  # noqa: BLE001
            pass
    return intra, delivered


def run(limit: int | None = None, log_every: int = 200) -> pd.DataFrame:
    if not TARBALL.exists():
        raise SystemExit(f"missing {TARBALL} -- download it first (scripts/download_data.py W1 note)")
    heavy = _heavy_lookup()
    print(f"loaded {len(heavy)} (system,method) heavy-atom counts for ligand selection", flush=True)
    true_contacts = {}
    if GROUND_TRUTH.exists():
        print("loading crystal contact fingerprints from ground_truth.tar.gz (E6b) ...", flush=True)
        true_contacts = ix.load_true_contacts(GROUND_TRUTH)
        print(f"  loaded true contacts for {len(true_contacts)} systems", flush=True)
    rows = []                       # intra-model rows
    cache: dict[tuple, dict] = {}   # (system, method) -> {pose, heavy}
    done: set = set()               # (system, method) already processed (resume)
    if not limit and CKPT_ROWS.exists() and CKPT_CACHE.exists():
        rows = pd.read_parquet(CKPT_ROWS).to_dict("records")
        with open(CKPT_CACHE, "rb") as fh:
            cache = pickle.load(fh)
        done = {(r["system_id"], r["method"]) for r in rows}
        print(f"resuming from checkpoint: {len(done)} groups done, {len(cache)} cached", flush=True)
    cur = None
    files: dict[str, bytes] = {}
    n_groups = len(done)

    def _save_ckpt():
        pd.DataFrame(rows).to_parquet(CKPT_ROWS, index=False)
        with open(CKPT_CACHE, "wb") as fh:
            pickle.dump(cache, fh, protocol=pickle.HIGHEST_PROTOCOL)

    def flush():
        nonlocal files, n_groups
        if cur is None or not files:
            return True
        tar_model, system = cur
        method = DIR_TO_METHOD.get(tar_model, tar_model)   # tarball dir -> parquet method
        files_local, files = files, {}
        if (system, method) in done:                       # already computed in a prior run
            return True
        exp = heavy.get((system, method))
        tfp = true_contacts.get(system, {}).get(exp) if exp is not None else None
        intra, delivered = _process_group(files_local, exp, tfp)
        if intra is not None:
            rows.append({"system_id": system, "method": method, **intra})
            done.add((system, method))
            if delivered is not None:
                cache[(system, method)] = {"pose": delivered, "heavy": exp}
        n_groups += 1
        if n_groups % log_every == 0:
            print(f"  processed {n_groups} groups, {len(cache)} delivered cached", flush=True)
        if not limit and n_groups % CKPT_EVERY == 0:
            _save_ckpt()
        return not (limit and n_groups >= limit)

    with tarfile.open(TARBALL, "r|gz") as t:
        for m in t:
            if not m.isfile():
                continue
            g = _group_key(m.name)
            if g is None:
                continue
            model, system, base = g
            if model not in ANALYZED_DIRS:      # not an analyzed method -> skip (advance past data)
                continue
            if cur != (model, system):
                if not flush():
                    break
                cur = (model, system)
            # on resume, skip reading members of already-done groups (advance past their data)
            if (system, DIR_TO_METHOD.get(model, model)) in done:
                continue
            files[base] = t.extractfile(m).read()
    flush()
    if not limit:
        _save_ckpt()   # persist all intra work before the (in-memory) cross-model pass

    intra_df = pd.DataFrame(rows)
    print(f"intra-model: {len(intra_df)} (system,model) rows; computing cross-model over "
          f"{len({s for s, _ in cache})} systems ...", flush=True)

    # cross-model: group cached delivered poses by system, compute agreement per model
    by_system: dict[str, dict] = {}
    heavy_by_system: dict[str, int] = {}
    for (system, model), v in cache.items():
        by_system.setdefault(system, {})[model] = v["pose"]
        if v["heavy"] is not None:
            heavy_by_system[system] = v["heavy"]
    xrows = []
    for system, delivered in by_system.items():
        if len(delivered) < 2:
            continue
        feats = pa.cross_model_pose_features(delivered, heavy_by_system.get(system))
        for model, f in feats.items():
            xrows.append({"system_id": system, "method": model, **f})
    xdf = pd.DataFrame(xrows)

    out = intra_df.merge(xdf, on=["system_id", "method"], how="outer") if len(xdf) else intra_df
    return out


LOCK = ROOT / "data" / "processed" / "_pose_driver.lock"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None, help="process only the first N groups (smoke test)")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    # single-writer lock: refuse to run a second driver (would race on the checkpoint). flock
    # auto-releases on process death, so a reaped run leaves no stale lock.
    lock_fh = None
    if not args.limit:
        import fcntl
        LOCK.parent.mkdir(parents=True, exist_ok=True)
        lock_fh = open(LOCK, "w")
        try:
            fcntl.flock(lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("another build_pose_features driver is already running; exiting.", flush=True)
            return
    df = run(limit=args.limit)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    if not args.limit:                       # final output written; drop the resume checkpoints
        CKPT_ROWS.unlink(missing_ok=True)
        CKPT_CACHE.unlink(missing_ok=True)
    print(f"\nwrote {args.out}  ({len(df)} rows)")
    cols = [c for c in df.columns if c not in ("system_id", "method")]
    print("coverage (non-null) per feature:")
    for c in cols:
        print(f"  {c:28} {df[c].notna().mean():.2f}")
    if "intra_model_pose_std" in df:
        print("\nintra_model_pose_std by method (median):")
        print(df.groupby("method")["intra_model_pose_std"].median().round(2).to_string())


if __name__ == "__main__":
    main()
