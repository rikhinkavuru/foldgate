"""E51 -- typed interaction-fingerprint recovery under the gate (reviewer D2).

E6b/E32's non-circular quality metric used an UNTYPED 4.5 A heavy-atom contact fingerprint,
which is dominated by hydrophobic bulk: nearly every pocket residue sits within 4.5 A of the
ligand through nonpolar carbon contacts, so "contact recovery" mostly measures whether the
ligand is roughly in the pocket. Reviewer D2 asks whether the gate's interaction-recovery
lift survives on the POLAR / DIRECTIONAL interaction classes a chemist actually reads
(H-bonds, salt bridges, pi-stacking) -- or is it only hydrophobic-contact count.

This script recomputes recovery with a TYPED fingerprint (see
foldgate.features.typed_interactions): per interaction class, recall of the crystal
interactions of that type by the delivered pose. It then repeats the E32 RMSD-conditioned,
leakage-free, target-grouped gate analysis PER TYPE and reports the accepted-vs-rejected
within-correct (sub-2 A) recovery gap with a 90% bootstrap CI, for AF3 + one more model.

Tooling / protonation: ProLIF 2.2.0 installed but AF3/Boltz/Chai predictions and the deposited
crystal coordinates carry no explicit hydrogens, so ProLIF's directional H-bond detectors
report ~0 bonds; we use a hydrogen-free heavy-atom typed scheme (see the module docstring for
the full statement of cutoffs and protonation/tautomer assumptions).

Output: results/e51_typed_ifp.json
"""

from __future__ import annotations

import io
import sys
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from experiments._common import ALPHA, DELTA, RESDIR, load_delivered, rng, save_json  # noqa: E402
from experiments.e32_rmsd_conditioned_ifp import _oof_accept, _gap  # noqa: E402
from foldgate.features import typed_interactions as ti  # noqa: E402
from foldgate.selective.metrics import bootstrap_ci  # noqa: E402

DELIVERED_TAR = ROOT / "data" / "processed" / "delivered_poses.tar.gz"
GROUND_TRUTH = ROOT / "data" / "raw" / "ground_truth.tar.gz"
CACHE = ROOT / "data" / "processed" / "e51_typed_recovery.parquet"

MODELS = ["af3", "chai"]      # AF3 + one more (Apache-licensed, comparable pose accuracy)
MIN_ROWS = 300
N_BOOT = 1000
TYPES = list(ti.TYPES)


def _heavy_lookup(df: pd.DataFrame) -> dict:
    d = df.dropna(subset=["ligand_num_heavy_atoms"])
    g = d.groupby(["system_id", "method"])["ligand_num_heavy_atoms"].first()
    return {k: int(v) for k, v in g.items()}


def compute_typed_recovery(df: pd.DataFrame) -> pd.DataFrame:
    """Stream crystals + delivered poses, return per-(system,method) typed recovery rows."""
    heavy = _heavy_lookup(df)
    want_systems = {s for (s, m) in heavy if m in MODELS}
    print(f"loading crystal typed fingerprints for {len(want_systems)} systems ...", flush=True)
    true_typed = ti.load_true_typed(GROUND_TRUTH, limit_systems=want_systems)
    print(f"  crystal typed FPs for {len(true_typed)} systems", flush=True)

    rows = []
    n_seen = 0
    with tarfile.open(DELIVERED_TAR, "r|gz") as t:
        for m in t:
            if not m.isfile() or not m.name.endswith(".cif"):
                continue
            parts = m.name.split("/")
            model = parts[0]
            if model not in MODELS:
                continue
            system = parts[-1][:-4]
            exp = heavy.get((system, model))
            crys = true_typed.get(system)
            if exp is None or crys is None or exp not in crys:
                continue
            raw = t.extractfile(m).read()
            try:
                pred_fps, pred_perc = ti.typed_fingerprint(raw, exp)
            except Exception:  # noqa: BLE001
                continue
            true_fps, true_perc = crys[exp]
            rec = ti.typed_recovery(pred_fps, true_fps, pred_perc, true_perc)
            rec.update({"system_id": system, "method": model,
                        "lig_perceived": bool(pred_perc and true_perc)})
            rows.append(rec)
            n_seen += 1
            if n_seen % 500 == 0:
                print(f"  processed {n_seen} delivered poses", flush=True)
    out = pd.DataFrame(rows)
    print(f"typed recovery for {len(out)} (system,method) poses", flush=True)
    return out


def _per_type_gap(sub: pd.DataFrame, accepted: np.ndarray, y_correct: np.ndarray) -> dict:
    """For each interaction type, within-correct accepted-vs-rejected recovery gap + CI."""
    ev = accepted >= 0
    acc = accepted[ev].astype(bool)
    corr = y_correct[ev].astype(bool)
    res = {}
    for t in TYPES:
        col = f"{t}_recall"
        v = sub[col].to_numpy(float)[ev]
        # within-correct subset with a defined recovery for this type
        m = corr & np.isfinite(v)
        acc_m = acc[m]
        v_m = v[m]
        n_true = sub[f"n_true_{t}"].to_numpy(float)[ev][m]
        entry = {
            "n_correct_with_type": int(m.sum()),
            "n_correct_accepted": int(acc_m.sum()),
            "n_correct_rejected": int((~acc_m).sum()),
            "median_n_true_interactions": (float(np.nanmedian(n_true)) if len(n_true) else float("nan")),
        }
        if acc_m.any() and (~acc_m).any() and m.sum() >= 20:
            gap = _gap(acc_m, v_m)
            lo, hi = bootstrap_ci(_gap, acc_m, v_m, n_boot=N_BOOT)
            entry.update({
                "accepted_recovery": float(np.nanmean(v_m[acc_m])),
                "rejected_recovery": float(np.nanmean(v_m[~acc_m])),
                "within_correct_gap": gap,
                "gap_ci90": [lo, hi],
                "gap_excludes_zero": bool(np.isfinite(lo) and lo > 0),
            })
        else:
            entry.update({"within_correct_gap": float("nan"), "gap_ci90": [float("nan"), float("nan")],
                          "gap_excludes_zero": False, "note": "too few typed poses in a class"})
        res[t] = entry
    return res


def run(recompute: bool = False) -> dict:
    df = load_delivered()
    if recompute or not CACHE.exists():
        rec = compute_typed_recovery(df)
        rec.to_parquet(CACHE, index=False)
    else:
        rec = pd.read_parquet(CACHE)
        print(f"loaded cached typed recovery ({len(rec)} rows) from {CACHE.name}", flush=True)

    out = {
        "alpha": ALPHA, "delta": DELTA,
        "tool": "hydrogen-free heavy-atom typed IFP (RDKit connectivity; ProLIF/PLIP H-bond "
                "detectors unusable without explicit hydrogens on H-free predictions + crystals)",
        "protocol": "E32 target-grouped LOTO gate (GroupKFold outer, grouped fit/cal inner), "
                    "combined score; within-correct (sub-2A) per-type recovery gap, 90% row bootstrap",
        "protonation_assumptions": (
            "no explicit H; protein Asp/Glu anionic, Arg/Lys/His cationic by residue rule, all "
            "backbone+sidechain N/O are H-bond partners; ligand aromatic rings = planar 5/6-rings "
            "and charged groups from RDKit connectivity heuristics (carboxylate/phosphate/sulfonate "
            "anionic, guanidinium/amidine/aliphatic-amine cationic); net charge not assumed"),
        "types": TYPES,
        "per_model": {},
    }
    _ = rng()
    for model in MODELS:
        base = df[df.method == model].copy()
        sub = base.merge(rec[rec.method == model], on=["system_id", "method"], how="inner")
        sub = sub.dropna(subset=["rmsd", "system_id"]).reset_index(drop=True)
        if len(sub) < MIN_ROWS:
            out["per_model"][model] = {"n": int(len(sub)), "note": "too few rows"}
            continue
        y_correct = sub["correct"].to_numpy().astype(int)
        accepted = _oof_accept(sub, y_correct, ALPHA, DELTA)
        ev = accepted >= 0
        if ev.sum() < MIN_ROWS or accepted[ev].sum() in (0, ev.sum()):
            out["per_model"][model] = {"n": int(len(sub)), "note": "no usable evaluated gate split"}
            continue

        # untyped 4.5A reference (parquet ifp_recall), within-correct gap, for comparison
        untyped = {}
        if "ifp_recall" in sub.columns:
            acc = accepted[ev].astype(bool)
            corr = y_correct[ev].astype(bool)
            v = sub["ifp_recall"].to_numpy(float)[ev]
            mm = corr & np.isfinite(v)
            if acc[mm].any() and (~acc[mm]).any():
                g = _gap(acc[mm], v[mm])
                lo, hi = bootstrap_ci(_gap, acc[mm], v[mm], n_boot=N_BOOT)
                untyped = {"within_correct_gap": g, "gap_ci90": [lo, hi],
                           "accepted_recovery": float(np.nanmean(v[mm][acc[mm]])),
                           "rejected_recovery": float(np.nanmean(v[mm][~acc[mm]])),
                           "gap_excludes_zero": bool(np.isfinite(lo) and lo > 0)}

        out["per_model"][model] = {
            "n_total": int(len(sub)),
            "n_evaluated": int(ev.sum()),
            "n_correct_evaluated": int((y_correct[ev] == 1).sum()),
            "coverage_evaluated": float(accepted[ev].mean()),
            "lig_perceived_frac": float(sub["lig_perceived"].mean()),
            "untyped_4.5A_within_correct": untyped,
            "typed": _per_type_gap(sub, accepted, y_correct),
        }
    return out


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--recompute", action="store_true")
    args = ap.parse_args()
    res = run(recompute=args.recompute)
    save_json(res, RESDIR / "e51_typed_ifp.json")
    print(f"\nE51 -- typed IFP recovery under the gate (alpha={ALPHA}, delta={DELTA})")
    print(res["tool"], "\n")
    for model, r in res["per_model"].items():
        if "typed" not in r:
            print(f"{model}: {r.get('note')}")
            continue
        u = r.get("untyped_4.5A_within_correct", {})
        print(f"== {model}  cov={r['coverage_evaluated']:.2f}  n_correct={r['n_correct_evaluated']}  "
              f"lig_perceived={r['lig_perceived_frac']:.2f}")
        if u:
            ci = u["gap_ci90"]
            print(f"   untyped-4.5A   gap {u['within_correct_gap']:+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}]"
                  f"{'*' if u['gap_excludes_zero'] else ' '}  (acc {u['accepted_recovery']:.3f} / rej {u['rejected_recovery']:.3f})")
        for t in TYPES:
            e = r["typed"][t]
            ci = e["gap_ci90"]
            g = e["within_correct_gap"]
            gtxt = f"{g:+.3f}" if np.isfinite(g) else "  nan "
            citxt = f"[{ci[0]:+.3f},{ci[1]:+.3f}]" if np.isfinite(ci[0]) else "[  nan ,  nan ]"
            star = "*" if e["gap_excludes_zero"] else " "
            print(f"   {t:12} gap {gtxt} {citxt}{star}  n_corr={e['n_correct_with_type']:4d} "
                  f"(acc {e.get('accepted_recovery', float('nan')):.3f} / rej {e.get('rejected_recovery', float('nan')):.3f}) "
                  f"med_true={e['median_n_true_interactions']:.0f}")
    print("\n* = within-correct accepted-minus-rejected recovery CI excludes 0.")


if __name__ == "__main__":
    main()
