"""D1 step 1: rebuild pose geometry in a SINGLE common frame, and verify the frame is clean.

This is the Week-1 gate. Everything in D1 rests on T1's triangle inequality, which is only a
theorem if the two models' poses and the crystal pose are three points in ONE metric space. The
delivered `xmodel_pose_rmsd_*` features are NOT: `pose_agreement.cross_model_pose_features`
superposes onto each reference model's pocket in turn, so its distances live in one frame per
reference model, and RNP's shipped `rmsd` label uses a third convention. A floor built on those
numbers would be an unlabelled quantity, so D1 recomputes the geometry from the raw coordinates.

Method (see `foldgate.features.single_frame` for why each step is forced):
  * the pocket is defined ONCE per system by the CRYSTAL ligand;
  * every model's receptor is Kabsch-superposed onto the crystal receptor on those shared pocket
    C-alpha, carrying its ligand along;
  * in that frozen frame we recompute BOTH the label RMSD(x_m, y*) and every pairwise
    RMSD(x_a, x_b) with identical symmetry-corrected settings.

The run streams `delivered_poses.tar.gz` once (built by `d1_extract_delivered.py`) and projects
each pose into its system's crystal frame immediately, keeping only the framed ligand, so the
receptors never have to be held in memory at once.

FOUR CHECKS DECIDE WHETHER D1 PROCEEDS, and all four are reported here rather than assumed:

  0. BIJECTIVE FRAME. C-alpha are keyed on (chain ordinal, seqid), which corresponds only when
     both sides carry the same protein chains. They often do not: RNP's `receptor.cif` ships only
     the system's receptor while a co-folding model predicts the full assembly, so for 8ttz the
     crystal has one chain and AF3 predicts the homodimer. The ordinal then superposes onto
     whichever protomer comes first, and since the protomers are identical the fit is excellent
     while the ligand is carried into the wrong site tens of Angstrom away. Instances without an
     unambiguous correspondence (chain-count mismatch, or several interchangeable predicted copies
     of the ligand) are EXCLUDED and counted, not guessed.
  1. FRAME SANITY (pass/fail, no tolerance). In one frame the triangle inequality is an identity,
     so 1{RMSD(x_a,x_b) > 2*rho} <= e_a + e_b must hold for EVERY pair. A single violation means
     the frame or the metric is wrong and T1 is void. This is necessary but NOT sufficient, and
     the distinction is load-bearing: a pose displaced into the wrong protomer sits far from
     everything, so its trigger fires and the inequality holds VACUOUSLY. This check passed at
     100% while a fifth of the labels were wrong, which is precisely why check 2 exists.
  2. LABEL AGREEMENT (the decisive check). The recomputed label is checked against RNP's shipped
     `rmsd`. The conventions differ in detail so they need not be identical, but a high rank
     correlation and agreeing correctness calls are the only real evidence that the recomputation
     measures the intended physical quantity rather than a frame artefact.
  3. EFFECTIVE n AND THE SLACK. The pair-skip rate, the per-model pocket-superposition residual
     eps (which instantiates T1's deployment-time 2*eps slack), and the surviving counts per
     novelty stratum -- the numbers that decide which cells are certifiable at all. Note eps is
     NOT a frame-correctness diagnostic: the wrong-protomer failures have small eps by
     construction.

Outputs `data/processed/d1_single_frame.parquet` (per model-instance) and
`data/processed/d1_pairs.parquet` (per pair), plus `results/d1_frame_check.json`.
"""

from __future__ import annotations

import argparse
import sys
import tarfile
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from experiments._common import RESDIR, load_delivered, save_json  # noqa: E402
from foldgate.features import single_frame as sf  # noqa: E402
from foldgate.features.pose_agreement import _pose_from_structure, parse_pose_str, select_ligand  # noqa: E402

GT = ROOT / "data" / "raw" / "ground_truth.tar.gz"
POSES = ROOT / "data" / "processed" / "delivered_poses.tar.gz"
OUT_MODELS = ROOT / "data" / "processed" / "d1_single_frame.parquet"
OUT_PAIRS = ROOT / "data" / "processed" / "d1_pairs.parquet"


def load_crystals(system_ids: set[str]) -> dict:
    """{system_id: {'crystal': parsed receptor, 'ligs': {chain: (el, xyz)}}} from ground truth."""
    import gemmi
    out: dict[str, dict] = {}
    cur: dict[str, dict] = {}
    t0 = time.time()
    with tarfile.open(GT, "r|gz") as t:
        for m in t:
            if not m.isfile():
                continue
            parts = m.name.split("/")
            if len(parts) < 3:
                continue
            sysid = parts[1]
            if sysid not in system_ids:
                continue
            base = parts[-1]
            if base == "receptor.cif":
                try:
                    doc = gemmi.cif.read_string(t.extractfile(m).read().decode())
                    pose = _pose_from_structure(gemmi.make_structure_from_block(doc.sole_block()))
                except Exception:  # noqa: BLE001 - a malformed receptor drops the system
                    continue
                cur.setdefault(sysid, {})["crystal"] = {"ca": pose["ca"], "cak": pose["cak"]}
            elif base.endswith(".sdf") and "ligand_files" in parts:
                lig = sf.parse_sdf(t.extractfile(m).read().decode())
                if lig is not None:
                    cur.setdefault(sysid, {}).setdefault("ligs", {})[base[:-4]] = lig
    for sysid, v in cur.items():
        if "crystal" in v and v.get("ligs"):
            out[sysid] = v
    print(f"  crystals: {len(out)} systems with receptor + >=1 ligand SDF "
          f"({time.time()-t0:.0f}s)", flush=True)
    return out


def frame_poses(crystals: dict, heavy: dict) -> dict:
    """Stream delivered poses; return {(system, lig_chain, method): (el, xyz_in_crystal_frame, eps, n_pocket)}.

    Each pose is projected into its system's crystal frame on arrival and the receptor is dropped,
    so peak memory holds framed ligands only.

    The stream is always run to completion: `delivered_poses.tar.gz` is laid out method-major
    (every af3 system, then every boltz1 system, ...), so stopping early would yield one model per
    system and therefore no pairs at all. Subsetting is done by restricting `crystals` instead,
    which skips the parse -- the part that actually costs.
    """
    pockets = {}   # (system, lig_chain) -> pocket keys, computed once from the crystal ligand
    framed: dict[tuple, tuple] = {}
    n_seen = n_ok = 0
    t0 = time.time()
    with tarfile.open(POSES, "r|gz") as t:
        for m in t:
            if not m.isfile() or not m.name.endswith(".cif"):
                continue
            method, sysid = m.name.split("/")[0], Path(m.name).stem
            if sysid not in crystals:
                continue
            n_seen += 1
            try:
                pose = parse_pose_str(t.extractfile(m).read().decode())
            except Exception:  # noqa: BLE001
                continue
            cry = crystals[sysid]["crystal"]
            for chain, (el_t, xyz_t) in crystals[sysid]["ligs"].items():
                key_pk = (sysid, chain)
                if key_pk not in pockets:
                    pockets[key_pk] = sf.pocket_keys_from_crystal(cry, xyz_t)
                pk = pockets[key_pk]
                exp = heavy.get((sysid, chain, method))
                sel = select_ligand(pose, exp)
                if sel is None:
                    continue
                el_m, xyz_m, _, _ = sel
                got = sf.to_crystal_frame(pose, cry, pk, xyz_m)
                if got is None:
                    continue
                xyz_f, eps, n_shared = got
                framed[(sysid, chain, method)] = (
                    el_m, xyz_f, eps, n_shared,
                    sf.n_protein_chains(pose), sf.n_protein_chains(cry),
                    sf.n_ligand_candidates(pose, exp),
                    sf.frame_is_bijective(pose, cry, exp),
                )
                n_ok += 1
            if n_seen % 1000 == 0:
                print(f"  framed {n_ok} instances over {n_seen} poses "
                      f"({time.time()-t0:.0f}s)", flush=True)
    print(f"  framed {n_ok} model-instances over {n_seen} parsed poses "
          f"({time.time()-t0:.0f}s)", flush=True)
    return framed


def run(limit: int | None = None) -> dict:
    dl = load_delivered()
    heavy = {}
    for r in dl[["system_id", "ligand_instance_chain", "method", "ligand_num_heavy_atoms"]].dropna().itertuples():
        heavy[(r.system_id, r.ligand_instance_chain, r.method)] = int(r.ligand_num_heavy_atoms)
    want = set(dl.system_id.unique())
    if limit:
        want = set(sorted(want)[:limit])
    print(f"loading crystals for {len(want)} systems ...", flush=True)
    crystals = load_crystals(want)
    print("framing delivered poses into the crystal frame ...", flush=True)
    framed = frame_poses(crystals, heavy)

    # --- per-model table: the recomputed single-frame label ---------------------------------
    rows = []
    for (sysid, chain, method), v in framed.items():
        el_m, xyz_f, eps, n_pocket, n_prot_p, n_prot_c, n_lig_cand, ok = v
        el_t, xyz_t = crystals[sysid]["ligs"][chain]
        rows.append({
            "system_id": sysid, "ligand_instance_chain": chain, "method": method,
            "rmsd_sf": sf.sym_rmsd(el_t, xyz_t, el_m, xyz_f),
            "eps": eps, "n_pocket": n_pocket, "n_heavy": int(len(el_m)),
            "n_prot_chain_pred": n_prot_p, "n_prot_chain_crystal": n_prot_c,
            "n_lig_candidates": n_lig_cand, "frame_ok": bool(ok),
        })
    md = pd.DataFrame(rows)
    md["correct_sf"] = (md["rmsd_sf"] <= sf.RHO).astype("Int64")
    md.loc[md["rmsd_sf"].isna(), "correct_sf"] = pd.NA

    # --- pair table: single-frame pairwise disagreement -------------------------------------
    # Pairs are formed ONLY between instances whose frame is bijective on both sides. A pose that
    # was carried into the wrong protomer is not a point in this system's frame at all, so a
    # distance to it is meaningless -- and, worse, it would look like informative disagreement.
    by_inst: dict[tuple, dict] = {}
    for (sysid, chain, method), v in framed.items():
        if v[7]:
            by_inst.setdefault((sysid, chain), {})[method] = v
    prows = []
    for (sysid, chain), models in by_inst.items():
        ms = sorted(models)
        for i, a in enumerate(ms):
            for b in ms[i + 1:]:
                el_a, xyz_a = models[a][0], models[a][1]
                el_b, xyz_b = models[b][0], models[b][1]
                prows.append({
                    "system_id": sysid, "ligand_instance_chain": chain,
                    "model_a": a, "model_b": b,
                    "pair_rmsd": sf.sym_rmsd(el_a, xyz_a, el_b, xyz_b),
                })
    pd_pairs = pd.DataFrame(
        prows, columns=["system_id", "ligand_instance_chain", "model_a", "model_b", "pair_rmsd"])

    OUT_MODELS.parent.mkdir(parents=True, exist_ok=True)
    md.to_parquet(OUT_MODELS, index=False)
    pd_pairs.to_parquet(OUT_PAIRS, index=False)

    # --- CHECK 1: frame sanity. Expected violations: exactly zero. ---------------------------
    lab = md.set_index(["system_id", "ligand_instance_chain", "method"])["rmsd_sf"]
    j = pd_pairs.join(lab.rename("rmsd_a"), on=["system_id", "ligand_instance_chain", "model_a"])
    j = j.join(lab.rename("rmsd_b"), on=["system_id", "ligand_instance_chain", "model_b"])
    ok = j.dropna(subset=["pair_rmsd", "rmsd_a", "rmsd_b"]).copy()
    ok["e_a"] = (ok.rmsd_a > sf.RHO).astype(int)
    ok["e_b"] = (ok.rmsd_b > sf.RHO).astype(int)
    ok["trigger"] = (ok.pair_rmsd > sf.DISAGREE).astype(int)
    ok["violation"] = ok.trigger > (ok.e_a + ok.e_b)
    # The continuous form is the sharper implementation check: the metric triangle inequality
    # d(a,b) <= d(a,y*) + d(b,y*) must hold pointwise, not merely its thresholded consequence.
    ok["tri_slack"] = (ok.rmsd_a + ok.rmsd_b) - ok.pair_rmsd
    n_viol = int(ok.violation.sum())
    n_tri_viol = int((ok.tri_slack < -1e-6).sum())

    # --- CHECK 2: agreement with RNP's shipped label ------------------------------------------
    # Reported on the bijective-frame subset AND on everything, because the contrast between the
    # two IS the evidence that the exclusion rule targets the right instances.
    ship = dl.groupby(["system_id", "ligand_instance_chain", "method"])["rmsd"].first()
    cmp_all = md.join(ship, on=["system_id", "ligand_instance_chain", "method"]).dropna(
        subset=["rmsd_sf", "rmsd"])

    def _agree(c: pd.DataFrame) -> dict:
        if len(c) <= 2:
            return {"n": int(len(c)), "spearman": float("nan"), "pearson": float("nan"),
                    "correctness_call_agreement": float("nan"), "median_abs_diff_A": float("nan"),
                    "frac_abs_diff_gt_5A": float("nan")}
        return {
            "n": int(len(c)),
            "spearman": float(pd.Series(c.rmsd_sf).corr(pd.Series(c.rmsd), method="spearman")),
            "pearson": float(pd.Series(c.rmsd_sf).corr(pd.Series(c.rmsd))),
            "correctness_call_agreement": float(((c.rmsd_sf <= sf.RHO) == (c.rmsd <= sf.RHO)).mean()),
            "median_abs_diff_A": float((c.rmsd_sf - c.rmsd).abs().median()),
            "frac_abs_diff_gt_5A": float(((c.rmsd_sf - c.rmsd).abs() > 5.0).mean()),
        }

    agree_ok = _agree(cmp_all[cmp_all.frame_ok])
    agree_all = _agree(cmp_all)

    res = {
        "n_systems_with_crystal": len(crystals),
        "n_model_instances": int(len(md)),
        "n_instances_with_label": int(md.rmsd_sf.notna().sum()),
        "label_nan_rate": float(md.rmsd_sf.isna().mean()) if len(md) else float("nan"),
        "n_pairs": int(len(pd_pairs)),
        "n_pairs_with_rmsd": int(pd_pairs.pair_rmsd.notna().sum()),
        "pair_skip_rate": float(pd_pairs.pair_rmsd.isna().mean()) if len(pd_pairs) else float("nan"),
        "frame_check": {
            "n_pairs_checked": int(len(ok)),
            "n_threshold_violations": n_viol,
            "n_triangle_violations": n_tri_viol,
            "min_triangle_slack": float(ok.tri_slack.min()) if len(ok) else float("nan"),
            "passes": bool(n_viol == 0 and n_tri_viol == 0),
        },
        "label_agreement_vs_rnp": agree_ok,
        "label_agreement_vs_rnp_unfiltered": agree_all,
        "frame_exclusions": {
            "n_instances": int(len(md)),
            "n_frame_ok": int(md.frame_ok.sum()),
            "frac_frame_ok": float(md.frame_ok.mean()) if len(md) else float("nan"),
            "n_excluded_chain_count_mismatch": int(
                (md.n_prot_chain_pred != md.n_prot_chain_crystal).sum()),
            "n_excluded_ambiguous_ligand_copy": int((md.n_lig_candidates != 1).sum()),
        },
        "eps": {
            "median": float(md[md.frame_ok].eps.median()) if md.frame_ok.any() else float("nan"),
            "p90": float(md[md.frame_ok].eps.quantile(0.90)) if md.frame_ok.any() else float("nan"),
            "p99": float(md[md.frame_ok].eps.quantile(0.99)) if md.frame_ok.any() else float("nan"),
            "max": float(md[md.frame_ok].eps.max()) if md.frame_ok.any() else float("nan"),
            "by_method": {k: float(v) for k, v in
                          md[md.frame_ok].groupby("method").eps.median().items()},
            "note": "reported on the bijective-frame subset; eps is meaningless where the frame is not",
        },
        "disagreement_rate_marginal": float((pd_pairs.pair_rmsd > sf.DISAGREE).mean())
        if len(pd_pairs) else float("nan"),
    }
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None, help="only the first N systems (smoke test)")
    args = ap.parse_args()
    res = run(limit=args.limit)
    save_json(res, RESDIR / "d1_frame_check.json")

    fc = res["frame_check"]
    la = res["label_agreement_vs_rnp"]
    lu = res["label_agreement_vs_rnp_unfiltered"]
    fe = res["frame_exclusions"]
    print("\n=== D1 single-frame build ===")
    print(f"model-instances framed : {res['n_model_instances']} "
          f"({res['n_instances_with_label']} with a label, "
          f"{res['label_nan_rate']:.1%} NaN)")
    print(f"pairs (bijective only) : {res['n_pairs']} "
          f"({res['pair_skip_rate']:.1%} skipped on ligand mismatch)")
    print("\nCHECK 0 bijective single frame")
    print(f"  frame_ok             : {fe['n_frame_ok']}/{fe['n_instances']} "
          f"({fe['frac_frame_ok']:.1%})")
    print(f"  excluded, chain-count mismatch : {fe['n_excluded_chain_count_mismatch']}")
    print(f"  excluded, ambiguous lig copy   : {fe['n_excluded_ambiguous_ligand_copy']}")
    print("\nCHECK 1 frame sanity (expected violations: exactly 0)")
    print(f"  pairs checked        : {fc['n_pairs_checked']}")
    print(f"  triangle violations  : {fc['n_triangle_violations']} "
          f"(min slack {fc['min_triangle_slack']:+.4f} A)")
    print(f"  T1 trigger violations: {fc['n_threshold_violations']}")
    print(f"  -> {'PASS' if fc['passes'] else 'FAIL -- the frame or the metric is wrong; T1 is void'}")
    print("  (necessary, NOT sufficient: a pose displaced into the wrong protomer is far from")
    print("   everything, so it satisfies the inequality vacuously. CHECK 2 is what catches that.)")
    print("\nCHECK 2 recomputed label vs RNP shipped rmsd  [the decisive check]")
    print(f"  bijective frame : n={la['n']:>6}  spearman={la['spearman']:.4f}  "
          f"pearson={la['pearson']:.4f}  calls agree={la['correctness_call_agreement']:.4f}  "
          f"med|diff|={la['median_abs_diff_A']:.3f} A  frac|diff|>5A={la['frac_abs_diff_gt_5A']:.4f}")
    print(f"  unfiltered      : n={lu['n']:>6}  spearman={lu['spearman']:.4f}  "
          f"pearson={lu['pearson']:.4f}  calls agree={lu['correctness_call_agreement']:.4f}  "
          f"med|diff|={lu['median_abs_diff_A']:.3f} A  frac|diff|>5A={lu['frac_abs_diff_gt_5A']:.4f}")
    print("\nCHECK 3 pocket-superposition residual eps (the deployment 2*eps slack)")
    e = res["eps"]
    print(f"  median={e['median']:.3f}  p90={e['p90']:.3f}  p99={e['p99']:.3f}  max={e['max']:.3f} A")
    print(f"  by method: { {k: round(v,3) for k,v in e['by_method'].items()} }")
    print(f"\nmarginal pair disagreement rate (>{sf.DISAGREE} A): "
          f"{res['disagreement_rate_marginal']:.4f}")
    print(f"\nwrote {OUT_MODELS}\nwrote {OUT_PAIRS}")


if __name__ == "__main__":
    main()
