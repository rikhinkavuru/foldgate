"""E60 -- does the coverage break reproduce beyond Runs N' Poses? (reviewer D3)

The D3 generality test. We regenerate Boltz-2 2.2.1 and Chai-1 0.6.1 predictions WITH
confidence on two external benchmarks that ship no released predictions -- PoseBusters-V2
(308) and PLINDER-test (346) -- self-score them (pose-superposed symmetry-corrected
ligand-RMSD vs the deposited crystal, same 2 A convention as RNP), stratify by ligand
novelty, and run the SAME feasibility-frontier and break machinery the RNP headline uses
(experiments/d2_feasibility_map). If the frontier collapses on novel external targets the
way it does on RNP, the impossibility is a property of co-folding confidence, not of one
benchmark.

This is the scaffold: it is runnable end to end the moment scored.csv lands. It reuses
d2's `_cell` / `_frontier` verbatim so the external numbers are apples-to-apples with the
RNP frontier (results/d2_feasibility_map.json, results/e50_score_family_frontier.json).

Inputs (produced on the GPU box by scripts/gpu_d3/selfscore.py):
  results/gpu_d3/scored.csv  columns: dataset, model, target_id, pdb_id, ligand_ccd,
     <confidence fields>, ligand_rmsd, correct, status
  scripts/gpu_d3/manifests/{posebusters_v2,plinder_subset}.csv  (carry ligand_smiles)

Novelty (the one piece that is not shipped and must be computed):
  - PLINDER-test: prefer the shipped ligand-similarity-to-training metadata when the
    PLINDER annotation table is present (--plinder-sim PATH); else fall back to the
    computed ECFP4 route below.
  - PoseBusters-V2: no shipped novelty, so we compute max ECFP4 Tanimoto of each ligand
    to a reference "seen" ligand pool (--ref-smiles PATH, one SMILES per line). Until the
    reference is wired the novelty column is left NaN and the frontier is reported only on
    the pooled set with a loud caveat, never silently.

Usage:
  .venv/bin/python -m experiments.e60_external_transfer \
      --scored results/gpu_d3/scored.csv \
      [--ref-smiles data/external/seen_ligands.smi] [--plinder-sim data/external/plinder_sim.parquet]
  # smoke-test the pipeline with no real data:
  .venv/bin/python -m experiments.e60_external_transfer --synthetic

Output: results/e60_external_transfer.json, results/figures/e60_external_transfer.png
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import pandas as pd

from experiments._common import CONF, DELTA, RESDIR, save_json, rng
from experiments.d2_feasibility_map import (
    COVERAGE_GRID,
    MIN_ACCEPT,
    SOURCE_STRATUM,
    _cell,
    _frontier,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCORED_DEFAULT = ROOT / "results" / "gpu_d3" / "scored.csv"
MANIFESTS = ROOT / "scripts" / "gpu_d3" / "manifests"
ALPHAS = (0.20, 0.10)
N_STRATA = 4          # S0..S3 quartiles; NaN-similarity -> S4 (matches RNP)

# Per-model confidence field that plays the role of RNP's global ranking_score, and the
# interface score. Boltz emits confidence_score; Chai emits aggregate_score. Both emit iptm.
RANKING_FIELD = {"boltz2": "confidence_score", "chai": "aggregate_score"}
IFACE_FIELD = "iptm"


# --------------------------------------------------------------------- loading
def load_scored(path: pathlib.Path) -> pd.DataFrame:
    """Normalize the regenerated scored.csv into the column names d2 expects."""
    df = pd.read_csv(path)
    df = df[df["status"].fillna("ok").eq("ok")] if "status" in df else df
    out = []
    for model, sub in df.groupby("model"):
        rank_col = RANKING_FIELD.get(model)
        if rank_col is None or rank_col not in sub:
            raise SystemExit(f"scored.csv missing ranking field {rank_col!r} for model {model}")
        m = pd.DataFrame({
            "dataset": sub["dataset"],
            "system_id": sub["target_id"],
            "method": model,
            "pdb_id": sub.get("pdb_id"),
            "ligand_ccd": sub.get("ligand_ccd"),
            CONF: sub[rank_col].astype(float),
            "iface_iptm": sub[IFACE_FIELD].astype(float) if IFACE_FIELD in sub else np.nan,
            "ligand_rmsd": sub["ligand_rmsd"].astype(float),
            "correct": sub["correct"].astype(int),
        })
        out.append(m)
    return pd.concat(out, ignore_index=True)


# --------------------------------------------------------------------- novelty
def _ecfp(smiles: str):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)


def _max_tanimoto(smiles: str, ref_fps) -> float:
    from rdkit.Chem import DataStructs
    fp = _ecfp(smiles)
    if fp is None or not ref_fps:
        return float("nan")
    return float(max(DataStructs.TanimotoSimilarity(fp, r) for r in ref_fps))


def compute_novelty(df: pd.DataFrame, ref_smiles: pathlib.Path | None,
                    plinder_sim: pathlib.Path | None) -> pd.DataFrame:
    """Attach a `ligand_similarity` (higher = more familiar) and quartile `novelty_stratum`.

    Similarity source, in order of preference:
      1. PLINDER shipped similarity for plinder_test rows (real training-set similarity).
      2. Computed max ECFP4 Tanimoto to a reference seen-ligand pool for everything else.
    Rows with no computable similarity go to S4 (the no-analog bin), exactly like RNP.
    """
    df = df.copy()
    df["ligand_similarity"] = np.nan

    # ligand SMILES from the manifests
    smi = {}
    for name in ("posebusters_v2", "plinder_subset"):
        p = MANIFESTS / f"{name}.csv"
        if p.exists():
            man = pd.read_csv(p)
            smi.update(dict(zip(man["target_id"], man["ligand_smiles"])))
    df["ligand_smiles"] = df["system_id"].map(smi)

    # 1. PLINDER shipped similarity
    if plinder_sim and pathlib.Path(plinder_sim).exists():
        ps = pd.read_parquet(plinder_sim)
        # expected columns: system_id (or target_id), ligand_similarity in [0,1]
        key = "system_id" if "system_id" in ps else "target_id"
        m = dict(zip(ps[key], ps["ligand_similarity"]))
        mask = df["dataset"].eq("plinder_test")
        df.loc[mask, "ligand_similarity"] = df.loc[mask, "system_id"].map(m)

    # 2. computed ECFP4 Tanimoto to a reference pool for the still-missing rows
    todo = df["ligand_similarity"].isna() & df["ligand_smiles"].notna()
    if ref_smiles and pathlib.Path(ref_smiles).exists():
        ref_lines = [l.strip().split()[0] for l in open(ref_smiles) if l.strip()]
        ref_fps = [fp for fp in (_ecfp(s) for s in ref_lines) if fp is not None]
        uniq = df.loc[todo, "ligand_smiles"].dropna().unique()
        tan = {s: _max_tanimoto(s, ref_fps) for s in uniq}
        df.loc[todo, "ligand_similarity"] = df.loc[todo, "ligand_smiles"].map(tan)
    elif todo.any():
        print(f"[e60] WARNING: no reference SMILES wired; {int(todo.sum())} rows have no "
              f"computable novelty and will fall into S4. Pass --ref-smiles to stratify them.")

    # quartile strata on the SIMILARITY marginal (a covariate, not the label), NaN -> S4
    s = df["ligand_similarity"]
    valid = s.notna()
    df["novelty_stratum"] = N_STRATA  # default S4
    if valid.sum() >= N_STRATA:
        # low similarity = high novelty; edges from the valid marginal (matches RNP convention)
        q = pd.qcut(s[valid].rank(method="first"), N_STRATA, labels=False)
        # rank ascending on similarity -> stratum 0 = most familiar (highest similarity)
        order = s[valid].rank(method="first", ascending=False)
        df.loc[valid, "novelty_stratum"] = pd.qcut(order, N_STRATA, labels=False).astype(int)
    return df


# ---------------------------------------------------------------- the analysis
def _stratum_cells(g: pd.DataFrame, score: str, alpha: float, delta: float) -> dict:
    """d2's coverage sweep for one (dataset, model, stratum, score): tau grid -> cells."""
    d = g.dropna(subset=[score, "correct"]).sort_values(score, ascending=False)
    n = len(d)
    cells = {}
    if n == 0:
        return cells
    loss_sorted = (1 - d["correct"].to_numpy())  # accept the top-c fraction by score
    for c in COVERAGE_GRID:
        k = int(round(c * n))
        if k < 1:
            continue
        cells[float(c)] = _cell(loss_sorted[:k], n_stratum=n, alpha=alpha, delta=delta)
    return cells


def external_frontier(df: pd.DataFrame, delta: float = DELTA) -> dict:
    """Per (dataset, model, score, alpha): per-stratum frontier c* and feasibility."""
    scores = [CONF, "iface_iptm"]
    res = {}
    for (dataset, model), dm in df.groupby(["dataset", "method"]):
        for score in scores:
            if dm[score].notna().sum() < MIN_ACCEPT:
                continue
            for alpha in ALPHAS:
                key = f"{dataset}|{model}|{score}|a{alpha}"
                per_stratum = {}
                for s in range(N_STRATA + 1):
                    g = dm[dm["novelty_stratum"] == s]
                    if len(g) < MIN_ACCEPT and s < N_STRATA:
                        per_stratum[str(s)] = {"n": int(len(g)), "frontier": None,
                                               "underpowered": True}
                        continue
                    cells = _stratum_cells(g, score, alpha, delta)
                    per_stratum[str(s)] = {
                        "n": int(len(g)),
                        "frontier": _frontier(cells, "feasible"),
                        "frontier_certified": _frontier(cells, "certified"),
                        "base_correct": float(g["correct"].mean()) if len(g) else float("nan"),
                    }
                res[key] = per_stratum
    return res


def break_check(df: pd.DataFrame, delta: float = DELTA) -> dict:
    """Calibrate the gate on familiar strata (S0,S1), deploy on novel (S2,S3), realized error."""
    out = {}
    for (dataset, model), dm in df.groupby(["dataset", "method"]):
        fam = dm[dm["novelty_stratum"].isin([0, 1])].dropna(subset=[CONF, "correct"])
        nov = dm[dm["novelty_stratum"].isin([2, 3])].dropna(subset=[CONF, "correct"])
        if len(fam) < MIN_ACCEPT or len(nov) < MIN_ACCEPT:
            out[f"{dataset}|{model}"] = {"underpowered": True,
                                         "n_fam": int(len(fam)), "n_nov": int(len(nov))}
            continue
        # marginally-valid gate: tau that hits alpha=0.20 on familiar, then read novel error
        alpha = 0.20
        fam_sorted = fam.sort_values(CONF, ascending=False)
        fam_loss = 1 - fam_sorted["correct"].to_numpy()
        tau = None
        for k in range(len(fam_sorted), 0, -1):
            if fam_loss[:k].mean() <= alpha:
                tau = float(fam_sorted[CONF].iloc[k - 1]); break
        if tau is None:
            out[f"{dataset}|{model}"] = {"no_familiar_gate": True}
            continue
        acc = nov[nov[CONF] >= tau]
        out[f"{dataset}|{model}"] = {
            "tau": tau,
            "novel_realized_error": float((1 - acc["correct"]).mean()) if len(acc) else float("nan"),
            "novel_coverage": float(len(acc) / len(nov)),
            "n_accept_novel": int(len(acc)),
            "target_alpha": alpha,
        }
    return out


def compare_to_rnp(ext_frontier: dict) -> dict:
    """Agreement of the external ABSTAIN pattern with the RNP frontier (results/e50)."""
    rnp_path = RESDIR / "e50_score_family_frontier.json"
    if not rnp_path.exists():
        return {"note": "RNP frontier json not found; run e50 first"}
    rnp = json.loads(rnp_path.read_text())
    # external zero-frontier fraction per (dataset, alpha), to line up against RNP's ~0.4
    frac = {}
    for key, strata in ext_frontier.items():
        dataset, model, score, a = key.split("|")
        if score != CONF:
            continue
        zero = sum(1 for s, v in strata.items()
                   if v.get("frontier") == 0.0 and not v.get("underpowered"))
        tot = sum(1 for v in strata.values() if not v.get("underpowered"))
        frac.setdefault(f"{dataset}|{a}", []).append((zero, tot))
    summary = {k: {"zero_cells": sum(z for z, _ in v), "total_cells": sum(t for _, t in v)}
               for k, v in frac.items()}
    return {
        "external_zero_frontier": summary,
        "rnp_reference": {
            "n_nonreference_cells": rnp["per_alpha"]["0.2"]["n_nonreference_cells"],
            "rnp_zero_ranking": rnp["per_alpha"]["0.2"]["zero_frontier_per_score"]["ranking_score"],
            "rnp_zero_full_family": rnp["per_alpha"]["0.2"]["intersection_full_family_incl_ligand_local"],
        },
        "interpretation": "If external zero-frontier fraction is comparable to RNP's, the "
                          "break reproduces beyond RNP; report per-dataset and note power.",
    }


# ----------------------------------------------------------------- synthetic smoke
def synthetic_scored(seed: int = 20260722) -> pd.DataFrame:
    """A fake scored.csv with the real schema, so the pipeline can be smoke-tested dry."""
    g = rng(seed)
    rows = []
    for dataset, n in [("posebusters_v2", 308), ("plinder_test", 346)]:
        for model in ("boltz2", "chai"):
            for i in range(n):
                sim = float(g.uniform(0, 1))
                # base correctness falls with novelty (low similarity), like the real data
                p = 0.85 * sim + 0.15
                correct = int(g.uniform() < p)
                conf = float(np.clip(g.normal(0.8 * p + 0.15, 0.1), 0, 1))
                rows.append({
                    "dataset": dataset, "model": model,
                    "target_id": f"{dataset[:2].upper()}{i:04d}",
                    "pdb_id": f"X{i:03d}", "ligand_ccd": "LIG",
                    RANKING_FIELD[model]: conf, "iptm": conf * 0.98,
                    "ligand_rmsd": float(g.exponential(1.5) + (0 if correct else 3)),
                    "correct": correct, "status": "ok", "_sim": sim,
                })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", type=pathlib.Path, default=SCORED_DEFAULT)
    ap.add_argument("--ref-smiles", type=pathlib.Path, default=None)
    ap.add_argument("--plinder-sim", type=pathlib.Path, default=None)
    ap.add_argument("--synthetic", action="store_true",
                    help="smoke-test the pipeline with generated data (no GPU output needed)")
    args = ap.parse_args()

    if args.synthetic:
        raw = synthetic_scored()
        df = load_scored_frame(raw)
        # synthetic carries its own similarity so the strata are meaningful in the dry run
        df = df.merge(raw[["target_id", "_sim"]].rename(columns={"target_id": "system_id"})
                      .drop_duplicates("system_id"), on="system_id", how="left")
        df["ligand_similarity"] = df["_sim"]
        s = df["ligand_similarity"]
        order = s.rank(method="first", ascending=False)
        df["novelty_stratum"] = pd.qcut(order, N_STRATA, labels=False).astype(int)
        provenance = "SYNTHETIC smoke test -- numbers are not real"
    else:
        if not args.scored.exists():
            raise SystemExit(f"{args.scored} not found. Run the D3 job + selfscore.py first, "
                             "or pass --synthetic to smoke-test.")
        df = load_scored(args.scored)
        df = compute_novelty(df, args.ref_smiles, args.plinder_sim)
        provenance = f"real: {args.scored}"

    frontier = external_frontier(df)
    result = {
        "provenance": provenance,
        "n_targets": {k: int(v) for k, v in df.groupby("dataset")["system_id"].nunique().items()},
        "stratum_counts": {f"{d}|{m}": g["novelty_stratum"].value_counts().sort_index().to_dict()
                           for (d, m), g in df.groupby(["dataset", "method"])},
        "external_frontier": frontier,
        "break": break_check(df),
        "vs_rnp": compare_to_rnp(frontier),
    }
    out = RESDIR / "e60_external_transfer.json"
    save_json(result, out)
    print(f"wrote {out.relative_to(ROOT)}")
    for k, v in result["break"].items():
        if "novel_realized_error" in v:
            print(f"  BREAK {k}: novel realized error {v['novel_realized_error']:.3f} "
                  f"(target {v['target_alpha']}), coverage {v['novel_coverage']:.2f}")


# load_scored operates on a path; this variant takes an in-memory frame (synthetic path)
def load_scored_frame(raw: pd.DataFrame) -> pd.DataFrame:
    tmp = ROOT / "results" / "gpu_d3"
    tmp.mkdir(parents=True, exist_ok=True)
    p = tmp / "_synthetic_scored.csv"
    raw.to_csv(p, index=False)
    return load_scored(p)


if __name__ == "__main__":
    main()
