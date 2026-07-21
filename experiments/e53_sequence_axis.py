"""E53 -- protein sequence-identity as a third novelty axis (reviewer D19).

The money figure (E2) shows a native iid-calibrated gate under-controls error on
novel *ligands* and novel *pockets*. Reviewer D19 asks whether the break also
appears on protein *sequence* novelty. We merge `protein_seqsim_max` (max sequence
identity to any training protein, 0-100) from annotations, build quartile strata
S0..S3 with the no-analog NaN systems as a top stratum S4 (mirroring
foldgate.features.novelty.make_strata), and re-run the E2 conditional break: fit
one global native LTT gate at alpha=0.20 on a random calibration half, deploy it,
and read the realized selective risk *per sequence stratum*.

For a like-for-like verdict we run the same conditional break on all three axes
(ligand, pocket, sequence) in one pass, so "does sequence break comparably" is a
direct number-to-number comparison, not a recollection of E2.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from experiments._common import (
    ALPHA,
    CONF,
    DELTA,
    RESDIR,
    load_delivered,
    methods_with_enough,
    rng,
    save_json,
)
from foldgate.conformal import ltt_threshold
from foldgate.features.novelty import make_strata
from foldgate.selective import evaluate_gate

SEQ_COL = "protein_seqsim_max"          # primary sequence-identity axis
SEQ_FALLBACK = "protein_fident_max"     # alternative if the primary is absent
AXES = {
    "ligand": "novelty_stratum",
    "pocket": "pocket_novelty_stratum",
    "sequence": "sequence_stratum",
}
HIGH_MIN = 3     # strata >= this = "high novelty" for the break summary


def attach_sequence_stratum(df: pd.DataFrame) -> tuple[pd.DataFrame, str, dict]:
    """Left-join protein sequence identity by system_id and build strata + diagnostics.

    make_strata quantile-bins identity (low identity -> high novelty) and puts the
    no-analog NaN systems in their own top stratum. Sequence identity in RNP is
    heavily tied at the ceiling (many test proteins have a near-identical training
    protein), so the quantile bins collapse: we report how much sequence novelty the
    benchmark actually contains, since a non-break on a nearly-degenerate axis is a
    property of the data, not just of the gate.
    """
    ann = pd.read_csv("data/raw/annotations.csv", low_memory=False)
    col = SEQ_COL if SEQ_COL in ann.columns else SEQ_FALLBACK
    seq = ann[["system_id", col]].drop_duplicates("system_id")
    df = df.merge(seq, on="system_id", how="left")
    df["sequence_stratum"] = make_strata(df, col=col, n_bins=4, no_analog_stratum=True)

    raw = pd.to_numeric(df[col], errors="coerce")
    nn = raw.dropna()
    ceil_val = float(nn.max()) if len(nn) else float("nan")
    diag = {
        "identity_scale_max": ceil_val,
        "frac_no_analog_nan": float(raw.isna().mean()),
        "frac_at_identity_ceiling": float((nn >= ceil_val - 1e-9).mean()) if len(nn) else float("nan"),
        "identity_quartiles": [float(np.nanpercentile(nn, q)) for q in (0, 25, 50, 75, 100)] if len(nn) else [],
        "n_realized_strata": int(df["sequence_stratum"].nunique()),
        "top_stratum_is_no_analog": bool(
            df.loc[raw.isna(), "sequence_stratum"].nunique() == 1
            and (raw.isna().any())
            and df.loc[raw.isna(), "sequence_stratum"].iloc[0] == df["sequence_stratum"].max()
        ),
    }
    return df, col, diag


def conditional_break(s, y, strat, g, n_repeats=300):
    """Per-stratum realized risk under one global iid-calibrated LTT tau (E2 recipe)."""
    n = len(s)
    levels = sorted(np.unique(strat).tolist())
    acc_err = {k: 0 for k in levels}
    acc_n = {k: 0 for k in levels}
    tot_n = {k: int((strat == k).sum()) for k in levels}
    correct_rate = {k: float(y[strat == k].mean()) if tot_n[k] else float("nan") for k in levels}
    per_repeat_risk = {k: [] for k in levels}
    per_repeat_cov = {k: [] for k in levels}
    marg_risks = []

    for _ in range(n_repeats):
        perm = g.permutation(n)
        cal, test = perm[: n // 2], perm[n // 2:]
        tau = ltt_threshold(s[cal], y[cal], alpha=ALPHA, delta=DELTA)
        if tau is None:
            continue
        acc = s[test] >= tau
        marg = evaluate_gate(s[test], y[test], tau)
        if marg["n_accept"]:
            marg_risks.append(marg["selective_risk"])
        for k in levels:
            in_k = strat[test] == k
            n_test_k = int(in_k.sum())
            if not n_test_k:
                continue
            mk = acc & in_k
            nk = int(mk.sum())
            per_repeat_cov[k].append(nk / n_test_k)
            if nk:
                ek = int((1 - y[test][mk]).sum())
                acc_err[k] += ek
                acc_n[k] += nk
                per_repeat_risk[k].append(ek / nk)

    out = {}
    for k in levels:
        risk = acc_err[k] / acc_n[k] if acc_n[k] else float("nan")
        pr = np.array(per_repeat_risk[k], dtype=float)
        out[str(k)] = {
            "n_stratum": tot_n[k],
            "base_correct": correct_rate[k],
            "pooled_selective_risk": float(risk),
            "risk_p05": float(np.nanpercentile(pr, 5)) if len(pr) else float("nan"),
            "risk_p95": float(np.nanpercentile(pr, 95)) if len(pr) else float("nan"),
            "mean_coverage": float(np.mean(per_repeat_cov[k])) if per_repeat_cov[k] else 0.0,
        }
    return out, float(np.nanmean(marg_risks)) if marg_risks else float("nan")


def axis_break_magnitude(cond: dict) -> dict:
    """Worst high-novelty realized risk and how far it exceeds alpha."""
    highs = [v for k, v in cond.items() if int(k) >= HIGH_MIN and np.isfinite(v["pooled_selective_risk"])]
    if not highs:
        return {"worst_high_risk": float("nan"), "excess_over_alpha": float("nan")}
    worst = max(h["pooled_selective_risk"] for h in highs)
    return {"worst_high_risk": float(worst), "excess_over_alpha": float(worst - ALPHA)}


def run(n_repeats: int = 300) -> dict:
    df = load_delivered()
    df, seq_col, diag = attach_sequence_stratum(df)
    methods = methods_with_enough(df)
    g = rng()

    out = {"_meta": {"alpha": ALPHA, "delta": DELTA, "sequence_column": seq_col,
                     "n_repeats": n_repeats, "high_stratum_min": HIGH_MIN,
                     "sequence_novelty_diagnostics": diag,
                     "note": "sequence strata mirror make_strata: low identity -> high "
                             "novelty; NaN (no training protein analog) -> top stratum. "
                             "Ties at the identity ceiling collapse the quantile bins, so "
                             "the realized number of strata can be < 5."},
           "models": {}}

    for m in methods:
        base = df[df.method == m]
        model_out = {}
        for axis, col in AXES.items():
            sub = base.dropna(subset=[CONF, col]).reset_index(drop=True)
            s = sub[CONF].to_numpy()
            y = sub["correct"].to_numpy().astype(int)
            strat = sub[col].to_numpy().astype(int)
            cond, marg = conditional_break(s, y, strat, g, n_repeats)
            model_out[axis] = {
                "marginal_risk": marg,
                "conditional": cond,
                "break": axis_break_magnitude(cond),
            }
        out["models"][m] = model_out
    return out


def summarize(res: dict) -> dict:
    """Cross-model mean worst-high-stratum risk per axis + a comparability verdict."""
    axes = list(AXES.keys())
    summary = {}
    for axis in axes:
        excess = [res["models"][m][axis]["break"]["excess_over_alpha"]
                  for m in res["models"]
                  if np.isfinite(res["models"][m][axis]["break"]["excess_over_alpha"])]
        worst = [res["models"][m][axis]["break"]["worst_high_risk"]
                 for m in res["models"]
                 if np.isfinite(res["models"][m][axis]["break"]["worst_high_risk"])]
        summary[axis] = {
            "mean_worst_high_risk": float(np.mean(worst)) if worst else float("nan"),
            "mean_excess_over_alpha": float(np.mean(excess)) if excess else float("nan"),
        }
    seq_ex = summary["sequence"]["mean_excess_over_alpha"]
    struct_ex = np.nanmean([summary["ligand"]["mean_excess_over_alpha"],
                            summary["pocket"]["mean_excess_over_alpha"]])
    ratio = seq_ex / struct_ex if struct_ex and np.isfinite(struct_ex) and struct_ex != 0 else float("nan")
    summary["_verdict"] = {
        "sequence_excess": float(seq_ex),
        "structural_excess_mean": float(struct_ex),
        "sequence_to_structural_ratio": float(ratio),
        "sequence_breaks": bool(np.isfinite(seq_ex) and seq_ex > 0.02),
        "comparable_to_structural": bool(np.isfinite(ratio) and ratio >= 0.5),
    }
    return summary


def main() -> None:
    res = run()
    res["_summary"] = summarize(res)
    save_json(res, RESDIR / "e53_sequence_axis.json")

    d = res["_meta"]["sequence_novelty_diagnostics"]
    print("E53 -- sequence-identity novelty axis  (alpha=%.2f, delta=%.2f, col=%s)"
          % (ALPHA, DELTA, res["_meta"]["sequence_column"]))
    print(f"sequence novelty in RNP: {d['frac_at_identity_ceiling']:.1%} of systems at the "
          f"identity ceiling, {d['frac_no_analog_nan']:.1%} no-analog (NaN); "
          f"{d['n_realized_strata']} realized strata\n")
    for m, mo in res["models"].items():
        print(f"[{m}]")
        for axis in AXES:
            cond = mo[axis]["conditional"]
            brk = mo[axis]["break"]
            cells = " ".join(
                f"S{k}:{cond[k]['pooled_selective_risk']:.2f}(n={cond[k]['n_stratum']})"
                for k in sorted(cond, key=int)
            )
            print(f"   {axis:>9}  marg={mo[axis]['marginal_risk']:.3f}  "
                  f"worst_high={brk['worst_high_risk']:.3f}  | {cells}")
        print()
    s = res["_summary"]
    print("cross-model mean excess-over-alpha on high strata:")
    for axis in AXES:
        print(f"   {axis:>9}: {s[axis]['mean_excess_over_alpha']:+.3f}")
    v = s["_verdict"]
    print(f"\nverdict: sequence excess {v['sequence_excess']:+.3f} vs structural "
          f"{v['structural_excess_mean']:+.3f}  (ratio {v['sequence_to_structural_ratio']:.2f})")
    print(f"   sequence axis breaks: {v['sequence_breaks']};  "
          f"comparable to structural: {v['comparable_to_structural']}")


if __name__ == "__main__":
    main()
