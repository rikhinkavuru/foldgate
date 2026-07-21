"""E55 -- Boltz-2 as a fully governed model with its own 2023 reference (reviewer D26).

Boltz-2 has a later training cutoff (2023-06-30) than the 2021-era cohort, so its
novelty must be measured against a 2023 reference or the shift is understated. We run
the core reliability pipeline on Boltz-2 alone:

  * pocket novelty from `sucos_shape_pocket_qcov_2023` (2023-referenced pocket shape x
    coverage), ligand novelty from `morgan_tanimoto`, both binned with the standard
    make_strata (low similarity -> high novelty, NaN -> top no-analog stratum);
  * per-stratum base correctness;
  * the E2 conditional break: one global native LTT gate at alpha=0.20, its marginal
    realized risk and its per-stratum realized risk;
  * reliability drift D(nu) on the most-novel analog stratum S3 (mirrors E12): the
    target-mass-weighted P(correct | ranking_score) gap S0 -> S3, with a bootstrap CI.

The question is whether Boltz-2, despite its newer cutoff, shows the same
under-control-on-novelty break as the 2021-era models.
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
    rng,
    save_json,
)
from foldgate.conformal import ltt_threshold
from foldgate.features.novelty import make_strata
from foldgate.selective import evaluate_gate
from foldgate.selective.metrics import clopper_pearson

KEY = ["system_id", "ligand_instance_chain"]
POCKET_2023 = "sucos_shape_pocket_qcov_2023"
LIGAND_COL = "morgan_tanimoto"
AXES = {"ligand": "b2_ligand_stratum", "pocket_2023": "b2_pocket2023_stratum"}
HIGH_MIN = 3
N_BINS = 5
N_BOOT = 2000
DRIFT_REF = 0            # S0 reference
DRIFT_TARGET = 3         # S3 = most-novel analog stratum


def attach_boltz2_strata(b2: pd.DataFrame) -> pd.DataFrame:
    """Merge the 2023 pocket + Morgan ligand similarity and build Boltz-2's own strata."""
    ann = pd.read_csv("data/raw/annotations.csv", low_memory=False)
    cols = [c for c in (POCKET_2023, LIGAND_COL) if c in ann.columns]
    ann_key = ann[KEY + cols].drop_duplicates(KEY)
    b2 = b2.merge(ann_key, on=KEY, how="left")
    b2["b2_ligand_stratum"] = make_strata(b2, col=LIGAND_COL, n_bins=4, no_analog_stratum=True)
    b2["b2_pocket2023_stratum"] = make_strata(b2, col=POCKET_2023, n_bins=4, no_analog_stratum=True)
    return b2


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


def _edges(conf_s, conf_t):
    e = np.quantile(np.concatenate([conf_s, conf_t]), np.linspace(0, 1, N_BINS + 1))
    e[0], e[-1] = -np.inf, np.inf
    return e


def _drift(conf_s, y_s, conf_t, y_t, edges):
    """Target-mass-weighted signed and absolute P(correct|conf) gap, S0 -> Sk (E12)."""
    signed, absg, wts = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:], strict=False):
        ms = (conf_s >= lo) & (conf_s < hi)
        mt = (conf_t >= lo) & (conf_t < hi)
        if not ms.any() or not mt.any():
            continue
        ps, pt = float(y_s[ms].mean()), float(y_t[mt].mean())
        signed.append(ps - pt)
        absg.append(abs(ps - pt))
        wts.append(int(mt.sum()))
    if not wts:
        return float("nan"), float("nan")
    w = np.asarray(wts, float)
    return float(np.average(signed, weights=w)), float(np.average(absg, weights=w))


def drift_on_stratum(sub, col, g, ref=DRIFT_REF, target=DRIFT_TARGET):
    """S0 -> S_target reliability drift with a bootstrap 90% CI (mirror E12)."""
    s = sub.dropna(subset=[CONF, col])
    conf = s[CONF].to_numpy()
    y = s["correct"].to_numpy().astype(int)
    strat = s[col].to_numpy().astype(int)
    m_ref, m_tar = strat == ref, strat == target
    if m_ref.sum() < 20 or m_tar.sum() < 20:
        return {"available": False, "n_ref": int(m_ref.sum()), "n_target": int(m_tar.sum())}
    edges = _edges(conf[m_ref], conf[m_tar])
    d_signed, d_abs = _drift(conf[m_ref], y[m_ref], conf[m_tar], y[m_tar], edges)

    n = len(conf)
    bs = np.empty(N_BOOT)
    ba = np.empty(N_BOOT)
    for b in range(N_BOOT):
        bi = g.integers(0, n, n)
        cb, yb, stb = conf[bi], y[bi], strat[bi]
        rb, tb = stb == ref, stb == target
        if not rb.any() or not tb.any():
            bs[b], ba[b] = np.nan, np.nan
            continue
        ss, sa = _drift(cb[rb], yb[rb], cb[tb], yb[tb], edges)
        bs[b], ba[b] = ss, sa
    bsf, baf = bs[np.isfinite(bs)], ba[np.isfinite(ba)]
    return {
        "available": True,
        "ref_stratum": ref,
        "target_stratum": target,
        "n_ref": int(m_ref.sum()),
        "n_target": int(m_tar.sum()),
        "D_signed": d_signed,
        "D_signed_ci90": [float(np.quantile(bsf, 0.05)) if bsf.size else float("nan"),
                          float(np.quantile(bsf, 0.95)) if bsf.size else float("nan")],
        "D_abs": d_abs,
        "D_abs_ci90": [float(np.quantile(baf, 0.05)) if baf.size else float("nan"),
                       float(np.quantile(baf, 0.95)) if baf.size else float("nan")],
    }


def run(n_repeats: int = 300) -> dict:
    df = load_delivered()
    b2 = df[df.method == "boltz2"].reset_index(drop=True)
    b2 = attach_boltz2_strata(b2)
    g = rng()

    out = {
        "_meta": {
            "model": "boltz2",
            "alpha": ALPHA, "delta": DELTA, "conf": CONF,
            "n_poses": int(len(b2)),
            "base_correct": float(b2["correct"].mean()),
            "pocket_reference": POCKET_2023 + " (2023-referenced)",
            "ligand_reference": LIGAND_COL,
            "note": "Boltz-2 governed on its own 2023 pocket reference; strata mirror "
                    "make_strata; break + drift mirror E2 / E12.",
        },
        "axes": {},
    }
    for axis, col in AXES.items():
        sub = b2.dropna(subset=[CONF, col]).reset_index(drop=True)
        s = sub[CONF].to_numpy()
        y = sub["correct"].to_numpy().astype(int)
        strat = sub[col].to_numpy().astype(int)
        cond, marg = conditional_break(s, y, strat, g, n_repeats)
        highs = [v["pooled_selective_risk"] for k, v in cond.items()
                 if int(k) >= HIGH_MIN and np.isfinite(v["pooled_selective_risk"])]
        worst_high = float(max(highs)) if highs else float("nan")
        out["axes"][axis] = {
            "marginal_risk": marg,
            "worst_high_risk": worst_high,
            "excess_over_alpha": float(worst_high - ALPHA) if np.isfinite(worst_high) else float("nan"),
            "conditional": cond,
            "drift_S0_to_S3": drift_on_stratum(b2, col, g),
        }

    # break verdict: does Boltz-2 under-control on novelty like the 2021-era cohort?
    excesses = [out["axes"][a]["excess_over_alpha"] for a in AXES
                if np.isfinite(out["axes"][a]["excess_over_alpha"])]
    drift_sig = any(
        out["axes"][a]["drift_S0_to_S3"].get("available")
        and out["axes"][a]["drift_S0_to_S3"]["D_signed_ci90"][0] > 0
        for a in AXES
    )
    out["_verdict"] = {
        "max_excess_over_alpha": float(max(excesses)) if excesses else float("nan"),
        "any_high_stratum_break": bool(excesses and max(excesses) > 0.05),
        "concept_drift_S3_significant": bool(drift_sig),
        "same_break_as_2021_cohort": bool(excesses and max(excesses) > 0.05),
    }
    return out


def main() -> None:
    res = run()
    save_json(res, RESDIR / "e55_boltz2_governed.json")
    m = res["_meta"]
    print("E55 -- Boltz-2 governed on its own 2023 reference  "
          f"(alpha={ALPHA}, delta={DELTA})")
    print(f"{m['n_poses']} delivered poses, base correct = {m['base_correct']:.3f}\n")
    for axis, a in res["axes"].items():
        cond = a["conditional"]
        cells = " ".join(
            f"S{k}:{cond[k]['pooled_selective_risk']:.2f}"
            f"(bc={cond[k]['base_correct']:.2f},n={cond[k]['n_stratum']})"
            for k in sorted(cond, key=int)
        )
        print(f"[{axis}]  marg_risk={a['marginal_risk']:.3f}  "
              f"worst_high={a['worst_high_risk']:.3f} (excess {a['excess_over_alpha']:+.3f})")
        print(f"   per-stratum realized risk: {cells}")
        d = a["drift_S0_to_S3"]
        if d.get("available"):
            lo, hi = d["D_signed_ci90"]
            print(f"   drift S0->S3: D_signed={d['D_signed']:+.3f} [{lo:+.3f},{hi:+.3f}]  "
                  f"D_abs={d['D_abs']:.3f}  (n_ref={d['n_ref']}, n_S3={d['n_target']})")
        else:
            print(f"   drift S0->S3: unavailable (n_ref={d['n_ref']}, n_S3={d['n_target']})")
        print()
    v = res["_verdict"]
    print(f"verdict: max excess over alpha = {v['max_excess_over_alpha']:+.3f}; "
          f"high-stratum break = {v['any_high_stratum_break']}; "
          f"S3 concept drift significant = {v['concept_drift_S3_significant']}")
    print(f"   Boltz-2 shows the same break as the 2021-era cohort: "
          f"{v['same_break_as_2021_cohort']}")


if __name__ == "__main__":
    main()
