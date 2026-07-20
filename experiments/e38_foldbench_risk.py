"""E38 -- FoldBench arm as RISK CONTROL, split by low-homology (reviewer R2.10).

The E15b FoldBench arm reported AURC (a ranking-quality measure) and pooled the
52 low-homology targets with the 384 train-similar ones. Two fixes here:

1. RISK, not ranking. Deploy the FROZEN RNP Protenix interface-ipTM LTT gate
   threshold (tau ~= 0.989, the same tau E15b calibrates on RNP and transfers) on
   the regenerated FoldBench Protenix table and report the REALIZED selective risk
   (error rate = fraction with ligand-RMSD > 2 A among accepted) and coverage at
   that frozen tau, with a Clopper-Pearson 90% interval and accepted n, computed
   SEPARATELY on the 52 low-homology and the 384 train-similar targets. This is the
   guarantee-relevant quantity: does the frozen gate hold alpha=0.20 on the
   low-homology subset, or does coverage collapse / risk overshoot there?

2. CIs on the AURCs. Recompute the pooled risk-coverage AURC for the interface-ipTM
   gate and the matched ranking_score control and attach 90% bootstrap CIs to both
   (resample targets), then state whether the intervals overlap.

3. Restate the top-1 caveat. Self-scored top-1 success (~0.401) vs FoldBench
   released (0.567): feature AND label come from the same regenerated Protenix run,
   so they are self-consistent and the gate comparison is valid within that run even
   though absolute success differs from the released number. This limits the arm to
   a DIRECTIONAL transfer check, not an absolute-accuracy claim.

Reuses the E15b protocol exactly (same REGEN_CSV, same load, same frozen-tau path).
Does not modify any existing file.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from experiments._common import (  # noqa: E402
    ALPHA,
    DELTA,
    RESDIR,
    load_delivered,
    save_json,
)
from foldgate.conformal import ltt_threshold  # noqa: E402
from foldgate.selective.metrics import (  # noqa: E402
    aurc,
    bootstrap_ci,
    clopper_pearson,
    evaluate_gate,
)

REGEN_CSV = "data/external/foldbench/foldbench_protenix_regen.csv"
CI = 0.90
FOLDBENCH_RELEASED_TOP1 = 0.567  # FoldBench-reported Protenix protein-ligand top-1 success
N_BOOT = 5000
BOOT_SEED = 20260720


def load_regen() -> pd.DataFrame:
    df = pd.read_csv(REGEN_CSV)
    df = df.dropna(subset=["lrmsd", "iptm_iface"]).reset_index(drop=True)
    df["correct"] = (df["lrmsd"] <= 2.0).astype(int)
    df["is_unseen_protein"] = df["is_unseen_protein"].astype(bool)
    return df


def frozen_gate_report(scores: np.ndarray, correct: np.ndarray, tau: float) -> dict:
    """Realized selective risk + coverage of the frozen tau, with CP 90% intervals."""
    scores = np.asarray(scores, dtype=float)
    correct = np.asarray(correct, dtype=int)
    r = evaluate_gate(scores, correct, tau)
    n, n_acc = r["n"], r["n_accept"]
    accept = scores >= tau
    n_err = int((1 - correct[accept]).sum()) if n_acc else 0
    risk_ci = clopper_pearson(n_err, n_acc, CI) if n_acc else (float("nan"), float("nan"))
    cov_ci = clopper_pearson(n_acc, n, CI) if n else (float("nan"), float("nan"))
    return {
        "n": n,
        "n_accept": n_acc,
        "n_errors": n_err,
        "base_success_rate": float(correct.mean()) if n else float("nan"),
        "coverage": r["coverage"],
        "coverage_ci90": [float(cov_ci[0]), float(cov_ci[1])],
        "selective_risk": r["selective_risk"],
        "selective_risk_ci90": [float(risk_ci[0]), float(risk_ci[1])],
        # guarantee target is alpha; flag the point estimate AND the CP upper bound
        "risk_point_over_alpha": (
            None if n_acc == 0 else bool(r["selective_risk"] > ALPHA)
        ),
        "risk_ci_hi_over_alpha": (
            None if n_acc == 0 else bool(risk_ci[1] > ALPHA)
        ),
    }


def aurc_with_ci(scores: np.ndarray, correct: np.ndarray) -> dict:
    scores = np.asarray(scores, dtype=float)
    correct = np.asarray(correct, dtype=int)
    lo, hi = bootstrap_ci(aurc, scores, correct, n_boot=N_BOOT, ci=CI, seed=BOOT_SEED)
    return {
        "aurc": float(aurc(scores, correct)),
        "aurc_ci90": [float(lo), float(hi)],
        "n": int(len(scores)),
    }


def run() -> dict:
    # --- frozen RNP Protenix interface-ipTM LTT threshold (same path as E15b) ---
    rnp = load_delivered()
    rnp_prot = rnp[rnp.method == "protenix"].dropna(subset=["iface_iptm", "correct"])
    rnp_prot = rnp_prot.reset_index(drop=True)
    s_rnp = rnp_prot["iface_iptm"].to_numpy(dtype=float)
    y_rnp = rnp_prot["correct"].to_numpy(dtype=int)
    tau = ltt_threshold(s_rnp, y_rnp, alpha=ALPHA, delta=DELTA)
    if tau is None:
        raise SystemExit("RNP Protenix interface-ipTM gate did not certify a tau")
    tau = float(tau)

    # --- deploy the frozen tau on regenerated FoldBench Protenix ---
    fb = load_regen()
    unseen = fb["is_unseen_protein"].to_numpy(dtype=bool)
    s_fb_iface = fb["iptm_iface"].to_numpy(dtype=float)
    s_fb_rank = fb["ranking_score"].to_numpy(dtype=float)
    y_fb = fb["correct"].to_numpy(dtype=int)

    n = len(fb)
    n_unseen = int(unseen.sum())

    frozen = {
        "gate": "RNP Protenix interface-ipTM LTT (alpha=0.20, delta=0.10)",
        "tau": tau,
        "rnp_n_calib": int(len(rnp_prot)),
        "rnp_home": frozen_gate_report(s_rnp, y_rnp, tau),
        # guarantee-relevant split
        "low_homology": frozen_gate_report(
            s_fb_iface[unseen], y_fb[unseen], tau
        ),
        "train_similar": frozen_gate_report(
            s_fb_iface[~unseen], y_fb[~unseen], tau
        ),
        "pooled": frozen_gate_report(s_fb_iface, y_fb, tau),
    }

    # --- AURCs with bootstrap CIs (pooled 436, resample targets) ---
    iface_aurc = aurc_with_ci(s_fb_iface, y_fb)
    rank_aurc = aurc_with_ci(s_fb_rank, y_fb)
    i_lo, i_hi = iface_aurc["aurc_ci90"]
    r_lo, r_hi = rank_aurc["aurc_ci90"]
    overlap = not (i_hi < r_lo or r_hi < i_lo)

    aurcs = {
        "note": "pooled over all scored FoldBench Protenix targets; bootstrap resamples targets.",
        "n_boot": N_BOOT,
        "interface_iptm": iface_aurc,
        "ranking_score": rank_aurc,
        "ci_intervals_overlap": bool(overlap),
        "aurc_gap": float(rank_aurc["aurc"] - iface_aurc["aurc"]),
    }

    caveat = {
        "self_scored_top1_success": float(fb["correct"].mean()),
        "foldbench_released_top1_success": FOLDBENCH_RELEASED_TOP1,
        "statement": (
            "Self-scored top-1 success is {:.3f} vs FoldBench-released {:.3f}. The "
            "feature (interface-ipTM) and the label (self-scored ligand-RMSD <= 2 A) "
            "come from the SAME regenerated Protenix run, so they are self-consistent "
            "within that run and the gate comparison is valid; the absolute-success gap "
            "means this arm is a DIRECTIONAL transfer check, not an absolute-accuracy "
            "claim."
        ).format(float(fb["correct"].mean()), FOLDBENCH_RELEASED_TOP1),
    }

    return {
        "reviewer": "R2.10",
        "alpha": ALPHA,
        "delta": DELTA,
        "ci": CI,
        "novelty_axis": "is_unseen_protein (1 = low-homology unseen protein)",
        "model": "protenix (regenerated on FoldBench, self-scored ligand-RMSD)",
        "n_targets_scored": n,
        "n_low_homology": n_unseen,
        "n_train_similar": n - n_unseen,
        "frozen_threshold_risk": frozen,
        "aurc_with_ci": aurcs,
        "top1_caveat": caveat,
    }


def _fmt(v, spec=".3f"):
    return "nan" if v is None or (isinstance(v, float) and not np.isfinite(v)) else format(v, spec)


def _print_split(name: str, s: dict) -> None:
    lo, hi = s["selective_risk_ci90"]
    clo, chi = s["coverage_ci90"]
    flag = "" if s["risk_ci_hi_over_alpha"] is None else (
        "  CI-hi>alpha" if s["risk_ci_hi_over_alpha"] else "  holds"
    )
    print(
        f"  {name:14} n={s['n']:>4}  accept={s['n_accept']:>3}  err={s['n_errors']:>2}  "
        f"risk={_fmt(s['selective_risk']):>5} CI[{_fmt(lo,'.2f')},{_fmt(hi,'.2f')}]  "
        f"cov={_fmt(s['coverage'],'.3f')} CI[{_fmt(clo,'.3f')},{_fmt(chi,'.3f')}]{flag}"
    )


def main() -> None:
    res = run()
    save_json(res, RESDIR / "e38_foldbench_risk.json")

    print("E38 -- FoldBench arm as risk control (reviewer R2.10)")
    print(f"alpha={ALPHA} delta={DELTA}  |  Protenix regenerated on FoldBench")
    f = res["frozen_threshold_risk"]
    print(f"\n[1] FROZEN RNP Protenix interface-ipTM gate  tau={f['tau']:.6f}  "
          f"(RNP calib n={f['rnp_n_calib']})")
    _print_split("RNP home", f["rnp_home"])
    _print_split("low-homology", f["low_homology"])
    _print_split("train-similar", f["train_similar"])
    _print_split("pooled(436)", f["pooled"])

    a = res["aurc_with_ci"]
    print("\n[2] Risk-coverage AURC (pooled, target bootstrap, lower=better)")
    ai, ar = a["interface_iptm"], a["ranking_score"]
    print(f"  interface-ipTM  AURC={ai['aurc']:.3f}  "
          f"CI90[{ai['aurc_ci90'][0]:.3f},{ai['aurc_ci90'][1]:.3f}]")
    print(f"  ranking_score   AURC={ar['aurc']:.3f}  "
          f"CI90[{ar['aurc_ci90'][0]:.3f},{ar['aurc_ci90'][1]:.3f}]")
    print(f"  intervals overlap: {a['ci_intervals_overlap']}  (gap={a['aurc_gap']:.3f})")

    c = res["top1_caveat"]
    print("\n[3] Top-1 caveat")
    print(f"  self-scored {c['self_scored_top1_success']:.3f} vs "
          f"released {c['foldbench_released_top1_success']:.3f} -> directional transfer check")

    print(f"\nsaved {RESDIR / 'e38_foldbench_risk.json'}")


if __name__ == "__main__":
    main()
