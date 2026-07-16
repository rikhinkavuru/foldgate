"""E15b -- feature-matched frozen transfer using recovered interface-ipTM.

E15 could only transfer ``ranking_score`` because FoldBench's public table omits the
interface signal. This session regenerated the Protenix predictions for the FoldBench
protein-ligand targets (matched config: seeds 42, cycle 10, step 200, sample 5, ColabFold
MSA server) and recovered the protein-ligand chain-pair ipTM (``iface_iptm``) that RNP
calibrates on. Crucially, both the feature AND the pose-correctness label come from the
same regenerated run -- we self-score ligand-RMSD against the deposited assembly, so
feature and label are self-consistent and we never borrow FoldBench's released poses
(which a different Protenix version / MSA snapshot produced).

E15b freezes an LTT selective-risk threshold calibrated on RNP Protenix ``iface_iptm``
and deploys it, unchanged, on the regenerated FoldBench Protenix table. As a matched
control it also transfers the RNP ``ranking_score`` gate on the SAME regenerated set, so
the two signals are compared apples-to-apples on identical targets. Everything is split by
``is_unseen_protein`` (the low-homology axis), the second-dataset replication of the E2
break.

Parity level (stated, not hidden). Feature parity is interface-ipTM + ranking_score, the
two signals shared with RNP; PoseBusters, ensemble spread, and cross-model agreement were
not reconstructed on FoldBench, so the full learned combiner still does not transfer. This
is Protenix-only: the other FoldBench models were not regenerated and remain ranking_score
-only (E15). Protein sequences and ligand stoichiometry come from the deposited assemblies
FoldBench used; sequences may omit unmodeled expression tags. See
scripts/build_foldbench_af3_inputs.py and scripts/score_foldbench_lrmsd.py.
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
    FIGDIR,
    RESDIR,
    load_delivered,
    save_json,
)
from foldgate.conformal import ltt_threshold  # noqa: E402
from foldgate.selective.metrics import aurc, clopper_pearson, evaluate_gate  # noqa: E402

REGEN_CSV = "data/external/foldbench/foldbench_protenix_regen.csv"
CI = 0.90


def _report(scores: np.ndarray, correct: np.ndarray, tau: float | None) -> dict:
    scores = np.asarray(scores, dtype=float)
    correct = np.asarray(correct, dtype=int)
    r = evaluate_gate(scores, correct, tau)
    n, n_acc = r["n"], r["n_accept"]
    accept = (scores >= tau) if tau is not None else np.zeros(n, dtype=bool)
    n_err = int((1 - correct[accept]).sum()) if n_acc else 0
    risk_ci = clopper_pearson(n_err, n_acc, CI) if n_acc else (float("nan"), float("nan"))
    cov_ci = clopper_pearson(n_acc, n, CI) if n else (float("nan"), float("nan"))
    return {
        "n": n,
        "n_accept": n_acc,
        "n_errors": n_err,
        "base_correct": float(correct.mean()) if n else float("nan"),
        "coverage": r["coverage"],
        "coverage_ci90": [float(cov_ci[0]), float(cov_ci[1])],
        "selective_risk": r["selective_risk"],
        "selective_risk_ci90": [float(risk_ci[0]), float(risk_ci[1])],
        "risk_over_alpha": bool(r["selective_risk"] > ALPHA) if n_acc else None,
        "risk_ci_lo_over_alpha": bool(risk_ci[0] > ALPHA) if n_acc else None,
        "aurc": float(aurc(scores, correct)) if n >= 1 else float("nan"),
    }


def _transfer(rnp_sub: pd.DataFrame, feat: str, fb: pd.DataFrame, fb_feat: str) -> dict:
    r = rnp_sub.dropna(subset=[feat, "correct"]).reset_index(drop=True)
    s_r = r[feat].to_numpy(dtype=float)
    y_r = r["correct"].to_numpy(dtype=int)
    tau = ltt_threshold(s_r, y_r, alpha=ALPHA, delta=DELTA)

    f = fb.dropna(subset=[fb_feat, "correct", "is_unseen_protein"]).reset_index(drop=True)
    s_f = f[fb_feat].to_numpy(dtype=float)
    y_f = f["correct"].to_numpy(dtype=int)
    unseen = f["is_unseen_protein"].to_numpy(dtype=bool)
    return {
        "feature": feat,
        "tau": float(tau) if tau is not None else None,
        "rnp_n_calib": int(len(r)),
        "rnp_home": _report(s_r, y_r, tau),
        "overall": _report(s_f, y_f, tau),
        "seen": _report(s_f[~unseen], y_f[~unseen], tau),
        "unseen": _report(s_f[unseen], y_f[unseen], tau),
    }


def load_regen() -> pd.DataFrame:
    df = pd.read_csv(REGEN_CSV)
    # keep only scored targets with a recovered interface signal
    df = df.dropna(subset=["lrmsd", "iptm_iface"]).reset_index(drop=True)
    df["correct"] = (df["lrmsd"] <= 2.0).astype(int)
    df["is_unseen_protein"] = df["is_unseen_protein"].astype(bool)
    return df


def run() -> dict:
    rnp = load_delivered()
    rnp_prot = rnp[rnp.method == "protenix"].copy()
    fb = load_regen()

    n = len(fb)
    n_unseen = int(fb["is_unseen_protein"].sum())
    out = {
        "alpha": ALPHA,
        "delta": DELTA,
        "novelty_axis": "is_unseen_protein (1 = low-homology unseen protein)",
        "model": "protenix (regenerated on FoldBench, self-scored ligand-RMSD)",
        "n_targets_scored": n,
        "n_unseen": n_unseen,
        "n_seen": n - n_unseen,
        "regen_success_rate": float(fb["correct"].mean()),
        "feature_parity_note": (
            "interface-ipTM + ranking_score are the RNP-shared signals recovered by "
            "regeneration. Feature and label are self-consistent (both from the same "
            "Protenix v0.5.5 run; ligand-RMSD self-scored vs the deposited assembly). "
            "PoseBusters / ensemble / cross-model were not reconstructed; other FoldBench "
            "models were not regenerated (still ranking_score-only in E15)."
        ),
        "iptm_transfer": _transfer(rnp_prot, "iface_iptm", fb, "iptm_iface"),
        "ranking_transfer": _transfer(rnp_prot, "ranking_score", fb, "ranking_score"),
    }
    return out


def make_figure(res: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    feats = [("iptm_transfer", "interface-ipTM (recovered)"),
             ("ranking_transfer", "ranking_score (E15 signal)")]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    x = np.arange(len(feats))
    w = 0.35

    def nz(v):
        return 0.0 if v is None or not np.isfinite(v) else v

    seen = [nz(res[k]["seen"]["selective_risk"]) for k, _ in feats]
    unseen = [nz(res[k]["unseen"]["selective_risk"]) for k, _ in feats]
    seen_cov = [nz(res[k]["seen"]["coverage"]) for k, _ in feats]
    unseen_cov = [nz(res[k]["unseen"]["coverage"]) for k, _ in feats]
    ax.bar(x - w / 2, seen, w, label="seen protein", color="#48c")
    ax.bar(x + w / 2, unseen, w, label="unseen protein (novel)", color="#c44")
    ax.axhline(ALPHA, ls="--", color="k", lw=1)
    ax.text(-0.4, ALPHA + 0.01, f"target alpha = {ALPHA}", fontsize=9)
    for xi, r, c in zip(x - w / 2, seen, seen_cov, strict=False):
        ax.text(xi, r + 0.005, f"cov\n{c:.2f}", ha="center", va="bottom", fontsize=7, color="#246")
    for xi, r, c in zip(x + w / 2, unseen, unseen_cov, strict=False):
        ax.text(xi, r + 0.005, f"cov\n{c:.2f}", ha="center", va="bottom", fontsize=7, color="#622")
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in feats])
    ax.set_ylabel("realized selective risk among accepted")
    ax.set_title("E15b: frozen RNP Protenix gate on regenerated FoldBench Protenix")
    ax.legend(fontsize=8)
    fig.tight_layout()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGDIR / "e15b_foldbench_iptm_transfer.png", dpi=150)
    print(f"saved {FIGDIR / 'e15b_foldbench_iptm_transfer.png'}")


def _fmt(v, spec=".3f"):
    return "  nan" if v is None or not np.isfinite(v) else format(v, spec)


def main() -> None:
    res = run()
    save_json(res, RESDIR / "e15b_foldbench_iptm_transfer.json")

    print("E15b -- feature-matched frozen transfer (recovered interface-ipTM)")
    print(f"alpha={ALPHA} delta={DELTA}  |  Protenix regenerated on FoldBench")
    print(f"targets scored: {res['n_targets_scored']}  "
          f"(seen {res['n_seen']}, unseen {res['n_unseen']})  "
          f"regen success rate {res['regen_success_rate']:.3f}\n")

    hdr = (f"{'feature':22} {'split':7} {'n':>4} {'nacc':>4} {'cov':>5} "
           f"{'risk':>6} {'risk CI90':>15} {'>a?':>4} {'AURC':>6}")
    print(hdr)
    print("-" * len(hdr))
    for key, label in (("iptm_transfer", "interface-ipTM"),
                       ("ranking_transfer", "ranking_score")):
        t = res[key]
        print(f"{label:22} tau={_fmt(t['tau'])}  (RNP calib n={t['rnp_n_calib']}, "
              f"home risk={_fmt(t['rnp_home']['selective_risk'])} "
              f"cov={_fmt(t['rnp_home']['coverage'],'.2f')})")
        for split in ("overall", "seen", "unseen"):
            s = t[split]
            lo, hi = s["selective_risk_ci90"]
            ci = f"[{_fmt(lo,'.2f')},{_fmt(hi,'.2f')}]"
            over = "" if s["risk_over_alpha"] is None else ("YES" if s["risk_over_alpha"] else "no")
            print(f"{'':22} {split:7} {s['n']:>4} {s['n_accept']:>4} "
                  f"{_fmt(s['coverage'],'.2f'):>5} {_fmt(s['selective_risk'],'.3f'):>6} "
                  f"{ci:>15} {over:>4} {_fmt(s['aurc'],'.3f'):>6}")
        print()

    make_figure(res)


if __name__ == "__main__":
    main()
