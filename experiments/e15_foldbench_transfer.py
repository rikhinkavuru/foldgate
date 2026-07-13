"""E15 -- frozen calibrate-on-RNP / deploy-on-FoldBench transfer.

E10 refit the gate inside FoldBench, which risks a small-n overfit confound. E15
removes that confound: it FREEZES a threshold calibrated on RNP and deploys it on
FoldBench unchanged. Per model we calibrate an LTT selective-risk threshold tau on
RNP ranking_score at ALPHA / DELTA, freeze it, and read off the realized selective
risk and coverage on the FoldBench top-1 table, overall and split by
is_unseen_protein. The unseen (low-homology) split is a second-dataset replication
of the E2 break: a gate exchangeable with RNP is not guaranteed to control error on
FoldBench proteins the training set never saw, so under-control there is the
expected, honest outcome.

Feature-parity limit, stated up front. FoldBench publicly shares only ranking_score
per pose. Interface-ipTM, PoseBusters, and the ensemble and cross-model signals the
E5 ablation credits for the RNP gain are all unavailable, so the learned combiner
has no shared features to carry across and cannot be transferred. E15 therefore
tests the one shared signal, ranking_score. Any null or under-coverage here is
feature-parity-limited by construction. It is not evidence that the method fails to
generalize.

The FoldBench->RNP model map keeps only models present in both with enough RNP data:
AlphaFold 3->af3, Boltz-1->boltz1, Chai-1->chai, Protenix->protenix. HelixFold 3 has
no RNP counterpart and Boltz-2 is absent from the FoldBench confidence table, so both
are reported as skipped.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from experiments._common import (  # noqa: E402
    ALPHA,
    CONF,
    DELTA,
    FIGDIR,
    RESDIR,
    load_delivered,
    methods_with_enough,
    save_json,
)
from foldgate.conformal import ltt_threshold  # noqa: E402
from foldgate.io.foldbench import load_foldbench_novelty  # noqa: E402
from foldgate.selective.metrics import aurc, clopper_pearson, evaluate_gate  # noqa: E402

# FoldBench model name -> RNP method key. Boltz-2 is absent from the FoldBench
# confidence table; HelixFold 3 has no RNP counterpart. Both drop out below.
FB_TO_RNP = {
    "AlphaFold 3": "af3",
    "Boltz-1": "boltz1",
    "Boltz-2": "boltz2",
    "Chai-1": "chai",
    "Protenix": "protenix",
}
CI = 0.90


def _report(scores: np.ndarray, correct: np.ndarray, tau: float | None) -> dict:
    """Realized gate stats on one subset with Clopper-Pearson CIs and AURC."""
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
        # honest break flags: point estimate over alpha, and the stronger
        # certified-break check (CP lower bound over alpha).
        "risk_over_alpha": bool(r["selective_risk"] > ALPHA) if n_acc else None,
        "risk_ci_lo_over_alpha": bool(risk_ci[0] > ALPHA) if n_acc else None,
        "aurc": float(aurc(scores, correct)) if n >= 1 else float("nan"),
    }


def run() -> dict:
    rnp = load_delivered()
    fb = load_foldbench_novelty()
    rnp_ok = set(methods_with_enough(rnp))
    fb_models = set(fb["model"].unique())

    models, skipped = {}, {}
    for fb_name in sorted(fb_models):
        if fb_name not in FB_TO_RNP:
            skipped[fb_name] = "no RNP counterpart method"
    for fb_name, key in FB_TO_RNP.items():
        if fb_name not in fb_models:
            skipped[fb_name] = "absent from FoldBench confidence table"
            continue
        if key not in rnp_ok:
            skipped[fb_name] = f"RNP method '{key}' below MIN_METHOD_N or absent"
            continue

        # 1. calibrate + freeze tau on ALL RNP delivered poses for this method.
        rsub = rnp[rnp.method == key].dropna(subset=[CONF, "correct"]).reset_index(drop=True)
        s_r = rsub[CONF].to_numpy()
        y_r = rsub["correct"].to_numpy().astype(int)
        tau = ltt_threshold(s_r, y_r, alpha=ALPHA, delta=DELTA)

        # 2. deploy the frozen tau on FoldBench, overall + seen/unseen.
        fsub = fb[fb.model == fb_name].dropna(
            subset=["ranking_score", "correct", "is_unseen_protein"]
        ).reset_index(drop=True)
        s_f = fsub["ranking_score"].to_numpy()
        y_f = fsub["correct"].to_numpy().astype(int)
        unseen = fsub["is_unseen_protein"].to_numpy(dtype=bool)

        models[key] = {
            "fb_model_name": fb_name,
            "tau": float(tau) if tau is not None else None,
            "rnp_n_calib": int(len(rsub)),
            # in-sample RNP control at the frozen tau (the gate works at home).
            "rnp_home": _report(s_r, y_r, tau),
            "overall": _report(s_f, y_f, tau),
            "seen": _report(s_f[~unseen], y_f[~unseen], tau),
            "unseen": _report(s_f[unseen], y_f[unseen], tau),
        }

    return {
        "alpha": ALPHA,
        "delta": DELTA,
        "conf": CONF,
        "novelty_axis": "is_unseen_protein (1 = low-homology unseen protein)",
        "feature_parity_note": (
            "FoldBench publicly shares only ranking_score per pose. The combiner "
            "features (interface-ipTM, PoseBusters, ensemble, cross-model) are "
            "unavailable, so the combiner cannot transfer. E15 tests the one shared "
            "signal. A null or under-coverage on unseen is feature-parity-limited "
            "by construction, not evidence the method fails to generalize."
        ),
        "models": models,
        "skipped": skipped,
    }


def make_figure(res: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = res["models"]
    keys = list(models.keys())
    x = np.arange(len(keys))
    seen_risk = [models[k]["seen"]["selective_risk"] for k in keys]
    unseen_risk = [models[k]["unseen"]["selective_risk"] for k in keys]
    seen_cov = [models[k]["seen"]["coverage"] for k in keys]
    unseen_cov = [models[k]["unseen"]["coverage"] for k in keys]

    def _nz(v):
        return 0.0 if v is None or not np.isfinite(v) else v

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.bar(x - 0.2, [_nz(v) for v in seen_risk], 0.4, label="seen protein", color="#48c")
    ax.bar(x + 0.2, [_nz(v) for v in unseen_risk], 0.4, label="unseen protein (novel)", color="#c44")
    ax.axhline(ALPHA, ls="--", color="k", lw=1)
    ax.text(-0.45, ALPHA + 0.01, f"target alpha = {ALPHA}", fontsize=9)
    for xi, r, c in zip(x - 0.2, seen_risk, seen_cov, strict=False):
        ax.text(xi, _nz(r) + 0.005, f"cov\n{_nz(c):.2f}", ha="center", va="bottom", fontsize=7, color="#246")
    for xi, r, c in zip(x + 0.2, unseen_risk, unseen_cov, strict=False):
        ax.text(xi, _nz(r) + 0.005, f"cov\n{_nz(c):.2f}", ha="center", va="bottom", fontsize=7, color="#622")
    ax.set_xticks(x)
    ax.set_xticklabels(keys)
    ax.set_xlabel("model (RNP-calibrated tau, frozen, deployed on FoldBench)")
    ax.set_ylabel("realized selective risk among accepted")
    ax.set_title("E15: frozen RNP gate on FoldBench -- seen vs unseen protein")
    ax.legend(fontsize=8)
    fig.tight_layout()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGDIR / "e15_foldbench_transfer.png", dpi=150)
    print(f"saved {FIGDIR / 'e15_foldbench_transfer.png'}")


def _fmt(v, spec=".3f"):
    return "  nan" if v is None or not np.isfinite(v) else format(v, spec)


def main() -> None:
    res = run()
    save_json(res, RESDIR / "e15_foldbench_transfer.json")

    print(f"E15 -- frozen RNP gate deployed on FoldBench  (alpha={ALPHA}, delta={DELTA})\n")
    print("Feature-parity limit: only ranking_score is shared, so the combiner cannot")
    print("transfer. This tests the single shared signal. Numbers below are honest.\n")

    hdr = (f"{'model':9} {'split':7} {'n':>4} {'nacc':>4} {'cov':>5} "
           f"{'risk':>6} {'risk CI90':>15} {'>a?':>4} {'AURC':>6}")
    print(hdr)
    print("-" * len(hdr))
    for k, r in res["models"].items():
        print(f"{k:9} tau={_fmt(r['tau'])}  (RNP calib n={r['rnp_n_calib']}, "
              f"home risk={_fmt(r['rnp_home']['selective_risk'])} cov={_fmt(r['rnp_home']['coverage'],'.2f')})")
        for split in ("overall", "seen", "unseen"):
            s = r[split]
            lo, hi = s["selective_risk_ci90"]
            ci = f"[{_fmt(lo,'.2f')},{_fmt(hi,'.2f')}]"
            over = "" if s["risk_over_alpha"] is None else ("YES" if s["risk_over_alpha"] else "no")
            print(f"{'':9} {split:7} {s['n']:>4} {s['n_accept']:>4} {_fmt(s['coverage'],'.2f'):>5} "
                  f"{_fmt(s['selective_risk'],'.3f'):>6} {ci:>15} {over:>4} {_fmt(s['aurc'],'.3f'):>6}")
        print()

    if res["skipped"]:
        print("skipped models:")
        for name, why in res["skipped"].items():
            print(f"  {name}: {why}")
        print()

    make_figure(res)


if __name__ == "__main__":
    main()
