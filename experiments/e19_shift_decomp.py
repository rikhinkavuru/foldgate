"""E19 -- concept-vs-covariate decomposition of the novel-stratum risk gap.

E3b showed the weighted-LTT gate abstains on the novel-pocket strata rather than
certifying them, and CLAUDE.md poses the open question directly: is the shift pure
covariate shift (weighted CP applies) or does confidence reliability itself degrade
on novel chemotypes (concept shift, which weighted CP cannot fix)? E19 answers it
with a signed, confidence-interval'd decomposition instead of a bare assertion.

For each model, take SOURCE = the familiar strata {0, 1, 2} and TARGET = each novel
stratum. Fit the accept threshold tau by LTT on a held-out half of the source, then
on the accept region {confidence >= tau} decompose the realized selective-risk gap

    Gap_total(tau)   = R_target(tau) - R_source(tau),
    Gap_concept(tau) = the residual after the source is optimally reweighted on the
                       score to match the target score marginal (certified, with a
                       finite-sample bootstrap CI),
    Gap_covariate    = Gap_total - Gap_concept  (descriptive, uncertified).

Gap_concept is the part of the gap that no covariate reweighting on the score can
close (proposition in foldgate.conformal.shift_decomp; the score-space, accept-region
analogue of the Ben-David et al. 2010 domain-adaptation lambda term, and the term
Tibshirani et al. 2019 weighted CP assumes to be zero). The headline: on the moderate
novel stratum S3 the certified floor Gap_concept is non-zero, which is exactly why
weighted CP abstains and the group-conditional route (E3) is necessary; on the thin
extreme stratum S4 (n ~ 76 per model) the certificate is honestly wide and vacuous.
"""

from __future__ import annotations

import numpy as np

from experiments._common import (
    ALPHA,
    CONF,
    DELTA,
    FIGDIR,
    RESDIR,
    load_delivered,
    methods_with_enough,
    rng,
    save_json,
)
from foldgate.conformal.risk import ltt_threshold, naive_threshold
from foldgate.conformal.shift_decomp import shift_decomposition

STRAT = "novelty_stratum"
SOURCE_STRATA = (0, 1, 2)
TARGET_STRATA = (3, 4)
N_BOOT = 4000


def _split_and_fit(
    scores: np.ndarray, correct: np.ndarray, g: np.random.Generator
) -> tuple[float, str, np.ndarray]:
    """Split the source once; fit tau on one half, return the complementary eval half.

    A single permutation is split into fit and eval halves so the threshold is
    out-of-fold with respect to every quantity the decomposition later reads off the
    source, which keeps R_source honest (no threshold-selection optimism). tau is the
    LTT accept threshold on the fit half. When LTT certifies nothing (a model whose
    top-confidence poses are already miscalibrated above alpha), fall back to the
    naive empirical-risk threshold, then to the fit-half median, recording which rule
    set tau.
    """
    n = len(scores)
    perm = g.permutation(n)
    fit, eval_idx = perm[: n // 2], perm[n // 2:]
    sf, yf = scores[fit], correct[fit]
    tau = ltt_threshold(sf, yf, alpha=ALPHA, delta=DELTA)
    if tau is not None:
        return float(tau), "ltt", eval_idx
    tau = naive_threshold(sf, yf, alpha=ALPHA)
    if tau is not None:
        return float(tau), "naive", eval_idx
    return float(np.quantile(sf, 0.5)), "median", eval_idx


def _model_seed(m: str) -> int:
    """Deterministic per-model seed (stable across runs, unlike Python's hash)."""
    return 20260710 + sum(ord(ch) for ch in m)


def run() -> dict:
    df = load_delivered()
    methods = methods_with_enough(df)
    out: dict = {"alpha": ALPHA, "delta": DELTA, "n_boot": N_BOOT, "models": {}}

    for m in methods:
        sub = df[df.method == m].dropna(subset=[CONF, STRAT]).reset_index(drop=True)
        s = sub[CONF].to_numpy()
        y = sub["correct"].to_numpy().astype(int)
        strat = sub[STRAT].to_numpy().astype(int)

        src = np.isin(strat, SOURCE_STRATA)
        s_src_all, y_src_all = s[src], y[src]

        # One deterministic source split: fit tau on one half, read the
        # decomposition off the complementary half so the source side stays out-of-fold.
        g = rng(seed=_model_seed(m))
        tau, tau_rule, eval_idx = _split_and_fit(s_src_all, y_src_all, g)
        s_src, y_src = s_src_all[eval_idx], y_src_all[eval_idx]

        targets: dict = {}
        target_specs = [(str(k), strat == k) for k in TARGET_STRATA]
        target_specs.append(("3+4", np.isin(strat, TARGET_STRATA)))
        for name, mask in target_specs:
            s_tgt, y_tgt = s[mask], y[mask]
            dec = shift_decomposition(
                s_src, y_src, s_tgt, y_tgt,
                tau=tau, n_bins=5, delta=DELTA, n_boot=N_BOOT, seed=0,
            )
            dec["n_target_total"] = int(mask.sum())
            targets[name] = dec

        out["models"][m] = {
            "tau": tau,
            "tau_rule": tau_rule,
            "n_source_total": int(src.sum()),
            "n_source_eval": int(len(s_src)),
            "targets": targets,
        }
    return out


def make_figure(res: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = list(res["models"].keys())
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, tname, title in zip(
        axes, ("3", "4"),
        ("target S3 (moderate novelty)", "target S4 (thin: no training analog)"),
        strict=False,
    ):
        x = np.arange(len(models))
        concept = np.array([res["models"][m]["targets"][tname]["gap_concept"] for m in models])
        covar = np.array([res["models"][m]["targets"][tname]["gap_covariate"] for m in models])
        total = np.array([res["models"][m]["targets"][tname]["gap_total"] for m in models])
        ci = np.array([res["models"][m]["targets"][tname]["ci"] for m in models])
        nonvac = [res["models"][m]["targets"][tname]["concept_nonvacuous"] for m in models]
        lo_err = np.clip(concept - ci[:, 0], 0, None)
        hi_err = np.clip(ci[:, 1] - concept, 0, None)

        ax.bar(x - 0.2, concept, 0.4, color="#c44", label="Gap_concept (certified floor)")
        ax.errorbar(x - 0.2, concept, yerr=[lo_err, hi_err], fmt="none",
                    ecolor="k", elinewidth=1.2, capsize=3)
        ax.bar(x + 0.2, covar, 0.4, color="#48c", label="Gap_covariate (descriptive)")
        ax.plot(x, total, "kD", ms=6, label="Gap_total (realized)")
        ax.axhline(0.0, color="k", lw=0.8)
        for xi, cval, nv in zip(x, concept, nonvac, strict=False):
            if nv:
                ax.text(xi - 0.2, cval, "*", ha="center", va="bottom",
                        fontsize=15, color="#000")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=20, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("selective-risk gap (target - source), accept region")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("E19: concept-shift floor that no covariate reweighting on the score can close "
                 "(* = CI excludes 0)", fontsize=11)
    fig.tight_layout()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGDIR / "e19_shift_decomp.png", dpi=150)
    print(f"saved {FIGDIR / 'e19_shift_decomp.png'}")


def _synthetic_line() -> str:
    from foldgate.conformal.shift_decomp import _synthetic_check
    r = _synthetic_check(seed=0)
    c, k = r["covariate_only"], r["concept_only"]
    return (f"synthetic validity: covariate-null gap={c['gap_concept']:+.4f} "
            f"ci=[{c['ci'][0]:+.4f},{c['ci'][1]:+.4f}] covers0={c['covers_zero']}  |  "
            f"concept gap={k['gap_concept']:+.4f} floor={k['floor_lower']:+.4f} "
            f"nonvacuous={k['concept_nonvacuous']}")


def main() -> None:
    res = run()
    save_json(res, RESDIR / "e19_shift_decomp.json")
    print(f"E19 -- shift decomposition  (alpha={ALPHA}, delta={DELTA}, boot={N_BOOT})")
    print("Gap = selective-risk (error-rate) gap target-minus-source among accepted.\n")
    hdr = (f"{'model':>9} {'tgt':>4} {'n_src':>6} {'n_tgt':>6} "
           f"{'R_src':>6} {'R_tgt':>6} {'total':>7} {'concept':>8} {'ci_lo':>7} {'ci_hi':>7} "
           f"{'covar':>7} {'floor':>7} {'nonvac':>6}")
    for m, mr in res["models"].items():
        print(f"[{m}]  tau={mr['tau']:.4f} (rule={mr['tau_rule']})"
              f"  n_source_eval={mr['n_source_eval']}")
        print(hdr)
        for tname, d in mr["targets"].items():
            print(f"{m:>9} {tname:>4} {d['n_accept_source']:>6} {d['n_accept_target']:>6} "
                  f"{d['R_source']:>6.3f} {d['R_target']:>6.3f} {d['gap_total']:>7.3f} "
                  f"{d['gap_concept']:>8.3f} {d['ci'][0]:>7.3f} {d['ci'][1]:>7.3f} "
                  f"{d['gap_covariate']:>7.3f} {d['floor_lower']:>7.3f} "
                  f"{'yes' if d['concept_nonvacuous'] else 'no':>6}")
        print()
    print(_synthetic_line())
    make_figure(res)


if __name__ == "__main__":
    main()
