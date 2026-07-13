"""E12 -- reliability drift D(nu): the concept-shift diagnostic as a first-class result.

Weighted conformal is exact only under *pure covariate shift*, where the map
P(correct | confidence) is stable and only the marginal P(confidence) moves. On
novel pockets/chemotypes that map itself degrades (concept shift), so importance
reweighting controls an aligned distribution and cannot restore the target risk.
E3b showed this empirically (weighted-LTT abstains); here we promote it to a
measurable, CI'd quantity and derive the operational consequence.

Reliability drift at novelty stratum k, relative to the familiar reference S0:

    D_signed(k) = sum_bins  w_bin * ( P(correct|conf, S0) - P(correct|conf, Sk) )
    D_abs(k)    = sum_bins  w_bin * | P(correct|conf, S0) - P(correct|conf, Sk) |

binned on a shared confidence grid, weighted by the stratum-k bin mass. D_signed > 0
means: at the *same* reported confidence, poses in the novel stratum are less often
correct -- the confidence is optimistic exactly where it matters. We report D across
three novelty axes (ligand-, pocket-, temporal-) with a bootstrap CI, so the reader
sees drift rise on structural novelty and stay flat on recency, and we attach the
admissibility rule: where D is small and its CI covers 0, covariate reweighting is
admissible; where D is large, only group-conditional calibration or abstention is valid.

Multiplicity control (docs/theory/MULTIPLICITY_SPEC.md).
This is a discovery family -- "drift is present somewhere in the model x axis x
stratum grid" -- so per-cell 90% CIs excluding zero inflate the family-wise false
positive rate. We add three layers on top of the raw per-cell numbers:

  * Romano-Wolf step-down max-t bootstrap (primary): one-sided FWER control at 0.10
    across the whole cell family, exploiting the strong positive dependence between
    cells that share the S0 reference resample. Emits step-down adjusted p-values and
    a single simultaneous critical value that gives a joint 90% band.
  * Benjamini-Hochberg FDR at q = 0.10 (alternative): the right control when the
    claim is the collective "drift is elevated on structural novelty".
  * TOST equivalence (temporal cells): a "flat" verdict is a non-rejection and cannot
    prove a null, so we upgrade "recency does not degrade confidence" to a two-one-
    sided-tests rejection of |D| >= SMALL_DRIFT. The joint "temporal flat for every
    model" is then a conjunction of equivalence rejections, an intersection-union test
    that needs no penalty.

The per-cell resample is shared across the strata of an axis within a model (one
resample per bootstrap rep drives every k), so the four ligand cells that share the
ligand-S0 reference move together and Romano-Wolf sees that dependence. Across
(model, axis) groups the reps are drawn independently, which the family max-t treats
as independent cells; that is the conservative direction for a critical value, so the
FWER guarantee is kept (a hair of power is left on the table).
"""

from __future__ import annotations

import numpy as np

from experiments._common import (
    CONF,
    DELTA,
    FIGDIR,
    RESDIR,
    bh,
    load_delivered,
    methods_with_enough,
    rng,
    romano_wolf_stepdown,
    save_json,
    tost_equivalence,
)

AXES = {
    "ligand": "novelty_stratum",
    "pocket": "pocket_novelty_stratum",
    "temporal": "temporal_stratum",
}
N_BINS = 5
N_BOOT = 2000
# |D_signed| below this, with a CI covering it, reads as covariate-only (reweighting admissible).
# It is also the TOST equivalence margin for the temporal cells.
SMALL_DRIFT = 0.05
# Family-wise level for Romano-Wolf, and q for the BH alternative (one-sided).
FWER = DELTA


def _drift(conf_s, y_s, conf_t, y_t, edges):
    """Target-mass-weighted signed and absolute P(correct|conf) gap, S0 -> Sk."""
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


def _edges(conf_s, conf_t):
    e = np.quantile(np.concatenate([conf_s, conf_t]), np.linspace(0, 1, N_BINS + 1))
    e[0], e[-1] = -np.inf, np.inf
    return e


def _verdict(d_signed, ci_lo, ci_hi):
    """Raw per-cell verdict from the point estimate and its pointwise 90% CI."""
    if not np.isfinite(d_signed):
        return "undefined"
    if abs(d_signed) < SMALL_DRIFT and ci_lo <= 0 <= ci_hi:
        return "covariate-only (reweighting admissible)"
    if d_signed >= SMALL_DRIFT and ci_lo > 0:
        return "concept drift (group-conditional / abstain)"
    return "ambiguous"


def _adjusted_verdict(axis, d_signed, rw_reject, tost_equiv):
    """Verdict after multiplicity control (Romano-Wolf FWER + TOST for temporal)."""
    if rw_reject and d_signed > 0:
        return "concept drift (FWER-adjusted; group-conditional / abstain)"
    if axis == "temporal" and tost_equiv:
        return "flat (TOST-equivalent to zero; reweighting admissible)"
    return "inconclusive (no FWER drift, no equivalence)"


def run() -> dict:
    df = load_delivered()
    methods = methods_with_enough(df)
    g = rng()
    out = {m: {} for m in methods}

    # Every cell in the model x axis x stratum grid, collected flat so Romano-Wolf and
    # BH run once over the whole family. Each cell carries its point estimate, the
    # finite bootstrap draws (pointwise CI), and a fixed-length filled draw vector
    # (Romano-Wolf matrix + standard error).
    cells = []           # ordered metadata, one dict per cell
    boot_matrix_rows = []  # (B,) filled signed-drift draws, aligned with cells

    for m in methods:
        sub = df[df.method == m]
        for axis, col in AXES.items():
            s = sub.dropna(subset=[CONF, col])
            conf = s[CONF].to_numpy()
            y = s["correct"].to_numpy().astype(int)
            strat = s[col].to_numpy().astype(int)
            levels = sorted(np.unique(strat).tolist())
            ref = levels[0]
            m_ref = strat == ref

            ks = [k for k in levels if k != ref and int((strat == k).sum()) >= 20]
            if not ks:
                out[m][axis] = {}
                continue

            edges_by_k = {k: _edges(conf[m_ref], conf[strat == k]) for k in ks}
            point = {}
            for k in ks:
                d_s, d_a = _drift(conf[m_ref], y[m_ref], conf[strat == k], y[strat == k], edges_by_k[k])
                point[k] = (d_s, d_a)

            # Shared resample across this axis's strata: one bootstrap index per rep,
            # so all k cells reuse the same S0 reference draw and move together.
            n = len(conf)
            boot_s = {k: np.empty(N_BOOT) for k in ks}
            boot_a = {k: np.empty(N_BOOT) for k in ks}
            for b in range(N_BOOT):
                bi = g.integers(0, n, n)
                cb, yb, stb = conf[bi], y[bi], strat[bi]
                ref_b = stb == ref
                cref, yref = cb[ref_b], yb[ref_b]
                for k in ks:
                    km = stb == k
                    ss, sa = _drift(cref, yref, cb[km], yb[km], edges_by_k[k])
                    boot_s[k][b] = ss
                    boot_a[k][b] = sa

            axis_out = {}
            for k in ks:
                d_signed, d_abs = point[k]
                bs, ba = boot_s[k], boot_a[k]
                finite = np.isfinite(bs)
                bs_ci = bs[finite]
                ba_ci = ba[np.isfinite(ba)]
                lo = float(np.quantile(bs_ci, 0.05)) if bs_ci.size else float("nan")
                hi = float(np.quantile(bs_ci, 0.95)) if bs_ci.size else float("nan")
                # Fixed-length draws for the Romano-Wolf matrix: non-finite reps fall
                # back to the point estimate (a neutral, recenters-to-zero fill).
                bs_full = np.where(finite, bs, d_signed)

                cell = {
                    "model": m,
                    "axis": axis,
                    "k": int(k),
                    "D_signed": d_signed,
                    "D_signed_ci90": [lo, hi],
                    "D_abs": d_abs,
                    "D_abs_ci90": [
                        float(np.quantile(ba_ci, 0.05)) if ba_ci.size else float("nan"),
                        float(np.quantile(ba_ci, 0.95)) if ba_ci.size else float("nan"),
                    ],
                    "n_ref": int(m_ref.sum()),
                    "n_stratum": int((strat == k).sum()),
                    "verdict": _verdict(d_signed, lo, hi),
                }
                if axis == "temporal":
                    t_lo = float(np.quantile(bs_ci, 0.05)) if bs_ci.size else float("nan")
                    t_hi = float(np.quantile(bs_ci, 0.95)) if bs_ci.size else float("nan")
                    cell["tost_equivalent"] = bool(tost_equivalence(bs_full, SMALL_DRIFT, alpha=0.05))
                    cell["tost_interval90"] = [t_lo, t_hi]
                cells.append(cell)
                boot_matrix_rows.append(bs_full)
                axis_out[k] = cell
            out[m][axis] = axis_out

    # --- Family-wide multiplicity control over every cell -----------------------
    stat_hat = np.array([c["D_signed"] for c in cells], dtype=float)
    boot_matrix = np.vstack(boot_matrix_rows) if boot_matrix_rows else np.zeros((0, N_BOOT))
    se = boot_matrix.std(axis=1) if boot_matrix.size else np.zeros(0)

    rw = romano_wolf_stepdown(boot_matrix, stat_hat, se, level=FWER)
    se_safe = np.where(se > 0, se, np.finfo(float).tiny)
    t_obs = stat_hat / se_safe
    tstar = (boot_matrix - stat_hat[:, None]) / se_safe[:, None]
    p_one_sided = (1.0 + (tstar >= t_obs[:, None]).sum(axis=1)) / (1.0 + boot_matrix.shape[1]) \
        if boot_matrix.size else np.ones(len(cells))
    bh_reject = bh(p_one_sided, q=FWER) if len(cells) else np.zeros(0, dtype=bool)
    crit = rw["crit_value"]

    for i, c in enumerate(cells):
        band_lo = float(stat_hat[i] - crit * se[i])
        band_hi = float(stat_hat[i] + crit * se[i])
        c["t_stat"] = float(rw["t_stat"][i])
        c["se_boot"] = float(se[i])
        c["p_one_sided"] = float(p_one_sided[i])
        c["rw_p_adjusted"] = float(rw["p_adjusted"][i])
        c["rw_reject"] = bool(rw["reject"][i])
        c["bh_reject"] = bool(bh_reject[i])
        c["simultaneous_band90"] = [band_lo, band_hi]
        c["clears_simultaneous_band"] = bool(rw["t_stat"][i] >= crit)
        c["adjusted_verdict"] = _adjusted_verdict(
            c["axis"], c["D_signed"], c["rw_reject"], c.get("tost_equivalent", False)
        )

    # Joint temporal-equivalence statement, per temporal stratum, as an IUT: the claim
    # "temporal drift is practically zero for EVERY model at stratum k" holds when each
    # per-model TOST rejects non-equivalence, with no multiplicity penalty (Berger IUT).
    temporal_cells = [c for c in cells if c["axis"] == "temporal"]
    temporal_ks = sorted({c["k"] for c in temporal_cells})
    temporal_flat_all_models = {}
    for k in temporal_ks:
        at_k = [c for c in temporal_cells if c["k"] == k]
        temporal_flat_all_models[str(k)] = bool(at_k) and all(c["tost_equivalent"] for c in at_k)

    out["_multiplicity"] = {
        "family_size": len(cells),
        "n_boot": N_BOOT,
        "fwer_level": FWER,
        "romano_wolf_crit_value": float(crit),
        "romano_wolf_n_reject": int(np.count_nonzero(rw["reject"])) if len(cells) else 0,
        "bh_q": FWER,
        "bh_n_reject": int(np.count_nonzero(bh_reject)) if len(cells) else 0,
        "tost_margin": SMALL_DRIFT,
        "tost_alpha": 0.05,
        "temporal_flat_all_models": temporal_flat_all_models,
        "notes": (
            "Romano-Wolf step-down max-t (one-sided, H0: D_signed<=0) controls the "
            "family-wise error at fwer_level across every model x axis x stratum cell; "
            "BH is the FDR alternative; TOST turns temporal 'flat' into an equivalence "
            "rejection at margin tost_margin. The joint temporal-flat claim is an IUT."
        ),
    }
    return out


def make_figure(res: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = [k for k in res.keys() if not k.startswith("_")]
    colors = {"ligand": "#c44", "pocket": "#e80", "temporal": "#48c"}
    fig, axes = plt.subplots(1, len(methods), figsize=(3.2 * len(methods), 3.4), sharey=True)
    if len(methods) == 1:
        axes = [axes]
    for ax, m in zip(axes, methods, strict=False):
        for axis, col in colors.items():
            ks = sorted(res[m][axis].keys())
            if not ks:
                continue
            xs = list(ks)
            ys = [res[m][axis][k]["D_signed"] for k in ks]
            # simultaneous (family-wide) band, drawn wide and translucent behind the point CI
            sb_lo = [res[m][axis][k]["D_signed"] - res[m][axis][k]["simultaneous_band90"][0] for k in ks]
            sb_hi = [res[m][axis][k]["simultaneous_band90"][1] - res[m][axis][k]["D_signed"] for k in ks]
            ax.errorbar(xs, ys, yerr=[sb_lo, sb_hi], fmt="none", ecolor=col,
                        alpha=0.22, elinewidth=6, capsize=0)
            # pointwise 90% CI on top
            los = [res[m][axis][k]["D_signed"] - res[m][axis][k]["D_signed_ci90"][0] for k in ks]
            his = [res[m][axis][k]["D_signed_ci90"][1] - res[m][axis][k]["D_signed"] for k in ks]
            ax.errorbar(xs, ys, yerr=[los, his], marker="o", ms=4, capsize=2,
                        color=col, label=axis, lw=1.4)
        ax.axhline(0.0, ls="--", color="k", lw=0.8)
        ax.axhline(SMALL_DRIFT, ls=":", color="0.5", lw=0.7)
        ax.axhline(-SMALL_DRIFT, ls=":", color="0.5", lw=0.7)
        ax.set_title(m, fontsize=10)
        ax.set_xlabel("novelty stratum (0 familiar)")
    axes[0].set_ylabel("reliability drift D_signed(nu)")
    axes[-1].legend(fontsize=8, title="axis")
    crit = res.get("_multiplicity", {}).get("romano_wolf_crit_value", float("nan"))
    fig.suptitle(
        "E12: confidence reliability degrades on structural novelty, not recency "
        f"(pale band = Romano-Wolf simultaneous 90%, crit={crit:.2f})",
        fontsize=10,
    )
    fig.tight_layout()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGDIR / "e12_reliability_drift.png", dpi=150)
    print(f"saved {FIGDIR / 'e12_reliability_drift.png'}")


def main() -> None:
    res = run()
    save_json(res, RESDIR / "e12_reliability_drift.json")
    mult = res["_multiplicity"]
    print("E12 -- reliability drift D(nu)  (S0 reference, target-mass-weighted P(correct|conf) gap)")
    print(f"multiplicity: family={mult['family_size']} cells, N_BOOT={mult['n_boot']}, "
          f"Romano-Wolf FWER={mult['fwer_level']} crit={mult['romano_wolf_crit_value']:.2f}, "
          f"RW rejects={mult['romano_wolf_n_reject']}, BH(q={mult['bh_q']}) rejects={mult['bh_n_reject']}\n")
    for m in [k for k in res if not k.startswith("_")]:
        print(f"[{m}]")
        for axis, ks in res[m].items():
            for k, r in ks.items():
                lo, hi = r["D_signed_ci90"]
                flags = "RW" if r["rw_reject"] else "  "
                flags += "/BH" if r["bh_reject"] else "/  "
                if axis == "temporal":
                    flags += "/TOST=eq" if r.get("tost_equivalent") else "/TOST=no"
                print(f"   {axis:>9} S{k}: D_signed={r['D_signed']:+.3f} "
                      f"[{lo:+.3f},{hi:+.3f}]  padj={r['rw_p_adjusted']:.3f} [{flags}]  ({r['adjusted_verdict']})")
        print()
    print("temporal flat for every model (IUT of TOST equivalence, no penalty): "
          f"{mult['temporal_flat_all_models']}")
    make_figure(res)


if __name__ == "__main__":
    main()
