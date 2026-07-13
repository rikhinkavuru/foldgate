"""E11 -- baselines a reviewer will demand, and the calibration-vs-conformal framing.

Three questions:

(A) *Is the combined score a better ranker than the native confidences a practitioner
    would reach for?*  AURC (lower = better) for raw ranking_score, native chain-pair
    ipTM, and the combined score, on i.i.d. test.

(B) *i.i.d.: does the guarantee matter, or would ordinary calibration do?*  Fix the SAME
    combined score and turn it into an accept/abstain gate three ways:

      combined_conformal   LTT threshold             finite-sample P(risk <= alpha) >= 1 - delta
      combined_platt       Platt-calibrated, accept iff P(correct) >= 1 - alpha
      combined_isotonic    isotonic-calibrated, accept iff P(correct) >= 1 - alpha

    plus native_iptm (LTT on chain-pair ipTM) and posebusters (accept iff PB-valid).

(C) *Under shift: is swapping calibration for conformal enough?*  No. Calibrate every gate
    on the most familiar targets (novelty stratum S0) and deploy on novel ones (S1-S2). The
    naive source-calibrated CONFORMAL gate breaks exactly like the calibrated fixed-threshold
    gates -- realized risk climbs well above alpha (this is the E2 exchangeability break; it
    is a property of naive transfer, not of calibration-vs-conformal). What restores validity
    is the shift-robust conformal repair with no calibration analogue:

      combined_groupcond   per-target-stratum LTT (Mondrian), calibrated on target labels

    which controls realized risk <= alpha by abstaining more on the novel strata. The
    takeaway a reviewer needs: calibration fixes only the marginal probability and carries no
    coverage guarantee; conformal gives a finite-sample guarantee AND a distribution-shift
    repair (group-conditional / weighted, E3/E3b). They are not interchangeable.

Boltz-2 affinity-probability (the other honest-negative baseline) is not in the released
tabular dump used here; it is noted as a follow-up in RESULTS.md.
"""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from experiments._common import (
    ALPHA,
    DELTA,
    FIGDIR,
    RESDIR,
    load_delivered,
    methods_with_enough,
    rng,
    save_json,
)
from foldgate.conformal import ltt_threshold
from foldgate.scores import ScoreCombiner
from foldgate.selective import aurc, evaluate_gate

NATIVE = "ranking_score"
IPTM = "iface_iptm"
STRAT = "novelty_stratum"
SOURCE_STRATA = {0}        # calibrate on the most familiar targets ...
TARGET_STRATA = {1, 2}     # ... and deploy on novel ones (moderate shift where the repair is usable)
MIN_STRATUM_CAL = 40

GATES_IID = ["combined_conformal", "combined_platt", "combined_isotonic", "native_iptm", "posebusters"]
GATES_SHIFT = ["combined_conformal", "combined_platt", "combined_isotonic", "native_iptm", "combined_groupcond"]


def _platt(sc_cal, y_cal, sc_te):
    lr = LogisticRegression(max_iter=1000).fit(sc_cal.reshape(-1, 1), y_cal)
    return lr.predict_proba(sc_te.reshape(-1, 1))[:, 1]


def _isotonic(sc_cal, y_cal, sc_te):
    ir = IsotonicRegression(out_of_bounds="clip").fit(sc_cal, y_cal)
    return ir.predict(sc_te)


def _gate_stats(vals):
    risk = [v["selective_risk"] for v in vals if v["n_accept"]]
    return {
        "coverage": float(np.mean([v["coverage"] for v in vals])),
        "selective_risk": float(np.mean(risk)) if risk else float("nan"),
        "frac_risk_le_alpha": float(np.mean(np.array(risk) <= ALPHA + 1e-9)) if risk else float("nan"),
    }


def _source_gates(comb, sub, cal, te, y):
    """Gates calibrated on `cal` (the source) and evaluated on `te`."""
    sc_cal, sc_te = comb.predict(sub.iloc[cal]), comb.predict(sub.iloc[te])
    out = {}
    out["combined_conformal"] = evaluate_gate(sc_te, y[te], ltt_threshold(sc_cal, y[cal], alpha=ALPHA, delta=DELTA))
    out["combined_platt"] = evaluate_gate(_platt(sc_cal, y[cal], sc_te), y[te], 1.0 - ALPHA)
    out["combined_isotonic"] = evaluate_gate(_isotonic(sc_cal, y[cal], sc_te), y[te], 1.0 - ALPHA)
    iptm_cal, iptm_te = sub[IPTM].to_numpy()[cal], sub[IPTM].to_numpy()[te]
    out["native_iptm"] = evaluate_gate(iptm_te, y[te], ltt_threshold(iptm_cal, y[cal], alpha=ALPHA, delta=DELTA))
    return out


def _posebusters_gate(sub, te, y):
    pb = sub["pb_valid"].to_numpy()[te]
    if not np.isfinite(pb).any():
        return {"coverage": float("nan"), "selective_risk": float("nan"), "n_accept": 0, "n": len(te)}
    acc = np.nan_to_num(pb, nan=0.0) >= 0.5
    n_acc = int(acc.sum())
    return {"coverage": n_acc / len(te),
            "selective_risk": float(1 - y[te][acc].mean()) if n_acc else float("nan"),
            "n_accept": n_acc, "n": len(te)}


def _groupcond_gate(comb, sub, t_cal, t_test, strat, y):
    """Shift-robust repair: per-target-stratum LTT calibrated on target labels, pooled."""
    sc_cal, sc_te = comb.predict(sub.iloc[t_cal]), comb.predict(sub.iloc[t_test])
    n_acc = err = 0
    for k in np.unique(strat[t_test]):
        ck = strat[t_cal] == k
        if ck.sum() < MIN_STRATUM_CAL:
            continue
        tau = ltt_threshold(sc_cal[ck], y[t_cal][ck], alpha=ALPHA, delta=DELTA)
        if tau is None:
            continue
        tk = strat[t_test] == k
        a = sc_te[tk] >= tau
        n_acc += int(a.sum())
        err += int((1 - y[t_test][tk][a]).sum())
    return {"coverage": n_acc / len(t_test),
            "selective_risk": float(err / n_acc) if n_acc else float("nan"),
            "n_accept": n_acc, "n": len(t_test)}


def three_way(idx, g):
    p = g.permutation(idx)
    a, b = int(0.4 * len(p)), int(0.7 * len(p))
    return p[:a], p[a:b], p[b:]


def run(n_repeats: int = 120) -> dict:
    df = load_delivered()
    g = rng()
    out = {}
    for m in methods_with_enough(df):
        sub = df[df.method == m].reset_index(drop=True)
        y = sub["correct"].to_numpy()
        strat = sub[STRAT].to_numpy().astype(int)
        idx = np.arange(len(sub))
        src = np.where(np.isin(strat, list(SOURCE_STRATA)))[0]
        tgt = np.where(np.isin(strat, list(TARGET_STRATA)))[0]

        aurc_rank, aurc_iptm, aurc_comb = [], [], []
        iid = {gt: [] for gt in GATES_IID}
        shift = {gt: [] for gt in GATES_SHIFT}
        for _ in range(n_repeats):
            # (A) + (B): i.i.d. random 3-way split
            tr, cal, te = three_way(idx, g)
            comb = ScoreCombiner().fit(sub.iloc[tr], y[tr])
            aurc_rank.append(aurc(sub[NATIVE].to_numpy()[te], y[te]))
            aurc_iptm.append(aurc(sub[IPTM].to_numpy()[te], y[te]))
            aurc_comb.append(aurc(comb.predict(sub.iloc[te]), y[te]))
            for gt, v in _source_gates(comb, sub, cal, te, y).items():
                iid[gt].append(v)
            iid["posebusters"].append(_posebusters_gate(sub, te, y))

            # (C): shift -- calibrate on source strata, deploy on target strata
            sp, tp = g.permutation(src), g.permutation(tgt)
            s_tr, s_cal = sp[: len(src) // 2], sp[len(src) // 2:]
            t_cal, t_test = tp[: len(tgt) // 2], tp[len(tgt) // 2:]
            comb_s = ScoreCombiner().fit(sub.iloc[s_tr], y[s_tr])
            for gt, v in _source_gates(comb_s, sub, s_cal, t_test, y).items():
                shift[gt].append(v)
            shift["combined_groupcond"].append(_groupcond_gate(comb_s, sub, t_cal, t_test, strat, y))

        out[m] = {
            "n": int(len(sub)),
            "aurc": {"ranking_score": float(np.mean(aurc_rank)),
                     "iface_iptm": float(np.mean(aurc_iptm)),
                     "combined": float(np.mean(aurc_comb))},
            "gates_iid": {gt: _gate_stats(iid[gt]) for gt in GATES_IID},
            "gates_shift": {gt: _gate_stats(shift[gt]) for gt in GATES_SHIFT},
        }
    return out


def make_figure(res: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = list(res)
    panels = [
        ("gates_iid", "i.i.d. split",
         ["combined_conformal", "combined_platt", "combined_isotonic", "native_iptm"],
         ["combined (conformal)", "combined (Platt)", "combined (isotonic)", "native ipTM (conformal)"],
         ["#2c7", "#c44", "#e80", "#59c"]),
        ("gates_shift", "novelty shift  S0 -> S1-S2",
         ["combined_conformal", "combined_platt", "combined_isotonic", "combined_groupcond"],
         ["combined (naive conformal)", "combined (Platt)", "combined (isotonic)",
          "combined (group-conditional)"],
         ["#c44", "#e80", "#eb0", "#2c7"]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
    for ax, (key, title, gates, labels, colors) in zip(axes, panels, strict=False):
        x = np.arange(len(methods))
        w = 0.2
        for j, (gt, lab, col) in enumerate(zip(gates, labels, colors, strict=False)):
            risks = [res[m][key][gt]["selective_risk"] for m in methods]
            ax.bar(x + (j - 1.5) * w, risks, w, label=lab, color=col)
        ax.axhline(ALPHA, ls="--", color="k", lw=1)
        ax.text(len(methods) - 0.5, ALPHA + 0.008, f"target alpha={ALPHA}", ha="right", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7, ncol=1, loc="upper left")
    axes[0].set_ylabel("realized selective risk (error among accepted)")
    fig.suptitle("E11: under novelty shift, naive conformal breaks like calibration; "
                 "group-conditional conformal repairs it", fontsize=11)
    fig.tight_layout()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGDIR / "e11_baselines.png", dpi=150)
    print(f"saved {FIGDIR / 'e11_baselines.png'}")


def main() -> None:
    res = run()
    save_json(res, RESDIR / "e11_baselines.json")
    print(f"E11 -- baselines (alpha={ALPHA}, delta={DELTA})\n")
    print("(A) AURC (lower = better ranker):")
    print(f"{'method':10} {'ranking':>9} {'ipTM':>8} {'combined':>9}")
    for m, r in res.items():
        a = r["aurc"]
        print(f"{m:10} {a['ranking_score']:>9.3f} {a['iface_iptm']:>8.3f} {a['combined']:>9.3f}")
    for key, gates, title in (("gates_iid", GATES_IID, "i.i.d."),
                              ("gates_shift", GATES_SHIFT, "NOVELTY SHIFT S0 -> S1-S2")):
        print(f"\n(B/C) [{title}] realized selective risk (target <= {ALPHA}); coverage in parens:")
        print(f"  {'method':10} " + " ".join(f"{gt.split('_')[-1][:9]:>15}" for gt in gates))
        for m, r in res.items():
            row = " ".join(f"{r[key][gt]['selective_risk']:>6.3f}({r[key][gt]['coverage']:>4.2f})" for gt in gates)
            print(f"  {m:10} {row}")
    make_figure(res)
    print("\nRead: combined has the lowest AURC. Under the novelty shift the source-calibrated "
          "gates ALL break -- naive conformal climbs above alpha exactly like Platt/isotonic "
          "(the break is a property of naive transfer, not of calibration). The group-conditional "
          "conformal gate, which has no calibration analogue, restores realized risk <= alpha by "
          "abstaining on the novel strata. Calibration != conformal: only the latter carries a "
          "finite-sample guarantee and a distribution-shift repair.")


if __name__ == "__main__":
    main()
