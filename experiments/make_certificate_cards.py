"""Certificate cards (addition III.10): the shipped deliverable.

One standardized card per (model, ligand novelty stratum): alpha, delta, calibration n,
accepted n, realized risk, exact Clopper-Pearson risk upper bound, retained coverage,
reliability drift D, recommended repair, and a FEASIBLE / ABSTAIN verdict. This turns
the paper from an audit into a deliverable a practitioner reads before trusting a pose.

Sources (no recomputation): results/d2_feasibility_map.json (per (model,axis,stratum,
coverage) cells with n_accepted_targets, R_Q, R_Q_ci, feasible, certified, tau) and
results/e12_reliability_drift.json (per (model,axis,stratum) reliability drift D).

Output: results/certificate_cards.json  and  results/figures/certificate_cards.png
"""

from __future__ import annotations

import json

from experiments._common import ALPHA, DELTA, FIGDIR, RESDIR, save_json

ALPHA_KEY = "0.2"
AXIS = "ligand"
STRATA = ["0", "1", "2", "3", "4"]
SMALL_DRIFT = 0.05
MIN_USABLE_COV = 0.10


def _best_operating_point(strat_cells: dict):
    """Largest-coverage cell that is feasible AND certified at alpha; else the largest
    feasible cell; else None. Returns the chosen cell dict + the certified flag."""
    cells = strat_cells.get("cells", {})
    items = sorted(((float(c), v) for c, v in cells.items()), key=lambda t: t[0])
    best_cert = None
    best_feas = None
    for _c, v in items:
        if v.get("n_accepted_targets", 0) < 20:
            continue
        if v.get("feasible"):
            best_feas = v
            if v.get("certified"):
                best_cert = v
    return (best_cert, True) if best_cert is not None else (best_feas, False)


def _drift_for(drift, model, stratum):
    try:
        cell = drift[model][AXIS].get(stratum)
        return None if cell is None else cell.get("D_signed")
    except (KeyError, TypeError):
        return None


def _recommend(verdict, D):
    if verdict == "ABSTAIN":
        return "abstain (no certified operating point)"
    if D is None:
        return "group-conditional calibration"
    if abs(D) < SMALL_DRIFT:
        return "weighted conformal admissible (covariate-dominated)"
    return "group-conditional calibration (concept drift)"


def build_cards():
    d2 = json.load(open(RESDIR / "d2_feasibility_map.json"))
    drift = json.load(open(RESDIR / "e12_reliability_drift.json"))
    axis = d2["axes"][AXIS]["models"]
    cards = {}
    for model, mdata in axis.items():
        adata = mdata.get("alpha", {}).get(ALPHA_KEY, {})
        n_cal = mdata.get("n_source_cal")
        strata = adata.get("strata", {})
        cards[model] = {}
        for k in STRATA:
            sc = strata.get(k)
            D = _drift_for(drift, model, k)
            if sc is None:
                cards[model][k] = {"verdict": "ABSTAIN", "reason": "stratum absent",
                                   "drift_D": D, "recommended_repair": "abstain"}
                continue
            cell, certified = _best_operating_point(sc)
            if cell is None or cell.get("coverage", 0) < MIN_USABLE_COV or not certified:
                verdict = "ABSTAIN"
                card = {
                    "alpha": ALPHA, "delta": DELTA,
                    "calibration_n_targets": n_cal,
                    "verdict": verdict,
                    "accepted_n_targets": None if cell is None else cell.get("n_accepted_targets"),
                    "retained_coverage": None if cell is None else round(cell.get("coverage", 0), 3),
                    "realized_risk": None if cell is None else round(cell.get("R_Q", 0), 3),
                    "risk_cp_upper90": None if cell is None else round(cell.get("R_Q_ci", [0, 1])[1], 3),
                    "certified": bool(certified),
                    "drift_D": None if D is None else round(D, 3),
                }
            else:
                verdict = "FEASIBLE"
                card = {
                    "alpha": ALPHA, "delta": DELTA,
                    "calibration_n_targets": n_cal,
                    "verdict": verdict,
                    "accepted_n_targets": cell["n_accepted_targets"],
                    "retained_coverage": round(cell["coverage"], 3),
                    "realized_risk": round(cell["R_Q"], 3),
                    "risk_cp_upper90": round(cell["R_Q_ci"][1], 3),
                    "certified": True,
                    "drift_D": None if D is None else round(D, 3),
                }
            card["recommended_repair"] = _recommend(verdict, D)
            cards[model][k] = card
    meta = {"alpha": ALPHA, "delta": DELTA, "axis": AXIS,
            "note": "One card per (model, ligand novelty stratum). FEASIBLE = a certified "
                    "operating point at retained coverage >= 0.10; ABSTAIN otherwise."}
    return {"meta": meta, "cards": cards}


def render(cards_obj):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cards = cards_obj["cards"]
    models = list(cards.keys())
    fig, axes = plt.subplots(len(models), len(STRATA),
                             figsize=(1.85 * len(STRATA), 1.35 * len(models)))
    for i, m in enumerate(models):
        for j, k in enumerate(STRATA):
            ax = axes[i][j]
            ax.axis("off")
            c = cards[m][k]
            feasible = c["verdict"] == "FEASIBLE"
            fc = "#e4f3e4" if feasible else "#f6e4e4"
            ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                       facecolor=fc, edgecolor="#999", lw=0.8))
            head = f"{m}  S{k}"
            if feasible:
                body = (f"{c['verdict']}\ncov {c['retained_coverage']:.2f} "
                        f"(n={c['accepted_n_targets']})\nrisk {c['realized_risk']:.2f} "
                        f"(UB {c['risk_cp_upper90']:.2f})\nD={c['drift_D']}")
            else:
                dd = c.get("drift_D")
                body = f"{c['verdict']}\nno certified\noperating point\nD={dd}"
            ax.text(0.5, 0.80, head, ha="center", va="top", fontsize=7.2,
                    fontweight="bold", transform=ax.transAxes)
            ax.text(0.5, 0.58, body, ha="center", va="top", fontsize=6.0,
                    transform=ax.transAxes, linespacing=1.25)
    fig.suptitle("Certificate cards (ligand axis, α=0.20): FEASIBLE = certified operating point; "
                 "ABSTAIN = the certified action", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIGDIR / f"certificate_cards.{ext}", dpi=170, bbox_inches="tight")
    print(f"saved {FIGDIR/'certificate_cards.png'}")


def main():
    obj = build_cards()
    save_json(obj, RESDIR / "certificate_cards.json")
    render(obj)
    # print a summary: feasible/abstain per model
    for m, cc in obj["cards"].items():
        verdicts = [cc[k]["verdict"][0] for k in STRATA]  # F / A
        print(f"  {m:>9}: " + " ".join(f"S{k}:{cc[k]['verdict'][:4]}" for k in STRATA))


if __name__ == "__main__":
    main()
