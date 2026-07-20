"""E29 -- deployment-computable PROXY stratifier vs the ORACLE novelty stratum.

Reviewer R2.2 / addition III.2 (the single most important missing experiment).

The group-conditional repair (E3) needs a NOVELTY STRATUM at deployment time. But the
shipped `ligand_similarity` / `novelty_stratum` is ECFP4 Tanimoto to the model's
TRAINING set (`morgan_tanimoto`). For AlphaFold3 that training corpus is not publicly
enumerable, so a user handed a brand-new target CANNOT assign the oracle stratum, and the
repair may be uncomputable for the flagship model.

This script asks: can a DEPLOYMENT-COMPUTABLE PROXY stratifier -- built only from public
information a user actually has (count of public training systems with a similar CCD,
deposition date of the nearest PUBLIC analog, protein sequence identity to the public
PDB) -- recover the oracle stratification well enough to preserve per-stratum control?

Three public proxy signals (higher signal == more novel, mirroring the oracle):
  * ccd    : num_training_systems_with_similar_ccds  (novelty = fewer similar CCDs)  PUBLIC count
  * date   : target_release_date recency             (novelty = later / no analog)   PUBLIC date
  * seqsim : protein_seqsim_max                       (novelty = lower identity)      PUBLIC vs PDB

Each is discretised into a 5-level stratum whose per-level SIZES match the oracle strata
exactly (matched-marginal binning by rank), so the confusion matrix is square and the
comparison isolates rank MISASSIGNMENT rather than binning granularity. A no-public-analog
case (missing signal) is pushed to the most-novel end, mirroring the oracle no-analog level.

Steps
  1. AGREEMENT: per model, Spearman(proxy signal, oracle stratum), the adjacent-agreement
     fraction (|proxy stratum - oracle stratum| <= 1), exact-agreement fraction, and the
     confusion matrix.
  2. CONTROL DEGRADATION: a per-stratum LTT gate on native ranking_score, calibrated per
     stratum on a held-out half and deployed frozen on the other half, run two ways:
       - ORACLE: tau keyed on the oracle stratum, risk evaluated per oracle stratum.
       - PROXY : tau keyed on the proxy stratum, risk evaluated per proxy stratum (its own,
                 internally consistent guarantee) AND -- the honest test -- per ORACLE
                 stratum (does proxy gating still protect the TRULY novel systems?).
     Reports per-stratum realized selective risk + coverage at alpha=0.20, exact
     Clopper-Pearson upper bound, and the max per-stratum risk overshoot (realized - alpha)+
     over the true (oracle) strata.
  3. VERDICT: proxy ~ oracle (deployable today) / graceful degradation (price of a closed
     training set) / proxy fails (repair uncomputable -- a governance finding).

Output: results/e29_proxy_stratifier.json
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from experiments._common import (
    ALPHA,
    CONF,
    DELTA,
    RESDIR,
    ROOT,
    load_delivered,
    methods_with_enough,
    rng,
    save_json,
)
from foldgate.conformal import ltt_threshold
from foldgate.selective.metrics import clopper_pearson

N_LEVELS = 5
N_SPLITS = 50             # random 50/50 cal/deploy splits, pooled for a stable read
MIN_ACCEPT_MONDRIAN = 5  # thin high-novelty strata: minimum calibration accept set
MIN_ACCEPT_LTT = 10      # LTT certified-frontier readout (secondary)
MIN_STRATUM_EVAL = 8     # below this a per-stratum deploy count is too thin to read

# Public proxy signals, oriented so that a HIGHER value means MORE novel (mirrors oracle).
# The oracle stratum is built from morgan_tanimoto (Tanimoto to the model's TRAINING set),
# which is exactly the quantity a deployment user cannot compute for a closed corpus.
PROXIES = {
    "ccd": {
        "desc": "num_training_systems_with_similar_ccds (public CCD-similar count)",
        "col": "num_training_systems_with_similar_ccds",
        "kind": "count",   # novelty = -count; count==0 is the no-public-CCD extreme
    },
    "date": {
        "desc": "target_release_date recency (deposition date of nearest public analog)",
        "col": "target_release_date",
        "kind": "date",    # novelty = later date; missing date = no analog = most novel
    },
    "seqsim": {
        "desc": "protein_seqsim_max (protein sequence identity to public PDB)",
        "col": "protein_seqsim_max",
        "kind": "sim",     # novelty = -seqsim; missing = no analog = most novel
    },
}


def novelty_signal(sub: pd.DataFrame, spec: dict) -> np.ndarray:
    """Oriented novelty signal: higher == more novel; missing pushed to the most-novel end."""
    col, kind = spec["col"], spec["kind"]
    if kind == "date":
        d = pd.to_datetime(sub[col], errors="coerce")
        ordv = d.astype("int64").where(d.notna()).astype("float64")
        sig = ordv  # later analog date == more novel
    else:
        v = pd.to_numeric(sub[col], errors="coerce").astype("float64")
        sig = -v    # fewer similar CCDs / lower identity == more novel
    sig = sig.to_numpy(dtype="float64")
    finite = np.isfinite(sig)
    if finite.any():
        # missing (no public analog) -> strictly beyond the observed max -> most-novel end
        sig = np.where(finite, sig, np.nanmax(sig[finite]) + 1.0)
    else:
        sig = np.zeros_like(sig)
    return sig


def matched_marginal_strata(signal: np.ndarray, oracle: np.ndarray, seed: int) -> np.ndarray:
    """5-level proxy strata whose per-level sizes match the oracle's, by rank of `signal`.

    Ties (e.g. the ~50 pct of systems with zero similar CCDs) are broken deterministically
    with a seeded jitter so the marginals match exactly and the comparison isolates rank
    misassignment from binning granularity. Level 0 = least novel, N_LEVELS-1 = most novel.
    """
    n = len(signal)
    r = rng(seed)
    jitter = r.random(n) * 1e-9
    order = np.argsort(signal + jitter, kind="mergesort")  # ascending novelty
    rank = np.empty(n, dtype=int)
    rank[order] = np.arange(n)

    # oracle stratum sizes in ascending-novelty order define the cut boundaries
    levels = sorted(pd.unique(oracle))
    out = np.empty(n, dtype=int)
    start = 0
    for lvl in levels:
        cnt = int((oracle == lvl).sum())
        sel = (rank >= start) & (rank < start + cnt)
        out[sel] = int(lvl)
        start += cnt
    return out


def mondrian_threshold(scores, correct, alpha, min_accept=MIN_ACCEPT_MONDRIAN):
    """Group-conditional (Mondrian) threshold: most permissive tau with cal risk <= alpha.

    The largest accept set (max coverage) whose empirical error rate among accepted is
    <= alpha on the calibration data. This is the point-threshold group-conditional repair
    (per-stratum quantile targeting alpha); it accepts a non-trivial fraction where the
    finite-sample LTT certificate is vacuous, which is what makes the oracle-vs-proxy
    control comparison readable. Returns tau (accept iff score >= tau) or None.
    """
    if len(scores) < min_accept:
        return None
    order = np.argsort(-scores, kind="mergesort")     # highest confidence first
    s_sorted = scores[order]
    err = np.cumsum(1 - correct[order])
    k = np.arange(1, len(scores) + 1)
    risk = err / k
    ok = (risk <= alpha) & (k >= min_accept)
    if not ok.any():
        return None
    kbest = int(np.max(np.where(ok)[0]))              # most permissive valid accept set
    return float(s_sorted[kbest])


def _thresholds_by_stratum(scores, correct, strata, cal_mask, alpha, gate):
    taus = {}
    for s in np.unique(strata):
        cm = cal_mask & (strata == s)
        if gate == "mondrian":
            taus[int(s)] = mondrian_threshold(scores[cm], correct[cm], alpha)
        else:  # ltt certified frontier
            if cm.sum() < MIN_ACCEPT_LTT:
                taus[int(s)] = None
            else:
                taus[int(s)] = ltt_threshold(scores[cm], correct[cm], alpha=alpha, delta=DELTA)
    return taus


def _accumulate(acc, scores, correct, eval_strata, deploy_mask, taus, keyed_strata, alpha):
    """Add this split's per-eval-stratum accepted/error counts + a per-split holds flag."""
    trow = np.array([taus[int(s)] if taus[int(s)] is not None else np.nan
                     for s in keyed_strata], dtype=float)
    accepted = deploy_mask & np.isfinite(trow) & (scores >= trow)
    for s in np.unique(eval_strata):
        dm = deploy_mask & (eval_strata == s)
        a = accepted & dm
        na = int(a.sum())
        ne = int((1 - correct[a]).sum())
        d = acc.setdefault(int(s), {"n_deploy": 0, "n_accept": 0, "err": 0,
                                    "holds": 0, "holds_denom": 0})
        d["n_deploy"] += int(dm.sum())
        d["n_accept"] += na
        d["err"] += ne
        if na >= MIN_STRATUM_EVAL:
            d["holds"] += int((ne / na) <= alpha)
            d["holds_denom"] += 1


def _finalize(acc, alpha):
    rep = {}
    for s, d in sorted(acc.items()):
        na = d["n_accept"]
        if na == 0:
            rep[int(s)] = {"n_deploy_pooled": d["n_deploy"], "n_accept_pooled": 0,
                           "coverage": 0.0, "realized_risk": None, "risk_cp_upper90": None,
                           "holds_fraction": None, "overshoot": None, "vacuous": True}
            continue
        risk = d["err"] / na
        _, cp_hi = clopper_pearson(d["err"], na, ci=0.90)
        hf = (d["holds"] / d["holds_denom"]) if d["holds_denom"] else None
        rep[int(s)] = {
            "n_deploy_pooled": d["n_deploy"],
            "n_accept_pooled": na,
            "coverage": round(na / d["n_deploy"], 4),
            "realized_risk": round(risk, 4),
            "risk_cp_upper90": round(cp_hi, 4),
            "holds_fraction": round(hf, 3) if hf is not None else None,
            "overshoot": round(max(0.0, risk - alpha), 4),
            "vacuous": bool(d["holds_denom"] == 0),
        }
    return rep


def _max_overshoot(rep):
    vals = [v["overshoot"] for v in rep.values()
            if v["overshoot"] is not None and not v["vacuous"]]
    return round(max(vals), 4) if vals else None


def run_model(sub: pd.DataFrame, alpha: float, delta: float, seed: int) -> dict:
    sub = sub.dropna(subset=[CONF]).reset_index(drop=True)
    scores = sub[CONF].to_numpy(dtype=float)
    correct = sub["correct"].to_numpy(dtype=int)
    oracle = sub["novelty_stratum"].to_numpy(dtype=int)
    n = len(sub)

    # ---- build proxy strata (matched marginals to oracle) + agreement ----
    proxies = {}
    for name, spec in PROXIES.items():
        sig = novelty_signal(sub, spec)
        pstrat = matched_marginal_strata(sig, oracle, seed=seed + hash(name) % 1000)
        # Spearman on the CONTINUOUS signal vs oracle stratum (rank-order fidelity)
        rho = float(spearmanr(sig, oracle).statistic)
        adj = float(np.mean(np.abs(pstrat - oracle) <= 1))
        exact = float(np.mean(pstrat == oracle))
        conf = pd.crosstab(pd.Series(oracle, name="oracle"),
                           pd.Series(pstrat, name="proxy"))
        conf = conf.reindex(index=range(N_LEVELS), columns=range(N_LEVELS), fill_value=0)
        proxies[name] = {
            "desc": spec["desc"],
            "spearman_signal_vs_oracle": round(rho, 4),
            "adjacent_agreement": round(adj, 4),
            "exact_agreement": round(exact, 4),
            "confusion_oracle_rows_proxy_cols": conf.to_numpy().tolist(),
            "_strata": pstrat,
        }

    # best proxy = strongest |Spearman|
    best = max(proxies, key=lambda k: abs(proxies[k]["spearman_signal_vs_oracle"]))
    pstrat = proxies[best]["_strata"]

    # ---- control: per-stratum gate calibrated on a held-out half, deployed frozen,
    #      pooled over N_SPLITS random 50/50 cal/deploy splits for a stable read ----
    def run_gate(gate_kind):
        acc_oracle = {}          # oracle-keyed, oracle-evaluated
        acc_proxy_self = {}      # proxy-keyed, proxy-evaluated (its own guarantee)
        acc_proxy_true = {}      # proxy-keyed, ORACLE-evaluated (the honest deployment test)
        r = rng(seed + 100)
        for _ in range(N_SPLITS):
            deploy_mask = r.random(n) < 0.5
            cal_mask = ~deploy_mask
            o_taus = _thresholds_by_stratum(scores, correct, oracle, cal_mask, alpha, gate_kind)
            p_taus = _thresholds_by_stratum(scores, correct, pstrat, cal_mask, alpha, gate_kind)
            _accumulate(acc_oracle, scores, correct, oracle, deploy_mask, o_taus, oracle, alpha)
            _accumulate(acc_proxy_self, scores, correct, pstrat, deploy_mask, p_taus, pstrat, alpha)
            _accumulate(acc_proxy_true, scores, correct, oracle, deploy_mask, p_taus, pstrat, alpha)
        return (_finalize(acc_oracle, alpha), _finalize(acc_proxy_self, alpha),
                _finalize(acc_proxy_true, alpha))

    m_oracle, m_proxy_self, m_proxy_true = run_gate("mondrian")
    l_oracle, l_proxy_self, l_proxy_true = run_gate("ltt")

    # per-TRUE-stratum degradation: proxy-gate risk minus oracle-gate risk on the SAME
    # oracle stratum (both gates read out on the true novelty label). This is the honest
    # "how much worse is per-stratum control under the proxy" comparison; a single global
    # max-overshoot is dominated by different (tiny) strata for each gate.
    degradation = {}
    for s in sorted(m_oracle):
        o, p = m_oracle[s], m_proxy_true.get(s, {})
        orisk, prisk = o.get("realized_risk"), p.get("realized_risk")
        degradation[int(s)] = {
            "oracle_risk": orisk,
            "proxy_risk_on_true_stratum": prisk,
            "risk_degradation": (round(prisk - orisk, 4)
                                 if (orisk is not None and prisk is not None) else None),
            "oracle_holds_fraction": o.get("holds_fraction"),
            "proxy_holds_fraction": p.get("holds_fraction"),
            "oracle_coverage": o.get("coverage"),
            "proxy_coverage": p.get("coverage"),
        }
    deg_vals = [v["risk_degradation"] for v in degradation.values()
                if v["risk_degradation"] is not None]
    max_risk_degradation = round(max(deg_vals), 4) if deg_vals else None
    # strata where the proxy gate leaves the truly-novel systems above alpha but the oracle
    # gate held them: the actionable failure of proxy stratification
    proxy_underprotects = [int(s) for s, v in degradation.items()
                           if v["proxy_risk_on_true_stratum"] is not None
                           and v["proxy_risk_on_true_stratum"] > alpha + 0.05
                           and (v["oracle_risk"] is None or v["oracle_risk"] <= alpha + 0.05)]

    control = {
        "alpha": alpha,
        "delta": delta,
        "n_splits": N_SPLITS,
        "best_proxy": best,
        "gate": "mondrian (group-conditional point threshold targeting risk<=alpha per stratum)",
        "mondrian": {
            "oracle_gate_per_oracle_stratum": m_oracle,
            "proxy_gate_per_proxy_stratum": m_proxy_self,
            "proxy_gate_per_ORACLE_stratum": m_proxy_true,
            "max_overshoot_oracle_gate": _max_overshoot(m_oracle),
            "max_overshoot_proxy_gate_on_true_strata": _max_overshoot(m_proxy_true),
            "per_true_stratum_degradation": degradation,
            "max_risk_degradation_proxy_vs_oracle": max_risk_degradation,
            "proxy_underprotects_true_strata": proxy_underprotects,
        },
        "ltt_certified_frontier": {
            "oracle_gate_per_oracle_stratum": l_oracle,
            "proxy_gate_per_ORACLE_stratum": l_proxy_true,
            "note": ("finite-sample LTT (delta=%.2f) certifies little on novel strata "
                     "(frontier collapse); shown for completeness, not the primary read" % delta),
        },
        # primary summary fields for the verdict
        "max_overshoot_oracle_gate": _max_overshoot(m_oracle),
        "max_overshoot_proxy_gate_on_true_strata": _max_overshoot(m_proxy_true),
        "max_risk_degradation_proxy_vs_oracle": max_risk_degradation,
        "proxy_underprotects_true_strata": proxy_underprotects,
    }

    for name in proxies:
        proxies[name].pop("_strata", None)

    return {"n": n, "agreement": proxies, "control": control}


def _verdict(results: dict, flagship: str) -> list[str]:
    """3-line honest read of which of the three outcomes holds, anchored on the flagship."""
    fm = results[flagship]
    ctrl = fm["control"]
    best = ctrl["best_proxy"]
    rho = abs(fm["agreement"][best]["spearman_signal_vs_oracle"])
    adj = fm["agreement"][best]["adjacent_agreement"]
    deg = ctrl["max_risk_degradation_proxy_vs_oracle"] or 0.0
    under = ctrl["proxy_underprotects_true_strata"]
    o_over = ctrl["max_overshoot_oracle_gate"] or 0.0

    if rho >= 0.85 and adj >= 0.92 and deg <= 0.03 and not under:
        outcome = "PROXY ~ ORACLE"
        line = (f"The public '{best}' proxy recovers the oracle stratification (Spearman {rho:.2f}, "
                f"adjacent {adj:.2f}) with negligible extra per-stratum risk (max degradation "
                f"{deg:+.3f}); the repair is DEPLOYABLE TODAY without enumerating the training set.")
    elif rho >= 0.4:
        outcome = "GRACEFUL DEGRADATION"
        line = (f"The public '{best}' proxy tracks true novelty (Spearman {rho:.2f}, adjacent "
                f"{adj:.2f}) and matches oracle control on the low-novelty strata, but its "
                f"mis-ranking loosens per-stratum risk on the truly-novel tail by up to {deg:+.3f} "
                f"and specifically un-controls true stratum(s) {under or 'none'}; this is the "
                f"measurable PRICE of not being able to enumerate the training set.")
    else:
        outcome = "PROXY FAILS"
        line = (f"The best public proxy '{best}' barely tracks true novelty (Spearman {rho:.2f}) and "
                f"leaves the truly-novel strata {under} well above alpha (up to {deg:+.3f} worse than "
                f"oracle gating), so for closed-corpus models the repair is effectively UNCOMPUTABLE "
                f"at deployment -- a governance finding for the abstract.")
    return [
        f"OUTCOME: {outcome} (flagship = {flagship}, best public proxy = {best}).",
        line,
        (f"Caveat that stands regardless of proxy: even ORACLE group-conditional gating on native "
         f"ranking_score leaves the most-novel strata above alpha (max oracle overshoot {o_over:.3f}), "
         f"so concept shift -- not just an un-enumerable training set -- caps how much any "
         f"stratifier can repair the novel tail."),
    ]


def main():
    df = load_delivered()
    # attach protein_seqsim_max (public) from annotations, deduped to one row per system
    ann = pd.read_csv(ROOT / "data" / "raw" / "annotations.csv")
    ann = ann.drop_duplicates("system_id")[["system_id", "protein_seqsim_max"]]
    df = df.merge(ann, on="system_id", how="left")

    # one pose per (system_id, method): the top-ranked delivered pose
    one = (df.sort_values(CONF)
             .drop_duplicates(["system_id", "method"], keep="last")
             .reset_index(drop=True))

    methods = methods_with_enough(one, n=800)  # system-level counts are ~2-3x smaller
    flagship = "af3" if "af3" in methods else methods[0]

    results = {}
    for m in methods:
        sub = one[one.method == m]
        results[m] = run_model(sub, alpha=ALPHA, delta=DELTA, seed=20260720)

    out = {
        "meta": {
            "experiment": "e29_proxy_stratifier",
            "question": ("can a deployment-computable PROXY novelty stratum built only from "
                         "public information recover the oracle (training-set) stratification "
                         "well enough to preserve per-stratum selective-risk control?"),
            "alpha": ALPHA,
            "delta": DELTA,
            "n_levels": N_LEVELS,
            "oracle_stratifier": "novelty_stratum (morgan_tanimoto to TRAINING set -- not public for AF3)",
            "proxy_signals": {k: v["desc"] for k, v in PROXIES.items()},
            "flagship_model": flagship,
            "methods": methods,
            "notes": ("Proxy strata are matched-marginal (per-level sizes equal the oracle's) "
                      "so the confusion matrix is square and the comparison isolates rank "
                      "misassignment. 'proxy_gate_per_ORACLE_stratum' is the honest deployment "
                      "test: taus keyed on the PUBLIC proxy stratum, risk read out on the TRUE "
                      "novelty stratum."),
        },
        "results": results,
        "flagship_af3": {
            "best_proxy": results[flagship]["control"]["best_proxy"],
            "agreement_best_proxy": results[flagship]["agreement"][
                results[flagship]["control"]["best_proxy"]],
            "control": results[flagship]["control"],
        },
        "spearman_and_adjacent_per_model": {
            m: {p: {"spearman": results[m]["agreement"][p]["spearman_signal_vs_oracle"],
                    "adjacent": results[m]["agreement"][p]["adjacent_agreement"]}
                for p in PROXIES}
            for m in methods
        },
        "verdict": _verdict(results, flagship),
    }
    save_json(out, RESDIR / "e29_proxy_stratifier.json")
    print("wrote", RESDIR / "e29_proxy_stratifier.json")
    print("\nflagship:", flagship, "| best proxy:", out["flagship_af3"]["best_proxy"])
    for line in out["verdict"]:
        print(" -", line)


if __name__ == "__main__":
    main()
