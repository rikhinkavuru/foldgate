"""E31 -- the extra baselines a reviewer will demand (R1.10 / addition III.7).

E11 already covers raw ranking_score, native ipTM LTT, PoseBusters-pass, and
Platt/isotonic calibration. This script adds the baselines still missing from
that table, evaluated on the SAME target-grouped protocol so nothing gets an
unfair split:

  1. FIXED ipTM THRESHOLD (accept iface_iptm >= 0.8, no calibration) -- the
     literal field baseline, "what practitioners do".
  2. Per-stratum VENN-ABERS on the native ranking score -- inductive Venn-Abers
     (IVAP/ABERS) calibrated separately in each novelty stratum; accept when the
     VA point-probability of correctness >= 1 - alpha. If this matches the GBM
     combiner + group-conditional conformal repair, the GBM is unnecessary.
  3. LOCALIZED conformal keyed on CONTINUOUS novelty -- a randomly-localized
     (RLCP-style, Hore-Barber arXiv:2310.07850) threshold that slides smoothly
     along ligand_similarity instead of a piecewise-constant Mondrian stratum.
     Documented approximation: we reuse foldgate.conformal.localized_threshold,
     which gives EXACT marginal coverage of the score quantile via the RLCP
     randomization but treats the selective-RISK behaviour as an empirically
     validated locally-reweighted plug-in, not a new finite-sample theorem.
  4. ACCEPT-ALL / ABSTAIN-ALL anchors -- accept-all realized risk = the base
     error rate; abstain-all coverage = 0.

Protocol (mirrors E11 and the paper's target-grouped standard):
  * i.i.d.:  GroupKFold(5) on system_id. Each dev pool is split by SYSTEM into a
    combiner-train half and a calibration half; the GBM combiner is fit on
    training targets only; thresholds / Venn-Abers are calibrated on the
    calibration systems; accepts are pooled across the 5 held-out folds.
  * novelty shift:  calibrate on the FAMILIAR strata (S0-S1), deploy on the NOVEL
    strata (S2-S3). Naive-transfer gates (fixed ipTM, naive conformal) see only
    source-calibration labels -- they are expected to break. Repair gates
    (group-conditional conformal, per-stratum Venn-Abers, localized conformal)
    are allowed a labelled target-calibration split, the semi-supervised regime
    in which a repair is even possible.

Headline: does any simpler baseline hold realized risk <= alpha under shift the
way group-conditional conformal does (by abstaining more on novel poses)?

Focus is AF3 for the detailed baseline table; the fixed-ipTM and accept-all
anchors are reported for all five governed models.
"""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold

from experiments._common import (
    ALPHA,
    DELTA,
    RESDIR,
    load_delivered,
    methods_with_enough,
    rng,
    save_json,
)
from foldgate.conformal import ltt_threshold, localized_threshold
from foldgate.scores.combiner import ScoreCombiner, DEFAULT_FEATURES
from foldgate.selective.metrics import clopper_pearson

NATIVE = "ranking_score"   # native score Venn-Abers is calibrated on
IPTM = "iface_iptm"
STRAT = "novelty_stratum"
NOVELTY = "ligand_similarity"   # continuous localizer axis (max Tanimoto to train)
FIXED_IPTM_TAU = 0.80           # the actual field threshold

SOURCE_STRATA = {0, 1}          # familiar -> calibrate here
TARGET_STRATA = {2, 3}          # novel    -> deploy here
DEPLOY_STRATA = [2, 3]          # per-stratum reporting under shift
MIN_STRATUM_CAL = 40            # a stratum thinner than this cannot self-calibrate

N_IID_REPEATS = 6
N_SHIFT_REPEATS = 15
VA_GRID = 40                    # score grid on which Venn-Abers p is evaluated then interpolated


# =============================================================================
# Inductive Venn-Abers (IVAP / ABERS) -- implemented in-house (no venn-abers pkg)
# =============================================================================
# For a test object with score s, the Venn-Abers multiprobability is (p0, p1),
# where p_l is the isotonic-regression fit of the calibration set AUGMENTED with
# (s, l), read at s. p0 <= p1 always. The merged point prediction (Vovk, Petej &
# Fedorova 2015) is  p = p1 / (1 - p0 + p1). We build p on a score grid via two
# isotonic fits per grid point (label 0 and label 1) and interpolate to the test
# scores; p is monotone in s for an isotonic calibrator, so interpolation is safe
# and cheap. A self-check validates the augmented-isotonic value against a direct
# sklearn IsotonicRegression refit.

def _pava(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Pool-adjacent-violators isotonic (non-decreasing) fit; expanded to inputs."""
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    vals: list[float] = []
    wts: list[float] = []
    sizes: list[int] = []
    for val, wt in zip(v, w):
        vals.append(float(val))
        wts.append(float(wt))
        sizes.append(1)
        while len(vals) > 1 and vals[-2] >= vals[-1]:
            nw = wts[-2] + wts[-1]
            nv = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / nw
            vals[-2], wts[-2], sizes[-2] = nv, nw, sizes[-2] + sizes[-1]
            vals.pop(); wts.pop(); sizes.pop()
    out = np.empty(len(v), dtype=float)
    i = 0
    for val, sz in zip(vals, sizes):
        out[i:i + sz] = val
        i += sz
    return out


def _augmented_iso_value(cal_x: np.ndarray, cal_y: np.ndarray, q: float, label: int) -> float:
    """Isotonic fit of {(cal_x, cal_y)} U {(q, label)} evaluated at q."""
    pos = int(np.searchsorted(cal_x, q, side="right"))
    x_aug = np.insert(cal_x, pos, q)
    y_aug = np.insert(cal_y.astype(float), pos, float(label))
    fit = _pava(y_aug, np.ones_like(y_aug))
    return float(fit[pos])


def venn_abers_point_probs(cal_x: np.ndarray, cal_y: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """VA merged point probability p = p1/(1-p0+p1) at each query score."""
    order = np.argsort(cal_x, kind="mergesort")
    cx, cy = np.asarray(cal_x, dtype=float)[order], np.asarray(cal_y, dtype=int)[order]
    out = np.empty(len(queries), dtype=float)
    for j, q in enumerate(queries):
        p0 = _augmented_iso_value(cx, cy, float(q), 0)
        p1 = _augmented_iso_value(cx, cy, float(q), 1)
        denom = 1.0 - p0 + p1
        out[j] = p1 / denom if denom > 0 else p1
    return out


def _va_accept(cal_x: np.ndarray, cal_y: np.ndarray, test_x: np.ndarray, target: float) -> np.ndarray:
    """Accept mask: VA point-prob of correctness >= target, via grid + interpolation."""
    finite = np.isfinite(cal_x)
    cal_x, cal_y = cal_x[finite], cal_y[finite]
    if len(cal_x) < MIN_STRATUM_CAL:
        return np.zeros(len(test_x), dtype=bool)
    lo, hi = float(np.min(cal_x)), float(np.max(cal_x))
    if hi <= lo:
        return np.zeros(len(test_x), dtype=bool)
    grid = np.linspace(lo, hi, VA_GRID)
    p_grid = venn_abers_point_probs(cal_x, cal_y, grid)
    p_test = np.interp(np.clip(test_x, lo, hi), grid, p_grid)
    return p_test >= target


def _va_self_check() -> dict:
    """Validate the augmented-isotonic value against a direct sklearn refit."""
    g = np.random.default_rng(0)
    x = np.sort(g.uniform(0, 1, 120))
    y = (g.uniform(size=120) < (0.2 + 0.6 * x)).astype(int)
    max_err = 0.0
    for q in g.uniform(0, 1, 25):
        for lbl in (0, 1):
            mine = _augmented_iso_value(x, y, float(q), lbl)
            xa = np.insert(x, np.searchsorted(x, q, "right"), q)
            ya = np.insert(y.astype(float), np.searchsorted(x, q, "right"), float(lbl))
            ref = float(IsotonicRegression(out_of_bounds="clip").fit(xa, ya).predict([q])[0])
            max_err = max(max_err, abs(mine - ref))
    return {"max_abs_err_vs_sklearn": max_err, "ok": bool(max_err < 1e-6)}


# =============================================================================
# gate bookkeeping
# =============================================================================
def _gate(accept: np.ndarray, correct: np.ndarray, ci: float = 0.90) -> dict:
    accept = np.asarray(accept, dtype=bool)
    correct = np.asarray(correct, dtype=int)
    n = len(accept)
    n_acc = int(accept.sum())
    risk = float(1.0 - correct[accept].mean()) if n_acc else float("nan")
    cov = n_acc / n if n else float("nan")
    lo, hi = clopper_pearson(n_acc, n, ci) if n else (float("nan"), float("nan"))
    return {"coverage": cov, "coverage_ci": [lo, hi], "selective_risk": risk,
            "n_accept": n_acc, "n": n}


def _pool_gate(masks: list, corrects: list) -> dict:
    a = np.concatenate(masks) if masks else np.zeros(0, dtype=bool)
    c = np.concatenate(corrects) if corrects else np.zeros(0, dtype=int)
    return _gate(a, c)


def _mean_of_gates(gates: list) -> dict:
    """Average pooled per-repeat gate dicts into one summary with a risk spread."""
    cov = np.array([g["coverage"] for g in gates], dtype=float)
    risk = np.array([g["selective_risk"] for g in gates], dtype=float)
    nacc = np.array([g["n_accept"] for g in gates], dtype=float)
    valid = np.isfinite(risk)
    return {
        "coverage": float(np.nanmean(cov)),
        "selective_risk": float(np.nanmean(risk[valid])) if valid.any() else float("nan"),
        "risk_std": float(np.nanstd(risk[valid])) if valid.any() else float("nan"),
        "frac_repeats_risk_le_alpha": float(np.mean(risk[valid] <= ALPHA + 1e-9)) if valid.any() else float("nan"),
        "mean_n_accept": float(np.mean(nacc)),
        "n_repeats": int(len(gates)),
    }


def _split_by_group(idx: np.ndarray, groups: np.ndarray, frac: float, g) -> tuple:
    """Split rows into two halves by unique system_id (no system spans the split)."""
    uniq = np.unique(groups[idx])
    perm = g.permutation(uniq)
    cut = max(1, int(frac * len(perm)))
    left_sys, right_sys = set(perm[:cut].tolist()), set(perm[cut:].tolist())
    left = idx[np.isin(groups[idx], list(left_sys))]
    right = idx[np.isin(groups[idx], list(right_sys))]
    return left, right


# =============================================================================
# i.i.d. evaluation (GroupKFold(5) on system_id, accepts pooled over folds)
# =============================================================================
def _iid_once(sub, y, strat, groups, seed) -> dict:
    g = rng(seed)
    idx = np.arange(len(sub))
    gkf = GroupKFold(n_splits=5)

    per = {k: ([], []) for k in ("combined_conformal", "groupcond", "venn_abers", "localized")}
    # fixed_iptm / accept_all are calibration-free -> evaluated on the whole set once
    for dev, test in gkf.split(idx, y, groups):
        dev = idx[dev]
        te = idx[test]
        ctr, cal = _split_by_group(dev, groups, 0.6, g)
        if len(ctr) < 30 or len(cal) < 30:
            continue
        comb = ScoreCombiner(DEFAULT_FEATURES).fit(sub.iloc[ctr], y[ctr])
        sc_cal, sc_te = comb.predict(sub.iloc[cal]), comb.predict(sub.iloc[te])

        # (1) naive/global conformal on the combiner score
        tau = ltt_threshold(sc_cal, y[cal], alpha=ALPHA, delta=DELTA)
        per["combined_conformal"][0].append(sc_te >= tau if tau is not None else np.zeros(len(te), bool))
        per["combined_conformal"][1].append(y[te])

        # (2) group-conditional conformal (the repair reference): per-stratum LTT
        acc = np.zeros(len(te), dtype=bool)
        for k in np.unique(strat[te]):
            ck = strat[cal] == k
            if ck.sum() < MIN_STRATUM_CAL:
                continue
            tk = ltt_threshold(sc_cal[ck], y[cal][ck], alpha=ALPHA, delta=DELTA)
            if tk is None:
                continue
            m = strat[te] == k
            acc[m] = sc_te[m] >= tk
        per["groupcond"][0].append(acc)
        per["groupcond"][1].append(y[te])

        # (3) per-stratum Venn-Abers on the native score
        acc = np.zeros(len(te), dtype=bool)
        nat_cal, nat_te = sub[NATIVE].to_numpy()[cal], sub[NATIVE].to_numpy()[te]
        for k in np.unique(strat[te]):
            ck, tk = strat[cal] == k, strat[te] == k
            if ck.sum() < MIN_STRATUM_CAL:
                continue
            acc[tk] = _va_accept(nat_cal[ck], y[cal][ck], nat_te[tk], 1.0 - ALPHA)
        per["venn_abers"][0].append(acc)
        per["venn_abers"][1].append(y[te])

        # (4) localized conformal keyed on continuous ligand_similarity
        tau_loc = localized_threshold(
            sc_cal, y[cal], sub[NOVELTY].to_numpy()[cal], sub[NOVELTY].to_numpy()[te],
            alpha=ALPHA, generator=g,
        )
        per["localized"][0].append((sc_te >= tau_loc) & np.isfinite(tau_loc))
        per["localized"][1].append(y[te])

    out = {k: _pool_gate(m, c) for k, (m, c) in per.items()}
    # calibration-free anchors on the full method set
    out["fixed_iptm"] = _gate(sub[IPTM].to_numpy() >= FIXED_IPTM_TAU, y)
    out["accept_all"] = _gate(np.ones(len(sub), bool), y)
    out["abstain_all"] = _gate(np.zeros(len(sub), bool), y)
    return out


# =============================================================================
# novelty-shift evaluation (calibrate on S0-S1, deploy on S2-S3)
# =============================================================================
def _perstratum(accept, correct, strat, deploy) -> dict:
    out = {}
    for k in deploy:
        m = strat == k
        out[int(k)] = _gate(accept[m], correct[m]) if m.any() else {"coverage": float("nan"),
                                                                     "selective_risk": float("nan"),
                                                                     "n_accept": 0, "n": 0}
    return out


def _shift_once(sub, y, strat, groups, seed) -> dict:
    g = rng(seed)
    idx = np.arange(len(sub))
    src = idx[np.isin(strat, list(SOURCE_STRATA))]
    tgt = idx[np.isin(strat, list(TARGET_STRATA))]

    ctr, scal = _split_by_group(src, groups, 0.6, g)      # combiner-train / source-cal
    tcal, ttest = _split_by_group(tgt, groups, 0.5, g)    # target-cal (repair) / target-test

    comb = ScoreCombiner(DEFAULT_FEATURES).fit(sub.iloc[ctr], y[ctr])
    sc_scal = comb.predict(sub.iloc[scal])
    sc_tcal = comb.predict(sub.iloc[tcal])
    sc_tt = comb.predict(sub.iloc[ttest])
    y_tt, strat_tt = y[ttest], strat[ttest]

    res: dict = {}

    # ---- naive transfer regime (only source-calibration labels) ----
    res["fixed_iptm"] = _gate(sub[IPTM].to_numpy()[ttest] >= FIXED_IPTM_TAU, y_tt)
    res["fixed_iptm"]["per_stratum"] = _perstratum(
        sub[IPTM].to_numpy()[ttest] >= FIXED_IPTM_TAU, y_tt, strat_tt, DEPLOY_STRATA)
    tau = ltt_threshold(sc_scal, y[scal], alpha=ALPHA, delta=DELTA)
    acc = sc_tt >= tau if tau is not None else np.zeros(len(ttest), bool)
    res["naive_conformal"] = _gate(acc, y_tt)
    res["naive_conformal"]["per_stratum"] = _perstratum(acc, y_tt, strat_tt, DEPLOY_STRATA)

    # ---- anchors ----
    res["accept_all"] = _gate(np.ones(len(ttest), bool), y_tt)
    res["abstain_all"] = _gate(np.zeros(len(ttest), bool), y_tt)

    # ---- repair regime (target-calibration labels allowed) ----
    # group-conditional conformal (reference repair): per-target-stratum LTT
    acc = np.zeros(len(ttest), bool)
    for k in DEPLOY_STRATA:
        ck = strat[tcal] == k
        if ck.sum() < MIN_STRATUM_CAL:
            continue
        tk = ltt_threshold(sc_tcal[ck], y[tcal][ck], alpha=ALPHA, delta=DELTA)
        if tk is None:
            continue
        m = strat_tt == k
        acc[m] = sc_tt[m] >= tk
    res["groupcond"] = _gate(acc, y_tt)
    res["groupcond"]["per_stratum"] = _perstratum(acc, y_tt, strat_tt, DEPLOY_STRATA)

    # per-stratum Venn-Abers on native score, calibrated on target-cal per stratum
    acc = np.zeros(len(ttest), bool)
    nat_tcal, nat_tt = sub[NATIVE].to_numpy()[tcal], sub[NATIVE].to_numpy()[ttest]
    for k in DEPLOY_STRATA:
        ck, m = strat[tcal] == k, strat_tt == k
        if ck.sum() < MIN_STRATUM_CAL:
            continue
        acc[m] = _va_accept(nat_tcal[ck], y[tcal][ck], nat_tt[m], 1.0 - ALPHA)
    res["venn_abers"] = _gate(acc, y_tt)
    res["venn_abers"]["per_stratum"] = _perstratum(acc, y_tt, strat_tt, DEPLOY_STRATA)

    # localized conformal keyed on continuous novelty; cal pool spans source+target
    cal_pool = np.concatenate([scal, tcal])
    sc_pool = np.concatenate([sc_scal, sc_tcal])
    tau_loc = localized_threshold(
        sc_pool, y[cal_pool], sub[NOVELTY].to_numpy()[cal_pool], sub[NOVELTY].to_numpy()[ttest],
        alpha=ALPHA, generator=g,
    )
    acc = (sc_tt >= tau_loc) & np.isfinite(tau_loc)
    res["localized"] = _gate(acc, y_tt)
    res["localized"]["per_stratum"] = _perstratum(acc, y_tt, strat_tt, DEPLOY_STRATA)
    return res


def _mean_shift(runs: list, gate: str) -> dict:
    gates = [r[gate] for r in runs]
    summ = _mean_of_gates(gates)
    if "per_stratum" in runs[0][gate]:
        ps = {}
        for k in DEPLOY_STRATA:
            gk = [r[gate]["per_stratum"][k] for r in runs]
            ps[int(k)] = _mean_of_gates(gk)
        summ["per_stratum"] = ps
    return summ


# =============================================================================
# driver
# =============================================================================
def run() -> dict:
    df = load_delivered()
    methods = methods_with_enough(df)
    va_chk = _va_self_check()
    if not va_chk["ok"]:
        raise SystemExit(f"Venn-Abers self-check failed: {va_chk}")

    out = {
        "config": {
            "alpha": ALPHA, "delta": DELTA, "fixed_iptm_tau": FIXED_IPTM_TAU,
            "source_strata": sorted(SOURCE_STRATA), "target_strata": sorted(TARGET_STRATA),
            "n_iid_repeats": N_IID_REPEATS, "n_shift_repeats": N_SHIFT_REPEATS,
            "combiner_features": DEFAULT_FEATURES, "native_score": NATIVE,
            "localizer": NOVELTY, "venn_abers": "in-house IVAP/ABERS (grid=%d)" % VA_GRID,
        },
        "venn_abers_self_check": va_chk,
        "methods": methods,
        "fixed_iptm_all_models": {},
        "accept_all_all_models": {},
        "af3_detail": {},
        "all_models_shift": {},
    }

    detail_iid = ["fixed_iptm", "venn_abers", "localized", "groupcond",
                  "combined_conformal", "accept_all", "abstain_all"]
    detail_shift = ["fixed_iptm", "naive_conformal", "venn_abers", "localized",
                    "groupcond", "accept_all", "abstain_all"]

    for m in methods:
        sub = df[df.method == m].reset_index(drop=True)
        y = sub["correct"].to_numpy().astype(int)
        strat = sub[STRAT].to_numpy().astype(int)
        groups = sub["system_id"].to_numpy()

        iid_runs = [_iid_once(sub, y, strat, groups, 20260720 + r) for r in range(N_IID_REPEATS)]
        shift_runs = [_shift_once(sub, y, strat, groups, 730000 + r) for r in range(N_SHIFT_REPEATS)]

        iid_gates = set().union(*[set(r) for r in iid_runs])
        iid_summ = {gt: _mean_of_gates([r[gt] for r in iid_runs if gt in r]) for gt in iid_gates}
        shift_gates = set().union(*[set(r) for r in shift_runs])
        shift_summ = {gt: _mean_shift(shift_runs, gt) for gt in shift_gates}

        # anchors for all five models
        out["fixed_iptm_all_models"][m] = {
            "iid": {"coverage": iid_summ["fixed_iptm"]["coverage"],
                    "selective_risk": iid_summ["fixed_iptm"]["selective_risk"]},
            "shift": {"coverage": shift_summ["fixed_iptm"]["coverage"],
                      "selective_risk": shift_summ["fixed_iptm"]["selective_risk"],
                      "per_stratum": {str(k): {"coverage": shift_summ["fixed_iptm"]["per_stratum"][k]["coverage"],
                                               "selective_risk": shift_summ["fixed_iptm"]["per_stratum"][k]["selective_risk"]}
                                      for k in DEPLOY_STRATA}},
        }
        out["accept_all_all_models"][m] = {
            "iid_base_error": iid_summ["accept_all"]["selective_risk"],
            "shift_base_error": shift_summ["accept_all"]["selective_risk"],
        }
        out["all_models_shift"][m] = {
            gt: {"coverage": shift_summ[gt]["coverage"],
                 "selective_risk": shift_summ[gt]["selective_risk"],
                 "frac_repeats_risk_le_alpha": shift_summ[gt]["frac_repeats_risk_le_alpha"]}
            for gt in ("groupcond", "venn_abers", "localized", "naive_conformal", "fixed_iptm")
        }

        if m == "af3":
            out["af3_detail"] = {
                "iid": {gt: iid_summ[gt] for gt in detail_iid if gt in iid_summ},
                "shift": {gt: shift_summ[gt] for gt in detail_shift if gt in shift_summ},
            }

    # verdict: does Venn-Abers match the GBM group-conditional repair under shift?
    af3s = out["all_models_shift"]["af3"]
    va, gc = af3s["venn_abers"], af3s["groupcond"]
    va_holds = np.isfinite(va["selective_risk"]) and va["selective_risk"] <= ALPHA + 0.02
    gc_holds = np.isfinite(gc["selective_risk"]) and gc["selective_risk"] <= ALPHA + 0.02
    cov_gap = va["coverage"] - gc["coverage"]
    out["venn_abers_vs_gbm"] = {
        "af3_va_shift_risk": va["selective_risk"], "af3_gbm_shift_risk": gc["selective_risk"],
        "af3_va_shift_coverage": va["coverage"], "af3_gbm_shift_coverage": gc["coverage"],
        "va_holds_risk": bool(va_holds), "gbm_holds_risk": bool(gc_holds),
        "coverage_gap_va_minus_gbm": float(cov_gap),
        "va_matches_gbm": bool(va_holds and gc_holds and abs(cov_gap) <= 0.10),
    }

    # which baselines break vs hold under shift (AF3), by realized risk
    breaks, holds = [], []
    for gt in ("fixed_iptm", "naive_conformal", "venn_abers", "localized", "groupcond"):
        r = af3s[gt]["selective_risk"]
        (holds if (np.isfinite(r) and r <= ALPHA + 1e-9) else breaks).append(
            {"baseline": gt, "shift_risk": r})
    out["shift_verdict_af3"] = {"hold_risk_le_alpha": holds, "break_risk_gt_alpha": breaks}
    return out


def main() -> None:
    res = run()
    save_json(res, RESDIR / "e31_extra_baselines.json")

    a = ALPHA
    print(f"E31 -- extra baselines (alpha={a}, delta={DELTA}, fixed ipTM tau={FIXED_IPTM_TAU})")
    print(f"Venn-Abers self-check max |err| vs sklearn = {res['venn_abers_self_check']['max_abs_err_vs_sklearn']:.2e}\n")

    print("AF3 detailed baseline table (risk = error among accepted; cov = accepted fraction):")
    print(f"  {'baseline':20} {'iid risk':>9} {'iid cov':>8}   {'shift risk':>10} {'shift cov':>9}")
    d = res["af3_detail"]
    order = ["fixed_iptm", "naive_conformal", "venn_abers", "localized", "groupcond",
             "combined_conformal", "accept_all", "abstain_all"]
    for gt in order:
        ii = d["iid"].get(gt)
        sh = d["shift"].get(gt)
        ir = f"{ii['selective_risk']:.3f}" if ii and np.isfinite(ii['selective_risk']) else "  -  "
        ic = f"{ii['coverage']:.2f}" if ii else "  -  "
        sr = f"{sh['selective_risk']:.3f}" if sh and np.isfinite(sh['selective_risk']) else "  -  "
        sc = f"{sh['coverage']:.2f}" if sh else "  -  "
        print(f"  {gt:20} {ir:>9} {ic:>8}   {sr:>10} {sc:>9}")

    print(f"\nAccept-all base error rate per model (iid / shift-target):")
    for m, v in res["accept_all_all_models"].items():
        print(f"  {m:9} iid={v['iid_base_error']:.3f}  shift(S2-S3)={v['shift_base_error']:.3f}")

    print(f"\nFixed ipTM>= {FIXED_IPTM_TAU} gate, all models (risk/cov):")
    for m, v in res["fixed_iptm_all_models"].items():
        print(f"  {m:9} iid risk={v['iid']['selective_risk']:.3f} cov={v['iid']['coverage']:.2f}   "
              f"shift risk={v['shift']['selective_risk']:.3f} cov={v['shift']['coverage']:.2f}")

    print(f"\nUnder shift (AF3), realized risk vs target alpha={a}:")
    v = res["shift_verdict_af3"]
    print("  HOLD (risk<=alpha):  " + ", ".join(f"{h['baseline']}({h['shift_risk']:.3f})" for h in v["hold_risk_le_alpha"]) or "  none")
    print("  BREAK (risk>alpha): " + ", ".join(f"{h['baseline']}({h['shift_risk']:.3f})" for h in v["break_risk_gt_alpha"]) or "  none")

    vg = res["venn_abers_vs_gbm"]
    print(f"\nVenn-Abers vs GBM group-conditional (AF3, shift): "
          f"VA risk={vg['af3_va_shift_risk']:.3f} cov={vg['af3_va_shift_coverage']:.2f} | "
          f"GBM risk={vg['af3_gbm_shift_risk']:.3f} cov={vg['af3_gbm_shift_coverage']:.2f} | "
          f"matches={vg['va_matches_gbm']}")
    print(f"\nsaved {RESDIR / 'e31_extra_baselines.json'}")


if __name__ == "__main__":
    main()
