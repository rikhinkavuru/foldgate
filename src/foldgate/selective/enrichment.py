"""Virtual-screening enrichment metrics and the selective-screening (abstention) analysis (W2).

A screening library is ranked by a co-folding *screening score* (e.g. interface chain-pair
ipTM or min-iPAE, which separate binders far better than global pTM). The foldgate reliability
layer adds a second axis: abstain on poses it cannot certify, then screen only the retained
subset. The question W2 answers is whether that abstention *raises* enrichment -- i.e. whether
the layer removes decoys faster than actives.

Metrics (labels: 1 = active/binder, 0 = decoy/inactive; scores: higher = more likely active):
  * ``enrichment_factor`` -- EF@k%, the standard fold-over-random early-retrieval metric.
  * ``bedroc`` -- Boltzmann-enhanced discrimination (Truchon & Bayly 2007), early-weighted.
  * ``roc_auc`` / ``log_auc`` -- global and early-log-scaled ranking quality.

Selective-screening analysis (the W2 result):
  * ``selective_enrichment_curve`` -- gate on reliability at each coverage, then measure EF on
    the retained library at a FIXED retained size (so the EF denominator does not silently
    shift as you abstain).
  * ``active_retention_curve`` -- fraction of ACTIVES kept vs coverage; decoys should drop
    faster than actives.
  * ``random_abstention_ef`` -- EF under random abstention at matched coverage, the control
    that isolates the reliability signal from the mere act of shrinking the library.

Calibrate the reliability abstention threshold with LTT on a held-out split, never on the test
enrichment. Bootstrap CIs over ligands. Prefer property-matched decoys (DEKOIS 2.0) or real
inactives (LIT-PCBA) over analog-biased DUD-E.
"""

from __future__ import annotations

import numpy as np


def _order(scores: np.ndarray) -> np.ndarray:
    """Descending rank order of scores (ties broken by index for determinism)."""
    return np.argsort(-np.asarray(scores, dtype=float), kind="stable")


def enrichment_factor(scores: np.ndarray, labels: np.ndarray, frac: float = 0.01) -> float:
    """EF@frac: (active fraction in the top frac of the ranking) / (overall active fraction)."""
    labels = np.asarray(labels, dtype=int)
    n = len(labels)
    n_act = int(labels.sum())
    if n == 0 or n_act == 0:
        return float("nan")
    k = max(1, int(round(frac * n)))
    top = _order(scores)[:k]
    return float((labels[top].sum() / k) / (n_act / n))


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC-AUC via the Mann-Whitney U statistic (fraction of active>decoy score pairs)."""
    labels = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    pos, neg = s[labels == 1], s[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def log_auc(scores: np.ndarray, labels: np.ndarray, lam: float = 0.001) -> float:
    """Area under the ROC curve on a log10 x-axis from lam to 1 (early-recognition weighted)."""
    labels = np.asarray(labels, dtype=int)
    order = _order(scores)
    y = labels[order]
    n_act = int(y.sum())
    n = len(y)
    if n_act == 0 or n_act == n:
        return float("nan")
    fpr, tpr = [], []
    fp = tp = 0
    n_dec = n - n_act
    for yi in y:
        if yi == 1:
            tp += 1
        else:
            fp += 1
        fpr.append(fp / n_dec)
        tpr.append(tp / n_act)
    fpr = np.array([lam] + fpr)
    tpr = np.array([tpr[0]] + tpr)
    mask = fpr >= lam
    x = np.log10(np.clip(fpr[mask], lam, 1.0))
    yv = tpr[mask]
    area = np.trapezoid(yv, x)
    return float(area / (np.log10(1.0) - np.log10(lam)))


def bedroc(scores: np.ndarray, labels: np.ndarray, alpha: float = 80.5) -> float:
    """BEDROC (Truchon & Bayly, JCIM 2007): early-recognition-weighted enrichment in [0,1]."""
    labels = np.asarray(labels, dtype=int)
    n = len(labels)
    n_act = int(labels.sum())
    if n_act == 0 or n_act == n:
        return float("nan")
    ra = n_act / n
    ranks = np.where(labels[_order(scores)] == 1)[0] + 1        # 1-indexed ranks of actives
    s = np.sum(np.exp(-alpha * ranks / n))
    rie = s / (ra * (1 - np.exp(-alpha)) / (np.exp(alpha / n) - 1))
    return float(
        rie * ra * np.sinh(alpha / 2) / (np.cosh(alpha / 2) - np.cosh(alpha / 2 - alpha * ra))
        + 1.0 / (1 - np.exp(alpha * (1 - ra)))
    )


def selective_enrichment_curve(
    scores: np.ndarray, reliability: np.ndarray, labels: np.ndarray,
    coverages=(1.0, 0.9, 0.75, 0.5, 0.25), frac: float = 0.01, fixed_topk: bool = True,
) -> list[dict]:
    """Enrichment as a function of reliability-gated coverage.

    At each coverage c, keep the top-c fraction of the library by ``reliability`` (abstain on
    the rest), then rank the retained by ``scores``. With ``fixed_topk`` the EF top-k is fixed
    to k = frac * N_full (a *selective-EF at fixed retained-library size*), so shrinking the
    library cannot inflate EF by moving the denominator. Returns one row per coverage.
    """
    scores = np.asarray(scores, dtype=float)
    reliability = np.asarray(reliability, dtype=float)
    labels = np.asarray(labels, dtype=int)
    n = len(labels)
    n_act_full = int(labels.sum())
    rel_order = _order(reliability)
    out = []
    for c in coverages:
        keep_n = max(1, int(round(c * n)))
        keep = rel_order[:keep_n]
        s_k, y_k = scores[keep], labels[keep]
        k = max(1, int(round(frac * n))) if fixed_topk else max(1, int(round(frac * keep_n)))
        k = min(k, keep_n)
        top = _order(s_k)[:k]
        ef = float((y_k[top].sum() / k) / (n_act_full / n)) if n_act_full else float("nan")
        out.append({
            "coverage": float(keep_n / n),
            "retained_actives_frac": float(y_k.sum() / n_act_full) if n_act_full else float("nan"),
            "ef_at_frac": ef,
            "bedroc": bedroc(s_k, y_k) if y_k.sum() and y_k.sum() < len(y_k) else float("nan"),
            "n_retained": keep_n,
        })
    return out


def active_retention_curve(reliability: np.ndarray, labels: np.ndarray,
                           coverages=(1.0, 0.9, 0.75, 0.5, 0.25)) -> list[dict]:
    """Fraction of actives vs decoys retained at each reliability-gated coverage.

    A useful gate removes decoys faster than actives (decoy_retained < active_retained), so the
    accepted library is enriched. Reports both so silent discarding of true actives is visible.
    """
    reliability = np.asarray(reliability, dtype=float)
    labels = np.asarray(labels, dtype=int)
    n = len(labels)
    n_act, n_dec = int(labels.sum()), int((labels == 0).sum())
    rel_order = _order(reliability)
    out = []
    for c in coverages:
        keep = rel_order[: max(1, int(round(c * n)))]
        y = labels[keep]
        out.append({
            "coverage": float(len(keep) / n),
            "active_retained": float(y.sum() / n_act) if n_act else float("nan"),
            "decoy_retained": float((y == 0).sum() / n_dec) if n_dec else float("nan"),
        })
    return out


def random_abstention_ef(scores: np.ndarray, labels: np.ndarray, coverage: float,
                         frac: float = 0.01, n_boot: int = 200, seed: int = 0) -> tuple[float, float, float]:
    """EF under RANDOM abstention at matched coverage (mean, p05, p95) -- the control.

    If reliability-gated selective EF beats this band, the gain is from the reliability signal,
    not from merely shrinking the library.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    n = len(labels)
    n_act_full = int(labels.sum())
    keep_n = max(1, int(round(coverage * n)))
    k = min(max(1, int(round(frac * n))), keep_n)
    rng = np.random.default_rng(seed)
    efs = []
    for _ in range(n_boot):
        keep = rng.permutation(n)[:keep_n]
        s_k, y_k = scores[keep], labels[keep]
        top = _order(s_k)[:k]
        if n_act_full:
            efs.append((y_k[top].sum() / k) / (n_act_full / n))
    efs = np.array(efs)
    return float(efs.mean()), float(np.percentile(efs, 5)), float(np.percentile(efs, 95))
