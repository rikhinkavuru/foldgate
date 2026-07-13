"""Real public tabular loaders reduced to the (s, y, nu) benchmark triple.

Same reduction as the synthetic generator and the co-folding pipeline: a base
classifier f is trained on the SOURCE domain only; then

  s  = f's predicted probability of its predicted class (max-softmax confidence),
  y  = 1[f's prediction is correct],
  nu = the natural shift coordinate (state/year for ACS, temporal block for elec2).

Leakage-free means the split unit is the natural group: training, calibration,
and test rows for a stratum are disjoint and drawn by group, never by shuffling
individuals across a group boundary (BENCHMARK_SPEC 3). The base classifier is
sklearn HistGradientBoostingClassifier (torch-free); pass any sklearn estimator
to override.

Loaders raise SkipDataset when the data cannot be fetched (package missing and
un-installable, or no network) so callers can skip gracefully rather than crash.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier


class SkipDataset(Exception):
    """Raised when a real dataset is unavailable; callers should skip, not fail."""


def _new_classifier():
    return HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.1, max_depth=None, random_state=0
    )


def build_triple(
    X: pd.DataFrame | np.ndarray,
    y_true: np.ndarray,
    nu: np.ndarray,
    train_mask: np.ndarray,
    eval_mask: np.ndarray,
    clf=None,
) -> pd.DataFrame:
    """Train f on train_mask rows, emit the (s, y, nu) triple on eval_mask rows.

    s = predicted P(predicted class); y = 1[prediction correct]; nu carried
    through. train_mask and eval_mask must be disjoint (asserted).
    """
    X = np.asarray(X) if not isinstance(X, pd.DataFrame) else X.to_numpy()
    y_true = np.asarray(y_true)
    nu = np.asarray(nu)
    if np.any(train_mask & eval_mask):
        raise ValueError("train_mask and eval_mask overlap -- would leak")
    if train_mask.sum() == 0 or eval_mask.sum() == 0:
        raise SkipDataset("empty train or eval split after grouping")

    clf = clf if clf is not None else _new_classifier()
    clf.fit(X[train_mask], y_true[train_mask])

    proba = clf.predict_proba(X[eval_mask])
    pred = clf.classes_[np.argmax(proba, axis=1)]
    s = proba.max(axis=1)
    y = (pred == y_true[eval_mask]).astype(int)
    return pd.DataFrame({"s": s, "y": y, "nu": nu[eval_mask]})


def grouped_split(
    group: np.ndarray,
    source_groups,
    rng: np.random.Generator,
    cal_frac: float = 0.5,
):
    """Leakage-free masks: train on source-group rows; per non-source group split
    its rows into a calibration half and a test half (disjoint).

    Returns (train_mask, cal_mask, test_mask). Source rows are all training; every
    target group contributes to both cal and test so Mondrian has in-stratum
    calibration and Marginal has a source calibration slice available downstream.
    """
    group = np.asarray(group)
    source_groups = set(np.atleast_1d(source_groups).tolist())
    n = len(group)
    train_mask = np.array([g in source_groups for g in group])
    cal_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)
    for g in np.unique(group):
        if g in source_groups:
            continue
        idx = np.where(group == g)[0]
        rng.shuffle(idx)
        n_cal = int(cal_frac * len(idx))
        cal_mask[idx[:n_cal]] = True
        test_mask[idx[n_cal:]] = True
    return train_mask, cal_mask, test_mask


# --------------------------------------------------------------------------- #
# ACS Income (folktables) -- spatial/temporal shift
# --------------------------------------------------------------------------- #
def _pip_install(pkg: str) -> None:
    """Install pkg into the running interpreter, tolerating uv-managed venvs.

    A uv venv ships no pip module, so `python -m pip` fails; fall back to
    `uv pip install --python <this interpreter>`. Raises on total failure.
    """
    import shutil

    attempts = [[sys.executable, "-m", "pip", "install", "--quiet", pkg]]
    uv = shutil.which("uv")
    if uv:
        attempts.append([uv, "pip", "install", "--quiet", "--python", sys.executable, pkg])
    last = None
    for cmd in attempts:
        try:
            subprocess.run(cmd, check=True, timeout=300)
            return
        except Exception as e:  # noqa: BLE001
            last = e
    raise RuntimeError(last)


def _ensure_folktables():
    try:
        import folktables  # noqa: F401
        return
    except ImportError:
        pass
    try:
        _pip_install("folktables")
        import folktables  # noqa: F401
    except Exception as e:  # install failure, no network, resolver error
        raise SkipDataset(f"folktables unavailable and un-installable: {e}") from e


def acs_income(
    states=("CA", "TX", "FL", "NY", "PA", "SD", "WY", "MS"),
    year: str = "2018",
    horizon: str = "1-Year",
):
    """Raw ACS Income frame: (X, y_true, group=state).

    Downloads and caches PUMS person records via folktables (MIT; underlying data
    is US Census public microdata). Raises SkipDataset if folktables or the data
    cannot be fetched.

    The shift coordinate nu is the STATE, so we fetch each state separately and tag
    every kept row with its state code. folktables' ``ACSIncome.group`` is RAC1P
    (race), NOT the state, and ``df_to_pandas`` applies row filters, so reading the
    state off the raw ``ST`` column would misalign; per-state fetch is the clean fix.
    """
    _ensure_folktables()
    from folktables import ACSDataSource, ACSIncome

    Xs, ys, groups = [], [], []
    for st in states:
        try:
            ds = ACSDataSource(survey_year=year, horizon=horizon, survey="person")
            df = ds.get_data(states=[st], download=True)
        except Exception as e:  # network / cache failure
            raise SkipDataset(f"ACS data fetch failed for {st}: {e}") from e
        X, y_true, _ = ACSIncome.df_to_pandas(df)
        Xs.append(X.reset_index(drop=True))
        ys.append(np.asarray(y_true).ravel().astype(int))
        groups.append(np.full(len(X), st, dtype=object))

    X_all = pd.concat(Xs, ignore_index=True)
    y_all = np.concatenate(ys)
    grp = np.concatenate(groups)
    return X_all, y_all, grp


def acs_income_triple(
    source_states=("CA",),
    target_states=("SD", "WY", "MS"),
    year: str = "2018",
    seed: int = 20260712,
    cal_frac: float = 0.5,
    max_rows: int | None = 200_000,
) -> pd.DataFrame:
    """End-to-end ACS Income (s, y, nu=state) triple with a leakage-free split.

    Trains f on source-state rows, evaluates on the calibration+test rows of every
    target state. nu is the state code. Returns one frame with an extra 'split'
    column ('cal'/'test') so downstream Mondrian/Marginal can respect the split.

    Only the source + target states are fetched, so the source is always present in
    the fetched set (folktables downloads one file per state; smaller state sets keep
    the run fast).
    """
    states = tuple(dict.fromkeys((*source_states, *target_states)))  # dedup, ordered
    X, y_true, group = acs_income(states=states, year=year)
    rng = np.random.default_rng(seed)
    if max_rows is not None and len(y_true) > max_rows:
        idx = rng.choice(len(y_true), size=max_rows, replace=False)
        X, y_true, group = X.iloc[idx].reset_index(drop=True), y_true[idx], group[idx]

    train_mask, cal_mask, test_mask = grouped_split(group, source_states, rng, cal_frac)
    frames = []
    for split, mask in (("cal", cal_mask), ("test", test_mask)):
        tri = build_triple(X, y_true, group, train_mask, mask)
        tri["split"] = split
        frames.append(tri)
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- #
# Electricity / elec2 (OpenML) -- temporal concept drift
# --------------------------------------------------------------------------- #
def electricity(n_blocks: int = 6):
    """Raw elec2 frame: (X, y_true, group=temporal block index).

    Fetched via sklearn fetch_openml(name='electricity', version=1) (OpenML data
    id 151), cached to ~/scikit_learn_data. Rows are kept in time order and cut
    into n_blocks contiguous temporal blocks; the block index is the shift
    coordinate nu (BENCHMARK_SPEC 3.2 -- never shuffle across time). Raises
    SkipDataset on fetch failure (no network / OpenML unreachable).
    """
    from sklearn.datasets import fetch_openml

    try:
        d = fetch_openml(name="electricity", version=1, as_frame=True)
    except Exception as e:
        raise SkipDataset(f"electricity fetch failed: {e}") from e

    df = d.frame.copy()
    target_col = d.target_names[0]
    y_true = (df[target_col].astype(str).str.upper().str[0] == "U").astype(int).to_numpy()
    X = df.drop(columns=[target_col])
    # numeric coercion; elec2 features are all numeric after the class column.
    X = X.apply(pd.to_numeric, errors="coerce")
    # preserve time order (the frame is already time-ordered by 'date'/'period').
    n = len(df)
    block = (np.arange(n) * n_blocks // n).astype(int)
    return X.reset_index(drop=True), y_true, block


def electricity_triple(
    n_blocks: int = 6,
    seed: int = 20260712,
    cal_frac: float = 0.5,
) -> pd.DataFrame:
    """End-to-end elec2 (s, y, nu=block) triple, train on block 0, drift forward.

    Temporal-only split: train f on block 0; every later block is split into an
    in-block calibration half and a test half. Respects time order (no shuffling
    across blocks). Returns a frame with a 'split' column.
    """
    X, y_true, block = electricity(n_blocks)
    rng = np.random.default_rng(seed)
    train_mask, cal_mask, test_mask = grouped_split(block, source_groups=0,
                                                    rng=rng, cal_frac=cal_frac)
    frames = []
    for split, mask in (("cal", cal_mask), ("test", test_mask)):
        tri = build_triple(X, y_true, block, train_mask, mask)
        tri["split"] = split
        frames.append(tri)
    return pd.concat(frames, ignore_index=True)
