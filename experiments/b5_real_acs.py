"""B5 -- real-data generality of the worst-stratum selective-risk claim (ACS Income).

Spatial-shift companion to B6. folktables ACSIncome (US Census PUMS person records)
reduced to the same (s, y, nu) triple: a base HistGradientBoosting classifier is
trained on a SOURCE state only, then

  s  = max-softmax confidence,
  y  = 1[base prediction of "income > 50k" is correct],
  nu = state code (the spatial shift coordinate).

The state (nu) is the protected shift axis: the source state trains f, and every
target state supplies its own calibration and test rows; no state ever calibrates
another. MARGINAL / MONDRIAN / WEIGHTED are compared exactly as in B6.

Why this dataset complements B6: elec2 is temporal CONCEPT drift, where reweighting
on the score cannot repair a moving P(correct | s). ACS spatial shift is closer to
covariate shift (the feature mix changes across states while P(income | features) is
more stable), so WEIGHTED is expected to help MORE here than on elec2. The concept-
shift diagnostic is reported so "covariate vs concept" is measured, not assumed.

Robustness: folktables is installed and fetched on demand; if the package or the
census download is unavailable (or the fetch exceeds a wall-clock budget), a
"skipped" JSON is written and the run returns cleanly rather than hanging.

Honesty: folktables is MIT and the underlying ACS PUMS microdata is US Census public
data. If ACS is skipped it is stated plainly and B6 (electricity) carries the claim.
"""

from __future__ import annotations

import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments.b6_real_electricity import (  # noqa: E402
    RESDIR,
    analyze,
    blocks_from_triple,
    git_sha,
    print_report,
    save_json,
)
from foldgate.bench.realdata import SkipDataset, acs_income_triple  # noqa: E402

SOURCE_STATES = ("CA",)
TARGET_STATES = ("SD", "WY", "MS")     # low-population states, strong spatial shift from CA
YEAR = "2018"
FETCH_BUDGET_SEC = 420                  # hard wall-clock cap on install+download+fit


def run() -> dict:
    t0 = time.time()
    tri = acs_income_triple(
        source_states=SOURCE_STATES, target_states=TARGET_STATES,
        year=YEAR, seed=20260712, cal_frac=0.5,
    )
    blocks = blocks_from_triple(tri)
    meta = {
        "dataset": "folktables ACSIncome (US Census PUMS, 2018 1-Year)",
        "loader": "folktables.ACSDataSource + ACSIncome.df_to_pandas (per-state fetch)",
        "license": "folktables MIT; underlying ACS PUMS is US Census public data",
        "shift_type": "spatial (state) covariate shift",
        "base_classifier": "HistGradientBoostingClassifier (torch-free)",
        "nu": "state code; train on source state, calibrate/test on target states",
        "reference_stratum_meaning": "one reference target state, threshold reused for all",
        "source_states": list(SOURCE_STATES), "target_states": list(TARGET_STATES),
        "year": YEAR, "delta": 0.10, "alphas": [0.10, 0.20], "seed": 20260712,
        "git_sha": git_sha(), "n_eval": int(len(tri)),
        "runtime_sec": round(time.time() - t0, 1),
    }
    out = analyze(blocks, meta)
    out["meta"]["runtime_sec"] = round(time.time() - t0, 1)
    return out


def _skip(reason: str) -> None:
    res = {
        "skipped": True, "reason": reason,
        "meta": {
            "dataset": "folktables ACSIncome",
            "source_states": list(SOURCE_STATES), "target_states": list(TARGET_STATES),
            "note": "B6 (electricity) carries the worst-stratum claim when ACS is unavailable",
            "git_sha": git_sha(),
        },
    }
    save_json(res, RESDIR / "b5_real_acs.json")
    print(f"B5 SKIPPED (folktables/ACS unavailable): {reason}")


def _on_alarm(signum, frame):
    raise SkipDataset(f"ACS fetch/fit exceeded the {FETCH_BUDGET_SEC}s wall-clock budget")


def main() -> None:
    had_alarm = hasattr(signal, "SIGALRM")
    if had_alarm:
        signal.signal(signal.SIGALRM, _on_alarm)
        signal.alarm(FETCH_BUDGET_SEC)
    try:
        res = run()
    except SkipDataset as e:
        _skip(str(e))
        return
    except Exception as e:  # any fetch/parse failure -> skip, never crash the pipeline
        _skip(f"{type(e).__name__}: {e}")
        return
    finally:
        if had_alarm:
            signal.alarm(0)

    save_json(res, RESDIR / "b5_real_acs.json")
    print_report(res, "B5 -- real-data (ACS Income) worst-state selective risk", "reference state")
    print(f"\nsaved {RESDIR / 'b5_real_acs.json'}")


if __name__ == "__main__":
    main()
