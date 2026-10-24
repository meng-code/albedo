import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "idarb"))
from keypoint_split import bootstrap_mean, make_split, summarise  # noqa: E402


def test_split_is_deterministic_and_disjoint():
    groups = {"same": [f"p{i}" for i in range(13)], "cross": ["c1", "c2"]}
    a = make_split(groups)
    b = make_split(groups)
    assert a == b
    assert set(a["tune"]).isdisjoint(a["held_out"])
    assert sorted(a["tune"] + a["held_out"]) == sorted(groups["same"] + groups["cross"])
    assert len(a["held_out"]) == 7 + 1  # ceil of half of 13 and of 2


def test_summary_pools_by_keypoint_count():
    rows = [dict(group="same", pair="a", method="m", n="2", pck_2="1.0", pck_5="1.0"),
            dict(group="same", pair="b", method="m", n="6", pck_2="0.0", pck_5="0.5"),
            dict(group="cross", pair="c", method="m", n="4", pck_2="0.5", pck_5="0.5")]
    split = dict(tune=["a", "b"], held_out=["c"])
    out = summarise(rows, split, "f.csv")
    tune_all = [r for r in out if r["split"] == "tune" and r["group"] == "all"][0]
    assert tune_all["pairs"] == 2 and tune_all["keypoints"] == 8
    assert np.isclose(tune_all["mean_pck_2"], 0.5) and np.isclose(tune_all["pooled_pck_2"], 0.25)
    held = [r for r in out if r["split"] == "held_out" and r["group"] == "cross"][0]
    assert held["pairs"] == 1 and np.isclose(held["pooled_pck_2"], 0.5)


def test_bootstrap_interval_contains_mean():
    lo, hi = bootstrap_mean([0.2, 0.4, 0.6, 0.8])
    assert lo <= 0.5 <= hi and lo < hi
