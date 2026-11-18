"""The local mode of match2d.py: a render cell searches only inside the window around its prior
photo cell, so a more similar cell outside the window is ignored and the best one inside wins."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "idarb"))
import match2d  # noqa: E402


def test_match_grids_local_stays_inside_the_window():
    rng = np.random.default_rng(0)
    hp = wp = 20
    fp = rng.normal(size=(hp, wp, 8))
    fp /= np.linalg.norm(fp, axis=2, keepdims=True)
    mp = np.ones((hp, wp), dtype=bool)
    mp[0, :] = False  # the top row is background
    # one render cell whose true photo cell is (10, 10); a distractor identical at (3, 3)
    fr = np.zeros((2, 2, 8))
    fr[1, 1] = fp[10, 10]
    fp[3, 3] = fp[10, 10]
    mr = np.zeros((2, 2), dtype=bool)
    mr[1, 1] = True
    prior = np.full((2, 2, 2), np.nan)
    prior[1, 1] = [11.4, 8.6]  # the prior is off by about two cells
    rc, yx, sim, moved = match2d.match_grids_local(fp, mp, fr, mr, prior, radius=3.0)
    assert rc.tolist() == [[1, 1]]
    assert np.allclose(yx[0], [10.5, 10.5], atol=0.3)  # the true cell, not the distractor
    assert sim[0] > 0.99 and 1.5 < moved[0] < 2.5
    # a window too small to reach the true cell: the best cell inside wins, and it is worse
    rc2, yx2, sim2, _ = match2d.match_grids_local(fp, mp, fr, mr, prior, radius=0.6)
    assert len(rc2) == 1 and sim2[0] < sim[0]
    # no prior: nothing to match
    prior[1, 1] = np.nan
    rc3, _, _, _ = match2d.match_grids_local(fp, mp, fr, mr, prior, radius=3.0)
    assert len(rc3) == 0
