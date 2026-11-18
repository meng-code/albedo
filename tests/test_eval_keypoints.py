"""Keypoint errors of a correspondence: exact map gives zero, a shifted map gives the shift."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "idarb"))
import eval_keypoints as ek  # noqa: E402


def test_landmark_errors_and_summary():
    rng = np.random.default_rng(0)
    Vd = rng.random((50, 3))
    P = Vd + [1.0, 0.0, 0.0]  # the map: every target vertex to its own position shifted in x
    lm_t = Vd[[3, 7, 11]]
    lm_s = P[[3, 7, 11]].copy()
    lm_s[2] += [0.0, 0.5, 0.0]  # one annotated source point off by 0.5
    err = ek.landmark_errors(P, Vd, lm_s, lm_t, diag_s=10.0)
    assert np.allclose(err[:2], 0.0) and abs(err[2] - 0.05) < 1e-9
    s = ek.summary(err)
    assert s["pck_2"] == round(2 / 3, 4) and s["pck_10"] == 1.0 and len(s["per_landmark"]) == 3


def test_plane_error_ignores_depth():
    landmark_errors = ek.landmark_errors
    Vd = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    P = np.array([[0.0, 0.0, 0.5], [1.0, 0.0, 0.0]])  # the first vertex left the relief in z
    lm_t = Vd.copy()
    lm_s = Vd.copy()
    e3 = landmark_errors(P, Vd, lm_s, lm_t, 1.0)
    e2 = landmark_errors(P, Vd, lm_s, lm_t, 1.0, plane=True)
    assert np.isclose(e3[0], 0.5) and np.isclose(e2[0], 0.0) and e3[1] == e2[1] == 0.0
