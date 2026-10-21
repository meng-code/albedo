"""Correspondence refinement: candidates, mutual pairs, the smooth prior and the windowed choice,
on small synthetic data."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "idarb"))
import refine_correspondence as rc  # noqa: E402


def unit(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_topk_candidates_contain_the_best():
    rng = np.random.default_rng(0)
    fs = unit(rng.normal(size=(300, 16))).astype(np.float32)
    fd = unit(rng.normal(size=(40, 16))).astype(np.float32)
    idx, sim = rc.topk_candidates(fd, fs, 5, chunk=16)
    best = (fd @ fs.T).argmax(axis=1)
    assert all(best[i] in idx[i] for i in range(len(fd)))
    assert np.allclose(sim.max(axis=1), (fd @ fs.T).max(axis=1), atol=1e-5)


def test_mutual_pairs_are_symmetric():
    rng = np.random.default_rng(1)
    fs = unit(rng.normal(size=(50, 8))).astype(np.float32)
    fd = np.concatenate([fs[:10], unit(rng.normal(size=(20, 8))).astype(np.float32)])
    t, s, w = rc.mutual_pairs(fd, fs)
    # the ten copied features are mutual with themselves
    assert set(zip(t.tolist(), s.tolist())) >= {(i, i) for i in range(10)}
    assert np.all(w[t < 10] > 0.999)


def test_farthest_points_spread():
    X = np.array([[0, 0, 0], [0.1, 0, 0], [10, 0, 0], [0, 10, 0]], dtype=float)
    chosen = rc.farthest_points(X, 3, seed=0)
    assert len(set(chosen.tolist())) == 3
    assert {2, 3} <= set(chosen.tolist())  # the two far points are always taken


def test_fit_prior_recovers_affine_map():
    rng = np.random.default_rng(2)
    X = rng.random((20, 3))
    A = np.array([[2.0, 0.1, 0.0], [0.0, 1.5, 0.2], [0.1, 0.0, 0.5]])
    Y = X @ A + [1.0, -2.0, 0.5]
    predict = rc.fit_prior(X, Y, sigma=0.5, ridge=1e-8)
    Q = rng.random((10, 3))
    assert np.allclose(predict(Q), Q @ A + [1.0, -2.0, 0.5], atol=1e-3)


def test_refine_prefers_candidates_near_the_prior():
    # source: a line of points with identical features except a tiny noise; the prior says the
    # target vertex i sits at source point i, and the refinement must follow it
    rng = np.random.default_rng(3)
    n = 30
    src_xyz = np.stack([np.linspace(0, 1, n), np.zeros(n), np.zeros(n)], axis=1)
    fs = unit(np.ones((n, 4)) + 0.01 * rng.normal(size=(n, 4))).astype(np.float32)
    Vd = src_xyz + [0, 2.0, 0]
    Fd = np.array([[i, i + 1, (i + 2) % n] for i in range(n - 2)])
    fd = unit(np.ones((n, 4)) + 0.01 * rng.normal(size=(n, 4))).astype(np.float32)
    anchors = (Vd[::5], src_xyz[::5])
    choice = rc.refine(Vd, Fd, fd, src_xyz, fs, anchors, k=n, radius=0.1, iters=0)
    assert np.abs(choice - np.arange(n)).max() <= 1
    # without a prior the nearly identical features give an arbitrary match
    plain = rc.refine(Vd, Fd, fd, src_xyz, fs, None, k=n, radius=0.1, iters=0)
    assert np.abs(plain - np.arange(n)).max() > 1
