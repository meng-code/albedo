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
    choice, sim, _ = rc.refine(Vd, Fd, fd, src_xyz, fs, anchors, k=n, radius=0.1, iters=0)
    assert np.abs(choice - np.arange(n)).max() <= 1
    assert sim.shape == (n,) and np.all(sim > 0.9)
    # without a prior the nearly identical features give an arbitrary match
    plain, _, _ = rc.refine(Vd, Fd, fd, src_xyz, fs, None, k=n, radius=0.1, iters=0)
    assert np.abs(plain - np.arange(n)).max() > 1


def test_smooth_field_follows_confident_matches_and_smooths_the_rest():
    # a strip of vertices; matched positions are the vertex positions plus noise, with every
    # fifth vertex confident and exact: the field must stay close to the exact positions
    rng = np.random.default_rng(4)
    n = 60
    Vd = np.stack([np.linspace(0, 1, n), np.zeros(n), np.zeros(n)], axis=1)
    Fd = np.array([[i, i + 1, (i + 2) % n] for i in range(n - 2)])
    S = Vd + 0.2 * rng.normal(size=(n, 3))
    w = np.full(n, 0.05)
    S[::5] = Vd[::5]
    w[::5] = 10.0
    P = rc.smooth_field(Vd, Fd, S, w, lam=1.0)
    assert np.linalg.norm(P - Vd, axis=1).max() < 0.1
    assert np.linalg.norm(S - Vd, axis=1).max() > 0.3


def test_similarities_match_the_dot_products():
    rng = np.random.default_rng(5)
    fs = unit(rng.normal(size=(40, 8))).astype(np.float32)
    fd = unit(rng.normal(size=(7, 8))).astype(np.float32)
    cand = rng.integers(0, 40, size=(7, 5))
    sim = rc.similarities(fd, fs, cand, chunk=3)
    expect = np.take_along_axis(fd @ fs.T, cand, axis=1)
    assert np.allclose(sim, expect, atol=1e-6)


def test_k_prior_offers_the_point_the_prior_predicts():
    # the features of every source point are identical except one impostor far from the prior's
    # prediction that is slightly more similar: with k=1 the impostor is the only candidate, with
    # k_prior the point at the predicted position is offered too and wins through the prior term
    n = 30
    src_xyz = np.stack([np.linspace(0, 1, n), np.zeros(n), np.zeros(n)], axis=1)
    fs = unit(np.ones((n, 4))).astype(np.float32)
    fd = unit(np.ones((10, 4)) + np.array([0, 0, 0, 0.05])).astype(np.float32)
    fs[n - 1] = fd[0]  # the impostor: exactly the target's feature
    Vd = src_xyz[:10] + [0, 2.0, 0]
    Fd = np.array([[i, i + 1, (i + 2) % 10] for i in range(8)])
    anchors = (Vd[::3], src_xyz[:10:3])
    only_feat, _, _ = rc.refine(Vd, Fd, fd, src_xyz, fs, anchors, k=1, radius=0.1, iters=0)
    assert np.all(only_feat == n - 1)
    with_prior, _, _ = rc.refine(
        Vd, Fd, fd, src_xyz, fs, anchors, k=1, radius=0.1, iters=0, k_prior=4
    )
    assert np.abs(with_prior - np.arange(10)).max() <= 1


def grid_mesh(n=6, spacing=0.1):
    """A flat n x n grid in the xy plane facing +z (counter-clockwise faces seen from +z)."""
    xs, ys = np.meshgrid(np.arange(n) * spacing, np.arange(n) * spacing, indexing="ij")
    V = np.stack([xs.ravel(), ys.ravel(), np.zeros(n * n)], axis=1)
    F = []
    for i in range(n - 1):
        for j in range(n - 1):
            a, b, c, d = i * n + j, (i + 1) * n + j, (i + 1) * n + j + 1, i * n + j + 1
            F += [[a, b, c], [a, c, d]]
    return V, np.array(F)


def test_face_fans_and_mirrored_fraction_detect_a_mirrored_choice():
    V, F = grid_mesh()
    J, K, SGN = rc.face_fans(V, F, (0.0, 0.0, 1.0))
    assert (SGN[J >= 0] == 1).all()  # every face winds counter-clockwise on screen
    # candidates: the vertex's own position and its mirror image in x; with the neighbours at
    # their own positions the first mirrors nothing, with the neighbours mirrored the second
    # choice (mirrored with them) reverses every face of the fan
    Vm = V * [-1, 1, 1]
    C = np.stack([V, Vm], axis=1)
    inner = (J >= 0).sum(axis=1) == 6  # interior vertices with a full fan
    frac = rc.mirrored_fraction(C, V, J, K, SGN)
    assert np.all(frac[inner, 0] == 0.0)
    frac_m = rc.mirrored_fraction(C, Vm, J, K, SGN)
    assert np.all(frac_m[inner, 1] == 1.0)
    assert np.all(frac_m[inner, 0] < 1.0)
    # a mesh seen from behind has no camera-facing fan
    J2, _, _ = rc.face_fans(V, F, (0.0, 0.0, -1.0))
    assert (J2 < 0).all()


def test_arap_view_term_keeps_an_exact_rigid_map():
    # target = source turned by a rotation; the matches are exact, so the field must return
    # them, with and without the view pair (which must not inflate the scale)
    Vs, F = grid_mesh(7)
    ang = 0.6
    Rg = np.array([[np.cos(ang), 0, np.sin(ang)], [0, 1, 0], [-np.sin(ang), 0, np.cos(ang)]])
    Vd = Vs @ Rg.T + [1.0, 0.0, 0.0]
    S = Vs.copy()
    w = np.full(len(Vd), 1.0)
    view = Rg @ np.array([0.0, 0.0, 1.0])  # the source camera axis seen in target space
    rng = np.random.default_rng(0)
    P0 = S + 0.02 * rng.normal(size=S.shape)  # the alternation converges from nearby
    for w_view in (0.0, 1.0, 5.0):
        P = rc.arap_field(Vd, F, S, w, P0, lam=3.0, iters=20, view=view, w_view=w_view)
        assert np.linalg.norm(P - S, axis=1).max() < 1e-3, w_view
    # a view pair pointing the wrong way must pull the field off the exact solution
    P = rc.arap_field(Vd, F, S, w, P0, lam=3.0, iters=20, view=-view, w_view=5.0)
    assert np.linalg.norm(P - S, axis=1).max() > 1e-2


def test_hard_constraints_are_met_exactly_by_both_fields():
    rng = np.random.default_rng(6)
    n = 60
    Vd = np.stack([np.linspace(0, 1, n), np.zeros(n), np.zeros(n)], axis=1)
    Fd = np.array([[i, i + 1, (i + 2) % n] for i in range(n - 2)])
    S = Vd + 0.2 * rng.normal(size=(n, 3))
    w = np.full(n, 0.5)
    mask = np.zeros(n, dtype=bool)
    mask[::7] = True
    values = Vd + [0.0, 1.0, 0.0]  # pinned vertices sit on a shifted copy of the strip
    P = rc.smooth_field(Vd, Fd, S, w, lam=1.0, hard=(mask, values))
    assert np.allclose(P[mask], values[mask])
    assert np.linalg.norm(P[~mask] - S[~mask], axis=1).max() > 0.0
    P2 = rc.arap_field(Vd, Fd, S, w, P, lam=1.0, iters=3, hard=(mask, values))
    assert np.allclose(P2[mask], values[mask])
    # without constraints the field does not pass through those points
    P3 = rc.smooth_field(Vd, Fd, S, w, lam=1.0)
    assert not np.allclose(P3[mask], values[mask])


def test_rematch_within_picks_the_most_similar_near_candidate():
    P = np.zeros((2, 3))
    cand_xyz = np.array([[[0.1, 0, 0], [0.5, 0, 0], [0.02, 0, 0]],
                         [[0.9, 0, 0], [0.8, 0, 0], [0.7, 0, 0]]])
    sims = np.array([[0.9, 0.95, 0.8], [0.9, 0.9, 0.9]])
    best = rc.rematch_within(P, cand_xyz, sims, radius=0.2)
    assert best.tolist() == [0, -1]  # the 0.95 candidate is too far; nothing near for vertex 2
