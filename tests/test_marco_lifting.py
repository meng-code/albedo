import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "idarb"))
import marco_features as mf  # noqa: E402
import rerank_check as rc  # noqa: E402


def test_sphere_views_are_unit_and_start_along_z():
    v = mf.sphere_views([-30.0, 0.0, 30.0], [0.0, 90.0])
    assert v.shape == (6, 3)
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0)
    assert np.allclose(v[2], [0.0, 0.0, 1.0]) and np.allclose(v[3], [1.0, 0.0, 0.0])


def test_cell_coords_puts_a_cell_centre_on_its_index():
    # a 100 px crop starting at (10, 20) under a 10 x 10 grid: cell (2, 3) is centred on pixel
    # centre (10 + 25, 20 + 35), which is the pixel with index (34.5, 54.5) + 0.5
    rc_ = mf.cell_coords(np.array([34.5]), np.array([54.5]), (10, 20, 100), (10, 10))
    assert np.allclose(rc_, [[2.0, 3.0]])


def test_fill_unseen_copies_the_nearest_seen_row():
    xyz = np.array([[0.0, 0, 0], [1.0, 0, 0], [0.9, 0, 0]])
    feat = np.array([[1.0, 0], [0, 1.0], [0, 0]])
    out = mf.fill_unseen(feat, np.array([True, True, False]), xyz)
    assert out[2].tolist() == [0.0, 1.0]


def test_full_features_on_a_tetrahedron():
    V = np.array([[0.0, 0, 0], [1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
    F = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])

    def extract(image):  # a constant unit feature for every cell
        f = np.zeros((8, 8, 4), dtype=np.float32)
        f[..., 0] = 1.0
        return f

    views = mf.sphere_views([0.0, 30.0], [0.0, 90.0, 180.0, 270.0])
    feat, seen, count = mf.full_features(extract, V, F, np.arange(4), views, 64, 0.08)
    assert seen.all() and count.max() <= len(views)
    assert np.allclose(feat, [[1.0, 0, 0, 0]] * 4)


def test_rerank_methods():
    sim_a = np.array([[0.9, 0.8, 0.1, 0.0]])
    sim_b = np.array([[0.0, 0.5, 0.2, 0.9]])
    picks, cand_a, cand_b = rc.methods_of(sim_a, sim_b, 2, [0.5])
    assert picks["nn_a"][0] == 0 and picks["nn_b"][0] == 3
    assert picks["b_in_top_a"][0] == 1  # A's top two are 0 and 1; B prefers 1
    assert picks["a_in_top_b"][0] == 1  # B's top two are 3 and 1; A prefers 1
    assert picks["fused_0.5"][0] == 1
    assert sorted(cand_a[0].tolist()) == [0, 1] and sorted(cand_b[0].tolist()) == [1, 3]
