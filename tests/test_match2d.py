import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "idarb"))
import match2d as m2  # noqa: E402


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def test_match_grids_identity_is_mutual_and_centred():
    rng = np.random.default_rng(0)
    f = unit(rng.normal(size=(6, 6, 8))).astype(np.float32)
    mask = np.ones((6, 6), dtype=bool)
    rc, yx, sim, mutual = m2.match_grids(f, mask, f, mask)
    assert len(rc) == 36 and mutual.all() and np.allclose(sim, 1.0, atol=1e-5)
    assert np.abs(yx - (rc + 0.5)).max() < 0.2  # the soft-argmax stays at the cell centre


def test_turned_views_are_unit_and_turned():
    views = m2.turned_views(np.array([0.0, 0.0, 1.0]), turn=30.0, elev=10.0)
    assert len(views) == 5 and all(abs(np.linalg.norm(v) - 1) < 1e-9 for v in views)
    assert np.isclose(np.degrees(np.arccos(np.dot(views[0], views[1]))), 30.0)
    assert np.isclose(np.degrees(np.arccos(np.dot(views[0], views[3]))), 10.0)


def test_lift_matches_hits_vertices_and_relief_pixels():
    # one big triangle filling the render; a 4x4 relief grid
    Vd = np.array([[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.5, 0.0]])
    Fd = np.array([[0, 1, 2]])
    img, fid, w0, w1 = m2.render_target(Vd, Fd, np.array([0.0, 0.0, 1.0]), size=64)
    assert (fid >= 0).sum() > 500
    vert = -np.ones((4, 4), dtype=np.int64)
    vert[1:3, 1:3] = np.arange(4).reshape(2, 2)
    rc = np.array([[1, 0], [1, 1]])  # render cells on a 2x2 grid of the 64-px render
    yx = np.array([[1.2, 1.2], [2.4, 2.4]])  # photo positions on a 4x4 grid
    t_idx, s_idx, ok = m2.lift_matches(rc, yx, (2, 2), (4, 4), fid, w0, w1, Vd, Fd, vert, (4, 4))
    assert ok.all() and set(t_idx.tolist()) <= {0, 1, 2}
    assert s_idx.tolist() == [0, 3]


def test_albedo_image_maps_pixels_to_vertices():
    cols, rows = np.meshgrid(np.arange(5), np.arange(3))
    Vs = np.stack([cols.ravel() * 0.1, -rows.ravel() * 0.1, np.zeros(15)], axis=1)
    Cs = np.tile([[10, 20, 30]], (15, 1)).astype(np.uint8)
    image, vert = m2.albedo_image(Vs, Cs)
    assert image.shape == (3, 5, 3) and (vert >= 0).all()
    assert vert[0, 0] == 0 and vert[2, 4] == 14
