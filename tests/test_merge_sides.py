import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "idarb"))
from merge_sides import export_two_sided_obj, pack_side_by_side, remap_uv, side_views  # noqa: E402


def test_pack_and_remap_keep_each_pixel():
    a = np.zeros((4, 6, 3), dtype=np.uint8)
    b = np.full((8, 2, 3), 9, dtype=np.uint8)
    img, box_a, box_b = pack_side_by_side(a, b)
    assert img.shape == (8, 8, 3) and box_a == (0, 6, 4) and box_b == (6, 2, 8)
    assert (img[:4, :6] == 0).all() and (img[:, 6:] == 9).all() and (img[4:, :6] == 255).all()
    # the top-left texel of a (u 0, v 1) stays at the top-left; its bottom row maps to row 3
    uv = remap_uv(np.array([[0.0, 1.0], [1.0, 0.0]]), box_a, 8, 8)
    assert np.allclose(uv[0], [0.0, 1.0])
    assert np.allclose(uv[1], [6 / 8, 1.0 - 4 / 8])
    # b's bottom-right corner is the packed image's bottom-right corner
    uv = remap_uv(np.array([[1.0, 0.0]]), box_b, 8, 8)
    assert np.allclose(uv[0], [1.0, 0.0])


def test_two_sided_obj_indexes_both_uv_sets(tmp_path):
    V = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float)
    F = np.array([[0, 1, 2], [1, 3, 2]])
    uv_f = np.zeros((4, 2))
    uv_b = np.ones((4, 2))
    path = str(tmp_path / "m.obj")
    export_two_sided_obj(path, V, F, uv_f, uv_b, np.array([True, False]),
                         np.zeros((2, 2, 3), dtype=np.uint8))
    lines = open(path).read().splitlines()
    faces = [ln for ln in lines if ln.startswith("f ")]
    assert faces[0] == "f 1/1 2/2 3/3" and faces[1] == "f 2/6 4/8 3/7"
    assert sum(ln.startswith("vt ") for ln in lines) == 8
    assert os.path.exists(str(tmp_path / "m.mtl")) and os.path.exists(str(tmp_path / "m.png"))


def test_side_views_are_perpendicular():
    left, right = side_views((0.0, 0.0, 1.0))
    assert np.allclose(left, -right) and abs(float(np.dot(left, [0, 0, 1]))) < 1e-9
