import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "idarb"))
from bake_albedo import texel_points  # noqa: E402
from bake_target_atlas import dilate_filled, level_colours, per_face_atlas  # noqa: E402


def test_per_face_atlas_patches_do_not_overlap():
    F = np.arange(15).reshape(5, 3)
    uvc = per_face_atlas(F, size=256, inset=2.0)
    assert uvc.shape == (5, 3, 2) and (uvc >= 0).all() and (uvc <= 1).all()
    V = np.random.default_rng(0).normal(size=(15, 3))
    rows, cols, pts, fid, bary = texel_points(uvc, F, V, 256, 256, extra=True)
    cells = set(zip(rows.tolist(), cols.tolist()))
    assert len(cells) == len(rows)  # no texel claimed by two faces
    assert set(fid.tolist()) == set(range(5)) and np.allclose(bary.sum(axis=1), 1.0, atol=1e-6)


def test_level_colours_matches_statistics():
    rng = np.random.default_rng(1)
    src = rng.normal(100, 10, size=(500, 3))
    ref = rng.normal(150, 20, size=(500, 3))
    mapped = level_colours(src, ref)(src)
    assert np.allclose(mapped.mean(axis=0), ref.mean(axis=0), atol=2.0)
    assert np.allclose(mapped.std(axis=0), ref.std(axis=0), atol=2.0)


def test_dilate_fills_the_ring_around_filled_texels():
    tex = np.zeros((9, 9, 3), dtype=np.uint8)
    filled = np.zeros((9, 9), dtype=bool)
    tex[4, 4] = 200
    filled[4, 4] = True
    out = dilate_filled(tex, filled, passes=1)
    assert out[3, 4, 0] == 200 and out[5, 5, 0] == 200 and out[0, 0, 0] == 0
