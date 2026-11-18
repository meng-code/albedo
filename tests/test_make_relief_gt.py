"""Tier-B ground truth: a relief built from the render's own depth buffer must get the full
vertices behind its pixels."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "idarb"))
import make_relief_gt as mg  # noqa: E402
import transfer_texture as tt  # noqa: E402


def _sphere(n=400, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.normal(size=(n, 3))
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    from scipy.spatial import ConvexHull

    return p, ConvexHull(p).simplices.astype(np.int64)


def test_attach_gt_recovers_the_vertices_behind_the_pixels():
    V, F = _sphere()
    view = np.array([0.2, 0.3, 1.0])
    view /= np.linalg.norm(view)
    size = 64
    img, buf = mg.render_rgba(V, F, view, size)
    assert img.shape == (size, size, 4) and (img[..., 3] == 255).sum() == (buf["fid"] >= 0).sum()
    # a relief as normal_to_mesh.py would build it: one vertex per foreground pixel on the
    # pixel grid, x right, y up, depth from the buffer
    rr, cc = np.nonzero(buf["fid"] >= 0)
    scale = 2.0 / size
    Vr = np.stack([(cc - size / 2) * scale, (size / 2 - rr) * scale, buf["z"][rr, cc] * scale],
                  axis=1)
    Vr = Vr - Vr.mean(axis=0)  # normal_to_mesh.py recentres the relief
    orig, pts, row, col = mg.attach_gt(Vr, buf, V, F)
    assert (orig >= 0).all() and (row == rr).all() and (col == cc).all()
    # the true point projects to the vertex's pixel, and orig is a corner of that face
    xy, _ = tt.project(V, view, size)
    proj = (pts @ tt.view_frame(view).T)[:, :2]
    lo, hi = (V @ tt.view_frame(view).T)[:, :2].min(0), (V @ tt.view_frame(view).T)[:, :2].max(0)
    s = size * 0.9 / float((hi - lo).max())
    px = (proj - (lo + hi) / 2) * s + size / 2
    assert np.abs(px[:, 0] - (cc + 0.5)).max() < 1.0
    assert np.abs(size - px[:, 1] - (rr + 0.5)).max() < 1.0
    faces_of_pixels = F[buf["fid"][rr, cc]]
    assert ((faces_of_pixels == orig[:, None]).any(axis=1)).all()
