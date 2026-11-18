"""Texel filling of uv triangles: every texel inside a triangle gets a 3D point by barycentric
interpolation, and a triangle smaller than a texel still gets its centre texel."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "idarb"))
import bake_albedo as ba  # noqa: E402


def test_texel_points_fill_and_interpolate():
    # one triangle covering the lower-left half of a 16x16 texture, 3D points = (u, v, 0)
    uv = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    V = np.column_stack([uv, np.zeros(3)])
    F = np.array([[0, 1, 2]])
    rows, cols, pts = ba.texel_points(uv, F, V, 16, 16)
    assert 100 <= len(rows) <= 160  # about half of 256 texels, with the edge tolerance
    # the 3D point of a texel equals its uv position (u right, v up: row 0 is v = 1)
    u = (cols + 0.5) / 16
    v = 1.0 - (rows + 0.5) / 16
    assert np.allclose(pts[:, 0], u, atol=1e-6) and np.allclose(pts[:, 1], v, atol=1e-6)
    # a tiny triangle still yields one texel
    uv2 = np.array([[0.5, 0.5], [0.51, 0.5], [0.5, 0.51]])
    rows2, cols2, pts2 = ba.texel_points(uv2, F, np.column_stack([uv2, np.zeros(3)]), 16, 16)
    assert len(rows2) >= 1
