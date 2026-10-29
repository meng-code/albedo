"""Texture-map quality: conformal distortion and flips on hand-made triangles."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "idarb"))
import uv_metrics as um  # noqa: E402


def square_facing_camera():
    # a unit square in the xy plane, seen from +z, textured with its own xy as uv
    V = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
    F = np.array([[0, 1, 2], [0, 2, 3]])
    uv = V[:, :2]
    return V, F, uv[F]


def test_identity_map_has_no_distortion_and_no_flip():
    V, F, uvc = square_facing_camera()
    q = um.uv_quality(V, F, uvc, (0, 0, 1))
    assert q["triangles"] == 2
    assert abs(q["conformal_median"] - 1.0) < 1e-6
    assert q["flipped"] == 0.0 and q["flipped_area"] == 0.0


def test_mirrored_texture_is_flipped():
    V, F, uvc = square_facing_camera()
    uvc = uvc.copy()
    uvc[:, :, 0] = 1.0 - uvc[:, :, 0]  # mirror u
    q = um.uv_quality(V, F, uvc, (0, 0, 1))
    assert q["flipped"] == 1.0 and q["flipped_area"] == 1.0


def test_back_facing_triangle_with_projective_uv_is_not_flipped():
    # the same square with reversed winding (normal -z) seen from +z: its texture is what the
    # camera sees, so it is not mirrored; mirroring u then is a flip
    V, F, uvc = square_facing_camera()
    F = F[:, ::-1]
    uvc = uvc[:, ::-1]
    assert um.uv_quality(V, F, uvc, (0, 0, 1))["flipped"] == 0.0
    mirrored = uvc.copy()
    mirrored[:, :, 0] = 1.0 - mirrored[:, :, 0]
    assert um.uv_quality(V, F, mirrored, (0, 0, 1))["flipped"] == 1.0


def test_degenerate_triangle_is_neither_flipped_nor_measured():
    V, F, uvc = square_facing_camera()
    uvc = uvc.copy()
    uvc[1, 1] = uvc[1, 0]  # two corners of the second triangle on one texture point
    q = um.uv_quality(V, F, uvc, (0, 0, 1))
    assert q["degenerate"] == 0.5 and q["flipped"] == 0.0
    assert abs(q["conformal_median"] - 1.0) < 1e-6


def test_non_square_image_is_not_distortion():
    V, F, uvc = square_facing_camera()
    uvc = uvc.copy()
    uvc[:, :, 0] *= 0.5  # the square occupies 100 x 200 pixels of a 200 x 200 image
    assert abs(um.uv_quality(V, F, uvc, (0, 0, 1), (200.0, 200.0))["conformal_median"] - 2.0) < 1e-6
    # on a 400 x 200 image the same uv is a square patch again
    assert abs(um.uv_quality(V, F, uvc, (0, 0, 1), (400.0, 200.0))["conformal_median"] - 1.0) < 1e-6


def test_stretched_texture_has_distortion():
    V, F, uvc = square_facing_camera()
    uvc = uvc.copy()
    uvc[:, :, 0] *= 4.0  # stretched four times along u
    q = um.uv_quality(V, F, uvc, (0, 0, 1))
    assert abs(q["conformal_median"] - 4.0) < 1e-6
    assert q["flipped"] == 0.0


def test_blank_faces_are_skipped(tmp_path):
    V, F, uvc = square_facing_camera()
    uvc = uvc.copy()
    uvc[1] = [0.5, 0.01]  # second triangle collapsed on one point (distinct corners, same place)
    q = um.uv_quality(V, F, uvc, (0, 0, 1))
    assert q["triangles"] == 1 and q["collapsed"] == 0.5
    # round trip through an OBJ: a face whose corners share one vt index is blank, not collapsed
    path = str(tmp_path / "t.obj")
    with open(path, "w") as f:
        for v in V:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for v in V:
            f.write(f"vt {v[0]} {v[1]}\n")
        f.write("vt 0.5 0.01\n")
        f.write("f 1/1 2/2 3/3\nf 1/5 3/5 4/5\n")
    V2, F2, uvc2 = um.read_textured_obj(path)
    q2 = um.uv_quality(V2, F2, uvc2, (0, 0, 1))
    assert q2["triangles"] == 1 and q2["collapsed"] == 0.0
