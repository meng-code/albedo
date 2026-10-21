"""Texture transfer: the relief's pixel grid, triangle spans, the software rasteriser, visibility
and the OBJ export. Runs on small hand-made meshes."""

import os
import sys

import numpy as np
import trimesh

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "idarb"))
import transfer_texture as tt  # noqa: E402


def grid_relief(H=6, W=5, step=0.1):
    """A relief over every pixel of an H x W image: vertex = (column, -row), colour = pixel."""
    rr, cc = np.nonzero(np.ones((H, W), dtype=bool))
    V = np.stack([cc * step - 0.3, -rr * step + 0.2, np.zeros(len(rr))], axis=1)
    C = np.stack([rr * 40, cc * 50, np.full(len(rr), 7)], axis=1).astype(np.uint8)
    return V, C, rr, cc


def test_relief_grid_recovers_pixels():
    V, C, rr, cc = grid_relief()
    col, row = tt.relief_grid(V)
    assert np.array_equal(col, cc) and np.array_equal(row, rr)


def test_relief_grid_rejects_irregular_mesh():
    rng = np.random.default_rng(0)
    try:
        tt.relief_grid(rng.random((50, 3)))
    except ValueError:
        return
    raise AssertionError("an irregular mesh must not pass as a pixel grid")


def test_relief_texture_image_and_uv():
    V, C, rr, cc = grid_relief()
    keep = np.arange(len(V)) != 7  # one missing pixel: filled from a neighbour
    image, uv, blank_uv = tt.relief_texture(V[keep], C[keep], blank_rows=4)
    assert image.shape == (6 + 4, 5, 3)
    assert np.array_equal(image[rr[keep], cc[keep]], C[keep])
    assert image[rr[7], cc[7]].tolist() in [C[i].tolist() for i in range(len(C)) if i != 7]
    assert np.array_equal(image[6:], np.full((4, 5, 3), tt.BLANK))
    # u grows with the column, v with the upward direction (smaller row); the blank point is in
    # the strip
    assert np.all(np.diff(uv[:4, 0]) > 0) and uv[0, 1] > uv[-1, 1]
    row_blank = int((1 - blank_uv[1]) * image.shape[0])
    assert 6 <= row_blank < 10


def test_uv_span_fraction_of_diagonal():
    uv = np.array([[0.0, 0.0], [0.5, 0.0], [0.0, 0.0]])
    span = tt.uv_span(uv[np.array([[0, 1, 2]])], (100, 100))
    assert np.isclose(span[0], 50.0 / np.hypot(100, 100))


def test_source_usage_collapse_versus_full():
    rng = np.random.default_rng(1)
    uv_s = rng.random((4000, 2))
    assert tt.source_usage(uv_s, uv_s) == 1.0  # every source cell is hit
    collapsed = np.repeat(uv_s[:1], 4000, axis=0)  # every target vertex on one source point
    assert tt.source_usage(collapsed, uv_s) < 0.01


def square():
    V = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], dtype=float)
    F = np.array([[0, 1, 2], [0, 2, 3]])
    uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    image[0, 0], image[0, 1] = (255, 0, 0), (0, 255, 0)
    image[1, 0], image[1, 1] = (0, 0, 255), (255, 255, 0)
    return V, F, uv, image


def test_rasterize_samples_texture_inside_triangle():
    # a square facing +z, textured with a 2x2 image: the four quadrants show the four texels
    V, F, uv, image = square()
    img = tt.rasterize(V, F, uv, image, (0, 0, 1), size=64, fill=1.0)
    assert np.allclose(img[16, 16], (1, 0, 0))  # top-left of the render = image row 0, column 0
    assert np.allclose(img[16, 48], (0, 1, 0))
    assert np.allclose(img[48, 16], (0, 0, 1))
    assert np.allclose(img[48, 48], (1, 1, 0))


def test_rasterize_per_corner_uv_blank_face():
    V, F, uv, image = square()
    uvc = uv[F].copy()
    uvc[1] = [0.75, 0.25]  # the second (top-left) triangle points at the yellow texel everywhere
    img = tt.rasterize(V, F, uvc, image, (0, 0, 1), size=64, fill=1.0)
    assert np.allclose(img[16, 16], (1, 1, 0))  # top-left is now yellow
    assert np.allclose(img[48, 16], (0, 0, 1))  # first triangle: bottom-left stays blue
    assert np.allclose(img[16, 48], (0, 1, 0))  # first triangle: top-right stays green


def test_visible_faces_one_side_of_a_box():
    box = trimesh.creation.box(extents=(1, 1, 1))
    V, F = np.asarray(box.vertices, dtype=float), np.asarray(box.faces)
    vis = tt.visible_faces(V, F, (0, 0, 1), size=128)
    normals = np.asarray(box.face_normals)
    assert np.array_equal(vis, normals[:, 2] > 0.5)  # exactly the +z side (two triangles)


def test_export_textured_obj_roundtrip(tmp_path):
    # every vertex is used by a face: the OBJ reader drops unreferenced vertices
    V, C, rr, cc = grid_relief(H=2, W=2)
    image, uv, blank_uv = tt.relief_texture(V, C)
    F = np.array([[0, 1, 2], [1, 3, 2]])
    path = str(tmp_path / "textured.obj")
    tt.export_textured_obj(path, V, F, uv, image, np.array([True, True]), blank_uv)
    assert os.path.exists(str(tmp_path / "textured.mtl"))
    assert os.path.exists(str(tmp_path / "textured.png"))
    m = trimesh.load(path, process=False, force="mesh")
    assert m.visual.uv.shape == (len(V), 2)
    assert np.allclose(m.visual.uv, uv, atol=1e-6)
    assert m.visual.material.image.size == (image.shape[1], image.shape[0])
    # a blank face points at the strip: the OBJ gains one texture coordinate
    tt.export_textured_obj(path, V, F, uv, image, np.array([True, False]), blank_uv)
    with open(path) as f:
        text = f.read()
    assert text.count("\nvt ") == len(V) + 1
    assert f"/{len(V) + 1} " in text.splitlines()[-1]
