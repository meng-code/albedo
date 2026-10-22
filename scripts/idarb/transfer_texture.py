"""
Texture transfer through the semantic correspondence: the full target mesh gets texture coordinates
into the relief's albedo image, so the inside of a triangle shows the image itself and not a blend
of three vertex colours.

The relief is a height field over the albedo image: every relief vertex is one image pixel, so the
relief's texture coordinates are its pixel positions and its texture is the albedo image. The image
is rebuilt here from the relief's vertex colours on their pixel grid, with the background filled
from the nearest foreground pixel so that a triangle reaching outside the silhouette does not turn
black. Every target vertex is matched to a relief vertex (nearest neighbour in Diff3F feature
space, or a table given with --correspondence) and takes that vertex's texture coordinates. Only
the side of the target that faces the relief's camera is textured; the faces it cannot see stay
blank (light grey), as the supervisor asked, unless --both_sides is given. The textured target is
written as OBJ + MTL + PNG (open in MeshLab) and drawn with a small software rasteriser, since
matplotlib cannot sample a texture inside a triangle.

Consistency: a triangle whose three corners are matched to distant image points shows a stretched
streak of the image. The script reports the fraction of textured triangles whose corners lie within
2 % and 5 % of the image diagonal of each other and the fraction of the relief's image that is used
at all (the span alone rewards a matching that collapses onto a few points), and writes the span of
every triangle as a face colour (target_uv_span.ply: green small, red large). A better
correspondence shows up directly as higher fractions.

Usage (repository root, mesh environment):
    python scripts/idarb/transfer_texture.py \
        --source outputs/diff3f/features/cat_partial.npz \
        --source_mesh outputs/mesh_depth/cat/relief.ply \
        --target outputs/diff3f/features/lion_full.npz \
        --target_mesh third_party/Diff3F/meshes/lion.off \
        --transform results/diff3f/cat_lion/transform.json \
        --out_dir outputs/diff3f/03_texture_nearest_neighbour/same_category/cat_lion
"""

import argparse
import json
import os
import sys

import numpy as np
import trimesh
from PIL import Image
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import register_partial as rp  # noqa: E402
from correspondence_maps import mesh_features, nearest_in_features  # noqa: E402

BLANK = (217, 217, 217)  # the colour of the faces that are not textured
BLANK_ROWS = 16  # rows of that colour added below the relief's image


def relief_grid(V, tol=0.05):
    """Pixel (column, row) of every relief vertex, recovered from the regular grid the vertices sit
    on. Raises ValueError when the mesh is not a pixel grid (a synthetic or ordinary mesh)."""
    cols, rows = [], []
    for axis, out, flip in ((0, cols, False), (1, rows, True)):
        u = np.unique(np.round(V[:, axis], 6))
        step = np.diff(u).min()
        if flip:
            c = (V[:, axis].max() - V[:, axis]) / step
        else:
            c = (V[:, axis] - V[:, axis].min()) / step
        if np.abs(c - np.round(c)).max() > tol:
            raise ValueError("source mesh vertices do not sit on a pixel grid")
        out.append(np.round(c).astype(np.int64))
    return cols[0], rows[0]


def relief_frame(V):
    """The pixel grid behind a relief: x of column 0, y of row 0 and the pixel step."""
    col, row = relief_grid(V)
    step = float((V[:, 0].max() - V[:, 0].min()) / max(col.max(), 1))
    return float(V[:, 0].min()), float(V[:, 1].max()), step


def relief_uv_of_points(P, V, image_shape, blank_rows=BLANK_ROWS, scale=1):
    """Texture coordinates of arbitrary source-space points P (n,3) over the relief's image, the
    same convention as relief_texture's uv for the vertices themselves. A point beyond the relief's
    pixel grid (a field can overshoot it) is clamped to the grid, so it never lands in the blank
    strip below the image. `scale` is the texture's pixels per relief pixel (a higher-resolution
    albedo laid over the same grid)."""
    x0, y1, step = relief_frame(V)
    Ht, Wt = image_shape[:2]
    W, H = Wt // scale, (Ht - blank_rows) // scale
    col = np.clip((P[:, 0] - x0) / step, 0, W - 1)
    row = np.clip((y1 - P[:, 1]) / step, 0, H - 1)
    return np.stack([(col + 0.5) / W, 1.0 - (row + 0.5) * scale / Ht], axis=1)


def find_grid_offset(V, C, albedo, search=14, white=245):
    """Where the relief's pixel grid sits inside the decomposition image it was built from: the
    (row, column) of the grid's first pixel, found by matching the relief's vertex colours against
    the image around the object's bounding box. Returns (row, col, mean colour difference)."""
    col, row = relief_grid(V)
    W, H = int(col.max()) + 1, int(row.max()) + 1
    G = np.zeros((H, W, 3))
    M = np.zeros((H, W), dtype=bool)
    G[row, col] = C[:, :3]
    M[row, col] = True
    ys, xs = np.nonzero((albedo[:, :, :3] < white).any(axis=2))
    dy0, dx0 = int(ys.min()), int(xs.min())
    best = (None, None, np.inf)
    A = albedo[:, :, :3].astype(np.float64)
    for dy in range(max(dy0 - search, 0), dy0 + search + 1):
        for dx in range(max(dx0 - search, 0), dx0 + search + 1):
            if dy + H > A.shape[0] or dx + W > A.shape[1]:
                continue
            d = float(np.abs(A[dy : dy + H, dx : dx + W] - G)[M].mean())
            if d < best[2]:
                best = (dy, dx, d)
    return best


def hires_texture(V, C, albedo_hr, offset, scale, blank_rows=BLANK_ROWS):
    """relief_texture with a higher-resolution albedo image laid over the relief's pixel grid:
    `albedo_hr` is `scale` times the decomposition image the relief came from and `offset` is
    the grid's (row, col) in that image. Returns image, uv, blank_uv as relief_texture does."""
    col, row = relief_grid(V)
    W, H = int(col.max()) + 1, int(row.max()) + 1
    dy, dx = offset
    crop = albedo_hr[scale * dy : scale * (dy + H), scale * dx : scale * (dx + W), :3].copy()
    M = np.zeros((H, W), dtype=bool)
    M[row, col] = True
    Mhr = np.kron(M, np.ones((scale, scale), dtype=bool))
    fg = np.stack(np.nonzero(Mhr), axis=1)
    bg = np.stack(np.nonzero(~Mhr), axis=1)
    if len(bg):
        _, nn = cKDTree(fg).query(bg)
        crop[bg[:, 0], bg[:, 1]] = crop[fg[nn, 0], fg[nn, 1]]
    image = np.concatenate([crop, np.full((blank_rows, scale * W, 3), BLANK, dtype=np.uint8)])
    Ht = image.shape[0]
    uv = np.stack([(col + 0.5) / W, 1.0 - (row + 0.5) * scale / Ht], axis=1)
    blank_uv = np.array([0.5, (blank_rows / 2.0) / Ht])
    return image, uv, blank_uv


def relief_texture(V, C, blank_rows=BLANK_ROWS):
    """The albedo image behind a relief mesh (uint8 HxWx3, background filled from the nearest
    foreground pixel, plus `blank_rows` of light grey at the bottom for the untextured faces), the
    texture coordinates (N,2) of its vertices (u right, v up) and the coordinates of the blank
    strip's centre (None without a strip)."""
    col, row = relief_grid(V)
    W, H = int(col.max()) + 1, int(row.max()) + 1
    image = np.zeros((H, W, 3), dtype=np.uint8)
    image[row, col] = C[:, :3]
    fg = np.zeros((H, W), dtype=bool)
    fg[row, col] = True
    bg_r, bg_c = np.nonzero(~fg)
    if len(bg_r):
        _, nn = cKDTree(np.stack([row, col], axis=1)).query(np.stack([bg_r, bg_c], axis=1))
        image[bg_r, bg_c] = C[nn, :3]
    blank_uv = None
    if blank_rows:
        image = np.concatenate([image, np.full((blank_rows, W, 3), BLANK, dtype=np.uint8)])
        blank_uv = np.array([0.5, (blank_rows / 2.0) / image.shape[0]])
    Ht = image.shape[0]
    uv = np.stack([(col + 0.5) / W, 1.0 - (row + 0.5) / Ht], axis=1)
    return image, uv, blank_uv


def uv_span(uvc, image_shape):
    """Per triangle, the largest distance between its corners' texture positions (uvc: (M,3,2)), as
    a fraction of the image diagonal."""
    H, W = image_shape[:2]
    P = uvc * np.array([W, H])
    d01 = np.linalg.norm(P[:, 0] - P[:, 1], axis=1)
    d12 = np.linalg.norm(P[:, 1] - P[:, 2], axis=1)
    d20 = np.linalg.norm(P[:, 2] - P[:, 0], axis=1)
    return np.maximum(np.maximum(d01, d12), d20) / np.hypot(W, H)


def source_usage(uv_d, uv_s, grid=32):
    """Fraction of the source image's foreground cells (a grid x grid lattice over the texture) that
    at least one target vertex is matched into: near 1 when the whole relief is used, small when
    the matching collapses onto a few source points. The corner span alone rewards a collapse (all
    corners on one point span nothing), so the two numbers are read together."""
    cell_s = np.unique(np.floor(np.clip(uv_s, 0, 0.999999) * grid).astype(np.int64) @ [grid, 1])
    cell_d = np.unique(np.floor(np.clip(uv_d, 0, 0.999999) * grid).astype(np.int64) @ [grid, 1])
    return float(len(np.intersect1d(cell_s, cell_d)) / max(len(cell_s), 1))


def view_frame(view, up=(0.0, 1.0, 0.0)):
    """Rows: screen right, screen up, towards the viewer (the frame of render_registration.draw)."""
    d = np.asarray(view, dtype=np.float64)
    d = d / np.linalg.norm(d)
    r = np.cross(up, d)
    if np.linalg.norm(r) < 1e-6:
        r = np.cross((1.0, 0.0, 0.0), d)
    r /= np.linalg.norm(r)
    u = np.cross(d, r)
    return np.stack([r, u, d])


def project(V, view, size, fill=0.9):
    """Orthographic projection to pixel coordinates (x right, y down, image rows) and a depth that
    grows towards the viewer."""
    P = V @ view_frame(view).T
    lo, hi = P[:, :2].min(axis=0), P[:, :2].max(axis=0)
    s = size * fill / max(float((hi - lo).max()), 1e-9)
    xy = (P[:, :2] - (lo + hi) / 2) * s + size / 2
    xy[:, 1] = size - xy[:, 1]
    return xy, P[:, 2]


def raster_buffers(V, F, view, size=640, fill=0.9):
    """Depth-buffered rasterisation: per pixel the face in front (-1 for none), its first two
    barycentric coordinates and its depth."""
    xy, z = project(V, view, size, fill)
    fid = np.full((size, size), -1, dtype=np.int64)
    W0 = np.zeros((size, size))
    W1 = np.zeros((size, size))
    zbuf = np.full((size, size), -np.inf)
    tri = xy[F]
    for k in range(len(F)):
        (ax, ay), (bx, by), (cx, cy) = tri[k]
        area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
        if abs(area) < 1e-12:
            continue
        x0 = max(int(np.floor(min(ax, bx, cx))), 0)
        x1 = min(int(np.ceil(max(ax, bx, cx))), size - 1)
        y0 = max(int(np.floor(min(ay, by, cy))), 0)
        y1 = min(int(np.ceil(max(ay, by, cy))), size - 1)
        if x1 < x0 or y1 < y0:
            continue
        xs, ys = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
        px, py = xs + 0.5, ys + 0.5
        w0 = ((bx - px) * (cy - py) - (by - py) * (cx - px)) / area
        w1 = ((cx - px) * (ay - py) - (cy - py) * (ax - px)) / area
        w2 = 1.0 - w0 - w1
        i, j, m = F[k]
        depth = w0 * z[i] + w1 * z[j] + w2 * z[m]
        hit = (w0 >= 0) & (w1 >= 0) & (w2 >= 0) & (depth > zbuf[ys, xs])
        if not hit.any():
            continue
        fid[ys[hit], xs[hit]] = k
        W0[ys[hit], xs[hit]] = w0[hit]
        W1[ys[hit], xs[hit]] = w1[hit]
        zbuf[ys[hit], xs[hit]] = depth[hit]
    return dict(fid=fid, w0=W0, w1=W1, z=zbuf, xy=xy, zv=z)


def face_shade(V, F, view, two_sided=True):
    """Brightness by orientation, as render_registration.draw: 0.55 for a face seen edge-on, 1 for
    one facing the viewer."""
    P = V @ view_frame(view).T
    n = np.cross(P[F][:, 1] - P[F][:, 0], P[F][:, 2] - P[F][:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    nz = np.abs(n[:, 2]) if two_sided else np.clip(n[:, 2], 0.0, 1.0)
    return 0.55 + 0.45 * nz, n[:, 2]


def rasterize(V, F, uvc, image, view, size=640, fill=0.9):
    """Software render of a textured mesh seen from direction `view`: nearest-texel sampling
    inside every triangle with per-corner texture coordinates uvc (M,3,2) (a (N,2) per-vertex
    array is accepted too), a depth buffer and brightness by orientation. Returns float RGB
    (size, size, 3) on a white background."""
    uvc = np.asarray(uvc, dtype=np.float64)
    if uvc.ndim == 2:
        uvc = uvc[F]
    buf = raster_buffers(V, F, view, size, fill)
    shade, _ = face_shade(V, F, view)
    H, W = image.shape[:2]
    tex = image[:, :, :3].astype(np.float64) / 255.0
    out = np.ones((size, size, 3))
    hit = buf["fid"] >= 0
    k = buf["fid"][hit]
    w0, w1 = buf["w0"][hit], buf["w1"][hit]
    w2 = 1.0 - w0 - w1
    u = w0 * uvc[k, 0, 0] + w1 * uvc[k, 1, 0] + w2 * uvc[k, 2, 0]
    v = w0 * uvc[k, 0, 1] + w1 * uvc[k, 1, 1] + w2 * uvc[k, 2, 1]
    col = np.clip((u * W).astype(np.int64), 0, W - 1)
    row = np.clip(((1.0 - v) * H).astype(np.int64), 0, H - 1)
    out[hit] = tex[row, col] * shade[k][:, None]
    return out


def visible_faces(V, F, view, size=1024, depth_tol=0.01):
    """Faces seen from direction `view`: those that win a pixel of the depth buffer, plus the
    front-facing ones whose centroid is no deeper than the buffer at its pixel (small faces that
    fell between pixel centres)."""
    buf = raster_buffers(V, F, view, size)
    vis = np.zeros(len(F), dtype=bool)
    seen = np.unique(buf["fid"][buf["fid"] >= 0])
    vis[seen] = True
    xy, z = buf["xy"], buf["zv"]
    c = xy[F].mean(axis=1)
    cz = z[F].mean(axis=1)
    px = np.clip(np.floor(c[:, 0]).astype(np.int64), 0, size - 1)
    py = np.clip(np.floor(c[:, 1]).astype(np.int64), 0, size - 1)
    _, nz = face_shade(V, F, view)
    tol = depth_tol * float(z.max() - z.min())
    # the centroid test only for the small faces that fell between pixel centres, so that a
    # hidden sheet just behind the surface is not declared visible wholesale
    t = xy[F]
    area = 0.5 * np.abs(
        (t[:, 1, 0] - t[:, 0, 0]) * (t[:, 2, 1] - t[:, 0, 1])
        - (t[:, 1, 1] - t[:, 0, 1]) * (t[:, 2, 0] - t[:, 0, 0])
    )
    vis |= ~vis & (area < 4.0) & (nz > 0) & (cz >= buf["z"][py, px] - tol)
    return vis


def export_textured_obj(path, V, F, uv, image, textured=None, blank_uv=None):
    """OBJ with per-corner texture coordinates, its MTL and the texture PNG next to it. Faces with
    textured=False point all three corners at blank_uv (the grey strip of the texture)."""
    stem = os.path.splitext(os.path.basename(path))[0]
    out_dir = os.path.dirname(os.path.abspath(path))
    Image.fromarray(image).save(os.path.join(out_dir, stem + ".png"))
    with open(os.path.join(out_dir, stem + ".mtl"), "w") as f:
        f.write(f"newmtl {stem}\nKa 1.0 1.0 1.0\nKd 1.0 1.0 1.0\nKs 0.0 0.0 0.0\n")
        f.write(f"map_Kd {stem}.png\n")
    n = len(V)
    vt = np.asarray(uv, dtype=np.float64)
    if textured is None:
        textured = np.ones(len(F), dtype=bool)
    if blank_uv is None and not textured.all():
        raise ValueError("blank faces need a blank_uv to point at")
    if blank_uv is not None:
        vt = np.concatenate([vt, np.asarray(blank_uv, dtype=np.float64)[None]])
    corner = np.repeat(F + 1, 1, axis=0)  # 1-based vertex indices
    tex_idx = np.where(textured[:, None], F + 1, n + 1)  # blank faces use the extra vt
    lines = [f"mtllib {stem}.mtl", f"usemtl {stem}"]
    lines += [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in V]
    lines += [f"vt {u:.6f} {v:.6f}" for u, v in vt]
    lines += [
        f"f {a}/{ta} {b}/{tb} {c}/{tc}"
        for (a, b, c), (ta, tb, tc) in zip(corner.tolist(), tex_idx.tolist())
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--source", required=True, help="features of the relief (.npz)")
    ap.add_argument("--source_mesh", required=True, help="the relief mesh with albedo colours")
    ap.add_argument("--target", required=True, help="features of the full mesh (.npz)")
    ap.add_argument("--target_mesh", required=True)
    ap.add_argument(
        "--correspondence",
        help="npz with t2s (a source mesh vertex per target vertex); default: nearest neighbour "
        "in feature space",
    )
    ap.add_argument("--transform", help="transform.json of the registration (view and coverage)")
    ap.add_argument("--feature", choices=["full", "diffusion", "dino"], default="full")
    ap.add_argument("--view", type=float, nargs=3, default=(0.0, 0.0, 1.0))
    ap.add_argument("--target_view", type=float, nargs=3, default=None)
    ap.add_argument(
        "--both_sides", action="store_true", help="texture every face, not only the seen side"
    )
    ap.add_argument("--size", type=int, default=640, help="render size in pixels")
    ap.add_argument(
        "--albedo_image",
        help="a higher-resolution albedo of the same photo (same framing as the decomposition "
        "the relief came from) to use as the texture instead of the relief's own colours",
    )
    ap.add_argument(
        "--albedo_panel",
        help="the decomposition image the relief was built from (four panels or the albedo "
        "alone), to locate the relief's pixel grid in it; needed with --albedo_image",
    )
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    Vs, Fs, fs, cs = mesh_features(args.source, args.source_mesh, args.feature)
    Vd, Fd, fd, _ = mesh_features(args.target, args.target_mesh, args.feature)
    if cs is None:
        sys.exit("the source mesh has no colours; texture transfer needs the relief with albedo")
    scale = 1
    if args.albedo_image:
        panel = np.asarray(Image.open(args.albedo_panel).convert("RGB"))
        if panel.shape[1] == 4 * panel.shape[0]:  # four panels: albedo is the first
            panel = panel[:, : panel.shape[0]]
        hr = np.asarray(Image.open(args.albedo_image).convert("RGB"))
        if hr.shape[1] == 4 * hr.shape[0]:
            hr = hr[:, : hr.shape[0]]
        scale = hr.shape[1] // panel.shape[1]
        if scale * panel.shape[1] != hr.shape[1] or scale * panel.shape[0] != hr.shape[0]:
            sys.exit("--albedo_image must be an integer multiple of --albedo_panel in size")
        dy, dx, diff = find_grid_offset(Vs, cs, panel)
        print(f"relief grid at row {dy}, column {dx} of the panel (colour difference {diff:.2f})")
        if diff > 8:
            sys.exit("the relief does not match the panel: wrong --albedo_panel?")
        image, uv_s, blank_uv = hires_texture(Vs, cs, hr, (dy, dx), scale)
    else:
        image, uv_s, blank_uv = relief_texture(Vs, cs)
    Image.fromarray(image).save(os.path.join(args.out_dir, "source_texture.png"))
    print(
        f"source {len(Vs)} vertices on a {image.shape[1]}x{image.shape[0]} image, "
        f"target {len(Vd)} vertices"
    )

    src_pos = None
    if args.correspondence:
        z = np.load(args.correspondence)
        t2s = z["t2s"].astype(np.int64)
        if "src_pos" in z.files:  # a continuous field: texture coordinates straight from it
            src_pos = z["src_pos"].astype(np.float64)
    else:
        t2s = nearest_in_features(fd, fs)
    if src_pos is None:
        uv_d = uv_s[t2s]
    else:
        uv_d = relief_uv_of_points(src_pos, Vs, image.shape, scale=scale)
    uv_d = np.clip(uv_d, 0.0, 1.0)

    # the target's viewing direction: given, from the registration, or the source's
    tview = args.target_view
    stats = dict(faces=int(len(Fd)))
    if args.transform:
        with open(args.transform) as f:
            tr = json.load(f)
        R = np.array(tr["rotation"], dtype=np.float64)
        aligned = rp.transform(
            rp.apply_depth(Vs, tr.get("depth_factor", 1.0)),
            tr["scale"],
            R,
            np.array(tr["translation"]),
        )
        diag = float(np.linalg.norm(Vd.max(axis=0) - Vd.min(axis=0)))
        covered_v = cKDTree(aligned).query(Vd)[0] < 0.05 * diag
        covered_f = covered_v[Fd].all(axis=1)
        if tview is None:
            tview = R[:, 2]
    if tview is None:
        tview = (0.0, 0.0, 1.0)
    tview = np.asarray(tview, dtype=np.float64)

    # one side: only the faces the relief's camera sees are textured
    if args.both_sides:
        textured = np.ones(len(Fd), dtype=bool)
    else:
        textured = visible_faces(Vd, Fd, tview)
    uvc = uv_d[Fd].copy()
    uvc[~textured] = blank_uv
    export_textured_obj(
        os.path.join(args.out_dir, "target_textured.obj"), Vd, Fd, uv_d, image, textured, blank_uv
    )

    # consistency of the corners of every textured triangle, and how much of the relief is used
    span_all = uv_span(uv_d[Fd], image.shape)
    span = span_all[textured]
    stats.update(
        textured_faces=int(textured.sum()),
        span_median=round(float(np.median(span)), 4),
        within_2pct=round(float((span < 0.02).mean()), 4),
        within_5pct=round(float((span < 0.05).mean()), 4),
        source_used=round(source_usage(uvc[textured].reshape(-1, 2), uv_s), 4),
        all_within_2pct=round(float((span_all < 0.02).mean()), 4),
        all_within_5pct=round(float((span_all < 0.05).mean()), 4),
        all_source_used=round(source_usage(uv_d, uv_s), 4),
    )
    if args.transform:
        sel = textured & covered_f
        stats.update(
            covered_faces=int(sel.sum()),
            covered_within_2pct=round(float((span_all[sel] < 0.02).mean()), 4),
            covered_within_5pct=round(float((span_all[sel] < 0.05).mean()), 4),
        )
    print(stats)
    with open(os.path.join(args.out_dir, "texture_transfer.json"), "w") as f:
        json.dump(dict(stats, source=args.source, target=args.target), f, indent=1)
    t = np.clip(span_all / 0.1, 0, 1)[:, None]
    fc = (np.array([[60, 200, 60]]) * (1 - t) + np.array([[220, 40, 40]]) * t).astype(np.uint8)
    fc[~textured] = BLANK
    trimesh.Trimesh(vertices=Vd, faces=Fd, face_colors=fc, process=False).export(
        os.path.join(args.out_dir, "target_uv_span.ply")
    )

    # renders: the relief through its own texture, the target from the relief's side and from behind
    front = rasterize(Vd, Fd, uvc, image, tview, args.size)
    back = rasterize(Vd, Fd, uvc, image, -tview, args.size)
    panels = [
        ("source: albedo texture", rasterize(Vs, Fs, uv_s, image, args.view, args.size)),
        ("target: texture through the correspondence", front),
        ("target: the side the relief never saw", back),
        ("target: corner span (green < 2 %, red >= 10 %)", None),
    ]
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from render_registration import draw

    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.4))
    for ax, (title, img) in zip(axes, panels):
        if img is None:
            draw(ax, [(Vd, Fd, fc[:, None, :].repeat(3, axis=1).reshape(-1, 3) / 255.0)], tview)
        else:
            ax.imshow(img)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    out = os.path.join(args.out_dir, "texture_transfer.png")
    fig.savefig(out, dpi=90)
    print("saved", out)


if __name__ == "__main__":
    main()
