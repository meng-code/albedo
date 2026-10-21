"""
Texture transfer through the semantic correspondence: the full target mesh gets texture coordinates
into the relief's albedo image, so the inside of a triangle shows the image itself and not a blend
of three vertex colours.

The relief is a height field over the albedo image: every relief vertex is one image pixel, so the
relief's texture coordinates are its pixel positions and its texture is the albedo image. The image
is rebuilt here from the relief's vertex colours on their pixel grid, with the background filled
from the nearest foreground pixel so that a triangle reaching outside the silhouette does not turn
black. Every target vertex is matched to a relief vertex (nearest neighbour in Diff3F feature
space, or a table given with --correspondence) and takes that vertex's texture coordinates. The
textured target is written as OBJ + MTL + PNG (open in MeshLab) and drawn with a small software
rasteriser, since matplotlib cannot sample a texture inside a triangle.

Consistency: a triangle whose three corners are matched to distant image points shows a stretched
streak of the image. The script reports the fraction of triangles whose corners lie within 2 % and
5 % of the image diagonal of each other (all triangles, and only those the registration says the
relief covers), and writes the span of every triangle as a face colour (target_uv_span.ply: green
small, red large). A better correspondence shows up directly as a higher fraction.

Usage (repository root, mesh environment):
    python scripts/idarb/transfer_texture.py --source outputs/diff3f/cat_partial.npz \
        --source_mesh outputs/mesh_depth/cat/relief.ply --target outputs/diff3f/lion_full.npz \
        --target_mesh third_party/Diff3F/meshes/lion.off \
        --transform results/diff3f/cat_lion/transform.json --out_dir outputs/diff3f/maps_cat_lion
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


def relief_texture(V, C):
    """The albedo image behind a relief mesh (uint8 HxWx3, background filled from the nearest
    foreground pixel) and the texture coordinates (N,2) of its vertices, u right and v up."""
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
    uv = np.stack([(col + 0.5) / W, 1.0 - (row + 0.5) / H], axis=1)
    return image, uv


def uv_span(F, uv, image_shape):
    """Per triangle, the largest distance between its corners' texture positions, as a fraction of
    the image diagonal."""
    H, W = image_shape[:2]
    P = uv[F] * np.array([W, H])
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


def rasterize(V, F, uv, image, view, size=640, fill=0.9, two_sided=True):
    """Orthographic software render of a textured mesh seen from direction `view`: nearest-texel
    sampling inside every triangle, a depth buffer, and the same brightness-by-orientation shading
    as render_registration.draw. Returns float RGB (size, size, 3) on a white background."""
    P = V @ view_frame(view).T
    lo, hi = P[:, :2].min(axis=0), P[:, :2].max(axis=0)
    s = size * fill / max(float((hi - lo).max()), 1e-9)
    xy = (P[:, :2] - (lo + hi) / 2) * s + size / 2
    xy[:, 1] = size - xy[:, 1]  # image rows go down
    z = P[:, 2]
    H, W = image.shape[:2]
    tex = image[:, :, :3].astype(np.float64) / 255.0
    out = np.ones((size, size, 3))
    zbuf = np.full((size, size), -np.inf)
    tri = xy[F]
    n = np.cross(P[F][:, 1] - P[F][:, 0], P[F][:, 2] - P[F][:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    nz = np.abs(n[:, 2]) if two_sided else np.clip(n[:, 2], 0.0, 1.0)
    shade = 0.55 + 0.45 * nz
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
        u = w0 * uv[i, 0] + w1 * uv[j, 0] + w2 * uv[m, 0]
        v = w0 * uv[i, 1] + w1 * uv[j, 1] + w2 * uv[m, 1]
        col = np.clip((u * W).astype(np.int64), 0, W - 1)
        row = np.clip(((1.0 - v) * H).astype(np.int64), 0, H - 1)
        out[ys[hit], xs[hit]] = tex[row[hit], col[hit]] * shade[k]
        zbuf[ys[hit], xs[hit]] = depth[hit]
    return out


def export_textured_obj(path, V, F, uv, image):
    """OBJ with texture coordinates, its MTL and the texture PNG next to it."""
    mesh = trimesh.Trimesh(
        vertices=V,
        faces=F,
        visual=trimesh.visual.TextureVisuals(uv=uv, image=Image.fromarray(image)),
        process=False,
    )
    stem = os.path.splitext(os.path.basename(path))[0]
    mesh.visual.material.name = stem
    out_dir = os.path.dirname(os.path.abspath(path))
    text = trimesh.exchange.obj.export_obj(
        mesh,
        include_texture=True,
        write_texture=True,
        resolver=trimesh.resolvers.FilePathResolver(out_dir),
        mtl_name=stem + ".mtl",
    )
    with open(path, "w") as f:
        f.write(text)


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
    ap.add_argument("--size", type=int, default=640, help="render size in pixels")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    Vs, Fs, fs, cs = mesh_features(args.source, args.source_mesh, args.feature)
    Vd, Fd, fd, _ = mesh_features(args.target, args.target_mesh, args.feature)
    if cs is None:
        sys.exit("the source mesh has no colours; texture transfer needs the relief with albedo")
    image, uv_s = relief_texture(Vs, cs)
    Image.fromarray(image).save(os.path.join(args.out_dir, "source_texture.png"))
    print(
        f"source {len(Vs)} vertices on a {image.shape[1]}x{image.shape[0]} image, "
        f"target {len(Vd)} vertices"
    )

    if args.correspondence:
        t2s = np.load(args.correspondence)["t2s"].astype(np.int64)
    else:
        t2s = nearest_in_features(fd, fs)
    uv_d = uv_s[t2s]
    export_textured_obj(os.path.join(args.out_dir, "target_textured.obj"), Vd, Fd, uv_d, image)

    # consistency of the three corners of every triangle
    span = uv_span(Fd, uv_d, image.shape)
    stats = dict(
        faces=int(len(Fd)),
        span_median=round(float(np.median(span)), 4),
        within_2pct=round(float((span < 0.02).mean()), 4),
        within_5pct=round(float((span < 0.05).mean()), 4),
        source_used=round(source_usage(uv_d, uv_s), 4),
    )
    tview = args.target_view
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
        stats.update(
            covered_faces=int(covered_f.sum()),
            covered_within_2pct=round(float((span[covered_f] < 0.02).mean()), 4),
            covered_within_5pct=round(float((span[covered_f] < 0.05).mean()), 4),
        )
        if tview is None:
            tview = R[:, 2]
    if tview is None:
        tview = (0.0, 0.0, 1.0)
    tview = np.asarray(tview, dtype=np.float64)
    print(stats)
    with open(os.path.join(args.out_dir, "texture_transfer.json"), "w") as f:
        json.dump(dict(stats, source=args.source, target=args.target), f, indent=1)
    t = np.clip(span / 0.1, 0, 1)[:, None]
    fc = (np.array([[60, 200, 60]]) * (1 - t) + np.array([[220, 40, 40]]) * t).astype(np.uint8)
    trimesh.Trimesh(vertices=Vd, faces=Fd, face_colors=fc, process=False).export(
        os.path.join(args.out_dir, "target_uv_span.ply")
    )

    # renders: the relief through its own texture, the target from the relief's side and from behind
    front = rasterize(Vd, Fd, uv_d, image, tview, args.size)
    back = rasterize(Vd, Fd, uv_d, image, -tview, args.size)
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
