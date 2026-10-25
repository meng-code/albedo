"""
Matching on the 2D side. The relief's pixels are the photo's pixels, so instead of lifting
features onto the relief, the photo (the relief's albedo image) is matched directly against
renders of the target seen from the relief camera's direction and a few directions around it,
with a 2D vision model's patch features (DINOv2 or DINOv3): the photo keeps its full
resolution and its real eyes and noses, and rendering the target from the photo's side makes
the visible sides correspond, which removes most left-right confusion. Every render pixel that
finds a mutual match in the photo is lifted through the rasteriser (face and barycentric
coordinates) to a target vertex, and the photo pixel is a relief vertex; the best match per
target vertex over the views is kept.

Output: an npz in the layout refine_correspondence.py writes (t2s per target vertex, src_pos,
view) plus the matches themselves (t_idx, s_idx, sim, view_idx, matched mask); t2s / src_pos
of unmatched vertices are filled from the nearest matched vertex. Optionally a landmarks JSON
(--landmarks_out, the most similar mutual matches spread over the target) that
refine_correspondence.py --landmarks accepts, and a preview image.

Local mode (--prior): instead of a global match, every render cell looks for its photo cell
only within --window (fraction of the photo crop's side) of where a prior correspondence (a
refine_correspondence.py npz, src_pos per target vertex) already puts its target vertex, and
the vertices the renders do not reach keep the prior. This is the fine step of a coarse-to-fine
localisation: the 3D chain says roughly where, the photo's own high-resolution patch features
say exactly where.

Usage (repository root; needs torch, and the model weights on first use):
    python scripts/idarb/match2d.py --relief outputs/mesh_depth/cat/relief.ply \
        --target_mesh third_party/Diff3F/meshes/lion.off \
        --view_from outputs/diff3f/15_relief_zoom2/same/cat_lion/refined.npz \
        --extractor dinov2 --out outputs/diff3f/17_match2d/cat_lion/match2d.npz \
        --landmarks_out outputs/diff3f/17_match2d/cat_lion/landmarks.json \
        --preview outputs/diff3f/17_match2d/cat_lion/preview.png
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import transfer_texture as tt  # noqa: E402
from correspondence_maps import load_mesh  # noqa: E402

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def albedo_image(Vs, Cs):
    """The photo behind the relief: its albedo image (background filled from the nearest
    foreground pixel, like the texture) and the relief vertex of every pixel (-1 outside)."""
    image, _uv, _blank = tt.relief_texture(Vs, Cs, blank_rows=0)
    col, row = tt.relief_grid(Vs)
    vert = -np.ones(image.shape[:2], dtype=np.int64)
    vert[row, col] = np.arange(len(Vs))
    return image, vert


def square_crop(image, mask, margin=0.08, fill=255):
    """The image cropped to the mask's bounding box, padded to a square with `fill` and a margin
    on every side, so that the object fills the crop the same way in the photo and in the
    render. Returns the crop and its box (y0, x0, side): crop pixel (py, px) is image pixel
    (y0 + py, x0 + px)."""
    H, W = image.shape[:2]
    rows, cols = np.nonzero(mask)
    if len(rows) == 0:
        return image, (0, 0, max(H, W))
    y0, y1 = int(rows.min()), int(rows.max()) + 1
    x0, x1 = int(cols.min()), int(cols.max()) + 1
    side = int(np.ceil(max(y1 - y0, x1 - x0) * (1.0 + 2.0 * margin)))
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    oy, ox = cy - side // 2, cx - side // 2
    out = np.full((side, side) + image.shape[2:], fill, dtype=image.dtype)
    ys0, xs0 = max(oy, 0), max(ox, 0)
    ys1, xs1 = min(oy + side, H), min(ox + side, W)
    out[ys0 - oy:ys1 - oy, xs0 - ox:xs1 - ox] = image[ys0:ys1, xs0:xs1]
    return out, (oy, ox, side)


def cell_pixels(grid, box):
    """Image-pixel coordinates (rows, cols) of the centres of a feature grid laid over a crop."""
    h, w = grid
    oy, ox, side = box
    rows = oy + (np.arange(h) + 0.5) * side / h
    cols = ox + (np.arange(w) + 0.5) * side / w
    return rows, cols


def turned_views(view, turn=25.0, elev=0.0, up=(0.0, 1.0, 0.0)):
    """The view itself, then turned left and right about the up axis by `turn` degrees and, with
    elev > 0, up and down about the screen-right axis."""
    R = tt.view_frame(view, up)
    right, upv, d = R[0], R[1], R[2]

    def rot(axis, deg):
        a = np.deg2rad(deg)
        axis = axis / np.linalg.norm(axis)
        return (d * np.cos(a) + np.cross(axis, d) * np.sin(a)
                + axis * np.dot(axis, d) * (1 - np.cos(a)))

    views = [d]
    if turn > 0:
        views += [rot(upv, turn), rot(upv, -turn)]
    if elev > 0:
        views += [rot(right, elev), rot(right, -elev)]
    return [v / np.linalg.norm(v) for v in views]


def render_target(Vd, Fd, view, size=1024, colors=None):
    """A render of the target from `view` (orthographic, like the previews): a shaded grey
    image, or the per-vertex colours shaded, with the face and barycentric buffers for lifting.
    Returns image uint8 (size,size,3), fid (size,size), w0, w1."""
    buf = tt.raster_buffers(Vd, Fd, view, size)
    shade, _ = tt.face_shade(Vd, Fd, view)
    img = np.full((size, size, 3), 255, dtype=np.uint8)
    hit = buf["fid"] >= 0
    k = buf["fid"][hit]
    if colors is None:
        g = np.clip(shade[k] * 200, 0, 255)
        img[hit] = np.stack([g, g, g], axis=1).astype(np.uint8)
    else:
        C = np.asarray(colors, dtype=np.float64)[:, :3]
        w0, w1 = buf["w0"][hit], buf["w1"][hit]
        w2 = 1.0 - w0 - w1
        c = w0[:, None] * C[Fd[k, 0]] + w1[:, None] * C[Fd[k, 1]] + w2[:, None] * C[Fd[k, 2]]
        img[hit] = np.clip(c * shade[k][:, None], 0, 255).astype(np.uint8)
    return img, buf["fid"], buf["w0"], buf["w1"]


DINOV3_REPO = "facebook/dinov3-vitb16-pretrain-lvd1689m"


def make_extractor(name, device="cuda", resolution=518, dinov3_repo=DINOV3_REPO):
    """A function image (H,W,3) uint8 -> unit patch features (h,w,D) float32 from DINOv2
    (torch.hub, patch 14) or DINOv3 (transformers, patch 16)."""
    import torch

    if name == "dinov2":
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14").to(device).eval()
        patch = 14

        def tokens(x):
            return model.forward_features(x)["x_norm_patchtokens"]
    elif name == "dinov3":
        from transformers import AutoModel

        model = AutoModel.from_pretrained(dinov3_repo).to(device).eval()
        patch = 16
        n_special = 1 + getattr(model.config, "num_register_tokens", 4)

        def tokens(x):
            return model(pixel_values=x).last_hidden_state[:, n_special:]
    else:
        raise ValueError(name)
    res = (resolution // patch) * patch

    def extract(image):
        from PIL import Image

        im = Image.fromarray(image).convert("RGB").resize((res, res), Image.BICUBIC)
        x = (np.asarray(im, dtype=np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        x = torch.from_numpy(x).permute(2, 0, 1)[None].to(device)
        with torch.no_grad():
            f = tokens(x)[0].float().cpu().numpy()
        h = w = res // patch
        f = f.reshape(h, w, -1)
        return f / np.maximum(np.linalg.norm(f, axis=2, keepdims=True), 1e-9)

    return extract


def match_grids(fp, mp, fr, mr, temperature=0.05):
    """Render cells against photo cells. fp (hp,wp,D) and fr (hr,wr,D) unit features, mp / mr
    the cells that lie on the object. For every render cell on the object: its best photo cell,
    the photo position refined by a soft-argmax over the 3x3 window around it (cell units,
    (row, col) of the cell centre), the similarity, and whether the match is mutual (the photo
    cell's best render cell lies within one cell of it). Returns render (r,c), photo (y,x), sim,
    mutual, each (M,...)."""
    hp, wp = mp.shape
    hr, wr = mr.shape
    rr, rc = np.nonzero(mr)
    pr, pc = np.nonzero(mp)
    if len(rr) == 0 or len(pr) == 0:
        z = np.zeros((0,), dtype=np.int64)
        return np.zeros((0, 2), dtype=np.int64), np.zeros((0, 2)), z.astype(float), z.astype(bool)
    A = fr[rr, rc]  # (M,D)
    B = fp[pr, pc]  # (N,D)
    sim = A @ B.T
    best = sim.argmax(axis=1)
    back = sim.argmax(axis=0)  # best render cell of every photo cell
    br = np.stack([rr[back[best]], rc[back[best]]], axis=1)
    mutual = np.abs(br - np.stack([rr, rc], axis=1)).max(axis=1) <= 1
    # soft-argmax in the 3x3 window around the best photo cell
    photo = np.zeros((len(rr), 2))
    full = np.full((hp, wp), -np.inf)
    cell_index = -np.ones((hp, wp), dtype=np.int64)
    cell_index[pr, pc] = np.arange(len(pr))
    for i in range(len(rr)):
        r0, c0 = pr[best[i]], pc[best[i]]
        rs = slice(max(r0 - 1, 0), min(r0 + 2, hp))
        cs = slice(max(c0 - 1, 0), min(c0 + 2, wp))
        idx = cell_index[rs, cs]
        full[rs, cs] = np.where(idx >= 0, sim[i, np.maximum(idx, 0)], -np.inf)
        win = full[rs, cs]
        w = np.exp((win - win.max()) / temperature)
        w[~np.isfinite(win)] = 0.0
        w /= w.sum()
        gy, gx = np.mgrid[rs, cs]
        photo[i] = [(w * gy).sum() + 0.5, (w * gx).sum() + 0.5]
        full[rs, cs] = -np.inf
    return np.stack([rr, rc], axis=1), photo, sim[np.arange(len(rr)), best], mutual


def match_grids_local(fp, mp, fr, mr, prior_rc, radius, temperature=0.05):
    """Render cells against the photo cells within `radius` cells of their prior position
    (prior_rc (hr, wr, 2): the prior photo cell of every render cell, NaN where none). Returns
    render (r,c), photo (y,x), sim and the move from the prior for the cells that matched."""
    rr, rc = np.nonzero(mr & np.isfinite(prior_rc[..., 0]))
    photo, sims, moved, keep = search_window(fp, mp, fr[rr, rc], prior_rc[rr, rc], radius,
                                             temperature)
    return np.stack([rr, rc], axis=1)[keep], photo[keep], sims[keep], moved[keep]


def search_window(fp, mp, feats, prior_rc, radius, temperature=0.05):
    """For every query feature (M,D) with a prior photo cell (M,2: row, col, float), the best
    photo cell on the object within `radius` cells, refined by the 3x3 soft-argmax. Returns the
    photo positions (M,2), similarities (M,), moves from the prior in cells (M,) and a mask of
    the queries that found a cell."""
    hp, wp = mp.shape
    m = len(feats)
    photo = np.zeros((m, 2))
    sims = np.full(m, -np.inf)
    moved = np.zeros(m)
    keep = np.zeros(m, dtype=bool)
    r = int(np.ceil(radius))
    for i in range(m):
        pr, pc = prior_rc[i]
        if not np.isfinite(pr):
            continue
        r0, r1 = max(int(np.floor(pr - r)), 0), min(int(np.ceil(pr + r)) + 1, hp)
        c0, c1 = max(int(np.floor(pc - r)), 0), min(int(np.ceil(pc + r)) + 1, wp)
        if r1 <= r0 or c1 <= c0:
            continue
        gy, gx = np.mgrid[r0:r1, c0:c1]
        inside = mp[r0:r1, c0:c1] & (np.hypot(gy - pr, gx - pc) <= radius)
        if not inside.any():
            continue
        win = np.full(inside.shape, -np.inf)
        win[inside] = fp[r0:r1, c0:c1][inside] @ feats[i]
        k = np.unravel_index(win.argmax(), win.shape)
        rs = slice(max(k[0] - 1, 0), min(k[0] + 2, win.shape[0]))
        cs = slice(max(k[1] - 1, 0), min(k[1] + 2, win.shape[1]))
        sub = win[rs, cs]
        w = np.exp((sub - sub.max()) / temperature)
        w[~np.isfinite(sub)] = 0.0
        w /= w.sum()
        photo[i] = [(w * gy[rs, cs]).sum() + 0.5, (w * gx[rs, cs]).sum() + 0.5]
        sims[i] = win[k]
        moved[i] = float(np.hypot(photo[i, 0] - 0.5 - pr, photo[i, 1] - 0.5 - pc))
        keep[i] = True
    return photo, sims, moved, keep


def visible_vertices(Vd, Fd, buf, size, depth_tol=0.01):
    """Target vertices seen in a render: their projected pixel holds one of their faces, or
    their depth is within depth_tol of the depth range from the buffer's front there."""
    xy, zv, fid, zbuf = buf["xy"], buf["zv"], buf["fid"], buf["z"]
    px = np.clip(xy[:, 0].astype(np.int64), 0, size - 1)
    py = np.clip(xy[:, 1].astype(np.int64), 0, size - 1)
    front = fid[py, px]
    own = np.zeros(len(Vd), dtype=bool)
    hit = front >= 0
    own[hit] = (Fd[front[hit]] == np.arange(len(Vd))[hit, None]).any(axis=1)
    zr = float(zv.max() - zv.min()) if len(zv) else 1.0
    near = hit & (zv >= zbuf[py, px] - depth_tol * zr)
    return own | near, py, px


def bilinear_features(fr, rc):
    """Unit features of the render grid interpolated at fractional cell positions rc (M,2)."""
    hr, wr = fr.shape[:2]
    r = np.clip(rc[:, 0], 0, hr - 1)
    c = np.clip(rc[:, 1], 0, wr - 1)
    r0, c0 = np.floor(r).astype(np.int64), np.floor(c).astype(np.int64)
    r1, c1 = np.minimum(r0 + 1, hr - 1), np.minimum(c0 + 1, wr - 1)
    fr_, fc = (r - r0)[:, None], (c - c0)[:, None]
    f = ((1 - fr_) * (1 - fc) * fr[r0, c0] + (1 - fr_) * fc * fr[r0, c1]
         + fr_ * (1 - fc) * fr[r1, c0] + fr_ * fc * fr[r1, c1])
    return f / np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-12)


def photo_to_relief(photo_yx, grid_photo, box_photo, vert):
    """Photo grid positions (row, col in cells) -> the relief vertex of the nearest pixel that
    holds one, and whether that pixel lies within one cell."""
    hp, wp = grid_photo
    oy, ox, side = box_photo
    qy = oy + photo_yx[:, 0] * side / hp
    qx = ox + photo_yx[:, 1] * side / wp
    rows, cols = np.nonzero(vert >= 0)
    d, j = cKDTree(np.stack([rows, cols], axis=1)).query(np.stack([qy, qx], axis=1))
    return vert[rows[j], cols[j]], d <= side / min(hp, wp)


def render_cell_vertices(grid_render, box_render, fid, w0, w1, Vd, Fd, tree_d):
    """The target vertex under every render cell's centre (-1 where the cell is off the mesh)."""
    hr, wr = grid_render
    size = fid.shape[0]
    ry, rx = cell_pixels((hr, wr), box_render)
    ry = ry.astype(np.int64).clip(0, size - 1)
    rx = rx.astype(np.int64).clip(0, size - 1)
    k = fid[ry][:, rx]
    out = -np.ones((hr, wr), dtype=np.int64)
    hit = k >= 0
    kk = k[hit]
    a, b = w0[ry][:, rx][hit], w1[ry][:, rx][hit]
    pts = (a[:, None] * Vd[Fd[kk, 0]] + b[:, None] * Vd[Fd[kk, 1]]
           + (1 - a - b)[:, None] * Vd[Fd[kk, 2]])
    out[hit] = tree_d.query(pts)[1]
    return out


def prior_photo_cells(cell_vertex, prior_pos, Vs, vert, grid_photo, box_photo):
    """Where the prior puts every render cell's target vertex in the photo's feature grid: the
    prior source position -> nearest relief vertex -> its pixel -> crop cell (row, col, float).
    NaN where the cell is off the mesh."""
    hp, wp = grid_photo
    oy, ox, side = box_photo
    col, row = tt.relief_grid(Vs)
    out = np.full(cell_vertex.shape + (2,), np.nan)
    hit = cell_vertex >= 0
    sv = cKDTree(Vs).query(prior_pos[cell_vertex[hit]])[1]
    out[hit, 0] = (row[sv] - oy) * hp / side - 0.5
    out[hit, 1] = (col[sv] - ox) * wp / side - 0.5
    return out


def lift_matches(render_rc, photo_yx, grid_render, box_render, grid_photo, box_photo, fid, w0,
                 w1, Vd, Fd, vert, tree_d=None):
    """Render cells -> target vertices (through the face and barycentric buffers at the cell
    centre) and photo positions -> relief vertices (the nearest pixel that holds one); both
    grids lie over square crops given by their boxes (see square_crop). Returns t_idx, s_idx
    (relief vertex) and a validity mask."""
    size = fid.shape[0]
    hr, wr = grid_render
    oy, ox, side = box_render
    py = (oy + (render_rc[:, 0] + 0.5) * side / hr).astype(np.int64).clip(0, size - 1)
    px = (ox + (render_rc[:, 1] + 0.5) * side / wr).astype(np.int64).clip(0, size - 1)
    k = fid[py, px]
    ok = k >= 0
    kk = np.maximum(k, 0)
    a, b = w0[py, px], w1[py, px]
    pts = (a[:, None] * Vd[Fd[kk, 0]] + b[:, None] * Vd[Fd[kk, 1]]
           + (1 - a - b)[:, None] * Vd[Fd[kk, 2]])
    tree_d = tree_d or cKDTree(Vd)
    t_idx = tree_d.query(pts)[1]
    hp, wp = grid_photo
    oy, ox, side = box_photo
    qy = oy + photo_yx[:, 0] * side / hp
    qx = ox + photo_yx[:, 1] * side / wp
    rows, cols = np.nonzero(vert >= 0)
    tree_p = cKDTree(np.stack([rows, cols], axis=1))
    d, j = tree_p.query(np.stack([qy, qx], axis=1))
    s_idx = vert[rows[j], cols[j]]
    ok &= d <= side / min(hp, wp)  # within one cell of a relief pixel
    return t_idx, s_idx, ok


def spread_landmarks(t_idx, s_idx, sim, Vd, n=400, seed=0):
    """The most similar matches, thinned by farthest-point sampling on the target."""
    order = np.argsort(-sim)
    t_idx, s_idx = t_idx[order], s_idx[order]
    _, first = np.unique(t_idx, return_index=True)
    first = np.sort(first)
    t_idx, s_idx = t_idx[first], s_idx[first]
    if len(t_idx) > n:
        from refine_correspondence import farthest_points

        keep = farthest_points(Vd[t_idx], n, seed=seed)
        t_idx, s_idx = t_idx[keep], s_idx[keep]
    return t_idx, s_idx


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--relief", required=True, help="relief.ply with albedo colours")
    ap.add_argument("--target_mesh", required=True)
    ap.add_argument("--target_colors", help="PLY with vertex colours to render the target with "
                    "(a painted target); default: shaded grey")
    ap.add_argument("--view", type=float, nargs=3, help="relief camera direction in target space")
    ap.add_argument("--view_from", help="refined.npz whose stored view to use")
    ap.add_argument("--turn", type=float, default=25.0, help="extra views turned by +-deg")
    ap.add_argument("--elev", type=float, default=0.0, help="extra views tilted by +-deg")
    ap.add_argument("--extractor", choices=["dinov2", "dinov3"], default="dinov2")
    ap.add_argument("--resolution", type=int, default=518, help="model input size")
    ap.add_argument("--size", type=int, default=1024, help="render size")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--include_nonmutual", action="store_true",
                    help="keep every match, not only the mutual ones")
    ap.add_argument("--margin", type=float, default=0.08, help="crop margin around the object")
    ap.add_argument("--min_sim", type=float, default=0.0)
    ap.add_argument("--landmarks", type=int, default=400, help="matches in --landmarks_out")
    ap.add_argument("--landmarks_out")
    ap.add_argument("--prior", help="refine_correspondence.py npz: local mode, each render cell "
                    "searches only within --window of the prior's position for its vertex")
    ap.add_argument("--window", type=float, default=0.06,
                    help="with --prior: search radius as a fraction of the photo crop's side")
    ap.add_argument("--per_vertex", action="store_true",
                    help="with --prior: every target vertex visible in the render searches "
                    "(its feature interpolated from the render grid), not one vertex per cell")
    ap.add_argument("--preview")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    relief = load_mesh(args.relief)
    Vs = np.asarray(relief.vertices, dtype=np.float64)
    Cs = np.asarray(relief.visual.vertex_colors)[:, :3]
    target = load_mesh(args.target_mesh)
    Vd = np.asarray(target.vertices, dtype=np.float64)
    Fd = np.asarray(target.faces, dtype=np.int64)
    colors = None
    if args.target_colors:
        cm = load_mesh(args.target_colors)
        colors = np.asarray(cm.visual.vertex_colors)[:, :3]
        if len(colors) != len(Vd):
            colors = colors[cKDTree(np.asarray(cm.vertices)).query(Vd)[1]]
    view = np.asarray(args.view, dtype=np.float64) if args.view else None
    prior, prior_t2s = None, None
    if args.prior:
        z = np.load(args.prior)
        prior = z["src_pos"].astype(np.float64)
        prior_t2s = z["t2s"].astype(np.int64) if "t2s" in z.files else None
        if view is None and not args.view_from and "view" in z.files:
            view = z["view"].astype(np.float64)
    if view is None and args.view_from:
        view = np.load(args.view_from)["view"].astype(np.float64)
    if view is None:
        sys.exit("give --view or --view_from")

    photo, vert = albedo_image(Vs, Cs)
    fg = vert >= 0
    photo = photo.copy()
    photo[~fg] = 255  # the background is white, like the renders', not the texture's fill
    H, W = photo.shape[:2]
    crop_p, box_p = square_crop(photo, fg, args.margin)
    extract = make_extractor(args.extractor, args.device, args.resolution)
    fp = extract(crop_p)
    hp, wp = fp.shape[:2]
    # photo cells on the object: the cell centre's pixel holds a relief vertex
    cy, cx = cell_pixels((hp, wp), box_p)
    cy = cy.astype(np.int64).clip(0, H - 1)
    cx = cx.astype(np.int64).clip(0, W - 1)
    mp = vert[cy][:, cx] >= 0

    tree_d = cKDTree(Vd)
    all_t, all_s, all_sim, all_view, all_mut, all_moved = [], [], [], [], [], []
    views = turned_views(view, args.turn, args.elev)
    renders = []
    for vi, v in enumerate(views):
        img, fid, w0, w1 = render_target(Vd, Fd, v, args.size, colors)
        renders.append(img)
        crop_r, box_r = square_crop(img, fid >= 0, args.margin)
        fr = extract(crop_r)
        hr, wr = fr.shape[:2]
        ry, rx = cell_pixels((hr, wr), box_r)
        ry = ry.astype(np.int64).clip(0, args.size - 1)
        rx = rx.astype(np.int64).clip(0, args.size - 1)
        mr = fid[ry][:, rx] >= 0
        if prior is not None and args.per_vertex:
            buf = tt.raster_buffers(Vd, Fd, v, args.size)
            vis, py, px = visible_vertices(Vd, Fd, buf, args.size)
            idx = np.nonzero(vis)[0]
            oy_r, ox_r, side_r = box_r
            cell_rc = np.stack([(py[idx] + 0.5 - oy_r) * hr / side_r - 0.5,
                                (px[idx] + 0.5 - ox_r) * wr / side_r - 0.5], axis=1)
            feats = bilinear_features(fr, cell_rc)
            prior_rc = prior_photo_cells(idx[None, :], prior, Vs, vert, (hp, wp), box_p)[0]
            yx, sim, moved, keep_v = search_window(fp, mp, feats, prior_rc,
                                                   args.window * min(hp, wp))
            s_idx, near_px = photo_to_relief(yx, (hp, wp), box_p, vert)
            t_idx, ok = idx, keep_v & near_px
            mutual = np.ones(len(idx), dtype=bool)
            all_moved.append(moved[ok])
        elif prior is not None:
            cell_vertex = render_cell_vertices((hr, wr), box_r, fid, w0, w1, Vd, Fd, tree_d)
            prior_rc = prior_photo_cells(cell_vertex, prior, Vs, vert, (hp, wp), box_p)
            rc, yx, sim, moved = match_grids_local(fp, mp, fr, mr, prior_rc,
                                                   args.window * min(hp, wp))
            mutual = np.ones(len(rc), dtype=bool)
            all_moved.append(moved)
        else:
            rc, yx, sim, mutual = match_grids(fp, mp, fr, mr)
        if not (prior is not None and args.per_vertex):
            t_idx, s_idx, ok = lift_matches(rc, yx, (hr, wr), box_r, (hp, wp), box_p, fid, w0,
                                            w1, Vd, Fd, vert, tree_d)
        keep = ok & (sim >= args.min_sim)
        if not args.include_nonmutual and prior is None:
            keep &= mutual
        all_t.append(t_idx[keep])
        all_s.append(s_idx[keep])
        all_sim.append(sim[keep])
        all_mut.append(mutual[keep])
        all_view.append(np.full(int(keep.sum()), vi))
        print(json.dumps(dict(view=vi, direction=np.round(v, 4).tolist(),
                              render_cells=int(mr.sum()), matches=int(keep.sum()),
                              mutual=int((mutual & keep).sum()))), flush=True)
    t_idx = np.concatenate(all_t)
    s_idx = np.concatenate(all_s)
    sim = np.concatenate(all_sim)
    mutual = np.concatenate(all_mut)
    view_idx = np.concatenate(all_view)
    # the best match per target vertex over all views
    n = len(Vd)
    best_sim = np.full(n, -np.inf)
    best_s = -np.ones(n, dtype=np.int64)
    order = np.argsort(sim)  # ascending: later (higher) writes win
    np.maximum.at(best_sim, t_idx[order], sim[order])
    for i in order:
        if sim[i] >= best_sim[t_idx[i]]:
            best_s[t_idx[i]] = s_idx[i]
    matched = best_s >= 0
    t2s = best_s.copy()
    if prior is not None:
        # the vertices no render reached keep the prior's position (and its nearest relief vertex)
        src_pos = prior.copy()
        src_pos[matched] = Vs[best_s[matched]]
        if not matched.all():  # unmatched vertices keep the prior's own discrete map too
            if prior_t2s is not None and len(prior_t2s) == n:
                t2s[~matched] = prior_t2s[~matched]
            else:
                t2s[~matched] = cKDTree(Vs).query(prior[~matched])[1]
    else:
        if matched.any() and not matched.all():
            near = cKDTree(Vd[matched]).query(Vd[~matched])[1]
            t2s[~matched] = best_s[matched][near]
        src_pos = Vs[np.maximum(t2s, 0)]
    stats = dict(views=len(views), matches=int(len(t_idx)), mutual=int(mutual.sum()),
                 target_vertices_matched=round(float(matched.mean()), 4),
                 relief_vertices_used=int(len(np.unique(best_s[matched]))))
    if prior is not None:
        moved = np.concatenate(all_moved) if all_moved else np.zeros(0)
        diag_s = float(np.linalg.norm(Vs.max(axis=0) - Vs.min(axis=0)))
        shift = np.linalg.norm(src_pos[matched] - prior[matched], axis=1) / diag_s
        any_m = bool(matched.any())
        stats.update(window_cells=round(float(args.window * min(hp, wp)), 2),
                     moved_cells_median=round(float(np.median(moved)), 2) if len(moved) else None,
                     shift_median=round(float(np.median(shift)), 4) if any_m else None,
                     shift_over_2pct=round(float((shift > 0.02).mean()), 4) if any_m else None)
    print(json.dumps(stats), flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    np.savez_compressed(args.out, t2s=t2s, src_pos=src_pos.astype(np.float32), view=view,
                        matched=matched, t_idx=t_idx, s_idx=s_idx, sim=sim.astype(np.float32),
                        view_idx=view_idx, mutual=mutual, stats=json.dumps(stats),
                        options=json.dumps(vars(args)))
    if args.landmarks_out:
        sel = mutual
        lt, ls = spread_landmarks(t_idx[sel], s_idx[sel], sim[sel], Vd, args.landmarks)
        with open(args.landmarks_out, "w") as f:
            json.dump(dict(source=Vs[ls].tolist(), target=Vd[lt].tolist(),
                           source_vertex=ls.tolist(), target_vertex=lt.tolist(),
                           note="match2d.py mutual matches, farthest-point thinned"), f)
    if args.preview:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        sel = mutual
        lt, ls = spread_landmarks(t_idx[sel], s_idx[sel], sim[sel], Vd, 60)
        col, row = tt.relief_grid(Vs)
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(photo)
        axes[0].scatter(col[ls], row[ls], c=np.arange(len(ls)), cmap="hsv", s=18)
        axes[0].set_title("photo (relief albedo)", fontsize=9)
        xy, _ = tt.project(Vd, views[0], args.size)
        axes[1].imshow(renders[0])
        axes[1].scatter(xy[lt, 0], xy[lt, 1], c=np.arange(len(lt)), cmap="hsv", s=18)
        axes[1].set_title("target from the relief camera; same colour = matched", fontsize=9)
        for ax in axes:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(args.preview, dpi=100)
        plt.close(fig)


if __name__ == "__main__":
    main()
