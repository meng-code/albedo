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


def lift_matches(render_rc, photo_yx, grid_render, grid_photo, fid, w0, w1, Vd, Fd, vert,
                 image_shape, tree_d=None):
    """Render cells -> target vertices (through the face and barycentric buffers at the cell
    centre) and photo positions -> relief vertices (the nearest pixel that holds one). Returns
    t_idx, s_idx (relief vertex) and a validity mask."""
    size = fid.shape[0]
    hr, wr = grid_render
    py = ((render_rc[:, 0] + 0.5) * size / hr).astype(np.int64).clip(0, size - 1)
    px = ((render_rc[:, 1] + 0.5) * size / wr).astype(np.int64).clip(0, size - 1)
    k = fid[py, px]
    ok = k >= 0
    kk = np.maximum(k, 0)
    a, b = w0[py, px], w1[py, px]
    pts = (a[:, None] * Vd[Fd[kk, 0]] + b[:, None] * Vd[Fd[kk, 1]]
           + (1 - a - b)[:, None] * Vd[Fd[kk, 2]])
    tree_d = tree_d or cKDTree(Vd)
    t_idx = tree_d.query(pts)[1]
    H, W = image_shape
    hp, wp = grid_photo
    qy = (photo_yx[:, 0] * H / hp).clip(0, H - 1)
    qx = (photo_yx[:, 1] * W / wp).clip(0, W - 1)
    rows, cols = np.nonzero(vert >= 0)
    tree_p = cKDTree(np.stack([rows, cols], axis=1))
    d, j = tree_p.query(np.stack([qy, qx], axis=1))
    s_idx = vert[rows[j], cols[j]]
    ok &= d <= max(H / hp, W / wp)  # within one cell of a relief pixel
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
    ap.add_argument("--mutual_only", action="store_true", help="keep mutual matches only")
    ap.add_argument("--min_sim", type=float, default=0.0)
    ap.add_argument("--landmarks", type=int, default=400, help="matches in --landmarks_out")
    ap.add_argument("--landmarks_out")
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
    if view is None and args.view_from:
        view = np.load(args.view_from)["view"].astype(np.float64)
    if view is None:
        sys.exit("give --view or --view_from")

    photo, vert = albedo_image(Vs, Cs)
    extract = make_extractor(args.extractor, args.device, args.resolution)
    fp = extract(photo)
    hp, wp = fp.shape[:2]
    H, W = photo.shape[:2]
    # photo cells on the object: the cell centre's pixel holds a relief vertex
    cy = ((np.arange(hp) + 0.5) * H / hp).astype(np.int64)
    cx = ((np.arange(wp) + 0.5) * W / wp).astype(np.int64)
    mp = vert[cy][:, cx] >= 0

    tree_d = cKDTree(Vd)
    all_t, all_s, all_sim, all_view, all_mut = [], [], [], [], []
    views = turned_views(view, args.turn, args.elev)
    renders = []
    for vi, v in enumerate(views):
        img, fid, w0, w1 = render_target(Vd, Fd, v, args.size, colors)
        renders.append(img)
        fr = extract(img)
        hr, wr = fr.shape[:2]
        ry = ((np.arange(hr) + 0.5) * args.size / hr).astype(np.int64)
        rx = ((np.arange(wr) + 0.5) * args.size / wr).astype(np.int64)
        mr = fid[ry][:, rx] >= 0
        rc, yx, sim, mutual = match_grids(fp, mp, fr, mr)
        t_idx, s_idx, ok = lift_matches(rc, yx, (hr, wr), (hp, wp), fid, w0, w1, Vd, Fd, vert,
                                        (H, W), tree_d)
        keep = ok & (sim >= args.min_sim)
        if args.mutual_only:
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
    if matched.any() and not matched.all():
        near = cKDTree(Vd[matched]).query(Vd[~matched])[1]
        t2s[~matched] = best_s[matched][near]
    src_pos = Vs[np.maximum(t2s, 0)]
    stats = dict(views=len(views), matches=int(len(t_idx)), mutual=int(mutual.sum()),
                 target_vertices_matched=round(float(matched.mean()), 4),
                 relief_vertices_used=int(len(np.unique(best_s[matched]))))
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
