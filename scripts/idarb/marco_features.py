"""
MARCO features on the sampled points of a mesh, in the layout of the Diff3F feature files, so
that every tool that takes a --feature_dir (eval_features.py, keypoint_table.py,
debias_check.py, fuse_features.py) can use them as the similarity.

The points are those of an existing feature file of the same mesh (--points: its idx and xyz
are copied), so the two feature sets can be compared and fused point by point.

  partial mesh   the photo behind it (its albedo image on a white background, cropped to the
                 object) goes through MARCO once; every point takes the feature at its own
                 pixel (bilinear in the feature grid).
  full mesh      shaded grey renders from --elevations x --azimuths directions around the
                 mesh; every point takes the mean of the features at its projections in the
                 views that see it. Points no view sees take the nearest seen point's feature.

No painting model is involved: the full mesh is seen untextured, as in match2d.py.

Needs torch, a GPU (about 3.6 GB) and the MARCO clone and checkpoint (see NOTICE.md).

Usage (repository root, on the server):
    python scripts/idarb/marco_features.py --kind partial \
        --mesh outputs/mesh_depth/cat/relief.ply \
        --points outputs/diff3f/features_zoom2/cat_partial.npz \
        --out outputs/diff3f/features_marco/cat_partial.npz
    python scripts/idarb/marco_features.py --kind full \
        --mesh third_party/Diff3F/meshes/lion.off \
        --points outputs/diff3f/features_zoom2/lion_full.npz \
        --out outputs/diff3f/features_marco/lion_full.npz
    python scripts/idarb/marco_features.py --all --points_dir outputs/diff3f/features_zoom2 \
        --out_dir outputs/diff3f/features_marco      # every mesh of the pair tables
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_features as ef  # noqa: E402
import match2d as m2  # noqa: E402
import transfer_texture as tt  # noqa: E402
from correspondence_maps import load_mesh  # noqa: E402
from sweep_anchor_filter import git_commit  # noqa: E402


def sphere_views(elevations, azimuths):
    """Unit viewing directions (towards the viewer) for every elevation and azimuth in degrees;
    the y axis is up, azimuth 0 looks along +z."""
    out = []
    for e in elevations:
        for a in azimuths:
            er, ar = np.deg2rad(e), np.deg2rad(a)
            out.append([np.cos(er) * np.sin(ar), np.sin(er), np.cos(er) * np.cos(ar)])
    return np.asarray(out, dtype=np.float64)


def cell_coords(py, px, box, grid):
    """Pixel centres (rows py, columns px) -> fractional cell positions (row, col) of a feature
    grid laid over the square crop `box` (y0, x0, side)."""
    oy, ox, side = box
    h, w = grid
    return np.stack([(py + 0.5 - oy) * h / side - 0.5, (px + 0.5 - ox) * w / side - 0.5], axis=1)


def fill_unseen(feat, seen, xyz):
    """Rows of unseen points copied from the nearest seen point."""
    if seen.all() or not seen.any():
        return feat
    near = cKDTree(xyz[seen]).query(xyz[~seen])[1]
    feat = feat.copy()
    feat[~seen] = feat[seen][near]
    return feat


def partial_features(extract, V, C, idx, margin):
    photo, vert = m2.albedo_image(V, C)
    fg = vert >= 0
    photo = photo.copy()
    photo[~fg] = 255
    crop, box = m2.square_crop(photo, fg, margin)
    f = extract(crop)
    col, row = tt.relief_grid(V)
    rc = cell_coords(row[idx].astype(np.float64), col[idx].astype(np.float64), box, f.shape[:2])
    return m2.bilinear_features(f, rc), np.ones(len(idx), dtype=bool), 1


def full_features(extract, V, F, idx, views, size, margin):
    total = None
    count = np.zeros(len(idx))
    for v in views:
        buf = tt.raster_buffers(V, F, v, size)
        shade, _ = tt.face_shade(V, F, v)
        hit = buf["fid"] >= 0
        img = np.full((size, size, 3), 255, dtype=np.uint8)
        g = np.clip(shade[buf["fid"][hit]] * 200, 0, 255)
        img[hit] = np.stack([g, g, g], axis=1).astype(np.uint8)
        crop, box = m2.square_crop(img, hit, margin)
        f = extract(crop)
        vis, py, px = m2.visible_vertices(V, F, buf, size)
        sel = vis[idx]
        rc = cell_coords(py[idx][sel].astype(np.float64), px[idx][sel].astype(np.float64), box,
                         f.shape[:2])
        if total is None:
            total = np.zeros((len(idx), f.shape[2]), dtype=np.float64)
        total[sel] += m2.bilinear_features(f, rc)
        count[sel] += 1
    seen = count > 0
    feat = total / np.maximum(np.linalg.norm(total, axis=1, keepdims=True), 1e-12)
    return feat, seen, count


def one_mesh(extract, kind, mesh_path, points_path, out_path, args):
    z = np.load(points_path, allow_pickle=False)
    idx = z["idx"].astype(np.int64)
    mesh = load_mesh(mesh_path)
    V = np.asarray(mesh.vertices, dtype=np.float64)
    tol = 1e-4 * float(np.abs(V).max())
    vidx = idx
    if idx.max() >= len(V) or not np.allclose(V[idx], z["xyz"], atol=tol):
        # the feature file may number the corners of an OBJ that load_mesh merged (the man: 41 320
        # corners, 6 890 positions); then its points are found by position among the vertices
        d, vidx = cKDTree(V).query(z["xyz"])
        if d.max() > tol:
            raise ValueError("the points of " + points_path + " are not vertices of " + mesh_path)
    if kind == "partial":
        C = np.asarray(mesh.visual.vertex_colors)[:, :3]
        feat, seen, count = partial_features(extract, V, C, vidx, args.margin)
        n_views = 1
    else:
        F = np.asarray(mesh.faces, dtype=np.int64)
        views = sphere_views(args.elevations, args.azimuths)
        feat, seen, count = full_features(extract, V, F, vidx, views, args.size, args.margin)
        n_views = len(views)
    feat = fill_unseen(feat, seen, z["xyz"].astype(np.float64))
    meta = dict(feature_set="marco", feature_dim=int(feat.shape[1]), kind=kind, views=n_views,
                resolution=args.resolution, seen=round(float(seen.mean()), 4),
                views_per_point=round(float(np.mean(count)), 2), mesh=mesh_path,
                points=points_path, commit=git_commit(), command=" ".join(sys.argv))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    np.savez_compressed(out_path, idx=idx, xyz=z["xyz"], feat=feat.astype(np.float16),
                        colors=z["colors"], meta=json.dumps(meta))
    print(json.dumps(dict(out=out_path, points=len(idx), seen=meta["seen"],
                          views_per_point=meta["views_per_point"])), flush=True)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--kind", choices=["partial", "full"])
    ap.add_argument("--mesh")
    ap.add_argument("--points", help="an existing feature file of the same mesh (idx, xyz)")
    ap.add_argument("--out")
    ap.add_argument("--all", action="store_true",
                    help="every partial and full mesh of eval_features.py's pair tables")
    ap.add_argument("--points_dir", default="outputs/diff3f/features_zoom2")
    ap.add_argument("--out_dir", default="outputs/diff3f/features_marco")
    ap.add_argument("--resolution", type=int, default=770)
    ap.add_argument("--elevations", type=float, nargs="+", default=[-30.0, 0.0, 30.0])
    ap.add_argument("--azimuths", type=float, nargs="+",
                    default=[0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0])
    ap.add_argument("--size", type=int, default=1024, help="render size")
    ap.add_argument("--margin", type=float, default=0.08)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--marco_repo", default="~/semtex/marco/repo")
    ap.add_argument("--marco_ckpt", default="~/semtex/marco/ckpt/marco_ap_spair.pth")
    ap.add_argument("--marco_pylib", default="~/semtex/marco/pylib")
    args = ap.parse_args()

    jobs = []
    if args.all:
        seen_names = set()
        for _g, _n, s, src_mesh, t, _tr, _ex in ef.pair_table(["same", "cross"]):
            for kind, name, mesh in (("partial", s + "_partial", src_mesh),
                                     ("full", t + "_full", ef.TARGET_MESH[t])):
                points = os.path.join(args.points_dir, name + ".npz")
                if name in seen_names or not os.path.exists(points):
                    continue
                seen_names.add(name)
                jobs.append((kind, mesh, points, os.path.join(args.out_dir, name + ".npz")))
    else:
        if not (args.kind and args.mesh and args.points and args.out):
            sys.exit("give --all, or --kind, --mesh, --points and --out")
        jobs.append((args.kind, args.mesh, args.points, args.out))
    marco = dict(repo=args.marco_repo, ckpt=args.marco_ckpt, pylib=args.marco_pylib)
    extract = m2.make_extractor("marco", args.device, args.resolution, marco=marco)
    for kind, mesh, points, out in jobs:
        one_mesh(extract, kind, mesh, points, out, args)


if __name__ == "__main__":
    main()
