"""
Validation tier B of the adapter: reliefs made by the REAL pipeline from renders of full meshes,
with ground truth through the render.

The dataset's partials (make_partial_dataset.py) are exact cuts of the full mesh; a real relief
is IDArb's normal map integrated with a depth prior (normal_to_mesh.py), with its own depth
distortion. This script makes the inputs and the ground truth for reliefs of the dataset's
full meshes, so that the adapter can be validated on that distortion:

render   For each object: an RGBA render of the full mesh (vertex colours shaded) from a view,
         512 x 512 with the object centred as IDArb's preprocessing keeps it (long side 512,
         square, white background), plus the render's buffers: the face and barycentric
         coordinates behind every pixel, from which a pixel's true point on the full mesh
         follows. The view is the dataset partial's cutting camera (its _gt.npz rotation) when
         given, so tier B looks at the object from the same side as tier A.
attach   After IDArb (normal map), estimate_depth.py and normal_to_mesh.py have turned the
         render into relief.ply (one vertex per foreground pixel, on the pixel grid): every
         relief vertex gets the full-mesh vertex behind its pixel (the corner of the face with
         the largest barycentric weight) as orig_idx, written in make_partial_mesh.py's _gt.npz
         layout (orig_idx, s, R, t; plus the true points), so evaluate() and adapter_data work
         unchanged.

Usage (repository root, mesh environment):
    python scripts/idarb/make_relief_gt.py render --mesh full/<uid>.ply \
        --gt partial/<uid>_v0_gt.npz --out_dir outputs/relief_gt/<uid>
    # then on the server: IDArb on <uid>.png, estimate_depth.py, normal_to_mesh.py --depth ...
    python scripts/idarb/make_relief_gt.py attach --relief outputs/mesh_depth/<uid>/relief.ply \
        --buffers outputs/relief_gt/<uid>/<uid>_buffers.npz \
        --out outputs/relief_gt/<uid>/<uid>_relief_gt.npz
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import transfer_texture as tt  # noqa: E402


def load_mesh_raw(path):
    """Vertices, faces and vertex colours (None when absent) in the file's own numbering."""
    import trimesh

    m = trimesh.load(path, process=False, force="mesh")
    colors = None
    if m.visual.kind == "vertex" and len(m.visual.vertex_colors) == len(m.vertices):
        colors = np.asarray(m.visual.vertex_colors)[:, :3]
    return (np.asarray(m.vertices, dtype=np.float64), np.asarray(m.faces, dtype=np.int64),
            colors)


def view_from_gt(gt_npz):
    """The cutting camera's direction in the full mesh's frame: the partial lives in the camera
    frame with the camera on +z and full = s R partial + t, so the direction is R's third
    column."""
    R = np.asarray(np.load(gt_npz)["R"], dtype=np.float64)
    return R[:, 2] / np.linalg.norm(R[:, 2])


def render_rgba(V, F, view, size=512, colors=None, light=(0.3, 0.5, 0.8)):
    """A shaded render with the raster buffers: RGBA uint8 (size, size, 4) and the buffers."""
    buf = tt.raster_buffers(V, F, view, size)
    hit = buf["fid"] >= 0
    k = buf["fid"][hit]
    # a directional light in the view frame plus ambient, on the face normals
    Rf = tt.view_frame(view)
    P = V @ Rf.T
    n = np.cross(P[F][:, 1] - P[F][:, 0], P[F][:, 2] - P[F][:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    n[n[:, 2] < 0] *= -1  # two-sided
    lt = np.asarray(light, dtype=np.float64)
    lt /= np.linalg.norm(lt)
    shade = 0.35 + 0.65 * np.clip(n @ lt, 0, 1)
    base = np.full((len(V), 3), 180.0) if colors is None else colors.astype(np.float64)
    w0, w1 = buf["w0"][hit], buf["w1"][hit]
    w2 = 1.0 - w0 - w1
    c = w0[:, None] * base[F[k, 0]] + w1[:, None] * base[F[k, 1]] + w2[:, None] * base[F[k, 2]]
    img = np.full((size, size, 4), 255, dtype=np.uint8)
    img[hit, :3] = np.clip(c * shade[k][:, None], 0, 255).astype(np.uint8)
    img[..., 3] = np.where(hit, 255, 0).astype(np.uint8)
    return img, buf


def locate_relief(row_rel, col_rel, mask, search=6):
    """normal_to_mesh.py recentres the relief, so its vertices give pixel coordinates only up
    to an offset; the offset that best overlaps the relief's pixel set with the render's
    foreground mask (searched around the mask's top-left corner; the relief's mask is the
    input's eroded by a few pixels) gives the absolute (row, col). Returns row0, col0."""
    rr, cc = np.nonzero(mask)
    h, w = row_rel.max() + 1, col_rel.max() + 1
    best = None
    for r0 in range(max(rr.min() - search, 0), min(rr.min() + search, mask.shape[0] - h) + 1):
        for c0 in range(max(cc.min() - search, 0),
                        min(cc.min() + search, mask.shape[1] - w) + 1):
            overlap = int(mask[row_rel + r0, col_rel + c0].sum())
            if best is None or overlap > best[0]:
                best = (overlap, r0, c0)
    return best[1], best[2]


def attach_gt(relief_vertices, buffers, V, F):
    """orig_idx (full vertex behind every relief vertex's pixel, -1 where the pixel shows no
    face), the true points on the full mesh, and the pixel (row, col) of every relief vertex."""
    col_rel, row_rel = tt.relief_grid(relief_vertices)
    fid, w0, w1 = buffers["fid"], buffers["w0"], buffers["w1"]
    size = fid.shape[0]
    row0, col0 = locate_relief(row_rel, col_rel, fid >= 0)
    row = np.clip(row_rel + row0, 0, size - 1)
    col = np.clip(col_rel + col0, 0, size - 1)
    k = fid[row, col]
    ok = k >= 0
    kk = np.maximum(k, 0)
    a, b = w0[row, col], w1[row, col]
    bary = np.stack([a, b, 1.0 - a - b], axis=1)
    corner = F[kk, bary.argmax(axis=1)]
    orig = np.where(ok, corner, -1)
    pts = (a[:, None] * V[F[kk, 0]] + b[:, None] * V[F[kk, 1]]
           + (1 - a - b)[:, None] * V[F[kk, 2]])
    pts[~ok] = np.nan
    return orig, pts, row, col


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="step", required=True)
    r = sub.add_parser("render")
    r.add_argument("--mesh", required=True, help="full mesh PLY (vertex colours used if any)")
    r.add_argument("--gt", help="a partial's _gt.npz: its cutting camera becomes the view")
    r.add_argument("--view", type=float, nargs=3, help="view direction instead of --gt")
    r.add_argument("--size", type=int, default=512)
    r.add_argument("--out_dir", required=True)
    r.add_argument("--name", help="file stem (default: the mesh's)")
    a = sub.add_parser("attach")
    a.add_argument("--relief", required=True, help="relief.ply from normal_to_mesh.py")
    a.add_argument("--buffers", required=True, help="<name>_buffers.npz from the render step")
    a.add_argument("--out", required=True, help="<name>_relief_gt.npz")
    args = ap.parse_args()

    if args.step == "render":
        from PIL import Image

        V, F, colors = load_mesh_raw(args.mesh)
        if args.view:
            view = np.asarray(args.view, dtype=np.float64)
            view /= np.linalg.norm(view)
        elif args.gt:
            view = view_from_gt(args.gt)
        else:
            sys.exit("give --gt or --view")
        name = args.name or os.path.splitext(os.path.basename(args.mesh))[0]
        os.makedirs(args.out_dir, exist_ok=True)
        img, buf = render_rgba(V, F, view, args.size, colors)
        Image.fromarray(img, "RGBA").save(os.path.join(args.out_dir, name + ".png"))
        np.savez_compressed(os.path.join(args.out_dir, name + "_buffers.npz"),
                            fid=buf["fid"], w0=buf["w0"].astype(np.float32),
                            w1=buf["w1"].astype(np.float32), view=view, mesh=args.mesh,
                            size=args.size)
        print(json.dumps(dict(name=name, view=np.round(view, 4).tolist(),
                              foreground=int((buf["fid"] >= 0).sum()),
                              colours=colors is not None)))
    else:
        import trimesh

        z = np.load(args.buffers)
        V, F, _ = load_mesh_raw(str(z["mesh"]))
        rel = trimesh.load(args.relief, process=False, force="mesh")
        Vr = np.asarray(rel.vertices, dtype=np.float64)
        orig, pts, row, col = attach_gt(Vr, z, V, F)
        R = tt.view_frame(z["view"].astype(np.float64))  # rows: right, up, towards the viewer
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        np.savez(args.out, orig_idx=orig, s=1.0, R=R.T, t=np.zeros(3), points=pts,
                 row=row, col=col, relief=True,
                 meta=json.dumps(dict(relief=args.relief, buffers=args.buffers,
                                      n_relief_vertices=int(len(Vr)),
                                      with_gt=int((orig >= 0).sum()))))
        print(json.dumps(dict(relief_vertices=int(len(Vr)), with_gt=int((orig >= 0).sum()),
                              without=int((orig < 0).sum()))))


if __name__ == "__main__":
    main()
