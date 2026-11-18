"""
Bake the photo's albedo into a generated mesh's texture atlas: every texel of the GLB's texture
that the photo's camera can see takes the relief's pixel colour there; the rest keeps the
generated colour. The result is the source texture for "generate, then match": the photo's own
(lighting-free) colours on the side the photo saw, the generator's guess on the back.

Needs the alignment of align_generated.py (transform.json: camera rotation, s, R, t, depth
factor: relief -> generated mesh in the photo frame). For each uv triangle of the atlas the
texels inside it are found by a scanline fill in texture space, their 3D points interpolated
barycentrically, carried into the photo frame, tested for visibility (the first surface hit by
the camera ray through the point) and looked up in the relief's pixel grid.

Writes <out_dir>/albedo_texture.png (the atlas with the photo baked in), <out_dir>/baked.glb
(the generated mesh with that texture) and <out_dir>/bake.json (counts).

Usage (repository root, mesh environment):
    python scripts/idarb/bake_albedo.py --glb outputs/diff3f/gen3d/hunyuan3d/cat/textured.glb \
        --relief outputs/mesh_depth/cat/relief.ply \
        --transform outputs/diff3f/13_generated/hunyuan3d/cat/transform.json \
        --out_dir outputs/diff3f/13_generated/hunyuan3d/cat
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
from align_generated import front_depth  # noqa: E402


def texel_points(uv, F, V, H, W, extra=False):
    """For every texel inside a uv triangle (u right, v up): its pixel (row, col) and 3D point.
    Scanline fill per face in texture pixel space; faces smaller than a texel still give their
    centre texel. uv is per vertex (N,2) or per corner (M,3,2). With extra=True also returns the
    face index and the barycentric coordinates of every texel."""
    uv = np.asarray(uv, dtype=np.float64)
    per_corner = uv.ndim == 3
    px_v = None if per_corner else np.column_stack([uv[:, 0] * W - 0.5, (1.0 - uv[:, 1]) * H - 0.5])
    rows_out, cols_out, pts_out, fid_out, bary_out = [], [], [], [], []
    for fi, f in enumerate(F):
        if per_corner:
            t = np.column_stack([uv[fi, :, 0] * W - 0.5, (1.0 - uv[fi, :, 1]) * H - 0.5])
        else:
            t = px_v[f]
        x0, x1 = int(np.floor(t[:, 0].min())), int(np.ceil(t[:, 0].max()))
        y0, y1 = int(np.floor(t[:, 1].min())), int(np.ceil(t[:, 1].max()))
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, W - 1), min(y1, H - 1)
        if x1 < x0 or y1 < y0:
            continue
        xs, ys = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
        q = np.column_stack([xs.ravel(), ys.ravel()]).astype(np.float64)
        # barycentric coordinates in texture space
        d = t[1:] - t[0]
        den = d[0, 0] * d[1, 1] - d[1, 0] * d[0, 1]
        if abs(den) < 1e-12:
            continue
        r = q - t[0]
        b1 = (r[:, 0] * d[1, 1] - r[:, 1] * d[1, 0]) / den
        b2 = (d[0, 0] * r[:, 1] - d[0, 1] * r[:, 0]) / den
        b0 = 1.0 - b1 - b2
        inside = (b0 >= -1e-3) & (b1 >= -1e-3) & (b2 >= -1e-3)
        if not inside.any():
            c = np.round(t.mean(axis=0)).astype(int)
            if 0 <= c[0] < W and 0 <= c[1] < H:
                rows_out.append(np.array([c[1]]))
                cols_out.append(np.array([c[0]]))
                pts_out.append(V[f].mean(axis=0, keepdims=True))
                fid_out.append(np.array([fi]))
                bary_out.append(np.full((1, 3), 1.0 / 3.0))
            continue
        b = np.column_stack([b0, b1, b2])[inside]
        rows_out.append(q[inside, 1].astype(int))
        cols_out.append(q[inside, 0].astype(int))
        pts_out.append(b @ V[f])
        fid_out.append(np.full(len(b), fi))
        bary_out.append(b)
    if not rows_out:
        empty = np.zeros(0, dtype=int)
        out = (empty, empty, np.zeros((0, 3)))
        return out + (empty, np.zeros((0, 3))) if extra else out
    out = (np.concatenate(rows_out), np.concatenate(cols_out), np.concatenate(pts_out))
    if extra:
        out = out + (np.concatenate(fid_out), np.concatenate(bary_out))
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--glb", required=True, help="textured.glb of the generated mesh")
    ap.add_argument("--relief", required=True, help="the relief with its albedo colours")
    ap.add_argument("--transform", required=True, help="transform.json of align_generated.py")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    m = trimesh.load(args.glb, force="mesh", process=False)
    V = np.asarray(m.vertices, dtype=np.float64)
    F = np.asarray(m.faces)
    uv = np.asarray(m.visual.uv, dtype=np.float64)
    tex = np.asarray(m.visual.material.baseColorTexture.convert("RGB")).copy()
    H, W = tex.shape[:2]
    relief = trimesh.load(args.relief, process=False, force="mesh")
    Vr = np.asarray(relief.vertices, dtype=np.float64)
    Cr = np.asarray(relief.visual.vertex_colors)[:, :3]
    with open(args.transform) as f:
        tr = json.load(f)
    Rc = np.array(tr["camera_rotation"])
    s, a = tr["scale"], tr["depth_factor"]
    R, t = np.array(tr["rotation"]), np.array(tr["translation"])

    V_cam = V @ Rc.T
    F_cam = F
    diag = float(np.linalg.norm(V_cam.max(axis=0) - V_cam.min(axis=0)))
    rows, cols, pts = texel_points(uv, F, V_cam, H, W)
    # visibility: the texel's point must be the first surface the camera ray hits
    seen = front_depth(V_cam, F_cam, pts[:, :2])
    visible = np.isfinite(seen) & (np.abs(seen - pts[:, 2]) < 0.01 * diag)
    # into the relief's frame: relief = depth^-1( R^T (x - t) / s )
    q = ((pts - t) / s) @ R
    q[:, 2] /= a
    step = float(np.diff(np.unique(np.round(Vr[:, 0], 6))).min())
    d_xy, j = cKDTree(Vr[:, :2]).query(q[:, :2])
    take = visible & (d_xy < 0.75 * step)
    tex[rows[take], cols[take]] = Cr[j[take]]
    Image.fromarray(tex).save(os.path.join(args.out_dir, "albedo_texture.png"))
    mat = trimesh.visual.material.PBRMaterial(baseColorTexture=Image.fromarray(tex))
    out = trimesh.Trimesh(vertices=V, faces=F, process=False,
                          visual=trimesh.visual.TextureVisuals(uv=uv, material=mat))
    out.export(os.path.join(args.out_dir, "baked.glb"))
    stats = dict(texels=int(len(rows)), visible=int(visible.sum()), baked=int(take.sum()),
                 baked_fraction_of_visible=round(float(take.sum() / max(visible.sum(), 1)), 4))
    with open(os.path.join(args.out_dir, "bake.json"), "w") as f:
        json.dump(stats, f, indent=1)
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
