"""
One texture atlas of the target's own, baked per texel from two sources: the photo through the
relief correspondence on the side the photo's camera sees, and a painted copy of the target
(Hunyuan3D-Paint or TRELLIS.2 texturing of the target from the photo) everywhere else. Baking
per texel removes the artefacts of looking a foreign atlas up per vertex (a target triangle
whose corners fall in different charts of the painter's atlas showed a stretched smear): every
texel is evaluated at its own 3D point. At the silhouette the two sources are blended over a
band of face orientations, and the painted colours are levelled to the photo's statistics on
the faces where both sources exist (a global per-channel affine match).

The atlas comes from xatlas when it is installed (few charts), else every face gets its own
patch of the texture (no shared edges, so no chart seams in the mapping; patch borders are
dilated so the renderer's filtering does not bleed).

Usage (repository root, mesh environment; CPU, about a minute per pair):
    python scripts/idarb/bake_target_atlas.py --target_mesh third_party/Diff3F/meshes/lion.off \
        --relief outputs/mesh_depth/cat/relief.ply \
        --correspondence outputs/diff3f/15_relief_zoom2/same/cat_lion/refined.npz \
        --painted_glb outputs/diff3f/paint_baseline_meshes/hunyuan3d/cat_lion/textured.glb \
        --out_dir outputs/diff3f/21_baked/hunyuan3d/same/cat_lion
"""

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import transfer_texture as tt  # noqa: E402
from bake_albedo import texel_points  # noqa: E402
from correspondence_maps import load_mesh  # noqa: E402


def per_face_atlas(F, size, inset=2.0):
    """Every face its own right-triangle patch in a grid of square cells: per-corner uv (M,3,2),
    u right, v up. The cell count per side is ceil(sqrt(M)); `inset` texels keep the patch
    away from the cell border."""
    m = len(F)
    n = int(np.ceil(np.sqrt(m)))
    cell = size / n
    uvc = np.zeros((m, 3, 2))
    i = np.arange(m)
    r, c = i // n, i % n
    x0 = c * cell + inset
    y0 = r * cell + inset
    x1 = (c + 1) * cell - inset
    y1 = (r + 1) * cell - inset
    uvc[:, 0] = np.column_stack([x0, y0])
    uvc[:, 1] = np.column_stack([x1, y0])
    uvc[:, 2] = np.column_stack([x0, y1])
    uvc /= size
    uvc[:, :, 1] = 1.0 - uvc[:, :, 1]
    return uvc


def xatlas_atlas(V, F):
    """Per-corner uv from xatlas (returns None when it is not installed)."""
    try:
        import xatlas
    except ImportError:
        return None
    vmapping, indices, uvs = xatlas.parametrize(V, F)
    return uvs[indices]  # (M,3,2); indices index the new vertex array, same face order


def dilate_filled(tex, filled, passes=4):
    """Grow the filled texels outwards so that filtering at patch borders samples colour, not
    black."""
    out = tex.copy()
    mask = filled.copy()
    for _ in range(passes):
        grown = ndimage.binary_dilation(mask)
        ring = grown & ~mask
        if not ring.any():
            break
        for ch in range(3):
            near = ndimage.grey_dilation(np.where(mask, out[:, :, ch], 0), size=3)
            cnt = ndimage.uniform_filter(mask.astype(np.float64), size=3) * 9.0
            avg = ndimage.uniform_filter(np.where(mask, out[:, :, ch], 0).astype(np.float64),
                                         size=3) * 9.0 / np.maximum(cnt, 1)
            out[:, :, ch] = np.where(ring, np.where(cnt > 0, avg, near), out[:, :, ch])
        mask = grown
    return out


def level_colours(src, ref):
    """Per-channel affine map that gives `src` the mean and standard deviation of `ref`."""
    src = np.asarray(src, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    if len(src) < 10 or len(ref) < 10:
        return lambda x: x
    ms, ss = src.mean(axis=0), src.std(axis=0) + 1e-6
    mr, sr = ref.mean(axis=0), ref.std(axis=0) + 1e-6
    gain = np.clip(sr / ss, 0.5, 2.0)
    return lambda x: np.clip((np.asarray(x, dtype=np.float64) - ms) * gain + mr, 0, 255)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--target_mesh", required=True)
    ap.add_argument("--relief", required=True, help="relief.ply with albedo colours")
    ap.add_argument("--correspondence", required=True, help="refined.npz (src_pos, view)")
    ap.add_argument("--painted_glb", required=True, help="painted copy of the target (GLB)")
    ap.add_argument("--view", type=float, nargs=3, help="texture camera; default: the npz's")
    ap.add_argument("--size", type=int, default=2048, help="atlas size in texels")
    ap.add_argument("--band", type=float, nargs=2, default=(0.15, 0.4),
                    help="face cosines to the camera between which photo and paint are blended")
    ap.add_argument("--no_level", action="store_true", help="skip the colour levelling")
    ap.add_argument("--atlas", choices=["auto", "xatlas", "faces"], default="auto")
    ap.add_argument("--render_size", type=int, default=640)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    tm = load_mesh(args.target_mesh)
    V = np.asarray(tm.vertices, dtype=np.float64)
    F = np.asarray(tm.faces, dtype=np.int64)
    relief = load_mesh(args.relief)
    Vs = np.asarray(relief.vertices, dtype=np.float64)
    Cs = np.asarray(relief.visual.vertex_colors)[:, :3]
    photo, _uv, _b = tt.relief_texture(Vs, Cs, blank_rows=0)
    z = np.load(args.correspondence)
    P = z["src_pos"].astype(np.float64) if "src_pos" in z.files else Vs[z["t2s"]]
    view = np.asarray(args.view, dtype=np.float64) if args.view else z["view"].astype(np.float64)
    img_p, uv_p, _bl, V_p, F_p = tt.atlas_texture(args.painted_glb, blank_rows=0)

    uvc = None
    if args.atlas in ("auto", "xatlas"):
        uvc = xatlas_atlas(V, F)
        if uvc is None and args.atlas == "xatlas":
            sys.exit("xatlas is not installed")
    atlas_kind = "xatlas" if uvc is not None else "faces"
    if uvc is None:
        uvc = per_face_atlas(F, args.size)
    S = args.size
    rows, cols, pts, fid, bary = texel_points(uvc, F, V, S, S, extra=True)

    # which source each face takes: seen and well-oriented faces the photo, the rest the paint,
    # blended over the band of orientations
    visible = tt.visible_faces(V, F, view)
    _, nz = tt.face_shade(V, F, view)
    lo, hi = args.band
    w_face = np.clip((nz - lo) / max(hi - lo, 1e-6), 0.0, 1.0) * visible
    w = w_face[fid]

    # the photo colour at every texel: the field interpolated over the face, then the relief pixel
    Pf = np.einsum("nk,nkd->nd", bary, P[F[fid]])
    x0, y1, step = tt.relief_frame(Vs)
    H, W = photo.shape[:2]
    pc = np.clip(np.round((Pf[:, 0] - x0) / step), 0, W - 1).astype(int)
    pr = np.clip(np.round((y1 - Pf[:, 1]) / step), 0, H - 1).astype(int)
    col_photo = photo[pr, pc].astype(np.float64)
    # the painted colour at every texel: the painted copy's atlas at the texel's own 3D point
    uv_t = tt.uv_of_points_on_mesh(pts, V_p, F_p, uv_p)
    Hp, Wp = img_p.shape[:2]
    tc = np.clip((uv_t[:, 0] * Wp).astype(int), 0, Wp - 1)
    tr = np.clip(((1.0 - uv_t[:, 1]) * Hp).astype(int), 0, Hp - 1)
    col_paint = img_p[tr, tc].astype(np.float64)
    if not args.no_level:
        both = w > 0.5
        col_paint = level_colours(col_paint[both], col_photo[both])(col_paint)
    colour = w[:, None] * col_photo + (1.0 - w[:, None]) * col_paint

    tex = np.zeros((S, S, 3), dtype=np.uint8)
    filled = np.zeros((S, S), dtype=bool)
    tex[rows, cols] = np.clip(colour, 0, 255).astype(np.uint8)
    filled[rows, cols] = True
    tex = dilate_filled(tex, filled, passes=3 if atlas_kind == "faces" else 6)
    Image.fromarray(tex).save(os.path.join(args.out_dir, "target_baked.png"))
    stem = "target_baked"
    with open(os.path.join(args.out_dir, stem + ".mtl"), "w") as f:
        f.write(f"newmtl {stem}\nKa 1.0 1.0 1.0\nKd 1.0 1.0 1.0\nKs 0.0 0.0 0.0\n")
        f.write(f"map_Kd {stem}.png\n")
    lines = [f"mtllib {stem}.mtl", f"usemtl {stem}"]
    lines += [f"v {x:.6f} {y:.6f} {zz:.6f}" for x, y, zz in V]
    lines += [f"vt {u:.6f} {v:.6f}" for u, v in uvc.reshape(-1, 2)]
    lines += [f"f {a + 1}/{3 * i + 1} {b + 1}/{3 * i + 2} {c + 1}/{3 * i + 3}"
              for i, (a, b, c) in enumerate(F.tolist())]
    with open(os.path.join(args.out_dir, stem + ".obj"), "w") as f:
        f.write("\n".join(lines) + "\n")

    blend = (w_face > 0) & (w_face < 1)
    stats = dict(atlas=atlas_kind, size=S, texels=int(len(rows)),
                 faces_photo=int((w_face >= 0.5).sum()), faces_blend=int(blend.sum()),
                 faces_paint=int((w_face == 0).sum()), levelled=not args.no_level,
                 view=[round(float(v), 6) for v in view])
    print(json.dumps(stats))
    with open(os.path.join(args.out_dir, "bake.json"), "w") as f:
        json.dump(stats, f, indent=1)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from merge_sides import side_views

    left, right = side_views(view)
    panels = [("front: photo", view), ("left", left), ("back: painted", -view), ("right", right)]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.4))
    for ax, (title, v) in zip(axes, panels):
        ax.imshow(tt.rasterize(V, F, uvc, tex, v, args.render_size))
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "baked.png"), dpi=90)
    plt.close(fig)


if __name__ == "__main__":
    main()
