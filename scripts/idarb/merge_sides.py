"""
One textured target from two sources. The side the photo's camera sees takes the photo's
texture through the relief correspondence (as transfer_texture.py does); the rest, which the
photo never saw, takes the generated mesh's texture (its atlas, with the photo's albedo baked
into the visible side by bake_albedo.py) through the generated-mesh correspondence. The two
images are packed side by side into one texture, so the result is one OBJ + MTL + PNG, and the
preview shows the target from the front, the back and both sides.

Usage (repository root, mesh environment):
    python scripts/idarb/merge_sides.py --target_mesh third_party/Diff3F/meshes/lion.off \
        --front_source outputs/diff3f/features_zoom2/cat_partial.npz \
        --front_mesh outputs/mesh_depth/cat/relief.ply \
        --front_correspondence outputs/diff3f/15_relief_zoom2/same/cat_lion/refined.npz \
        --back_source outputs/diff3f/features_zoom2_gen/hunyuan3d/cat.npz \
        --back_mesh outputs/diff3f/features_zoom2_gen/hunyuan3d/cat_mesh.ply \
        --back_correspondence <gen out_dir>/same/cat_lion/refined.npz \
        --back_glb outputs/diff3f/13_generated/hunyuan3d/cat/baked.glb \
        --out_dir outputs/diff3f/14_generated_match/hunyuan3d/same/cat_lion/merged
The texture camera is the `view` stored in the front correspondence unless --view is given.
"""

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import transfer_texture as tt  # noqa: E402
from correspondence_maps import load_mesh, mesh_features  # noqa: E402


def pack_side_by_side(img_a, img_b):
    """Two images side by side in one (height = the taller, both top-aligned, the rest white).
    Returns the image and, for each input, the box (x0, width, height) it occupies."""
    Ha, Wa = img_a.shape[:2]
    Hb, Wb = img_b.shape[:2]
    H, W = max(Ha, Hb), Wa + Wb
    out = np.full((H, W, 3), 255, dtype=np.uint8)
    out[:Ha, :Wa] = img_a[:, :, :3]
    out[:Hb, Wa:] = img_b[:, :, :3]
    return out, (0, Wa, Ha), (Wa, Wb, Hb)


def remap_uv(uv, box, W, H):
    """Texture coordinates of one image (u right, v up) into the packed image."""
    x0, w, h = box
    uv = np.asarray(uv, dtype=np.float64)
    u = (x0 + uv[:, 0] * w) / W
    v = 1.0 - (1.0 - uv[:, 1]) * h / H
    return np.column_stack([u, v])


def source_positions(npz_path, V_source):
    """Source-space positions of the target vertices from a correspondence npz: the continuous
    field when it has one, else the matched vertices."""
    z = np.load(npz_path)
    if "src_pos" in z.files:
        P = z["src_pos"].astype(np.float64)
    else:
        P = V_source[z["t2s"].astype(np.int64)]
    view = z["view"].astype(np.float64) if "view" in z.files else None
    return P, view


def export_two_sided_obj(path, V, F, uv_front, uv_back, front, image):
    """OBJ whose front faces index one set of texture coordinates and whose other faces index
    a second set; one material, one packed texture."""
    stem = os.path.splitext(os.path.basename(path))[0]
    out_dir = os.path.dirname(os.path.abspath(path))
    Image.fromarray(image).save(os.path.join(out_dir, stem + ".png"))
    with open(os.path.join(out_dir, stem + ".mtl"), "w") as f:
        f.write(f"newmtl {stem}\nKa 1.0 1.0 1.0\nKd 1.0 1.0 1.0\nKs 0.0 0.0 0.0\n")
        f.write(f"map_Kd {stem}.png\n")
    n = len(V)
    vt = np.concatenate([uv_front, uv_back])
    tex_idx = np.where(front[:, None], F + 1, F + 1 + n)
    lines = [f"mtllib {stem}.mtl", f"usemtl {stem}"]
    lines += [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in V]
    lines += [f"vt {u:.6f} {v:.6f}" for u, v in vt]
    lines += [f"f {a}/{ta} {b}/{tb} {c}/{tc}"
              for (a, b, c), (ta, tb, tc) in zip((F + 1).tolist(), tex_idx.tolist())]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def side_views(view, up=(0.0, 1.0, 0.0)):
    """The camera's left and right directions, for the side renders."""
    view = np.asarray(view, dtype=np.float64)
    view = view / np.linalg.norm(view)
    up = np.asarray(up, dtype=np.float64)
    if abs(float(np.dot(up, view))) > 0.95:
        up = np.array([0.0, 0.0, 1.0])
    right = np.cross(up, view)
    right /= np.linalg.norm(right)
    return right, -right


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--target_mesh", required=True)
    ap.add_argument("--front_source", required=True, help="features of the relief (.npz)")
    ap.add_argument("--front_mesh", required=True, help="the relief with its albedo colours")
    ap.add_argument("--front_correspondence", required=True)
    ap.add_argument("--back_source", help="features of the generated mesh (unused with "
                    "--back_identity)")
    ap.add_argument("--back_mesh", help="the mesh the back correspondence indexes")
    ap.add_argument("--back_correspondence", help="npz with src_pos / t2s into --back_mesh")
    ap.add_argument("--back_glb", required=True, help="the textured GLB the unseen side is "
                    "read from: the generated mesh's baked.glb, or a painted copy of the "
                    "target itself with --back_identity")
    ap.add_argument("--back_identity", action="store_true",
                    help="the back GLB is the target itself (a painted copy, own vertices and "
                    "atlas): every target vertex takes the texture at its own position")
    ap.add_argument("--view", type=float, nargs=3, help="texture camera; default: the front "
                    "correspondence's stored view")
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    m = load_mesh(args.target_mesh)
    Vd = np.asarray(m.vertices, dtype=np.float64)
    Fd = np.asarray(m.faces, dtype=np.int64)
    # front: the relief's own image and pixel grid
    Vf, _Ff, _fs, cf = mesh_features(args.front_source, args.front_mesh)
    if cf is None:
        sys.exit("the relief has no colours")
    img_f, uv_f_src, _blank = tt.relief_texture(Vf, cf)
    P_f, view = source_positions(args.front_correspondence, Vf)
    uv_f = tt.relief_uv_of_points(P_f, Vf, img_f.shape)
    # back: the generated mesh's atlas, looked up by position on the GLB surface
    img_b, uv_b_src, _blank_b, V_glb, F_glb = tt.atlas_texture(args.back_glb, blank_rows=0)
    if args.back_identity:
        P_b = Vd  # the painted copy lies on the target: look its atlas up at the vertex itself
    else:
        if not (args.back_mesh and args.back_correspondence):
            sys.exit("give --back_mesh and --back_correspondence, or --back_identity")
        Vb = np.asarray(load_mesh(args.back_mesh).vertices, dtype=np.float64)
        P_b, _ = source_positions(args.back_correspondence, Vb)
    uv_b = tt.uv_of_points_on_mesh(P_b, V_glb, F_glb, uv_b_src)
    if args.view is not None:
        view = np.asarray(args.view, dtype=np.float64)
    if view is None:
        sys.exit("no view: the front correspondence stores none, give --view")

    image, box_f, box_b = pack_side_by_side(img_f, img_b)
    H, W = image.shape[:2]
    uv_front = np.clip(remap_uv(np.clip(uv_f, 0, 1), box_f, W, H), 0, 1)
    uv_back = np.clip(remap_uv(np.clip(uv_b, 0, 1), box_b, W, H), 0, 1)
    front = tt.visible_faces(Vd, Fd, view)
    uvc = np.where(front[:, None, None], uv_front[Fd], uv_back[Fd])
    export_two_sided_obj(os.path.join(args.out_dir, "target_merged.obj"), Vd, Fd, uv_front,
                         uv_back, front, image)

    span_b = tt.uv_span(uv_b[Fd], img_b.shape)[~front]
    stats = dict(faces=int(len(Fd)), front_faces=int(front.sum()),
                 back_faces=int((~front).sum()), back_identity=bool(args.back_identity),
                 back_within_5pct=round(float((span_b < 0.05).mean()), 4) if len(span_b) else None,
                 view=[round(float(v), 6) for v in view])
    print(json.dumps(stats))
    with open(os.path.join(args.out_dir, "merge.json"), "w") as f:
        json.dump(stats, f, indent=1)

    left, right = side_views(view)
    panels = [("front: photo through the relief", view), ("left", left),
              ("back: generated texture", -view), ("right", right)]
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.4))
    for ax, (title, v) in zip(axes, panels):
        ax.imshow(tt.rasterize(Vd, Fd, uvc, image, v, args.size))
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    out = os.path.join(args.out_dir, "merged.png")
    fig.savefig(out, dpi=90)
    plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    main()
