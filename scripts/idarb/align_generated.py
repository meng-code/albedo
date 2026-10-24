"""
Put a generated full mesh (Hunyuan3D-2.1 or TRELLIS.2, from the photo) into the photo's frame by
registering the relief to it, then carry the relief's keypoints and albedo onto it.

The relief is the photo's frame: x right, y up, z towards the camera, one vertex per pixel. The
generated mesh is first turned so that the fitted input camera (gen3d_render.py's azimuth and
elevation) looks down -z, then the relief is fitted to it: a similarity in x, y from the two
silhouettes' bounding boxes, a depth offset from the generated mesh's front surface under each
relief pixel, and ICP (register_partial.icp_rigid) that refines the rotation, translation and the
relief's depth factor. The result is the "easy case" of our registration (the same object seen
from the same side), and it gives:

  transform.json          relief -> generated mesh: camera rotation, s, R, t, depth factor,
                          residuals and the share of relief vertices within 2 % of the diagonal
  keypoints.json          the relief's annotated keypoints on the generated mesh (nearest vertex
                          of the transformed point), with the same names
  albedo_vc.ply           the generated mesh with the relief's albedo on the side the photo saw
                          (projective texturing: vertices the photo's camera sees, under a relief
                          pixel, take that pixel's colour; the rest keep the generated colour)
  check.png               the generated mesh (vertex colours, orthographic from +z) with the
                          keypoints, next to the relief with the same keypoints

Usage (repository root, mesh environment):
    python scripts/idarb/align_generated.py --relief outputs/mesh_depth/cat/relief.ply \
        --generated outputs/diff3f/gen3d/hunyuan3d/cat/textured_vc.ply \
        --view results/diff3f/gen3d/hunyuan3d/cat/renders_view.json \
        --keypoints results/diff3f/keypoints/cat_lion.json \
        --out_dir outputs/diff3f/13_generated/hunyuan3d/cat
"""

import argparse
import json
import os
import sys

import numpy as np
import trimesh
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import register_partial as rp  # noqa: E402
from transfer_texture import view_frame  # noqa: E402


def camera_rotation(azimuth, elevation):
    """Rows of the matrix that maps the generated mesh's frame to the photo frame, for a camera
    at PyTorch3D's (azimuth, elevation): the camera direction (sin az cos el, sin el, cos az cos el)
    becomes +z, with y kept up."""
    az, el = np.radians(azimuth), np.radians(elevation)
    d = np.array([np.sin(az) * np.cos(el), np.sin(el), np.cos(az) * np.cos(el)])
    return view_frame(d)


def front_depth(V, F, xy, direction=(0.0, 0.0, -1.0)):
    """Depth (z) of the first surface hit by a ray from far in front of the mesh at every xy,
    cast towards -z; NaN where the ray misses."""
    import open3d as o3d

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(
        o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(F))))
    z0 = float(V[:, 2].max()) + 1.0
    origins = np.column_stack([xy[:, 0], xy[:, 1], np.full(len(xy), z0)])
    rays = np.concatenate([origins, np.tile(direction, (len(xy), 1))], axis=1).astype(np.float32)
    hit = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
    depth = z0 - hit
    depth[~np.isfinite(hit)] = np.nan
    return depth


def carry_points(points, tr, Vg, normals_g):
    """Relief points (the photo's frame) onto the generated mesh with a stored transform.json:
    the transformed point's nearest camera-facing vertex. Returns the vertex indices, the
    vertices' positions in the generated mesh's own frame and the snap distances (fraction of
    the generated mesh's diagonal in the photo frame)."""
    Rc = np.array(tr["camera_rotation"])
    s, a = tr["scale"], tr["depth_factor"]
    R, t = np.array(tr["rotation"]), np.array(tr["translation"])
    Vg_cam = Vg @ Rc.T
    front = (normals_g @ Rc.T)[:, 2] > 0.2
    diag = float(np.linalg.norm(Vg_cam.max(axis=0) - Vg_cam.min(axis=0)))
    moved = rp.transform(rp.apply_depth(np.asarray(points, dtype=np.float64), a), s, R, t)
    idx = np.nonzero(front)[0][cKDTree(Vg_cam[front]).query(moved)[1]]
    snap = np.linalg.norm(Vg_cam[idx] - moved, axis=1) / diag
    return idx, Vg[idx], snap


def initial_fit(Vr, Vg_cam, Fg):
    """Similarity relief -> generated (in the photo frame) from the xy bounding boxes, with the
    depth offset from the generated front surface under the relief's pixels. Returns s, t."""
    lo_r, hi_r = Vr[:, :2].min(axis=0), Vr[:, :2].max(axis=0)
    lo_g, hi_g = Vg_cam[:, :2].min(axis=0), Vg_cam[:, :2].max(axis=0)
    s = float(np.mean((hi_g - lo_g) / np.maximum(hi_r - lo_r, 1e-9)))
    t = np.zeros(3)
    t[:2] = (lo_g + hi_g) / 2 - s * (lo_r + hi_r) / 2
    xy = s * Vr[:, :2] + t[:2]
    depth = front_depth(Vg_cam, Fg, xy)
    ok = np.isfinite(depth)
    t[2] = float(np.median(depth[ok] - s * Vr[ok, 2]))
    return s, t, float(ok.mean())


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--relief", required=True)
    ap.add_argument("--generated", required=True, help="textured_vc.ply of the generated mesh")
    ap.add_argument("--view", required=True, help="renders_view.json of gen3d_render.py")
    ap.add_argument("--keypoints", help="landmarks JSON whose 'source' points are on the relief")
    ap.add_argument("--icp_dist", type=float, default=0.05, help="ICP pairing distance, of diag")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    relief = trimesh.load(args.relief, process=False, force="mesh")
    gen = trimesh.load(args.generated, process=False, force="mesh")
    Vr = np.asarray(relief.vertices, dtype=np.float64)
    Cr = np.asarray(relief.visual.vertex_colors)[:, :3]
    Vg = np.asarray(gen.vertices, dtype=np.float64)
    Fg = np.asarray(gen.faces, dtype=np.int64)
    Cg = np.asarray(gen.visual.vertex_colors)[:, :3] if gen.visual.kind == "vertex" else None
    with open(args.view) as f:
        view = json.load(f)
    Rc = camera_rotation(view.get("azimuth", 0.0), view.get("elevation", 0.0))
    Vg_cam = Vg @ Rc.T  # the generated mesh seen from the photo's camera: +z towards it
    diag = float(np.linalg.norm(Vg_cam.max(axis=0) - Vg_cam.min(axis=0)))

    s, t, hit = initial_fit(Vr, Vg_cam, Fg)
    R = np.eye(3)
    # ICP: the relief's front vertices against the generated mesh's front surface (vertices whose
    # normal faces the camera), refining R, t and the relief's depth factor
    normals_g = np.asarray(gen.vertex_normals) @ Rc.T
    front = normals_g[:, 2] > 0.2
    R, t, a, rmse, kept, _ = rp.icp_rigid(
        Vr, Vg_cam[front], s, R, t, args.icp_dist * diag, iters=60, depth_scale=True
    )
    P = rp.transform(rp.apply_depth(Vr, a), s, R, t)  # relief vertices in the generated frame
    d_front = cKDTree(Vg_cam[front]).query(P)[0] / diag
    stats = dict(
        azimuth=view.get("azimuth", 0.0), elevation=view.get("elevation", 0.0),
        hit_fraction=round(hit, 4), scale=s, depth_factor=a, rotation=R.tolist(),
        translation=t.tolist(), camera_rotation=Rc.tolist(), icp_rmse=round(float(rmse), 5),
        icp_kept=round(float(kept), 4),
        within_2pct=round(float((d_front < 0.02).mean()), 4),
        within_5pct=round(float((d_front < 0.05).mean()), 4),
        rotation_deg=round(float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))), 2),
    )
    with open(os.path.join(args.out_dir, "transform.json"), "w") as f:
        json.dump(stats, f, indent=1)

    # albedo onto the generated mesh, as projective texturing: every vertex that the photo's
    # camera sees (the first surface hit by the ray through it) and that lies under a relief
    # pixel takes that pixel's colour
    step = float(np.diff(np.unique(np.round(Vr[:, 0], 6))).min()) * s
    tree_r = cKDTree(P[:, :2])
    d_xy, j = tree_r.query(Vg_cam[:, :2])
    seen = front_depth(Vg_cam, Fg, Vg_cam[:, :2])
    visible = np.isfinite(seen) & (np.abs(seen - Vg_cam[:, 2]) < 0.01 * diag)
    close = (d_xy < 0.75 * step) & visible
    colours = Cg.copy() if Cg is not None else np.full((len(Vg), 3), 180, dtype=np.uint8)
    colours[close] = Cr[j[close]]
    stats["albedo_vertices"] = int(close.sum())
    stats["albedo_fraction_of_front"] = round(float(close.sum() / max(front.sum(), 1)), 4)
    trimesh.Trimesh(vertices=Vg, faces=Fg, vertex_colors=colours, process=False).export(
        os.path.join(args.out_dir, "albedo_vc.ply")
    )
    # also the generated mesh in the photo frame (so later steps need no camera rotation)
    trimesh.Trimesh(vertices=Vg_cam, faces=Fg, vertex_colors=colours, process=False).export(
        os.path.join(args.out_dir, "albedo_vc_photo_frame.ply")
    )

    kp_out = None
    if args.keypoints:
        with open(args.keypoints) as f:
            lm = json.load(f)
        src = np.asarray(lm["source"], dtype=np.float64)
        moved = rp.transform(rp.apply_depth(src, a), s, R, t)
        idx = cKDTree(Vg_cam[front]).query(moved)[1]
        idx = np.nonzero(front)[0][idx]
        kp_out = dict(
            names=lm.get("names", [str(i) for i in range(len(src))]),
            generated_photo_frame=Vg_cam[idx].tolist(),
            generated=Vg[idx].tolist(),
            vertex=idx.tolist(),
            snap_distance=(np.linalg.norm(Vg_cam[idx] - moved, axis=1) / diag).round(4).tolist(),
        )
        with open(os.path.join(args.out_dir, "keypoints.json"), "w") as f:
            json.dump(kp_out, f, indent=1)
        stats["keypoint_snap_max"] = max(kp_out["snap_distance"])

    # check image: relief (colours, keypoints) | generated mesh in the photo frame (colours,
    # albedo-covered vertices, keypoints)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].scatter(Vr[:, 0], Vr[:, 1], c=Cr / 255.0, s=1)
    axes[0].set_title("relief (photo frame)")
    order = np.argsort(Vg_cam[:, 2])
    axes[1].scatter(Vg_cam[order, 0], Vg_cam[order, 1], c=colours[order] / 255.0, s=1)
    axes[1].set_title(f"generated, albedo on {stats['albedo_fraction_of_front']:.0%} of the front")
    axes[2].scatter(Vg_cam[order, 0], Vg_cam[order, 1], c=np.where(close[order], 0.2, 0.85),
                    cmap="gray", vmin=0, vmax=1, s=1)
    axes[2].set_title(f"albedo coverage; ICP rms {rmse / diag:.3f} diag, {kept:.0%} kept")
    if kp_out is not None:
        axes[0].scatter(src[:, 0], src[:, 1], c="red", s=25, marker="x")
        G = np.asarray(kp_out["generated_photo_frame"])
        for ax in axes[1:]:
            ax.scatter(G[:, 0], G[:, 1], c="red", s=25, marker="x")
        for i, (x, y) in enumerate(src[:, :2]):
            axes[0].annotate(str(i), (x, y), color="red", fontsize=8)
        for i, (x, y) in enumerate(G[:, :2]):
            axes[1].annotate(str(i), (x, y), color="red", fontsize=8)
    for ax in axes:
        ax.set_aspect("equal")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "check.png"), dpi=90)
    with open(os.path.join(args.out_dir, "transform.json"), "w") as f:
        json.dump(stats, f, indent=1)
    print(json.dumps({k: v for k, v in stats.items() if k not in ("rotation", "camera_rotation")}))


if __name__ == "__main__":
    main()
