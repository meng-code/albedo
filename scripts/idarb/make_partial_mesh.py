"""
Cut the part of a full mesh that one camera sees, as a synthetic partial mesh with known ground
truth, for testing partial-to-full registration (register_partial.py) without a domain gap.

The camera looks at the mesh centre from direction --view (a vector in the mesh's frame). A vertex
is visible when the ray from the camera to it first hits the mesh within a small tolerance of the
vertex (Open3D ray casting); faces whose three vertices are visible are kept. The partial mesh is
written in the camera frame (x right, y up, z towards the camera, like the relief meshes of
normal_to_mesh.py), multiplied by --scale and shifted by --shift, so that the similarity transform
that maps it back onto the full mesh is known.

Writes <out>.ply and <out>_gt.npz with orig_idx (the full-mesh vertex index of every partial-mesh
vertex) and s, R, t such that  full = s * R @ partial + t.

Usage (repository root, mesh environment):
    python scripts/idarb/make_partial_mesh.py --mesh third_party/Diff3F/meshes/cow.obj \
        --view 0.3 0.3 1 --out results/diff3f/synthetic/cow_synth
"""

import argparse
import json
import os

import numpy as np
import trimesh


def camera_rotation(view, up=(0.0, 1.0, 0.0)):
    """Rows: camera right, camera up, direction towards the camera. Maps mesh-frame offsets to the
    camera frame, where the camera sits on +z looking at the origin."""
    d = np.asarray(view, dtype=np.float64)
    d /= np.linalg.norm(d)
    up = np.asarray(up, dtype=np.float64)
    right = np.cross(up, d)
    if np.linalg.norm(right) < 1e-6:
        right = np.cross((1.0, 0.0, 0.0), d)
    right /= np.linalg.norm(right)
    cam_up = np.cross(d, right)
    return np.stack([right, cam_up, d])


def visible_vertices(verts, faces, origin, tol):
    """Boolean mask of the vertices the camera at origin sees (ray casting with Open3D)."""
    import open3d as o3d

    mesh = o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(verts.astype(np.float32)), o3d.core.Tensor(faces.astype(np.int32))
    )
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh)
    dirs = verts - origin
    dist = np.linalg.norm(dirs, axis=1)
    dirs = dirs / dist[:, None]
    rays = np.hstack([np.repeat(origin[None], len(verts), axis=0), dirs]).astype(np.float32)
    hit = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
    return np.isfinite(hit) & (hit >= dist - tol)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--out", required=True, help="output path without extension")
    ap.add_argument(
        "--view", type=float, nargs=3, default=(0.0, 0.0, 1.0), help="direction towards the camera"
    )
    ap.add_argument("--up", type=float, nargs=3, default=(0.0, 1.0, 0.0))
    ap.add_argument(
        "--scale", type=float, default=1.7, help="partial = scale * camera coords + shift"
    )
    ap.add_argument("--shift", type=float, nargs=3, default=(0.4, -0.2, 0.3))
    ap.add_argument(
        "--depth_factor",
        type=float,
        default=1.0,
        help="also multiply the partial's z (depth) by this, to imitate a wrong relief depth scale",
    )
    args = ap.parse_args()

    m = trimesh.load(args.mesh, process=False, force="mesh")
    verts = np.asarray(m.vertices, dtype=np.float64)
    faces = np.asarray(m.faces, dtype=np.int64)
    lo, hi = verts.min(0), verts.max(0)
    centre = (lo + hi) / 2.0
    diag = float(np.linalg.norm(hi - lo))
    R_cam = camera_rotation(args.view, args.up)
    origin = centre + R_cam[2] * 10.0 * diag  # far away: nearly parallel rays
    vis = visible_vertices(verts, faces, origin, tol=1e-3 * diag)
    keep_faces = vis[faces].all(axis=1)
    orig_idx = np.unique(faces[keep_faces])
    new_index = -np.ones(len(verts), dtype=np.int64)
    new_index[orig_idx] = np.arange(len(orig_idx))
    pfaces = new_index[faces[keep_faces]]

    cam = (verts[orig_idx] - centre) @ R_cam.T  # camera frame, z towards the camera
    cam[:, 2] *= args.depth_factor
    shift = np.asarray(args.shift, dtype=np.float64)
    pverts = args.scale * cam + shift

    # ground truth: full = s R partial + t (exact when depth_factor is 1)
    s = 1.0 / args.scale
    R = R_cam.T
    t = centre - R @ shift / args.scale

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    trimesh.Trimesh(vertices=pverts, faces=pfaces, process=False).export(args.out + ".ply")
    meta = dict(
        mesh=args.mesh,
        view=list(args.view),
        up=list(args.up),
        scale=args.scale,
        shift=list(args.shift),
        depth_factor=args.depth_factor,
        n_full_vertices=int(len(verts)),
        n_partial_vertices=int(len(orig_idx)),
        n_partial_faces=int(len(pfaces)),
    )
    np.savez(args.out + "_gt.npz", orig_idx=orig_idx, s=s, R=R, t=t, meta=json.dumps(meta))
    print(json.dumps(meta))
    print("saved", args.out + ".ply", args.out + "_gt.npz")


if __name__ == "__main__":
    main()
