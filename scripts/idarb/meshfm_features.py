"""
MeshFM semantic features at the same points as existing Diff3F feature files, so the correspondence
scripts can use them unchanged. Runs on the GPU server in the meshfm environment
(scripts/idarb/requirements-meshfm.txt); the method code is a clone of threedle/MeshFM.

MeshFM (Zhou, Liu, Lang and Hanocka, ECCV 2026) is a network that looks at a 3D point cloud once
and outputs a feature field: three 128 x 128 feature planes (a "triplane") around the shape. Any 3D
point can then be given a 1536-number feature by reading the three planes at its position. The
network was trained to reproduce DINOv2 image features that were painted onto Objaverse shapes, so
its features describe what a point is (an ear, a leg), like Diff3F's, but it needs no rendering and
no diffusion at test time, and it was trained with random rotations, so the features should not
depend much on the shape's orientation.

For every input .npz written by diff3f_features.py the script reads the mesh named in its meta,
normalises the mesh the way MeshFM does (bounding-box centre at the origin, farthest vertex on the
unit sphere), samples --num_sample_points surface points (MeshFM's own torch sampler, seed --seed),
runs the encoder once, and reads the feature field at the npz's xyz points, transformed into the
same normalised frame. --close_mirror closes a one-sided relief before encoding by adding a copy
mirrored through the plane z = (z_min + z_max) / 2 ("mid": front and copy interpenetrate, a thin
closed shell) or z = z_min ("zmin": the copy sits behind the front, twice the depth); only the
original points are queried. On the synthetic partials "mid" gained a little and "zmin" lost (see
the commit message), so the default is "none". The output .npz has the same idx, xyz and colors;
only feat changes (float16, N x 1536, unit length), and meta is a JSON string with model,
checkpoint, repo commit, command, source mesh, number of input points and git commit.

Usage (repository root, meshfm environment):
    python scripts/idarb/meshfm_features.py --src outputs/diff3f/cat_full.npz \
        --out_dir outputs/diff3f/features_meshfm
    python scripts/idarb/meshfm_features.py --src outputs/diff3f \
        --out_dir outputs/diff3f/features_meshfm
"""

import argparse
import glob
import json
import os
import shlex
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def git_commit(path):
    try:
        return subprocess.check_output(["git", "-C", path, "rev-parse", "--short", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


def source_mesh(meta):
    """The --mesh path of the diff3f_features.py command stored in an npz's meta."""
    args = shlex.split(json.loads(meta)["command"])
    return args[args.index("--mesh") + 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="a Diff3F .npz or a folder of them")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--meshfm_dir", default=os.path.expanduser("~/semtex/MeshFM"))
    ap.add_argument("--checkpoint", default=None,
                    help="default: <meshfm_dir>/checkpoints/meshfm.pth")
    ap.add_argument("--config", default=None,
                    help="default: <meshfm_dir>/configs/meshfm_1536d.yaml")
    ap.add_argument("--num_sample_points", type=int, default=100000, help="encoder input points")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--close_mirror", default="none", choices=["none", "mid", "zmin"],
                    help="close the shape with a mirrored copy before encoding (see above)")
    args = ap.parse_args()

    sys.path.insert(0, args.meshfm_dir)
    import torch
    import trimesh
    from meshfm import MeshFM, normalization_transform

    ckpt = args.checkpoint or os.path.join(args.meshfm_dir, "checkpoints", "meshfm.pth")
    cfg = args.config or os.path.join(args.meshfm_dir, "configs", "meshfm_1536d.yaml")
    if os.path.isdir(args.src):
        srcs = sorted(glob.glob(os.path.join(args.src, "*.npz")))
    else:
        srcs = [args.src]
    os.makedirs(args.out_dir, exist_ok=True)
    model = MeshFM.from_checkpoint(ckpt, cfg)
    command = " ".join(["scripts/idarb/meshfm_features.py"] + sys.argv[1:])
    info = dict(model="MeshFM 1536-d", checkpoint=os.path.relpath(ckpt, args.meshfm_dir),
                repo="threedle/MeshFM", repo_commit=git_commit(args.meshfm_dir), command=command,
                git_commit=git_commit(ROOT), seed=args.seed, close_mirror=args.close_mirror)

    for src in srcs:
        d = np.load(src, allow_pickle=True)
        mesh_path = source_mesh(str(d["meta"]))
        mesh = trimesh.load(mesh_path, process=False, force="mesh")
        verts = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        if args.close_mirror != "none":
            z0, z1 = verts[:, 2].min(), verts[:, 2].max()
            c = (z0 + z1) / 2 if args.close_mirror == "mid" else z0
            back = verts.copy()
            back[:, 2] = 2 * c - back[:, 2]
            faces = np.vstack([faces, faces[:, ::-1] + len(verts)])
            verts = np.vstack([verts, back])
        centre, scale = normalization_transform(verts)
        queries = (d["xyz"].astype(np.float64) - centre) / scale

        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        res = model.extract_mesh_features((verts, faces), mode="custom", query_points=queries,
                                          num_sample_points=args.num_sample_points, seed=args.seed)
        seconds = time.time() - t0
        feat = res.features.astype(np.float32)
        feat /= np.linalg.norm(feat, axis=1, keepdims=True) + 1e-8

        meta = dict(info, source_mesh=mesh_path, source_npz=os.path.relpath(src, ROOT),
                    n_input_points=int(args.num_sample_points), n_query_points=int(len(queries)),
                    seconds=round(seconds, 2),
                    peak_gpu_mib=round(torch.cuda.max_memory_allocated() / 2 ** 20))
        out = os.path.join(args.out_dir, os.path.basename(src))
        np.savez(out, idx=d["idx"], xyz=d["xyz"], feat=feat.astype(np.float16), colors=d["colors"],
                 meta=json.dumps(meta))
        print(json.dumps(dict(out=out, n=int(len(queries)), seconds=meta["seconds"],
                              peak_gpu_mib=meta["peak_gpu_mib"])), flush=True)


if __name__ == "__main__":
    main()
