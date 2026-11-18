"""
Where nearest-neighbour matching goes wrong with a given source, keypoint by keypoint. For every
annotated keypoint of a pair: the error of the target keypoint's nearest neighbour in feature
space (fraction of the source diagonal), whether that neighbour faces the photo's camera, the
rank of the true location among all source points by similarity (0 = the nearest neighbour is
the true location) and the similarity of the true location to the target keypoint's feature.
A low similarity and a high rank at the true location mean the source's feature there does not
look like the target's; a low rank with a large error means a near-duplicate elsewhere (the
other side of a symmetric object) won.

Sources: the relief (--source relief: <name>_partial.npz in --feature_dir, keypoints in
--keypoints) or a generated mesh (--source hunyuan3d|trellis2: <model>/<name>.npz and
<name>_mesh.ply in --gen_features, keypoints of gen_keypoints.py in --keypoints_gen, the camera
from --aligned/<model>/<name>/transform.json).

Usage (repository root, mesh environment):
    python scripts/idarb/keypoint_rank.py --source hunyuan3d --csv results/diff3f/rank_hunyuan3d.csv
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_features as ef  # noqa: E402
import register_partial as rp  # noqa: E402
from correspondence_maps import load_mesh, mesh_features  # noqa: E402

FIELDS = ["pair", "source", "keypoint", "error", "nn_faces_camera", "rank", "sim_true"]


def true_rank(sim, true_idx):
    """For every row of sim (keypoints x source points): how many source points score higher
    than the true one."""
    own = sim[np.arange(len(true_idx)), true_idx]
    return (sim > own[:, None]).sum(axis=1), own


def keypoint_rows(pair, source, ft, S_xyz, S_feat, facing, lm_s, diag_s, names):
    sim = ft @ S_feat.T
    nn = sim.argmax(axis=1)
    err = np.linalg.norm(S_xyz[nn] - lm_s, axis=1) / diag_s
    true_idx = cKDTree(S_xyz).query(lm_s)[1]
    rank, own = true_rank(sim, true_idx)
    return [dict(pair=pair, source=source, keypoint=names[k], error=round(float(err[k]), 4),
                 nn_faces_camera=bool(facing[nn[k]]), rank=int(rank[k]),
                 sim_true=round(float(own[k]), 4)) for k in range(len(nn))]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--source", default="relief", choices=["relief", "hunyuan3d", "trellis2"])
    ap.add_argument("--feature_dir", default="outputs/diff3f/features_zoom2")
    ap.add_argument("--gen_features", default="outputs/diff3f/features_zoom2_gen")
    ap.add_argument("--aligned", default="outputs/diff3f/13_generated")
    ap.add_argument("--keypoints", default="results/diff3f/keypoints")
    ap.add_argument("--landmarks", default="results/diff3f/landmarks")
    ap.add_argument("--keypoints_gen", default="results/diff3f/keypoints_gen")
    ap.add_argument("--pairs", nargs="*")
    ap.add_argument("--csv")
    args = ap.parse_args()
    rows = []
    for _g, name, src, _m, tgt, _tr, _e in ef.pair_table(["same", "cross"]):
        if args.pairs and name not in args.pairs:
            continue
        if args.source == "relief":
            kp_path = os.path.join(args.keypoints, name + ".json")
            if not os.path.exists(kp_path):
                kp_path = os.path.join(args.landmarks, name + ".json")
            npz = os.path.join(args.feature_dir, src + "_partial.npz")
            mesh = f"outputs/mesh_depth/{src}/relief.ply"
            tr_path = None
        else:
            kp_path = os.path.join(args.keypoints_gen, args.source, name + ".json")
            npz = os.path.join(args.gen_features, args.source, src + ".npz")
            mesh = os.path.join(args.gen_features, args.source, src + "_mesh.ply")
            tr_path = os.path.join(args.aligned, args.source, src, "transform.json")
        dst_npz = os.path.join(args.feature_dir, tgt + "_full.npz")
        needed = [kp_path, npz, mesh, dst_npz] + ([tr_path] if tr_path else [])
        if not all(os.path.exists(p) for p in needed):
            continue
        with open(kp_path) as f:
            lm = json.load(f)
        names = lm.get("names") or [str(i) for i in range(len(lm["source"]))]
        lm_s = np.asarray(lm["source"], dtype=np.float64)
        Vd, _, fd, _ = mesh_features(dst_npz, ef.TARGET_MESH[tgt])
        ft = fd[cKDTree(Vd).query(np.asarray(lm["target"], dtype=np.float64))[1]]
        S = rp.load_features(npz)
        m = load_mesh(mesh)
        Vs = np.asarray(m.vertices, dtype=np.float64)
        normals = np.asarray(m.vertex_normals, dtype=np.float64)
        if tr_path:
            with open(tr_path) as f:
                normals = normals @ np.array(json.load(f)["camera_rotation"]).T
        facing = normals[S["idx"], 2] > 0
        diag_s = float(np.linalg.norm(Vs.max(axis=0) - Vs.min(axis=0)))
        new = keypoint_rows(name, args.source, ft, S["xyz"], S["feat"], facing, lm_s, diag_s,
                            names)
        rows += new
        e = np.array([r["error"] for r in new])
        print("{:16s} n {:2d}  nn_pck_2 {:.2f}  median rank {:5.0f}  mean sim_true {:.3f}  "
              "nn facing camera {:.2f}".format(
                  name, len(new), float((e < 0.02).mean()),
                  float(np.median([r["rank"] for r in new])),
                  float(np.mean([r["sim_true"] for r in new])),
                  float(np.mean([r["nn_faces_camera"] for r in new]))))
    if args.csv and rows:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)) or ".", exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
