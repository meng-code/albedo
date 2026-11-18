"""
Keypoint files for the "generate, then match" pairs: the source side of every annotated pair
moved from the relief onto the generated mesh with align_generated.py's transform (nearest
camera-facing vertex of the transformed point, in the generated mesh's own frame), the target
side unchanged. One file per generator and pair under
results/diff3f/keypoints_gen/<model>/<pair>.json, in the layout eval_keypoints.py reads.

Usage (repository root, mesh environment):
    python scripts/idarb/gen_keypoints.py --aligned outputs/diff3f/13_generated \
        --keypoints results/diff3f/keypoints --out results/diff3f/keypoints_gen
"""

import argparse
import json
import os
import sys

import numpy as np
import trimesh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_features as ef  # noqa: E402
from align_generated import carry_points  # noqa: E402


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--aligned", default="outputs/diff3f/13_generated")
    ap.add_argument("--gen_meshes", default="outputs/diff3f/gen3d")
    ap.add_argument("--keypoints", default="results/diff3f/keypoints")
    ap.add_argument("--landmarks", default="results/diff3f/landmarks")
    ap.add_argument("--models", nargs="+", default=["hunyuan3d", "trellis2"])
    ap.add_argument("--out", default="results/diff3f/keypoints_gen")
    args = ap.parse_args()
    written = 0
    for _g, name, src, _m, _t, _tr, _e in ef.pair_table(["same", "cross"]):
        lm_path = os.path.join(args.keypoints, name + ".json")
        if not os.path.exists(lm_path):
            lm_path = os.path.join(args.landmarks, name + ".json")
        if not os.path.exists(lm_path):
            continue
        with open(lm_path) as f:
            lm = json.load(f)
        for model in args.models:
            tr_path = os.path.join(args.aligned, model, src, "transform.json")
            mesh_path = os.path.join(args.gen_meshes, model, src, "textured_vc.ply")
            if not (os.path.exists(tr_path) and os.path.exists(mesh_path)):
                continue
            with open(tr_path) as f:
                tr = json.load(f)
            gen = trimesh.load(mesh_path, process=False, force="mesh")
            idx, pts, snap = carry_points(
                lm["source"], tr, np.asarray(gen.vertices, dtype=np.float64),
                np.asarray(gen.vertex_normals, dtype=np.float64),
            )
            out = dict(source=pts.tolist(), target=lm["target"],
                       names=lm.get("names"), source_mesh=f"{model}/{src}",
                       vertex=idx.tolist(), relief_keypoints=lm_path,
                       snap_distance=np.round(snap, 4).tolist())
            os.makedirs(os.path.join(args.out, model), exist_ok=True)
            with open(os.path.join(args.out, model, name + ".json"), "w") as f:
                json.dump(out, f, indent=1)
            written += 1
    print(f"{written} keypoint files written under {args.out}")


if __name__ == "__main__":
    main()
