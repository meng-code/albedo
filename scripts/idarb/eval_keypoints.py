"""
Keypoint-level accuracy of a correspondence: where does it send the target's annotated points
(eye, nose, ear tip, ...) on the relief, and how far is that from the annotated source point?

This is the fine-scale measure the supervisor asked for ("eye to eye, nose to nose"). The span,
distortion and mirror numbers of transfer_texture.py / uv_metrics.py measure how smooth a map
is; this one measures whether it is right at the places a person would check.

Inputs: a landmarks JSON of pick_landmarks.py --merge (lists "source" and "target" of 3D points
in the same order; optional "names"), the two meshes, and one or more correspondence files
(refine_correspondence.py npz with t2s and, for a field, src_pos). The nearest-neighbour
correspondence is computed from the feature files when --source / --target are given.

For every landmark: the target vertex nearest to the annotated target point is mapped through
the correspondence to a source position (src_pos when present, else the matched source vertex);
the error is its distance to the annotated source point as a fraction of the relief's bounding
box diagonal. Reports every landmark's error, the mean, and the fraction within 2 % / 5 % (PCK).

Usage (repository root, mesh environment):
    python scripts/idarb/eval_keypoints.py --landmarks results/diff3f/landmarks/cat_lion.json \
        --source_mesh outputs/mesh_depth/cat/relief.ply \
        --target_mesh third_party/Diff3F/meshes/lion.off \
        --source outputs/diff3f/features/cat_partial.npz \
        --target outputs/diff3f/features/lion_full.npz \
        --correspondence arap=outputs/diff3f/07_texture_arap/same_category/cat_lion/refined.npz \
            new=outputs/diff3f/09_texture_view_consistent/same/cat_lion/refined.npz
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import register_partial as rp  # noqa: E402
from correspondence_maps import load_mesh, mesh_features, nearest_in_features  # noqa: E402


def landmark_errors(P_source_of_target, Vd, lm_source, lm_target, diag_s):
    """Errors (fraction of the source diagonal) of the mapped target landmarks."""
    t_idx = cKDTree(Vd).query(lm_target)[1]
    mapped = P_source_of_target[t_idx]
    return np.linalg.norm(mapped - lm_source, axis=1) / diag_s


def summary(err):
    return dict(
        mean=round(float(err.mean()), 4),
        median=round(float(np.median(err)), 4),
        pck_2=round(float((err < 0.02).mean()), 4),
        pck_5=round(float((err < 0.05).mean()), 4),
        pck_10=round(float((err < 0.10).mean()), 4),
        per_landmark=[round(float(e), 4) for e in err],
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--landmarks", required=True, help="merged landmarks JSON")
    ap.add_argument("--source_mesh", required=True, help="the relief")
    ap.add_argument("--target_mesh", required=True)
    ap.add_argument("--source", help="features of the relief, for the nearest-neighbour row")
    ap.add_argument("--target", help="features of the full mesh")
    ap.add_argument(
        "--correspondence",
        nargs="*",
        default=[],
        help="name=path of refine_correspondence.py npz files (t2s, optional src_pos)",
    )
    ap.add_argument("--csv", help="append one row per correspondence here")
    ap.add_argument("--label", default="", help="prefix for the CSV rows, e.g. the pair name")
    args = ap.parse_args()

    with open(args.landmarks) as f:
        lm = json.load(f)
    lm_s = np.asarray(lm["source"], dtype=np.float64)
    lm_t = np.asarray(lm["target"], dtype=np.float64)
    names = lm.get("names") or [str(i) for i in range(len(lm_s))]
    Vs = np.asarray(load_mesh(args.source_mesh).vertices, dtype=np.float64)
    Vd = np.asarray(load_mesh(args.target_mesh).vertices, dtype=np.float64)
    diag_s = float(np.linalg.norm(Vs.max(axis=0) - Vs.min(axis=0)))
    rows = {}
    if args.source and args.target:
        src = rp.load_features(args.source)
        _, _, fd, _ = mesh_features(args.target, args.target_mesh)
        nn = src["idx"][nearest_in_features(fd, src["feat"])]
        rows["nearest_neighbour"] = landmark_errors(Vs[nn], Vd, lm_s, lm_t, diag_s)
    for item in args.correspondence:
        name, path = item.split("=", 1) if "=" in item else (os.path.basename(item), item)
        z = np.load(path)
        P = z["src_pos"].astype(np.float64) if "src_pos" in z.files else Vs[z["t2s"]]
        rows[name] = landmark_errors(P, Vd, lm_s, lm_t, diag_s)
    out = {k: summary(v) for k, v in rows.items()}
    out["names"] = names
    print(json.dumps(out))
    if args.csv:
        import csv

        new = not os.path.exists(args.csv)
        with open(args.csv, "a", newline="") as f:
            w = csv.writer(f)
            cols = ["mean", "median", "pck_2", "pck_5", "pck_10"]
            if new:
                w.writerow(["pair", "correspondence", "n"] + cols)
            for k, v in rows.items():
                s = summary(v)
                w.writerow([args.label, k, len(v)] + [s[c] for c in cols])


if __name__ == "__main__":
    main()
