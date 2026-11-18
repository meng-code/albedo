"""
Keypoint accuracy of every annotated pair for several correspondence results at once: one row
per pair and method in a CSV, plus the means per group (the fine-scale table).

Reads results/diff3f/keypoints/<pair>.json (and the older results/diff3f/landmarks/<pair>.json
for the pairs that only have those), finds the pair's meshes and feature files the way
eval_features.py does, and runs eval_keypoints.py's measurement for the nearest neighbour and
for every --result name=<out_dir of eval_features.py> (the refined.npz under <group>/<pair>/).

Usage (repository root, mesh environment):
    python scripts/idarb/keypoint_table.py \
        --result arap_07=outputs/diff3f/08_features_diff3f \
        --result c2f3_09=outputs/diff3f/09_texture_view_consistent \
        --csv results/diff3f/keypoints/keypoint_accuracy.csv
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_features as ef  # noqa: E402
import register_partial as rp  # noqa: E402
from correspondence_maps import load_mesh, mesh_features, nearest_in_features  # noqa: E402
from eval_keypoints import landmark_errors, summary  # noqa: E402


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--feature_dir", default="outputs/diff3f/features")
    ap.add_argument("--result", action="append", default=[], help="name=<eval_features out_dir>")
    ap.add_argument("--keypoints", default="results/diff3f/keypoints")
    ap.add_argument("--landmarks", default="results/diff3f/landmarks")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--plane", action="store_true",
                    help="also report every method's error in the relief's image plane (_xy)")
    args = ap.parse_args()

    rows = []
    for group, name, src, src_mesh, target, _tr, _ex in ef.pair_table(["same", "cross"]):
        lm_path = os.path.join(args.keypoints, name + ".json")
        if not os.path.exists(lm_path):
            lm_path = os.path.join(args.landmarks, name + ".json")
        if not os.path.exists(lm_path):
            continue
        with open(lm_path) as f:
            lm = json.load(f)
        lm_s = np.asarray(lm["source"], dtype=np.float64)
        lm_t = np.asarray(lm["target"], dtype=np.float64)
        Vs = np.asarray(load_mesh(src_mesh).vertices, dtype=np.float64)
        Vd = np.asarray(load_mesh(ef.TARGET_MESH[target]).vertices, dtype=np.float64)
        diag_s = float(np.linalg.norm(Vs.max(axis=0) - Vs.min(axis=0)))
        src_npz = os.path.join(args.feature_dir, f"{src}_partial.npz")
        dst_npz = os.path.join(args.feature_dir, f"{target}_full.npz")
        if not (os.path.exists(src_npz) and os.path.exists(dst_npz)):
            continue  # a feature set computed for part of the pairs only
        S = rp.load_features(src_npz)
        _, _, fd, _ = mesh_features(dst_npz, ef.TARGET_MESH[target])
        nn = S["idx"][nearest_in_features(fd, S["feat"])]
        positions = {"nearest_neighbour": Vs[nn]}
        for item in args.result:
            rname, out_dir = item.split("=", 1)
            path = os.path.join(out_dir, group, name, "refined.npz")
            if not os.path.exists(path):
                continue
            z = np.load(path)
            positions[rname] = z["src_pos"].astype(np.float64) if "src_pos" in z.files \
                else Vs[z["t2s"]]
        results = {}
        for rname, P in positions.items():
            results[rname] = landmark_errors(P, Vd, lm_s, lm_t, diag_s)
            if args.plane:
                # the same error in the relief's image plane (x, y only)
                results[rname + "_xy"] = landmark_errors(P, Vd, lm_s, lm_t, diag_s, plane=True)
        for rname, err in results.items():
            s = summary(err)
            rows.append(dict(group=group, pair=name, method=rname, n=len(err), mean=s["mean"],
                             median=s["median"], pck_2=s["pck_2"], pck_5=s["pck_5"],
                             pck_10=s["pck_10"]))
            print(json.dumps(rows[-1]), flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    # group means, weighted by pair (each pair counts once)
    print("\nmeans per group and method (mean error, PCK@2 %, PCK@5 %, PCK@10 %):")
    for group in ("same", "cross"):
        methods = sorted({r["method"] for r in rows if r["group"] == group}, key=str)
        for m in methods:
            rs = [r for r in rows if r["group"] == group and r["method"] == m]
            if not rs:
                continue
            mean = np.mean([r["mean"] for r in rs])
            p2, p5, p10 = (np.mean([r[k] for r in rs]) for k in ("pck_2", "pck_5", "pck_10"))
            print(f"  {group:<6} {m:<18} pairs {len(rs):2d}  mean {mean:.3f}  "
                  f"pck2 {p2:.2f}  pck5 {p5:.2f}  pck10 {p10:.2f}")


if __name__ == "__main__":
    main()
