"""
Candidate upper bound of the correspondence: for every annotated target keypoint (its nearest
full sample), is ANY of the k most similar relief samples within 2 % of the relief diagonal of
the annotated relief point, and at which rank is the first? The nearest neighbour is k = 1; the
chain chooses among k = 64. If the correct region is inside the candidates but the chain does
not pick it, a re-ranking with context has headroom; if it is not, the features themselves must
change.

Usage (repository root, mesh environment):
    python scripts/idarb/candidate_bound.py --feature_dir outputs/diff3f/features_zoom2 \
        --csv results/diff3f/candidate_bound.csv
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
from correspondence_maps import load_mesh  # noqa: E402

KS = (1, 4, 16, 64, 256, 1024)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feature_dir", default="outputs/diff3f/features_zoom2")
    ap.add_argument("--keypoints", default="results/diff3f/keypoints")
    ap.add_argument("--landmarks", default="results/diff3f/landmarks")
    ap.add_argument("--threshold", type=float, default=0.02)
    ap.add_argument("--csv", required=True)
    args = ap.parse_args()
    rows, ranks = [], []
    for name, s, t in ef.SAME:
        kp = os.path.join(args.keypoints, f"{name}.json")
        if not os.path.exists(kp):
            kp = os.path.join(args.landmarks, f"{name}.json")
        if not os.path.exists(kp):
            continue
        d = json.load(open(kp))
        lm_s, lm_t = np.asarray(d["source"], float), np.asarray(d["target"], float)
        src = np.load(os.path.join(args.feature_dir, f"{s}_partial.npz"))
        dst = np.load(os.path.join(args.feature_dir, f"{t}_full.npz"))
        fs, xs = src["feat"].astype(np.float32), src["xyz"].astype(float)
        fd, xd = dst["feat"].astype(np.float32), dst["xyz"].astype(float)
        Vs = np.asarray(load_mesh(f"outputs/mesh_depth/{s}/relief.ply").vertices, float)
        diag_s = float(np.linalg.norm(Vs.max(0) - Vs.min(0)))
        t_idx = cKDTree(xd).query(lm_t)[1]
        order = np.argsort(-(fd[t_idx] @ fs.T), axis=1)
        dist = np.linalg.norm(xs[order] - lm_s[:, None], axis=2) / diag_s
        within = dist <= args.threshold
        first = np.where(within.any(axis=1), within.argmax(axis=1) + 1, np.inf)
        region = (np.linalg.norm(xs[None] - lm_s[:, None], axis=2) / diag_s
                  <= args.threshold).sum(axis=1)
        ranks += list(first)
        row = dict(pair=name, n=len(lm_t), region_samples_median=int(np.median(region)))
        for k in KS:
            row[f"top{k}"] = int((first <= k).sum())
        rows.append(row)
        print(json.dumps(row))
    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    n = sum(r["n"] for r in rows)
    r = np.array(ranks)
    fin = r[np.isfinite(r)]
    print(json.dumps(dict(
        keypoints=n, **{f"top{k}": round(sum(r_[f"top{k}"] for r_ in rows) / n, 3) for k in KS},
        first_correct_rank_median=int(np.median(fin)), rank_75th=int(np.percentile(fin, 75)),
        never=int((~np.isfinite(r)).sum()))))


if __name__ == "__main__":
    main()
