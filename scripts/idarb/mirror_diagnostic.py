"""
Is left-right mirroring what the correspondence gets wrong at the keypoints?

Three measurements on the annotated same-category pairs, with the target's symmetry plane found
by refine_correspondence.symmetry_plane and the relief's side of it taken from the filtered
mutual anchors' similarity transform (side_coordinates):

1. nearest neighbour: the share of target samples whose most similar source sample lies on the
   other side of the plane, and the keypoint accuracy of a side-constrained nearest neighbour
   (the most similar same-side candidate among the top k) against the plain one;
2. the chain's misses (keypoints farther than 2 % of the relief diagonal from the annotation):
   how many land within 2 % of the relief position of the keypoint's left-right partner (the
   annotated keypoint nearest the mirror image of the target keypoint), which is only testable
   for keypoints that have such a partner;
3. the chain's error bands (2-5 %, 5-10 %, over 10 %), to see whether the misses are near
   misses or gross ones.

Usage (repository root, mesh environment, after eval_features.py wrote
<result_dir>/same/<pair>/refined.npz):
    python scripts/idarb/mirror_diagnostic.py --result_dir outputs/diff3f/17_mirror_diag/rm5x3 \
        --csv results/diff3f/mirror_diagnostic.csv
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
import refine_correspondence as rc  # noqa: E402
from correspondence_maps import load_mesh  # noqa: E402


def keypoint_file(name, keypoints, landmarks):
    for d in (keypoints, landmarks):
        p = os.path.join(d, f"{name}.json")
        if os.path.exists(p):
            return p
    return None


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--result_dir", required=True,
                    help="eval_features out_dir with same/<pair>/refined.npz")
    ap.add_argument("--feature_dir", default="outputs/diff3f/features_zoom2")
    ap.add_argument("--keypoints", default="results/diff3f/keypoints")
    ap.add_argument("--landmarks", default="results/diff3f/landmarks")
    ap.add_argument("--band", type=float, default=0.03,
                    help="no side test within this fraction of the plane")
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--csv", required=True)
    args = ap.parse_args()

    rows = []
    for name, s, t in ef.SAME:
        kp = keypoint_file(name, args.keypoints, args.landmarks)
        refined = os.path.join(args.result_dir, "same", name, "refined.npz")
        if kp is None or not os.path.exists(refined):
            continue
        d = json.load(open(kp))
        lm_s, lm_t = np.asarray(d["source"], float), np.asarray(d["target"], float)
        src = np.load(os.path.join(args.feature_dir, f"{s}_partial.npz"))
        dst = np.load(os.path.join(args.feature_dir, f"{t}_full.npz"))
        fs, xs = src["feat"].astype(np.float32), src["xyz"].astype(float)
        fd, xd = dst["feat"].astype(np.float32), dst["xyz"].astype(float)
        Vd = np.asarray(load_mesh(ef.TARGET_MESH[t]).vertices, float)
        Vs = np.asarray(load_mesh(f"outputs/mesh_depth/{s}/relief.ply").vertices, float)
        diag_s = float(np.linalg.norm(Vs.max(0) - Vs.min(0)))
        diag_d = float(np.linalg.norm(Vd.max(0) - Vd.min(0)))

        # 1. nearest neighbour on the samples, plain and side-constrained
        ti, si, _ = rc.mutual_pairs(fd, fs)
        keep = rc.farthest_points(xd[ti], 150)
        T, S = xd[ti[keep]], xs[si[keep]]
        ok = rc.filter_anchors(T, S, "rigid", 0.1)
        side_s, side_t, ax, resid = rc.side_coordinates(xd, xs, T[ok], S[ok])
        cand, _ = rc.topk_candidates(fd, fs, args.k)
        pen = rc.side_penalty(cand, side_s, side_t, args.band)
        nn = cand[:, 0]
        allowed = pen == 0
        first = allowed.argmax(axis=1)
        side_nn = np.where(allowed.any(axis=1), cand[np.arange(len(cand)), first], nn)
        k_idx = cKDTree(xd).query(lm_t)[1]
        e_nn = np.linalg.norm(xs[nn[k_idx]] - lm_s, axis=1) / diag_s
        e_sd = np.linalg.norm(xs[side_nn[k_idx]] - lm_s, axis=1) / diag_s

        # 2. the chain's misses and their mirror partners
        z = np.load(refined)
        P = z["src_pos"].astype(float)
        e_ch = np.linalg.norm(P[cKDTree(Vd).query(lm_t)[1]] - lm_s, axis=1) / diag_s
        c = Vd.mean(axis=0)
        M = lm_t.copy()
        M[:, ax] = 2 * c[ax] - M[:, ax]
        D = np.linalg.norm(lm_t[None] - M[:, None], axis=2) / diag_d
        np.fill_diagonal(D, 9)
        partner, has = D.argmin(axis=1), D.min(axis=1) < 0.03
        miss = e_ch > 0.02
        t_idx = cKDTree(Vd).query(lm_t)[1]
        mirror = [
            bool(has[i] and np.linalg.norm(P[t_idx[i]] - lm_s[partner[i]]) / diag_s <= 0.02)
            for i in range(len(lm_t)) if miss[i]
        ]
        rows.append(dict(
            pair=name, n=len(lm_t), with_partner=int(has.sum()), plane="xyz"[ax],
            mirror_residual=round(resid, 4), nn_other_side=round(float(pen[:, 0].mean()), 3),
            nn_pck_2=round(float((e_nn <= 0.02).mean()), 3),
            side_nn_pck_2=round(float((e_sd <= 0.02).mean()), 3),
            nn_mean=round(float(e_nn.mean()), 4), side_nn_mean=round(float(e_sd.mean()), 4),
            chain_miss=int(miss.sum()), chain_mirror=int(sum(mirror)),
            chain_2_5=int(((e_ch > 0.02) & (e_ch <= 0.05)).sum()),
            chain_5_10=int(((e_ch > 0.05) & (e_ch <= 0.10)).sum()),
            chain_over_10=int((e_ch > 0.10).sum()),
        ))
        print(json.dumps(rows[-1]))

    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    n = sum(r["n"] for r in rows)
    counts = ("with_partner", "chain_miss", "chain_mirror",
              "chain_2_5", "chain_5_10", "chain_over_10")
    tot = {k: sum(r[k] for r in rows) for k in counts}
    print(json.dumps(dict(
        pairs=len(rows), keypoints=n, **tot,
        nn_other_side_mean=round(float(np.mean([r["nn_other_side"] for r in rows])), 3),
        nn_pck_2=round(sum(r["nn_pck_2"] * r["n"] for r in rows) / n, 3),
        side_nn_pck_2=round(sum(r["side_nn_pck_2"] * r["n"] for r in rows) / n, 3),
        nn_mean=round(sum(r["nn_mean"] * r["n"] for r in rows) / n, 4),
        side_nn_mean=round(sum(r["side_nn_mean"] * r["n"] for r in rows) / n, 4),
    )))


if __name__ == "__main__":
    main()
