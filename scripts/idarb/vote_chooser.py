"""
A training-free chooser over the top-k feature candidates, to test whether the candidate
headroom (candidate_bound.py: the correct region is among the top 64 for 81 % of the
keypoints) can be taken by neighbour consistency alone. Every candidate of a target sample
scores its similarity plus lambda times the mean, over the sample's 3D neighbours, of the best
support a neighbour's own candidate list gives to a relief position near the candidate
(Gaussian in relief distance, weighted by the neighbour's candidate similarity); optional
rounds afterwards let the neighbours' chosen positions vote. lambda = 0 is the nearest
neighbour. Keypoint PCK@2 on the 13 same-category pairs.

Usage (repository root, mesh environment; about 7 minutes):
    python scripts/idarb/vote_chooser.py --csv results/diff3f/vote_chooser.csv
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_features as ef  # noqa: E402
import refine_correspondence as rc  # noqa: E402
from correspondence_maps import load_mesh  # noqa: E402


def choose(cand, sim, xs, xd, lam, sigma, nb, rounds):
    """The chosen relief sample per target sample after the one-shot vote (round 0) and after
    each refinement round, as a list of index arrays."""
    n, k = cand.shape
    rows = np.arange(n)
    C = xs[cand]  # (n, k, 3)
    nbr = cKDTree(xd).query(xd, nb + 1)[1][:, 1:]  # (n, nb)
    w = np.clip(sim, 0, None)
    score = sim.astype(np.float64).copy()
    if lam > 0:
        vote = np.zeros((n, k))
        for j in range(nb):
            u = nbr[:, j]
            for a in range(0, n, 2000):  # chunked: (2000, k, k, 3) at a time
                b = min(a + 2000, n)
                d2 = ((C[a:b, :, None, :] - C[u[a:b]][:, None, :, :]) ** 2).sum(-1)
                vote[a:b] += (w[u[a:b]][:, None, :] * np.exp(-d2 / (2 * sigma**2))).max(-1)
        score = score + lam * vote / nb
    choice = score.argmax(1)
    out = [cand[rows, choice]]
    for _ in range(rounds):
        if lam <= 0:
            out.append(out[-1])
            continue
        Pn = C[rows, choice][nbr]  # (n, nb, 3): the neighbours' chosen positions
        d2 = ((C[:, :, None, :] - Pn[:, None, :, :]) ** 2).sum(-1)  # (n, k, nb)
        choice = (sim + lam * np.exp(-d2 / (2 * sigma**2)).mean(-1)).argmax(1)
        out.append(cand[rows, choice])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feature_dir", default="outputs/diff3f/features_zoom2")
    ap.add_argument("--keypoints", default="results/diff3f/keypoints")
    ap.add_argument("--landmarks", default="results/diff3f/landmarks")
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--neighbours", type=int, default=12)
    ap.add_argument("--sigma", type=float, default=0.03, help="fraction of the relief diagonal")
    ap.add_argument("--lambdas", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0, 4.0])
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--csv", required=True)
    args = ap.parse_args()
    rows, n_tot, t0 = [], 0, time.time()
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
        cand, sim = rc.topk_candidates(fd, fs, args.k)
        k_idx = cKDTree(xd).query(lm_t)[1]
        row = dict(pair=name, n=len(lm_t))
        for lam in args.lambdas:
            chosen = choose(cand, sim, xs, xd, lam, args.sigma * diag_s, args.neighbours,
                            args.rounds)
            for r, ch in enumerate(chosen):
                err = np.linalg.norm(xs[ch[k_idx]] - lm_s, axis=1) / diag_s
                row[f"lam{lam:g}_round{r}"] = int((err <= 0.02).sum())
        n_tot += len(lm_t)
        rows.append(row)
        print(json.dumps(row), f"{time.time() - t0:.0f}s", flush=True)
    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    keys = [k for k in rows[0] if k.startswith("lam")]
    print(json.dumps(dict(keypoints=n_tot, k=args.k, neighbours=args.neighbours,
                          sigma=args.sigma,
                          **{k: round(sum(r[k] for r in rows) / n_tot, 3) for k in keys})))


if __name__ == "__main__":
    main()
