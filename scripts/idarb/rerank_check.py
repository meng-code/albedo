"""
Two feature sets on the same sampled points, compared and combined at the annotated keypoints
without running the chain. For every keypoint of every annotated pair (A = --a, B = --b):

  nn_a, nn_b        the nearest neighbour in each set
  b_in_top_a        the most similar sample by B among the --k most similar by A (B as the
                    selector over A's candidates), and a_in_top_b the other way round
  fused_<w>         the nearest neighbour of the weighted mean of the two cosine similarities
                    (weight w on A), for every --weights value
  bound_a, bound_b  a sample within the threshold of the truth is among the top --k of the set

A hit is an error below --threshold (2 %) of the partial mesh's bounding-box diagonal. One row
per pair and method in per_pair.csv; summary.csv pools the same-category pairs and gives the
paired difference against nn_a with a 95 % interval from a bootstrap over pairs. CPU only.

Usage (repository root, mesh environment):
    python scripts/idarb/rerank_check.py --a outputs/diff3f/features_zoom2 \
        --b outputs/diff3f/features_marco --csv_dir results/diff3f/marco_lifted/rerank
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
from correspondence_maps import load_mesh  # noqa: E402
from sweep_anchor_filter import git_commit, paired_interval  # noqa: E402


def top_k(sim, k):
    """Column indices of the k largest entries of every row (unordered)."""
    k = min(k, sim.shape[1])
    return np.argpartition(-sim, k - 1, axis=1)[:, :k]


def select_within(sim_pick, cand):
    """Per row, the candidate (column index) with the largest sim_pick."""
    return cand[np.arange(len(cand)), np.take_along_axis(sim_pick, cand, 1).argmax(axis=1)]


def methods_of(sim_a, sim_b, k, weights):
    """The chosen source sample per keypoint for every method, and the two candidate sets."""
    cand_a, cand_b = top_k(sim_a, k), top_k(sim_b, k)
    out = dict(nn_a=sim_a.argmax(axis=1), nn_b=sim_b.argmax(axis=1),
               b_in_top_a=select_within(sim_b, cand_a), a_in_top_b=select_within(sim_a, cand_b))
    for w in weights:
        out[f"fused_{w:g}"] = (w * sim_a + (1.0 - w) * sim_b).argmax(axis=1)
    return out, cand_a, cand_b


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", default="outputs/diff3f/features_zoom2")
    ap.add_argument("--b", default="outputs/diff3f/features_marco")
    ap.add_argument("--keypoints", default="results/diff3f/keypoints")
    ap.add_argument("--landmarks", default="results/diff3f/landmarks")
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--weights", type=float, nargs="*", default=[0.25, 0.5, 0.75])
    ap.add_argument("--threshold", type=float, default=0.02)
    ap.add_argument("--csv_dir", default="results/diff3f/marco_lifted/rerank")
    args = ap.parse_args()

    per_pair = []
    for group, name, s, src_mesh, t, _tr, _ex in ef.pair_table(["same", "cross"]):
        kp = os.path.join(args.keypoints, name + ".json")
        if not os.path.exists(kp):
            kp = os.path.join(args.landmarks, name + ".json")
        files = [os.path.join(d, n) for d in (args.a, args.b)
                 for n in (s + "_partial.npz", t + "_full.npz")]
        if not (os.path.exists(kp) and all(os.path.exists(f) for f in files)):
            continue
        with open(kp) as f:
            d = json.load(f)
        lm_s, lm_t = np.asarray(d["source"], float), np.asarray(d["target"], float)
        sa, da, sb, db = (rp.load_features(f) for f in files)
        if len(sa["xyz"]) != len(sb["xyz"]) or len(da["xyz"]) != len(db["xyz"]):
            raise ValueError("the two feature sets of " + name + " have different points")
        Vs = np.asarray(load_mesh(src_mesh).vertices, dtype=np.float64)
        diag = float(np.linalg.norm(Vs.max(axis=0) - Vs.min(axis=0)))
        t_idx = cKDTree(da["xyz"]).query(lm_t)[1]
        dist = np.linalg.norm(sa["xyz"][None] - lm_s[:, None], axis=2) / diag
        sim_a = da["feat"][t_idx] @ sa["feat"].T
        sim_b = db["feat"][t_idx] @ sb["feat"].T
        picks, cand_a, cand_b = methods_of(sim_a, sim_b, args.k, args.weights)
        rows = np.arange(len(t_idx))
        res = {m: dist[rows, p] for m, p in picks.items()}
        bounds = dict(bound_a=(np.take_along_axis(dist, cand_a, 1) < args.threshold).any(axis=1),
                      bound_b=(np.take_along_axis(dist, cand_b, 1) < args.threshold).any(axis=1))
        for m, err in res.items():
            per_pair.append(dict(group=group, pair=name, method=m, n=len(err),
                                 hits=int((err < args.threshold).sum()),
                                 mean_error=round(float(err.mean()), 4)))
        for m, ok in bounds.items():
            per_pair.append(dict(group=group, pair=name, method=m, n=len(ok),
                                 hits=int(ok.sum()), mean_error=""))
        print(json.dumps({r["method"]: r["hits"] for r in per_pair[-len(res) - 2:]},
                         sort_keys=False), name, flush=True)
    if not per_pair:
        sys.exit("no annotated pair with both feature sets")

    same = [r for r in per_pair if r["group"] == "same"]
    methods = list(dict.fromkeys(r["method"] for r in same))
    by = {m: {r["pair"]: r for r in same if r["method"] == m} for m in methods}
    pairs = sorted(by["nn_a"])
    n = np.array([by["nn_a"][p]["n"] for p in pairs])
    base = np.array([by["nn_a"][p]["hits"] for p in pairs])
    summary = []
    for m in methods:
        hits = np.array([by[m][p]["hits"] for p in pairs])
        errs = [by[m][p]["mean_error"] for p in pairs if by[m][p]["mean_error"] != ""]
        row = dict(method=m, pairs=len(pairs), keypoints=int(n.sum()), hits=int(hits.sum()),
                   pck2=round(100.0 * hits.sum() / n.sum(), 1),
                   mean_error=round(float(np.mean(errs)), 4) if errs else "",
                   diff_vs_nn_a="", ci_low="", ci_high="")
        if m != "nn_a" and not m.startswith("bound"):
            diff, lo, hi = paired_interval(base, hits, n)
            row.update(diff_vs_nn_a=round(diff, 1), ci_low=round(lo, 1), ci_high=round(hi, 1))
        summary.append(row)
    os.makedirs(args.csv_dir, exist_ok=True)
    for fname, rows in (("summary.csv", summary), ("per_pair.csv", per_pair)):
        with open(os.path.join(args.csv_dir, fname), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    with open(os.path.join(args.csv_dir, "run.json"), "w") as f:
        json.dump(dict(commit=git_commit(), command=" ".join(sys.argv), a=args.a, b=args.b,
                       k=args.k, weights=args.weights, threshold=args.threshold), f, indent=1)
    cols = list(summary[0].keys())
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for r in summary:
        print("| " + " | ".join(str(r[c]) for c in cols) + " |")


if __name__ == "__main__":
    main()
