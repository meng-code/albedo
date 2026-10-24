"""
A fixed split of the keypoint benchmark into a tuning half and a held-out half, and the keypoint
accuracy reported per half with a bootstrap interval. The split is drawn once with a fixed seed
and stored (results/diff3f/keypoints/split.json); from then on every setting decision is made
on the tuning pairs and the held-out pairs are only reported.

Input: keypoint-accuracy CSVs as keypoint_table.py / eval_keypoints.py write them
(group, pair, method, n, mean, median, pck_2, pck_5, pck_10). Output: one row per CSV, method,
split and group with the number of pairs, the mean over pairs of PCK@2 / 5, the pooled PCK@2
(keypoints within 2 % over all keypoints of the split) and a 95 % bootstrap interval of the
mean PCK@2 over pairs.

Usage (repository root, mesh environment):
    python scripts/idarb/keypoint_split.py \
        results/diff3f/kp_variants/keypoint_accuracy_zoom2_all.csv \
        --csv results/diff3f/keypoints/split_accuracy.csv
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_features as ef  # noqa: E402

SPLIT_PATH = "results/diff3f/keypoints/split.json"
FIELDS = ["file", "method", "split", "group", "pairs", "keypoints", "mean_pck_2", "mean_pck_5",
          "pooled_pck_2", "ci95_low", "ci95_high"]


def make_split(pairs_by_group, held_out_fraction=0.5, seed=0):
    """Seeded shuffle per group; the first ceil(fraction * n) pairs are held out."""
    rng = np.random.default_rng(seed)
    out = dict(seed=seed, held_out_fraction=held_out_fraction, tune=[], held_out=[])
    for group in sorted(pairs_by_group):
        pairs = sorted(pairs_by_group[group])
        order = rng.permutation(len(pairs))
        k = int(np.ceil(held_out_fraction * len(pairs)))
        out["held_out"] += [pairs[i] for i in order[:k]]
        out["tune"] += [pairs[i] for i in order[k:]]
    out["tune"].sort()
    out["held_out"].sort()
    return out


def bootstrap_mean(values, n_boot=2000, seed=0):
    """95 % percentile interval of the mean over pairs."""
    v = np.asarray(values, dtype=np.float64)
    if len(v) < 2:
        return float(v.mean()), float(v.mean())
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarise(rows, split, name):
    """Rows of one CSV -> summary rows per method, split half and group ('all' included)."""
    out = []
    methods = sorted({r["method"] for r in rows})
    for method in methods:
        for half in ("tune", "held_out"):
            members = set(split[half])
            for group in ("all", "same", "cross"):
                rs = [r for r in rows if r["method"] == method and r["pair"] in members
                      and (group == "all" or r["group"] == group)]
                if not rs:
                    continue
                n = np.array([int(r["n"]) for r in rs], dtype=np.float64)
                p2 = np.array([float(r["pck_2"]) for r in rs])
                p5 = np.array([float(r["pck_5"]) for r in rs])
                lo, hi = bootstrap_mean(p2)
                out.append(dict(file=name, method=method, split=half, group=group,
                                pairs=len(rs), keypoints=int(n.sum()),
                                mean_pck_2=round(float(p2.mean()), 4),
                                mean_pck_5=round(float(p5.mean()), 4),
                                pooled_pck_2=round(float((p2 * n).sum() / n.sum()), 4),
                                ci95_low=round(lo, 4), ci95_high=round(hi, 4)))
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("csvs", nargs="+", help="keypoint-accuracy CSVs")
    ap.add_argument("--split", default=SPLIT_PATH)
    ap.add_argument("--csv", help="write the summary rows here")
    args = ap.parse_args()
    if os.path.exists(args.split):
        with open(args.split) as f:
            split = json.load(f)
    else:
        groups = {}
        for g, name, _s, _m, _t, _tr, _e in ef.pair_table(["same", "cross"]):
            if os.path.exists(os.path.join("results/diff3f/keypoints", name + ".json")) or \
                    os.path.exists(os.path.join("results/diff3f/landmarks", name + ".json")):
                groups.setdefault(g, []).append(name)
        split = make_split(groups)
        os.makedirs(os.path.dirname(args.split), exist_ok=True)
        with open(args.split, "w") as f:
            json.dump(split, f, indent=1)
        print(f"split written to {args.split}: tune {split['tune']}, held out "
              f"{split['held_out']}")
    out = []
    for path in args.csvs:
        with open(path) as f:
            rows = list(csv.DictReader(f))
        out += summarise(rows, split, os.path.basename(path))
    for r in out:
        if r["group"] == "all":
            print(f"{r['file']:36s} {r['method']:20s} {r['split']:9s} pairs {r['pairs']:2d} "
                  f"mean PCK@2 {r['mean_pck_2']:.3f} [{r['ci95_low']:.3f}, {r['ci95_high']:.3f}]"
                  f"  pooled {r['pooled_pck_2']:.3f}  PCK@5 {r['mean_pck_5']:.3f}")
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(out)


if __name__ == "__main__":
    main()
