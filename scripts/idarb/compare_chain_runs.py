"""
Paired comparison of two chain runs that were each evaluated by their own sweep (for example
the chain on two feature sets): keypoint hits pooled over the annotated same-category pairs
(PCK@2, @5, @10) with the paired difference at 2 % and its 95 % interval from a bootstrap over
pairs, and the per-pair differences of the texture measures (triangles within 2 %, mirrored
area) with a bootstrap interval of their mean and the number of pairs that go up and down.

Each run is a folder with keypoint_accuracy.csv and <config>.csv as written by sweep_debias.py
or sweep_anchor_filter.py; --config names the method / file in both (default: none).

Usage (repository root, mesh environment):
    python scripts/idarb/compare_chain_runs.py --base results/diff3f/debias_chain \
        --other results/diff3f/marco_lifted/chain_fused
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sweep_anchor_filter import paired_interval, read_csv  # noqa: E402


def mean_interval(d, draws=20000, seed=0):
    """Mean of per-pair differences and its 95 % bootstrap interval."""
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(draws, len(d)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--other", required=True)
    ap.add_argument("--config", default="none")
    args = ap.parse_args()

    def keypoints(folder):
        return {r["pair"]: r for r in read_csv(os.path.join(folder, "keypoint_accuracy.csv"))
                if r["group"] == "same" and r["method"] == args.config}

    def texture(folder):
        return {r["pair"]: r for r in read_csv(os.path.join(folder, args.config + ".csv"))
                if r["group"] == "same" and r["variant"] == "arap"}

    kb, ko = keypoints(args.base), keypoints(args.other)
    pairs = sorted(set(kb) & set(ko))
    n = np.array([int(kb[p]["n"]) for p in pairs])

    def hits(rows, key):
        return np.array([round(float(rows[p][key]) * int(rows[p]["n"])) for p in pairs])

    print(f"keypoints: {int(n.sum())} on {len(pairs)} pairs")
    for key in ("pck_2", "pck_5", "pck_10"):
        print(f"  {key}: base {int(hits(kb, key).sum())}, other {int(hits(ko, key).sum())}")
    d, lo, hi = paired_interval(hits(kb, "pck_2"), hits(ko, "pck_2"), n)
    print(f"  pck_2 difference {d:+.1f} points [{lo:.1f}, {hi:.1f}]")
    tb, to = texture(args.base), texture(args.other)
    pairs_t = sorted(set(tb) & set(to))
    print(f"texture: {len(pairs_t)} pairs")
    for key in ("within_2pct", "within_5pct", "flipped_area"):
        diff = 100.0 * np.array([float(to[p][key]) - float(tb[p][key]) for p in pairs_t])
        m, lo, hi = mean_interval(diff)
        print(f"  {key}: mean difference {m:+.1f} points [{lo:.1f}, {hi:.1f}], "
              f"up on {int((diff > 0).sum())} pairs, down on {int((diff < 0).sum())}")


if __name__ == "__main__":
    main()
