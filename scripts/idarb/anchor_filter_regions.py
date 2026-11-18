"""
Where the anchor filter drops anchors: counts and their spatial distribution per pair.

For every pair it rebuilds the chain's mutual anchors exactly as refine_correspondence.py does
(mutual nearest neighbours of the features, thinned to --max_anchors by farthest-point
sampling on the target) and applies filter_anchors for every setting (none, rigid:<thr>,
loo:<thr>). Both shapes are cut into spatial regions (farthest-point seeds over the feature
samples, every sample to its nearest seed; the regions have no names, they only show where
anchors are) and per setting it reports:

  before, kept           anchors before the filter and kept
  t_regions_with         target regions that hold an anchor before the filter
  t_regions_emptied      of those, the regions left without any anchor
  t_per_region           kept/before for every target region
  s_regions_*            the same on the partial mesh
  largest_gap            the largest distance from a target sample to its nearest kept anchor,
                         as a fraction of the target's bounding-box diagonal (how far the
                         prior has to extrapolate)

The question it answers: on pairs whose body layouts differ (an upright toy against a
quadruped), does one similarity transform (rigid) remove all anchors of whole regions, and
does the non-rigid mode (loo) keep them? With --figure_dir one image per pair: kept anchors
green, dropped red, on the target (two principal axes) and on the partial mesh (image plane).

CPU only, a few seconds per pair and setting (loo: about a minute).

Usage (repository root, mesh environment):
    python scripts/idarb/anchor_filter_regions.py --groups cross \
        --settings none rigid:0.1 rigid:0.2 rigid:0.3 loo:0.1 loo:0.2 \
        --csv results/diff3f/anchor_filter_cross/regions.csv \
        --figure_dir results/diff3f/anchor_filter_cross/figures
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
from refine_correspondence import farthest_points, filter_anchors, mutual_pairs  # noqa: E402
from sweep_anchor_filter import git_commit  # noqa: E402


def regions(xyz, k, seed=0):
    """A region id for every point: its nearest of k farthest-point seeds."""
    seeds = farthest_points(xyz, k, seed=seed)
    return cKDTree(xyz[seeds]).query(xyz)[1]


def region_counts(region_of_anchor, keep, k):
    """Per region: anchors before and kept; the number of regions with an anchor before, and
    how many of those have none left."""
    before = np.bincount(region_of_anchor, minlength=k)
    kept = np.bincount(region_of_anchor[keep], minlength=k)
    return before, kept, int((before > 0).sum()), int(((before > 0) & (kept == 0)).sum())


def largest_gap(xyz, anchors):
    """The largest distance from a point of xyz to its nearest anchor, over the diagonal."""
    if len(anchors) == 0:
        return float("nan")
    diag = float(np.linalg.norm(xyz.max(axis=0) - xyz.min(axis=0)))
    return float(cKDTree(anchors).query(xyz)[0].max() / diag)


def principal_plane(xyz):
    """The two directions of largest extent of a point set (rows), for a flat picture."""
    c = xyz - xyz.mean(axis=0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    return vt[:2]


def draw(path, name, src_xyz, dst_xyz, S, T, masks):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    axes_t = principal_plane(dst_xyz)
    fig, axes = plt.subplots(2, len(masks), figsize=(3.2 * len(masks), 6.6), squeeze=False)
    for j, (setting, keep) in enumerate(masks.items()):
        for i, (pts, anc, proj) in enumerate(((dst_xyz, T, axes_t), (src_xyz, S, np.eye(3)[:2]))):
            ax = axes[i, j]
            p, a = pts @ proj.T, anc @ proj.T
            ax.scatter(p[:, 0], p[:, 1], s=0.3, c="#cccccc")
            ax.scatter(a[~keep, 0], a[~keep, 1], s=9, c="#d62728")
            ax.scatter(a[keep, 0], a[keep, 1], s=9, c="#2ca02c")
            ax.set_aspect("equal")
            ax.axis("off")
            ax.set_title(f"{setting}: kept {int(keep.sum())} of {len(keep)}\n"
                         + ("target" if i == 0 else "partial mesh"), fontsize=8)
    fig.suptitle(name + " (green kept, red dropped)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--groups", nargs="+", default=["cross"], choices=["same", "cross"])
    ap.add_argument("--pairs", nargs="*", help="only these pair names")
    ap.add_argument("--settings", nargs="+",
                    default=["none", "rigid:0.1", "rigid:0.2", "rigid:0.3", "loo:0.1", "loo:0.2"])
    ap.add_argument("--feature_dir", default="outputs/diff3f/features_zoom2")
    ap.add_argument("--max_anchors", type=int, default=400, help="as in the chain")
    ap.add_argument("--target_regions", type=int, default=12)
    ap.add_argument("--source_regions", type=int, default=8)
    ap.add_argument("--csv", default="results/diff3f/anchor_filter_cross/regions.csv")
    ap.add_argument("--figure_dir")
    args = ap.parse_args()

    rows = []
    for group, name, s, _mesh, t, _tr, _ex in ef.pair_table(args.groups):
        if args.pairs and name not in args.pairs:
            continue
        src_npz = os.path.join(args.feature_dir, s + "_partial.npz")
        dst_npz = os.path.join(args.feature_dir, t + "_full.npz")
        if not (os.path.exists(src_npz) and os.path.exists(dst_npz)):
            continue
        src, dst = rp.load_features(src_npz), rp.load_features(dst_npz)
        ti, si, _w = mutual_pairs(dst["feat"], src["feat"])
        thin = farthest_points(dst["xyz"][ti], args.max_anchors)
        ti, si = ti[thin], si[thin]
        T, S = dst["xyz"][ti], src["xyz"][si]
        reg_t = regions(dst["xyz"], args.target_regions)[ti]
        reg_s = regions(src["xyz"], args.source_regions)[si]
        masks = {}
        for setting in args.settings:
            if setting == "none":
                keep = np.ones(len(T), dtype=bool)
            else:
                mode, thr = setting.split(":")
                keep = filter_anchors(T, S, mode, float(thr))
            masks[setting] = keep
            bt, kt, with_t, empty_t = region_counts(reg_t, keep, args.target_regions)
            bs, ks, with_s, empty_s = region_counts(reg_s, keep, args.source_regions)
            row = dict(
                group=group, pair=name, setting=setting, mutual_pairs=int(len(_w)),
                before=int(len(T)), kept=int(keep.sum()),
                t_regions_with=with_t, t_regions_emptied=empty_t,
                t_per_region=" ".join(f"{a}/{b}" for a, b in zip(kt, bt)),
                s_regions_with=with_s, s_regions_emptied=empty_s,
                s_per_region=" ".join(f"{a}/{b}" for a, b in zip(ks, bs)),
                largest_gap=round(largest_gap(dst["xyz"], T[keep]), 4),
            )
            rows.append(row)
            print(json.dumps(row), flush=True)
        if args.figure_dir:
            os.makedirs(args.figure_dir, exist_ok=True)
            draw(os.path.join(args.figure_dir, name + ".png"), name, src["xyz"], dst["xyz"],
                 S, T, masks)
    if not rows:
        sys.exit("no pair with features found")
    os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.splitext(args.csv)[0] + "_run.json", "w") as f:
        json.dump(dict(commit=git_commit(), command=" ".join(sys.argv), settings=args.settings,
                       max_anchors=args.max_anchors, target_regions=args.target_regions,
                       source_regions=args.source_regions), f, indent=1)


if __name__ == "__main__":
    main()
