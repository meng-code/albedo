"""
Two-sphere occlusion experiment: a small sphere floating in front of a large one, with exact
normals.

For three distances between the spheres it reports the true gap at the small sphere's outline, the
gap recovered from normals only, the gap with an ideal depth prior, and the recovered overall depth
range divided by the true one. A single large sphere without occlusion is the control.

Writes results/analysis/occlusion_gaps.csv.

Usage (from the repository root, in the mesh environment):
    python scripts/idarb/eval_occlusion.py
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import normal_to_mesh as ntm  # noqa: E402

FIELDS = ["scene", "gap_true", "gap_normals", "gap_depth", "ratio_normals", "ratio_depth"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/analysis/occlusion_gaps.csv")
    ap.add_argument(
        "--centers",
        type=float,
        nargs="+",
        default=[100.0, 150.0, 200.0],
        help="distance between the sphere centres along depth, in pixels",
    )
    args = ap.parse_args()

    rows = []
    for c in args.centers:
        r = ntm.occlusion_check(c)
        rows.append(
            dict(
                scene=f"two spheres, centres {c:.0f} px apart",
                gap_true=round(r["gap_true"], 1),
                gap_normals=round(r["gap_normals"], 1),
                gap_depth=round(r["gap_depth"], 1),
                ratio_normals=round(r["ratio_normals"], 2),
                ratio_depth=round(r["ratio_depth"], 2),
            )
        )
    r = ntm.occlusion_check(with_front=False)
    rows.append(
        dict(
            scene="single sphere (no occlusion)",
            gap_true="",
            gap_normals="",
            gap_depth="",
            ratio_normals=round(r["ratio_normals"], 2),
            ratio_depth=round(r["ratio_depth"], 2),
        )
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    for row in rows:
        print(row)
    print("saved", args.out)


if __name__ == "__main__":
    main()
