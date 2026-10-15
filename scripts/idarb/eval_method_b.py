"""
Distance between the method-B surface (screened Poisson surface reconstruction) and the method-A
surface (Poisson normal integration) on sample images.

Method B gets the method-A points with IDArb's normals and is trimmed as in normal_to_mesh.py. For
every method-B vertex the script measures the distance to the nearest method-A point, in pixels,
and reports the median and the 90th percentile.

Writes results/analysis/method_b_vs_a.csv.

Usage (from the repository root, in the mesh environment, needs open3d):
    python scripts/idarb/eval_method_b.py --names panda chair1 bird
"""

import argparse
import csv
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import normal_to_mesh as ntm  # noqa: E402

FIELDS = ["name", "verts_a", "verts_b", "median_px", "p90_px"]


def measure(name, root, erode_px=2):
    albedo, normal_rgb, _, _ = ntm.load_result(
        os.path.join(root, "decomposition", name + "_00.png")
    )
    size = normal_rgb.shape[:2]
    mask = ntm.erode(ntm.load_mask(os.path.join(root, "inputs", name + ".png"), size), erode_px)
    n = ntm.decode_normal(normal_rgb)
    h = ntm.integrate_normals(n, mask)  # method A, normals only, as in normal_to_mesh.py
    va, _, vcol = ntm.heightmap_to_mesh(h, mask, albedo)
    vb, _, _ = ntm.screened_poisson(va, n[mask], vcol)
    scale = 2.0 / max(size)  # heightmap_to_mesh scales pixels by this factor
    dist, _ = cKDTree(va).query(vb)
    dist = dist / scale
    return dict(
        name=name,
        verts_a=len(va),
        verts_b=len(vb),
        median_px=round(float(np.median(dist)), 3),
        p90_px=round(float(np.percentile(dist, 90)), 3),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="+", default=["panda", "chair1", "bird"])
    ap.add_argument("--root", default="results/idarb_single")
    ap.add_argument("--out", default="results/analysis/method_b_vs_a.csv")
    args = ap.parse_args()

    rows = [measure(name, args.root) for name in args.names]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)
    print("saved", args.out)


if __name__ == "__main__":
    main()
