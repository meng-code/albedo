"""
Per-image numbers for all sample images, with and without the depth prior.

For every image in results/idarb_single/ it reports:
  mean_nz            mean z component of IDArb's normals inside the mask (1 = everything faces
                     the camera); images with mean_nz >= 0.95 are counted as failed normals
  scale_s            depth-prior scale s in pixels
  dw_normals         depth/width ratio, normals only (depth range / object width, both in pixels)
  dw_depth           depth/width ratio, normals + depth prior
  angle_normals      mean angle (deg) between IDArb's normals and the normals recomputed from the
                     integrated depth, normals only
  angle_depth        the same angle with the depth prior
  parts              number of connected parts after cliff cuts and fragment merging

Writes results/analysis/sample_metrics.csv and prints a summary over the usable images (medians
of the depth/width ratios, means of the angles).

Usage (from the repository root, in the mesh environment):
    python scripts/idarb/eval_samples.py
"""

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import normal_to_mesh as ntm  # noqa: E402

FIELDS = [
    "name",
    "mean_nz",
    "usable",
    "scale_s",
    "dw_normals",
    "dw_depth",
    "angle_normals",
    "angle_depth",
    "parts",
]


def evaluate(name, root, erode_px=2, nz_failed=0.95):
    _, normal_rgb, _, _ = ntm.load_result(os.path.join(root, "decomposition", name + "_00.png"))
    size = normal_rgb.shape[:2]
    mask = ntm.erode(ntm.load_mask(os.path.join(root, "inputs", name + ".png"), size), erode_px)
    n = ntm.decode_normal(normal_rgb)
    d = ntm.load_depth(os.path.join(root, "depth", name + ".png"), size)
    h_n = ntm.integrate_normals(n, mask)
    h_d, info = ntm.integrate_normals_with_depth(n, d, mask)

    cols = np.nonzero(mask.any(axis=0))[0]
    width = float(cols.max() - cols.min())
    inner = ntm.erode(mask, 1)
    mean_nz = float(n[..., 2][mask].mean())
    return dict(
        name=name,
        mean_nz=round(mean_nz, 3),
        usable=int(mean_nz < nz_failed),
        scale_s=round(float(info["scale"]), 1),
        dw_normals=round((np.nanmax(h_n) - np.nanmin(h_n)) / width, 3),
        dw_depth=round((np.nanmax(h_d) - np.nanmin(h_d)) / width, 3),
        angle_normals=round(ntm.angular_error(n, ntm.normals_from_depth(h_n), inner)[0], 2),
        angle_depth=round(ntm.angular_error(n, ntm.normals_from_depth(h_d), inner)[0], 2),
        parts=int(info["parts"]),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/idarb_single")
    ap.add_argument("--out", default="results/analysis/sample_metrics.csv")
    args = ap.parse_args()

    dec = os.path.join(args.root, "decomposition")
    names = sorted(f[: -len("_00.png")] for f in os.listdir(dec) if f.endswith("_00.png"))
    rows = [evaluate(name, args.root) for name in names]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    for r in rows:
        print(
            "{name:32s} nz {mean_nz:.3f}  s {scale_s:6.1f}  d/w {dw_normals:.2f} -> {dw_depth:.2f}"
            "  angle {angle_normals:5.1f} -> {angle_depth:5.1f}  parts {parts}".format(**r)
        )
    ok = [r for r in rows if r["usable"]]
    failed = [r["name"] for r in rows if not r["usable"]]
    print(f"\nimages: {len(rows)}, usable: {len(ok)}, failed normals: {failed}")
    print(
        "usable, median depth/width: normals only {:.2f}, with depth prior {:.2f}".format(
            np.median([r["dw_normals"] for r in ok]), np.median([r["dw_depth"] for r in ok])
        )
    )
    print(
        "usable, mean angle (deg):   normals only {:.1f}, with depth prior {:.1f}".format(
            np.mean([r["angle_normals"] for r in ok]), np.mean([r["angle_depth"] for r in ok])
        )
    )
    print("saved", args.out)


if __name__ == "__main__":
    main()
