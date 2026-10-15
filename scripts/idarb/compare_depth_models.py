"""
Compare depth maps from two Depth Anything V2 variants (for example Small and Large).

Both folders hold the 16-bit PNGs written by estimate_depth.py for the same input images. For every
image present in both, the script computes the Pearson correlation of the two depth maps inside the
object mask.

Producing the Large depth maps needs the GPU server (the model is about 1.3 GB, CC-BY-NC-4.0):
    python scripts/idarb/estimate_depth.py --input_dir results/idarb_single/inputs \
        --out_dir outputs/depth_large --model depth-anything/Depth-Anything-V2-Large-hf
The comparison itself needs only NumPy and Pillow:
    python scripts/idarb/compare_depth_models.py --a results/idarb_single/depth \
        --b outputs/depth_large --names panda chair1 bird superman labubu

Writes results/analysis/depth_small_vs_large.csv.
"""

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import normal_to_mesh as ntm  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="first depth folder (e.g. Small)")
    ap.add_argument("--b", required=True, help="second depth folder (e.g. Large)")
    ap.add_argument("--inputs", default="results/idarb_single/inputs", help="RGBA inputs for masks")
    ap.add_argument("--names", nargs="*", help="images to compare (default: all in both folders)")
    ap.add_argument("--out", default="results/analysis/depth_small_vs_large.csv")
    args = ap.parse_args()

    names = args.names or sorted(
        f[:-4]
        for f in os.listdir(args.a)
        if f.endswith(".png") and os.path.exists(os.path.join(args.b, f))
    )
    rows = []
    for name in names:
        _, mask = ntm.load_input_512(os.path.join(args.inputs, name + ".png"))
        da = ntm.load_depth(os.path.join(args.a, name + ".png"), mask.shape)
        db = ntm.load_depth(os.path.join(args.b, name + ".png"), mask.shape)
        corr = float(np.corrcoef(da[mask], db[mask])[0, 1])
        rows.append(dict(name=name, correlation=round(corr, 4)))
        print(f"{name:32s} {corr:.4f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "correlation"])
        w.writeheader()
        w.writerows(rows)
    print("saved", args.out)


if __name__ == "__main__":
    main()
