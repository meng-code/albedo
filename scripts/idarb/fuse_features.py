"""
Join two feature sets of the same sampled points into one: for every file name present in both
folders, the unit-length feature vectors are concatenated, each side scaled so that the cosine
similarity of the joined vectors is the weighted mean of the two cosine similarities
(weight w for the first set, 1 - w for the second). The joined files keep idx, xyz, colors and
can be evaluated with eval_features.py like any other feature folder.

Usage (repository root, mesh environment):
    python scripts/idarb/fuse_features.py --a outputs/diff3f/features \
        --b outputs/diff3f/features_meshfm --w 0.5 --out outputs/diff3f/features_fused
"""

import argparse
import json
import os

import numpy as np


def fuse(fa, fb, w):
    """Concatenate unit features so that the dot product is w * <a, a'> + (1 - w) * <b, b'>."""
    fa = fa.astype(np.float32)
    fb = fb.astype(np.float32)
    fa /= np.maximum(np.linalg.norm(fa, axis=1, keepdims=True), 1e-12)
    fb /= np.maximum(np.linalg.norm(fb, axis=1, keepdims=True), 1e-12)
    return np.concatenate([np.sqrt(w) * fa, np.sqrt(1.0 - w) * fb], axis=1)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--a", required=True, help="first feature folder")
    ap.add_argument("--b", required=True, help="second feature folder")
    ap.add_argument("--w", type=float, default=0.5, help="weight of the first set (0 to 1)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    names = sorted(set(os.listdir(args.a)) & set(os.listdir(args.b)))
    names = [n for n in names if n.endswith(".npz")]
    for name in names:
        za = np.load(os.path.join(args.a, name))
        zb = np.load(os.path.join(args.b, name))
        if not np.array_equal(za["idx"], zb["idx"]):
            raise ValueError(f"{name}: the two sets sample different points")
        feat = fuse(za["feat"], zb["feat"], args.w)
        meta = dict(fused_from=[args.a, args.b], w=args.w, meta_a=str(za["meta"]),
                    meta_b=str(zb["meta"]))
        np.savez_compressed(
            os.path.join(args.out, name),
            idx=za["idx"],
            xyz=za["xyz"],
            feat=feat.astype(np.float16),
            colors=za["colors"],
            meta=json.dumps(meta),
        )
        print(name, feat.shape)
    print(f"{len(names)} files written to {args.out}")


if __name__ == "__main__":
    main()
