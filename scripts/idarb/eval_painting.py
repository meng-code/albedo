"""
Albedo transfer (painting) with ground truth, on the synthetic partial meshes.

The full mesh is given a known colour field: its position inside the bounding box, with the
coordinate across the symmetry plane folded (|t|), so the field is mirror-symmetric like most real
albedos. The synthetic partial mesh (make_partial_mesh.py) carries the colours of the vertices it
was cut from, is registered to the full mesh and painted back onto it with paint_target(); the
painted colours are then compared with the field at every painted vertex (mean absolute RGB error,
0 to 255, and the fraction of vertices within 20).

Variants: nearest aligned point only (feature_weight 0), feature-guided, feature-guided with the
part penalty; each with the global and the piecewise alignment. The mirrored vertices (far side,
filled through the symmetry plane) are scored separately.

Writes results/diff3f/painting_eval.csv.

Usage (repository root, mesh environment):
    python scripts/idarb/eval_painting.py
"""

import argparse
import csv
import os
import sys

import numpy as np
import trimesh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import register_partial as rp  # noqa: E402

PAIRS = [
    # name, partial npz, full npz, partial mesh, gt npz, full mesh
    (
        "cow_synth",
        "cow_synth_partial.npz",
        "cow_full.npz",
        "results/diff3f/synthetic/cow_synth.ply",
        "results/diff3f/synthetic/cow_synth_gt.npz",
        "third_party/Diff3F/meshes/cow.obj",
    ),
    (
        "cat_synth",
        "cat_synth_partial.npz",
        "cat_full.npz",
        "results/diff3f/synthetic/cat_synth.ply",
        "results/diff3f/synthetic/cat_synth_gt.npz",
        "third_party/Diff3F/meshes/cat.off",
    ),
]

VARIANTS = [
    # name, feature_weight, part_penalty
    ("nearest", 0.0, 0.0),
    ("joint", 1.0, 0.0),
    ("joint+parts", 1.0, 0.5),
]

FIELDS = [
    "pair", "alignment", "variant", "covered_frac", "mirrored_frac", "plane_err",
    "err_covered", "within20_covered", "err_mirrored", "within20_mirrored", "err_feature_only",
]  # fmt: skip


class ColourField:
    """Mirror-symmetric position colours over a mesh."""

    def __init__(self, verts):
        self.lo = verts.min(axis=0)
        self.hi = np.maximum(verts.max(axis=0) - self.lo, 1e-9)
        self.c, self.n, self.err = rp.symmetry_plane(verts)

    def __call__(self, p):
        t = (p - self.c) @ self.n
        folded = p - ((t - np.abs(t))[:, None]) * self.n[None, :]  # reflect the negative side
        return (np.clip((folded - self.lo) / self.hi, 0, 1) * 255).astype(np.uint8)


def score(painted, truth, mask):
    if mask.sum() == 0:
        return None, None
    err = np.abs(painted[mask].astype(float) - truth[mask].astype(float)).mean(axis=1)
    return round(float(err.mean()), 1), round(float((err < 20).mean()), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature_dir", default="outputs/diff3f")
    ap.add_argument("--out", default="results/diff3f/painting_eval.csv")
    args = ap.parse_args()

    rows = []
    for name, src_file, dst_file, src_mesh, gt_file, dst_mesh in PAIRS:
        src_path = os.path.join(args.feature_dir, src_file)
        dst_path = os.path.join(args.feature_dir, dst_file)
        if not (os.path.exists(src_path) and os.path.exists(dst_path)):
            print(f"{name}: feature files missing, skipped")
            continue
        src = rp.load_features(src_path, "full", src_mesh)
        dst = rp.load_features(dst_path, "full", dst_mesh)
        gt = dict(np.load(gt_file, allow_pickle=False))
        full = trimesh.load(dst_mesh, process=False, force="mesh")
        field = ColourField(np.asarray(full.vertices, dtype=np.float64))
        # the partial mesh's sampled points take the colour of the full-mesh vertex they came from
        rows_gt = rp.gt_targets(src, dst, gt)
        src["colors"] = field(dst["xyz"][np.maximum(rows_gt, 0)])
        res = rp.register(src, dst, gt, piecewise=True)
        s, R, t, a = res["fine"]
        aligned = rp.transform(rp.apply_depth(src["xyz"], a), s, R, t)
        for alignment, pts in [("global", aligned), ("piecewise", res["aligned_piecewise"])]:
            for variant, fw, pp in VARIANTS:
                paint = rp.paint_target(
                    dst_mesh,
                    dst["xyz"],
                    dst["feat"],
                    src["feat"],
                    pts,
                    src["colors"],
                    res["diag"],
                    src_parts=res["src_parts"],
                    dst_parts=res["dst_parts"],
                    part_penalty=pp,
                    feature_weight=fw,
                    mirror=True,
                )
                truth = field(paint["V"])
                e_cov, w_cov = score(paint["registered"], truth, paint["covered"])
                e_mir, w_mir = score(paint["mirror"], truth, paint["mirrored"])
                e_feat, _ = score(paint["feature"], truth, np.ones(len(truth), dtype=bool))
                row = dict(
                    pair=name,
                    alignment=alignment,
                    variant=variant,
                    covered_frac=round(float(paint["covered"].mean()), 3),
                    mirrored_frac=round(float(paint["mirrored"].mean()), 3),
                    plane_err=None if paint["plane"] is None else round(paint["plane"][2], 4),
                    err_covered=e_cov,
                    within20_covered=w_cov,
                    err_mirrored=e_mir,
                    within20_mirrored=w_mir,
                    err_feature_only=e_feat,
                )
                rows.append(row)
                print(row)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print("saved", args.out)


if __name__ == "__main__":
    main()
