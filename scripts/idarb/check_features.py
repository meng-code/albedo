"""
Quick checks that compare two sets of per-point features (for example Diff3F and MeshFM) written in
the diff3f_features.py .npz format, before they go into the correspondence pipeline.

For each pair "partial:full" (both names without .npz) and each feature set:
  - mutual nearest neighbours: the share of partial points whose most similar full point (cosine
    similarity of the features) has that same partial point as its most similar partial point. A
    rough measure of how distinctive and consistent the features are; no ground truth needed.
  - for synthetic partials cut from the full mesh (results/diff3f/synthetic/<name>_gt.npz holds the
    full-mesh vertex of every partial vertex): the distance between the matched full point and the
    true one, as a fraction of the full mesh's bounding-box diagonal (median, share under 5 % and
    under 10 %).
  - --pca_dir: a joint PCA of both meshes' features (fitted on the two point sets together, so equal
    colours mean similar features), first three components as RGB, written as coloured .ply point
    clouds and a .png with three axis-aligned views of each mesh.

Usage (repository root, any environment with numpy, scipy, trimesh, matplotlib):
    python scripts/idarb/check_features.py --sets diff3f=outputs/diff3f \
        meshfm=outputs/diff3f/features_meshfm --pairs cat_partial:cat_full \
        cow_synth_partial:cow_full cat_synth_partial:cat_full --pca_dir outputs/diff3f/check
"""

import argparse
import json
import os

import numpy as np

SYNTH_GT = {"cow_synth_partial": "results/diff3f/synthetic/cow_synth_gt.npz",
            "cat_synth_partial": "results/diff3f/synthetic/cat_synth_gt.npz"}


def load(path):
    d = np.load(path, allow_pickle=True)
    f = d["feat"].astype(np.float32)
    f /= np.linalg.norm(f, axis=1, keepdims=True) + 1e-8
    return d["idx"], d["xyz"].astype(np.float64), f


def nn(a, b, chunk=4096):
    """Index in b of the most cosine-similar row for every row of a."""
    out = np.empty(len(a), np.int64)
    for i in range(0, len(a), chunk):
        out[i:i + chunk] = (a[i:i + chunk] @ b.T).argmax(1)
    return out


def pca_rgb(feats):
    """Joint PCA over a list of feature arrays; returns one uint8 RGB array per input."""
    allf = np.concatenate(feats, 0)
    mean = allf.mean(0)
    _, _, vt = np.linalg.svd(allf[::max(1, len(allf) // 20000)] - mean, full_matrices=False)
    proj = [(f - mean) @ vt[:3].T for f in feats]
    allp = np.concatenate(proj, 0)
    lo, hi = np.percentile(allp, 2, 0), np.percentile(allp, 98, 0)
    return [(np.clip((p - lo) / (hi - lo + 1e-8), 0, 1) * 255).astype(np.uint8) for p in proj]


def write_ply(path, xyz, rgb):
    with open(path, "w") as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(xyz)}\nproperty float x\n"
                "property float y\nproperty float z\nproperty uchar red\nproperty uchar green\n"
                "property uchar blue\nend_header\n")
        for p, c in zip(xyz, rgb):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {c[0]} {c[1]} {c[2]}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", required=True, help="name=folder of .npz files")
    ap.add_argument("--pairs", nargs="+", required=True, help="partial:full, names without .npz")
    ap.add_argument("--pca_dir", default=None)
    args = ap.parse_args()
    sets = dict(s.split("=", 1) for s in args.sets)
    report = {}
    for pair in args.pairs:
        pa, fu = pair.split(":")
        report[pair] = {}
        pca_inputs = {}
        for name, folder in sets.items():
            ia, xa, fa = load(os.path.join(folder, pa + ".npz"))
            ib, xb, fb = load(os.path.join(folder, fu + ".npz"))
            ab, ba = nn(fa, fb), nn(fb, fa)
            r = dict(mutual_nn=round(float((ba[ab] == np.arange(len(fa))).mean()), 4))
            if pa in SYNTH_GT:
                gt_full_vertex = np.load(SYNTH_GT[pa])["orig_idx"][ia]
                row_of_vertex = {int(v): k for k, v in enumerate(ib)}
                keep = np.array([int(v) in row_of_vertex for v in gt_full_vertex])
                gt_row = np.array([row_of_vertex[int(v)] for v in gt_full_vertex[keep]])
                diag = np.linalg.norm(xb.max(0) - xb.min(0))
                err = np.linalg.norm(xb[ab[keep]] - xb[gt_row], axis=1) / diag
                r.update(gt_points=int(keep.sum()), median_err=round(float(np.median(err)), 4),
                         within_5pct=round(float((err < 0.05).mean()), 4),
                         within_10pct=round(float((err < 0.10).mean()), 4))
            report[pair][name] = r
            pca_inputs[name] = (xa, fa, xb, fb)
        if args.pca_dir:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            os.makedirs(args.pca_dir, exist_ok=True)
            fig, axes = plt.subplots(len(sets), 6, figsize=(24, 4 * len(sets)), squeeze=False)
            for row, (name, (xa, fa, xb, fb)) in enumerate(pca_inputs.items()):
                ca, cb = pca_rgb([fa, fb])
                write_ply(os.path.join(args.pca_dir, f"{pa}__{name}_pca.ply"), xa, ca)
                write_ply(os.path.join(args.pca_dir, f"{fu}__{name}_pca.ply"), xb, cb)
                for col, (x, c, title) in enumerate([(xa, ca, pa), (xb, cb, fu)]):
                    for k, (i, j) in enumerate([(0, 1), (2, 1), (0, 2)]):
                        ax = axes[row, col * 3 + k]
                        ax.scatter(x[:, i], x[:, j], c=c / 255.0, s=1)
                        ax.set_aspect("equal")
                        ax.set_axis_off()
                        ax.set_title(f"{name}: {title} ({'xyz'[i]}{'xyz'[j]})", fontsize=9)
            fig.tight_layout()
            fig.savefig(os.path.join(args.pca_dir, f"pca_{pa}__{fu}.png"), dpi=80)
            plt.close(fig)
    print(json.dumps(report, indent=1))
    if args.pca_dir:
        with open(os.path.join(args.pca_dir, "check_features.json"), "w") as f:
            json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
