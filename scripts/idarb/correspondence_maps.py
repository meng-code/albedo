"""
Semantic correspondence maps between a partial mesh and a full mesh, from Diff3F features alone
(no registration): the "same part, same colour" pictures.

Two colourings, each written as a PLY pair (open in MeshLab) and rendered side by side:
  pca    the two feature sets are projected together on their first three principal components,
         which become RGB. No matching is involved: a part gets the same colour on both meshes
         only if its features are the same, so this shows whether the semantic maps agree.
  nn     the source is coloured by position (a rainbow over its bounding box) and every target
         vertex takes the colour of its nearest neighbour in feature space (highest cosine
         similarity). This shows the nearest-neighbour correspondence itself: coherent colours
         where it is right, jumps and left-right mix-ups where it is not.
With --gt (synthetic pairs) a third colouring shows the nearest-neighbour error on the target:
the distance between the matched source vertex's true position and the vertex, as a fraction of
the diagonal (white = exact, red = far).

Features exist on the sampled vertices only; every mesh vertex takes the feature of its nearest
sampled point.

Usage (repository root, mesh environment):
    python scripts/idarb/correspondence_maps.py --source outputs/diff3f/cow_partial.npz \
        --source_mesh outputs/mesh_depth/cow/relief.ply --target outputs/diff3f/cow_full.npz \
        --target_mesh third_party/Diff3F/meshes/cow.obj --out_dir outputs/diff3f/cow_maps
"""

import argparse
import os
import sys

import numpy as np
import trimesh
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import register_partial as rp  # noqa: E402
from render_registration import draw  # noqa: E402


def mesh_features(npz_path, mesh_path, feature="full"):
    """Mesh vertices, faces and a feature per vertex (nearest sampled point)."""
    z = rp.load_features(npz_path, feature)
    m = trimesh.load(mesh_path, process=False, force="mesh")
    V = np.asarray(m.vertices, dtype=np.float64)
    F = np.asarray(m.faces, dtype=np.int64)
    _, nn = cKDTree(z["xyz"]).query(V)
    return V, F, z["feat"][nn], z["idx"][nn]


def pca_colors(fs, fd):
    """Joint 3-component PCA of both feature sets, scaled to 0-255 per component."""
    X = np.concatenate([fs, fd]).astype(np.float64)
    mean = X.mean(axis=0)
    _, _, vt = np.linalg.svd(X - mean, full_matrices=False)
    P = (X - mean) @ vt[:3].T
    lo = np.percentile(P, 2, axis=0)
    hi = np.percentile(P, 98, axis=0)
    C = (np.clip((P - lo) / np.maximum(hi - lo, 1e-9), 0, 1) * 255).astype(np.uint8)
    return C[: len(fs)], C[len(fs) :]


def nearest_in_features(fd, fs, chunk=2048):
    out = np.empty(len(fd), dtype=np.int64)
    for i in range(0, len(fd), chunk):
        out[i : i + chunk] = (fd[i : i + chunk] @ fs.T).argmax(axis=1)
    return out


def save_colored(path, V, F, C):
    trimesh.Trimesh(vertices=V, faces=F, vertex_colors=C, process=False).export(path)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--source", required=True, help="features of the partial mesh (.npz)")
    ap.add_argument("--source_mesh", required=True)
    ap.add_argument("--target", required=True, help="features of the full mesh (.npz)")
    ap.add_argument("--target_mesh", required=True)
    ap.add_argument("--gt", help="_gt.npz of a synthetic pair, for the error map")
    ap.add_argument("--feature", choices=["full", "diffusion", "dino"], default="full")
    ap.add_argument(
        "--view",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 1.0),
        help="viewing direction for the source render (its camera side is +z)",
    )
    ap.add_argument(
        "--target_view",
        type=float,
        nargs=3,
        default=None,
        help="viewing direction for the target render (default: the registration's)",
    )
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    Vs, Fs, fs, idx_s = mesh_features(args.source, args.source_mesh, args.feature)
    Vd, Fd, fd, idx_d = mesh_features(args.target, args.target_mesh, args.feature)
    print(f"source {len(Vs)} vertices, target {len(Vd)} vertices")

    # 1. feature PCA colours
    cs_pca, cd_pca = pca_colors(fs, fd)
    save_colored(os.path.join(args.out_dir, "pca_source.ply"), Vs, Fs, cs_pca)
    save_colored(os.path.join(args.out_dir, "pca_target.ply"), Vd, Fd, cd_pca)

    # 2. nearest-neighbour match colours
    cs_pos = rp.position_colors(Vs)
    t2s = nearest_in_features(fd, fs)
    save_colored(os.path.join(args.out_dir, "nn_source.ply"), Vs, Fs, cs_pos)
    save_colored(os.path.join(args.out_dir, "nn_target.ply"), Vd, Fd, cs_pos[t2s])
    np.savez_compressed(os.path.join(args.out_dir, "nn_target_to_source.npz"), t2s=t2s)

    # a view for the target: where the registration put the relief's camera, if available
    tview = args.target_view
    if tview is None:
        try:
            src = rp.load_features(args.source, args.feature)
            dst = rp.load_features(args.target, args.feature)
            res = rp.register(src, dst)
            tview = res["fine"][1][:, 2]
        except Exception as e:  # noqa: BLE001
            print("no registration view:", e)
            tview = (0.0, 0.0, 1.0)
    panels = [
        ("source: feature PCA", [(Vs, Fs, cs_pca / 255.0)], args.view),
        ("target: feature PCA", [(Vd, Fd, cd_pca / 255.0)], tview),
        ("source: position colours", [(Vs, Fs, cs_pos / 255.0)], args.view),
        ("target: colour of its nearest neighbour", [(Vd, Fd, cs_pos[t2s] / 255.0)], tview),
    ]

    # 3. error map with ground truth
    if args.gt:
        gt = np.load(args.gt)
        orig = gt["orig_idx"]  # source vertex -> target vertex
        diag = float(np.linalg.norm(Vd.max(0) - Vd.min(0)))
        err = np.linalg.norm(Vd[orig[t2s]] - Vd, axis=1) / diag
        has = np.zeros(len(Vd), dtype=bool)
        has[orig] = True
        acc1, acc5 = (err[has] < 0.01).mean(), (err[has] < 0.05).mean()
        print(
            "nearest neighbour on the target vertices the source contains: "
            f"{acc1:.1%} within 1 %, {acc5:.1%} within 5 %, median {np.median(err[has]):.4f}"
        )
        t = np.clip(err / 0.1, 0, 1)[:, None]
        c_err = (np.array([[255, 255, 255]]) * (1 - t) + np.array([[200, 30, 30]]) * t).astype(
            np.uint8
        )
        c_err[~has] = 150
        save_colored(os.path.join(args.out_dir, "nn_error_target.ply"), Vd, Fd, c_err)
        panels.append(("target: NN error (white 0, red >= 10 %)", [(Vd, Fd, c_err / 255.0)], tview))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.4), squeeze=False)
    for ax, (title, meshes, view) in zip(axes[0], panels):
        draw(ax, meshes, view)
        ax.set_title(title, fontsize=9)
    fig.tight_layout()
    out = os.path.join(args.out_dir, "correspondence_maps.png")
    fig.savefig(out, dpi=90)
    print("saved", out)


if __name__ == "__main__":
    main()
