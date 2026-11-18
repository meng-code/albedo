"""
Software renders (matplotlib, no GPU, no OpenGL) of a register_partial.py result folder: the
source mesh on the target before and after the per-part refinement, from the relief's camera
direction and from the side, and the painted target meshes.

Reads transform.json (for the camera direction and the target mesh path recorded in target_meta),
aligned_source.ply, aligned_source_piecewise.ply (if present),
target_from_source_colors_registered.ply and target_from_source_colors_combined.ply. Writes
render.png into the folder (or --out).

Usage (repository root, mesh environment):
    python scripts/idarb/render_registration.py results/diff3f/cow results/diff3f/cat
"""

import argparse
import json
import os

import matplotlib
import numpy as np
import trimesh

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402


def load(path):
    m = trimesh.load(path, process=False, force="mesh")
    V = np.asarray(m.vertices, dtype=np.float64)
    F = np.asarray(m.faces, dtype=np.int64)
    C = None
    if m.visual.kind == "vertex":
        C = np.asarray(m.visual.vertex_colors)[:, :3] / 255.0
    return V, F, C


def draw(ax, meshes, view, up=(0.0, 1.0, 0.0)):
    """Painter's-algorithm render of (V, F, colors or None) meshes seen from direction `view`."""
    d = np.asarray(view, dtype=np.float64)
    d /= np.linalg.norm(d)
    r = np.cross(up, d)
    if np.linalg.norm(r) < 1e-6:
        r = np.cross((1.0, 0.0, 0.0), d)
    r /= np.linalg.norm(r)
    u = np.cross(d, r)
    tris, cols, depths = [], [], []
    for V, F, C in meshes:
        P = np.stack([V @ r, V @ u, V @ d], axis=1)
        tri = P[F]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
        shade = 0.55 + 0.45 * np.clip(n[:, 2], 0.0, 1.0)
        fc = np.full((len(F), 3), 0.78) if C is None else C[F].mean(axis=1)
        tris.append(tri[:, :, :2])
        cols.append(fc * shade[:, None])
        depths.append(tri[:, :, 2].mean(axis=1))
    tris = np.concatenate(tris)
    cols = np.concatenate(cols)
    order = np.argsort(np.concatenate(depths))
    ax.add_collection(PolyCollection(tris[order], facecolors=cols[order], edgecolors="none"))
    allp = tris.reshape(-1, 2)
    lo, hi = allp.min(axis=0), allp.max(axis=0)
    c = (lo + hi) / 2.0
    rr = (hi - lo).max() / 2.0 * 1.05
    ax.set_xlim(c[0] - rr, c[0] + rr)
    ax.set_ylim(c[1] - rr, c[1] + rr)
    ax.set_aspect("equal")
    ax.axis("off")


def render_dir(d, out=None, dpi=75):
    t = json.load(open(os.path.join(d, "transform.json")))
    R = np.array(t["rotation"])
    view = R[:, 2]  # the relief camera (+z of the source) in the target frame
    side = np.cross(view, (0.0, 1.0, 0.0))
    side /= np.linalg.norm(side)
    target_mesh = t["target_meta"]["mesh"]
    target = load(target_mesh)
    src = load(os.path.join(d, "aligned_source.ply"))
    panels = [
        ("source on target (front)", [target, src], view),
        ("source on target (side)", [target, src], side),
    ]
    pw_path = os.path.join(d, "aligned_source_piecewise.ply")
    if os.path.exists(pw_path):
        pw = load(pw_path)
        panels += [
            ("piecewise (front)", [target, pw], view),
            ("piecewise (side)", [target, pw], side),
        ]
    for name, title in [
        ("target_from_source_colors_registered.ply", "painted (covered part)"),
        ("target_from_source_colors_mirrored.ply", "painted (covered + mirrored)"),
        ("target_from_source_colors_combined.ply", "painted (combined)"),
    ]:
        p = os.path.join(d, name)
        if os.path.exists(p):
            panels.append((title, [load(p)], view))
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.4), squeeze=False)
    for ax, (title, meshes, v) in zip(axes[0], panels):
        draw(ax, meshes, v)
        ax.set_title(f"{os.path.basename(d.rstrip('/'))}: {title}", fontsize=9)
    fig.tight_layout()
    out = out or os.path.join(d, "render.png")
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("dirs", nargs="+", help="register_partial.py output folders")
    ap.add_argument("--out", help="output image (only with a single folder)")
    ap.add_argument("--dpi", type=int, default=75)
    args = ap.parse_args()
    for d in args.dirs:
        print("saved", render_dir(d, args.out if len(args.dirs) == 1 else None, args.dpi))


if __name__ == "__main__":
    main()
