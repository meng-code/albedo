"""
Landmarks without a 3D viewer: render a mesh with a pixel grid, read the pixel coordinates of the
points you want off the image, and let the script turn them into mesh vertices.

Two steps, both from the repository root in the mesh environment:

1. A grid image to read coordinates from (the relief with its albedo, any other mesh shaded grey):
       python scripts/idarb/pick_landmarks.py --mesh outputs/mesh_depth/cat/relief.ply \
           --out outputs/diff3f/landmarks/cat_grid.png
       python scripts/idarb/pick_landmarks.py --mesh third_party/Diff3F/meshes/lion.off \
           --view 0.9 0.1 0.4 --out outputs/diff3f/landmarks/lion_grid.png
   --view is the direction the camera looks from (a registration's transform.json gives the
   relief's camera as the third column of its rotation).
2. The same command with --pick x,y x,y ... (pixel coordinates in that image, same --view and
   --size) writes the 3D points and vertex indices to --points (JSON) and draws them, numbered, on
   the image for checking.

Pick the same features in the same order on both meshes, then pass the two JSON files to
refine_correspondence.py as --landmarks after merging them:
       python scripts/idarb/pick_landmarks.py --merge cat_points.json lion_points.json \
           --out outputs/diff3f/landmarks/cat_lion.json
MeshLab's PickPoints tool (.pp files) is the alternative; refine_correspondence.py reads those too.
"""

import argparse
import json
import os
import sys

import numpy as np
import trimesh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transfer_texture import face_shade, project, raster_buffers  # noqa: E402


def render_grid(V, F, view, size, grid, colors=None):
    """Shaded render (vertex colours when the mesh has them) as float RGB plus the buffers."""
    buf = raster_buffers(V, F, view, size)
    shade, _ = face_shade(V, F, view)
    out = np.ones((size, size, 3))
    hit = buf["fid"] >= 0
    k = buf["fid"][hit]
    if colors is None:
        base = np.full((len(k), 3), 0.8)
    else:
        w0, w1 = buf["w0"][hit], buf["w1"][hit]
        w2 = 1.0 - w0 - w1
        c = colors[:, :3].astype(np.float64) / 255.0
        base = w0[:, None] * c[F[k, 0]] + w1[:, None] * c[F[k, 1]] + w2[:, None] * c[F[k, 2]]
    out[hit] = base * shade[k][:, None]
    return out, buf


def pick_vertices(buf, F, picks, snap=0):
    """Pixel coordinates (x, y) -> the mesh vertex under them (the corner with the largest
    barycentric weight). A pixel on the background snaps to the nearest surface pixel within
    `snap` pixels (silhouette landmarks such as ear tips and paws), else gives -1."""
    out = []
    size = buf["fid"].shape[0]
    for x, y in picks:
        xi, yi = int(round(x)), int(round(y))
        k = buf["fid"][yi, xi]
        if k < 0 and snap > 0:
            y0, y1 = max(yi - snap, 0), min(yi + snap + 1, size)
            x0, x1 = max(xi - snap, 0), min(xi + snap + 1, size)
            win = buf["fid"][y0:y1, x0:x1]
            ys, xs = np.nonzero(win >= 0)
            if len(ys):
                j = int(np.argmin((ys + y0 - yi) ** 2 + (xs + x0 - xi) ** 2))
                yi, xi = int(ys[j] + y0), int(xs[j] + x0)
                k = buf["fid"][yi, xi]
        if k < 0:
            out.append(-1)
            continue
        w0, w1 = buf["w0"][yi, xi], buf["w1"][yi, xi]
        w = np.array([w0, w1, 1 - w0 - w1])
        out.append(int(F[k, int(w.argmax())]))
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--mesh")
    ap.add_argument("--view", type=float, nargs=3, default=(0.0, 0.0, 1.0))
    ap.add_argument("--size", type=int, default=800)
    ap.add_argument("--grid", type=int, default=50, help="grid spacing in pixels")
    ap.add_argument("--pick", nargs="*", default=[], help="pixel coordinates x,y of the landmarks")
    ap.add_argument(
        "--snap", type=int, default=15, help="snap a background pick to the surface within pixels"
    )
    ap.add_argument("--points", help="JSON to write the picked points to")
    ap.add_argument("--merge", nargs=2, metavar=("SOURCE_JSON", "TARGET_JSON"))
    ap.add_argument("--out", required=True, help="image, or the merged landmarks JSON")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    if args.merge:
        with open(args.merge[0]) as f:
            s = json.load(f)
        with open(args.merge[1]) as f:
            t = json.load(f)
        n = min(len(s["points"]), len(t["points"]))
        if len(s["points"]) != len(t["points"]):
            print(
                f"warning: {len(s['points'])} source and {len(t['points'])} target points, "
                f"using the first {n}"
            )
        with open(args.out, "w") as f:
            json.dump(dict(source=s["points"][:n], target=t["points"][:n]), f, indent=1)
        print("saved", args.out, n, "landmark pairs")
        return

    m = trimesh.load(args.mesh, process=False, force="mesh")
    V = np.asarray(m.vertices, dtype=np.float64)
    F = np.asarray(m.faces, dtype=np.int64)
    colors = None
    if getattr(m.visual, "kind", None) == "vertex":
        colors = np.asarray(m.visual.vertex_colors)
    img, buf = render_grid(V, F, args.view, args.size, args.grid, colors)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(args.size / 100, args.size / 100), dpi=100)
    ax.imshow(img, extent=(0, args.size, args.size, 0))
    ticks = np.arange(0, args.size + 1, args.grid)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.tick_params(labelsize=6)
    ax.grid(color="deepskyblue", linewidth=0.4, alpha=0.8)
    picks = [tuple(float(v) for v in p.split(",")) for p in args.pick]
    if picks:
        ids = pick_vertices(buf, F, picks, args.snap)
        xy, _ = project(V, args.view, args.size)
        for n, ((x, y), vid) in enumerate(zip(picks, ids)):
            if vid < 0:
                print(f"{n}: ({x}, {y}) is background")
                continue
            ax.plot(xy[vid, 0], xy[vid, 1], "o", color="red", markersize=4)
            ax.annotate(str(n), (xy[vid, 0], xy[vid, 1]), color="yellow", fontsize=8, weight="bold")
        pts = [V[v].tolist() for v in ids if v >= 0]
        kept = [v for v in ids if v >= 0]
        if args.points:
            with open(args.points, "w") as f:
                record = dict(mesh=args.mesh, view=list(args.view), vertices=kept, points=pts)
                json.dump(record, f, indent=1)
            print("saved", args.points, len(kept), "points")
    fig.savefig(args.out, dpi=100, bbox_inches="tight")
    print("saved", args.out)


if __name__ == "__main__":
    main()
