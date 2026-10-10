"""
Experiment (supervisor's suggestion, 2026-10-10): build a point cloud directly from the Depth
Anything depth map (x, y from the pixel, z = s * d), attach IDArb's normal to every point, and run
screened Poisson surface reconstruction on that cloud. Compare it with the other ways of building
the mesh.

For every variant the script prints one metric: the angle between the mesh's vertex normals and
IDArb's normals at the same pixel (smaller means the surface follows the normals, i.e. keeps the
surface detail), and the depth-to-width ratio (front-to-back structure).

Variants:
  relief, normals only        method A alone
  relief, normals + depth     method A with the depth prior (the current default)
  relief, depth only          the depth map itself connected into a mesh
  poisson, depth points       the supervisor's suggestion
  poisson, joint points       the current method B (points from the joint solve)

Usage (from the repository root, in the mesh environment):
    python scripts/idarb/poisson_on_depth_points.py --names chair1 superman panda \
        --out_dir outputs/poisson_on_depth
"""

import argparse
import os
import sys

import numpy as np
import trimesh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import normal_to_mesh as ntm  # noqa: E402


def normal_deviation(verts, faces, n, mask, scale, shift):
    """Mean and median angle (degrees) between mesh and IDArb normals, plus depth/width."""
    H, W = mask.shape
    vn = np.asarray(trimesh.Trimesh(vertices=verts, faces=faces, process=False).vertex_normals)
    v = verts + shift  # undo the centring so vertices map back to pixels
    cc = np.clip(np.round(v[:, 0] / scale + W / 2).astype(int), 0, W - 1)
    rr = np.clip(np.round(H / 2 - v[:, 1] / scale).astype(int), 0, H - 1)
    ok = mask[rr, cc] & np.isfinite(vn).all(axis=1)
    cos = np.clip(np.abs((vn[ok] * n[rr[ok], cc[ok]]).sum(axis=1)), 0, 1)  # abs: ignore front/back
    ang = np.degrees(np.arccos(cos))
    width = (cc[ok].max() - cc[ok].min()) * scale
    depth = v[ok, 2].max() - v[ok, 2].min()
    return float(ang.mean()), float(np.median(ang)), float(depth / width)


def run(name, root, out_dir, poisson_depth):
    result_path = os.path.join(root, "decomposition", name + "_00.png")
    albedo, normal_rgb, _, _ = ntm.load_result(result_path)
    size = normal_rgb.shape[:2]
    mask = ntm.erode(ntm.load_mask(os.path.join(root, "inputs", name + ".png"), size), 2)
    n = ntm.decode_normal(normal_rgb)
    d = ntm.load_depth(os.path.join(root, "depth", name + ".png"), size)
    scale = 2.0 / max(size)

    h_n = ntm.integrate_normals(n, mask)
    h_nd, info = ntm.integrate_normals_with_depth(n, d, mask)
    h_d = np.where(mask, info["scale"] * d, np.nan)

    out = os.path.join(out_dir, name)
    os.makedirs(out, exist_ok=True)
    print(f"\n{name}  (depth-prior scale s = {info['scale']:.0f} px)")
    print(f"  {'variant':28s} {'mean deg':>9s} {'median':>7s} {'depth/width':>12s}")

    reliefs = [
        ("relief, normals only", h_n, None),
        ("relief, normals + depth", h_nd, info),
        ("relief, depth only", h_d, None),
    ]
    for tag, h, cut in reliefs:
        v, f, c = ntm.heightmap_to_mesh(h, mask, albedo)
        if cut is not None:
            f = ntm.drop_cut_faces(f, cut["cut_i"], cut["cut_j"], len(v))
        shift = v.mean(axis=0)
        mean, med, dw = normal_deviation(v - shift, f, n, mask, scale, shift)
        print(f"  {tag:28s} {mean:9.1f} {med:7.1f} {dw:12.2f}")

    poissons = [("poisson, depth points", h_d), ("poisson, joint points", h_nd)]
    for tag, h in poissons:
        v, _, c = ntm.heightmap_to_mesh(h, mask, albedo)
        pv, pf, pc = ntm.screened_poisson(v, n[mask], c, depth=poisson_depth)
        shift = pv.mean(axis=0)
        mean, med, dw = normal_deviation(pv - shift, pf, n, mask, scale, shift)
        print(f"  {tag:28s} {mean:9.1f} {med:7.1f} {dw:12.2f}")
        fname = tag.replace("poisson, ", "poisson_").replace(" ", "_") + ".ply"
        ntm.save_mesh(os.path.join(out, fname), pv - shift, pf, pc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="+", default=["chair1", "superman", "panda"])
    ap.add_argument("--root", default="results/idarb_single")
    ap.add_argument("--out_dir", default="outputs/poisson_on_depth")
    ap.add_argument("--poisson_depth", type=int, default=9)
    args = ap.parse_args()
    for name in args.names:
        run(name, args.root, args.out_dir, args.poisson_depth)


if __name__ == "__main__":
    main()
