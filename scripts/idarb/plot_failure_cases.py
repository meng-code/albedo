"""
Draw a failure-case figure. Each object gets one row; from left to right: the input image, the
IDArb four-panel output (albedo, normal, metallic, roughness), the mesh from normals only, and the
mesh with the depth prior. The meshes are rebuilt with exactly the same functions and default
parameters as normal_to_mesh.py, and rendered from the side (rotated 90 degrees about the vertical
axis) to show whether the object has any thickness.

The script also prints a few diagnostic numbers for each object:
  mean nz         mean z component of the normals (the component towards the camera);
                  1 means the whole object faces the camera
  nz < 0.9        fraction of pixels whose normals are clearly tilted
  s               depth-prior scale in pixels, fitted by matching the depth map to the normal slopes
  depth range     difference between the farthest and nearest point of the reconstructed depth,
                  in pixels (normals only / with depth prior)
  width           width of the object in the image, in pixels

Usage (from the repository root, in the same environment as normal_to_mesh.py):
    python scripts/idarb/plot_failure_cases.py --names cat2 cat3 cat4 \
        --out results/analysis/failure_cats.png
"""

import argparse
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import normal_to_mesh as ntm  # noqa: E402

ROW_H = 256
YAW = 90


def reconstruct(name, root, erode_px=2):
    """Rebuild both meshes with the defaults of normal_to_mesh.process_one.

    Returns the two side-view renders and the diagnostic numbers.
    """
    albedo, normal_rgb, _, _ = ntm.load_result(
        os.path.join(root, "decomposition", name + "_00.png")
    )
    size = normal_rgb.shape[:2]
    mask = ntm.erode(ntm.load_mask(os.path.join(root, "inputs", name + ".png"), size), erode_px)
    n = ntm.decode_normal(normal_rgb)
    d = ntm.load_depth(os.path.join(root, "depth", name + ".png"), size)

    h_n = ntm.integrate_normals(n, mask)
    h_d, info = ntm.integrate_normals_with_depth(n, d, mask)

    renders = []
    for h, cut in ((h_n, None), (h_d, info)):
        verts, faces, vcol = ntm.heightmap_to_mesh(h, mask, albedo)
        if cut is not None:
            faces = ntm.drop_cut_faces(faces, cut["cut_i"], cut["cut_j"], len(verts))
        verts = verts - verts.mean(axis=0)
        renders.append(ntm.render_fast(verts, faces, vcol, YAW, res=ROW_H))

    nz = n[..., 2][mask]
    cols = np.nonzero(mask.any(axis=0))[0]
    stats = dict(
        nz_mean=float(nz.mean()),
        tilted=float((nz < 0.9).mean()),
        s=info["scale"],
        range_n=float(np.nanmax(h_n) - np.nanmin(h_n)),
        range_d=float(np.nanmax(h_d) - np.nanmin(h_d)),
        width=int(cols.max() - cols.min()),
    )
    return renders, stats


def input_on_white(path):
    """Input image resized, padded and put on white exactly as at IDArb inference.

    This keeps it pixel-aligned with the IDArb output.
    """
    rgb, _ = ntm.load_input_512(path)
    return Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="+", default=["cat2", "cat3", "cat4"])
    ap.add_argument("--root", default="results/idarb_single")
    ap.add_argument("--out", default="results/analysis/failure_cats.png")
    args = ap.parse_args()

    headers = [
        "input",
        "IDArb: albedo | normal | metallic | roughness",
        "mesh, normals only (side view)",
        "mesh, normals + depth prior (side view)",
    ]
    widths = [ROW_H, 4 * ROW_H, ROW_H, ROW_H]
    gap, top, left = 6, 24, 0
    W = sum(widths) + gap * (len(widths) - 1)
    rows = []
    print("object   mean nz  nz<0.9    s (px)  range normals only  with depth  width")
    for name in args.names:
        renders, st = reconstruct(name, args.root)
        print(
            "{:8s} {:7.3f} {:6.1f}% {:9.1f} {:19.1f} {:11.1f} {:6d}".format(
                name,
                st["nz_mean"],
                100 * st["tilted"],
                st["s"],
                st["range_n"],
                st["range_d"],
                st["width"],
            )
        )
        decomp = Image.open(os.path.join(args.root, "decomposition", name + "_00.png")).convert(
            "RGB"
        )
        tiles = [
            input_on_white(os.path.join(args.root, "inputs", name + ".png")).resize((ROW_H, ROW_H)),
            decomp.resize((4 * ROW_H, ROW_H)),
            Image.fromarray(renders[0]),
            Image.fromarray(renders[1]),
        ]
        label = "{}   mean nz {:.3f}   s = {:.1f} px   depth range {:.1f} / {:.1f} px".format(
            name, st["nz_mean"], st["s"], st["range_n"], st["range_d"]
        )
        rows.append((tiles, label))

    row_total = ROW_H + 20
    canvas = Image.new("RGB", (W, top + len(rows) * (row_total + gap)), (255, 255, 255))
    dr = ImageDraw.Draw(canvas)
    x = left
    for i, head in enumerate(headers):
        dr.text((x + 4, 6), head, fill=(0, 0, 0))
        x += widths[i] + gap
    for r, (tiles, label) in enumerate(rows):
        y = top + r * (row_total + gap)
        dr.text((4, y + 2), label, fill=(160, 0, 0))
        x = left
        for i, tile in enumerate(tiles):
            canvas.paste(tile, (x, y + 20))
            x += widths[i] + gap
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    canvas.save(args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
