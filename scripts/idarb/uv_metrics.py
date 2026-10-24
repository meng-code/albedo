"""
Quality of a transferred texture map, measured the way mesh parametrisations are judged: for every
textured triangle, the map from the triangle's own shape to its image in the texture is a linear
map with two singular values, and

  conformal distortion  sigma_max / sigma_min   (1: the triangle keeps its shape, large: a streak)
  flip                  the triangle lands on the texture mirrored, which shows as a reversed patch

The same two numbers (median conformal distortion and the flipped fraction) are what image-to-3D
attribute-transfer papers report (for example RealSkin, arXiv 2607.12495, mCD and Flip%), so they
put our results on the same footing. The orientation reference is the view the texture was taken
from: a triangle facing the relief's camera must map onto the image with the same winding it shows
on screen (u right, v up against screen right, screen up).

Reads the OBJ written by transfer_texture.py (per-corner texture coordinates; faces whose three
corners share the blank texture coordinate are skipped) and writes the numbers to stdout and,
with --csv, appends a row to a CSV.

Usage (repository root, mesh environment):
    python scripts/idarb/uv_metrics.py --obj <folder>/target_textured.obj --view x y z \
        [--label cat_lion/nn] [--csv results/diff3f/uv_quality.csv]
"""

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transfer_texture import view_frame  # noqa: E402


def read_textured_obj(path):
    """Vertices (N,3), faces (M,3), per-corner texture coordinates (M,3,2) of an OBJ with v/vt."""
    V, VT, F, FT = [], [], [], []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                V.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("vt "):
                VT.append([float(x) for x in line.split()[1:3]])
            elif line.startswith("f "):
                corners = [c.split("/") for c in line.split()[1:4]]
                F.append([int(c[0]) - 1 for c in corners])
                FT.append([int(c[1]) - 1 if len(c) > 1 and c[1] else -1 for c in corners])
    if not VT:
        raise ValueError(f"{path} has no texture coordinates")
    V, VT, F, FT = np.array(V), np.array(VT), np.array(F), np.array(FT)
    uvc = np.where(FT[:, :, None] >= 0, VT[np.maximum(FT, 0)], np.nan)
    # a blank face points its three corners at one and the same texture coordinate
    blank = (FT[:, 0] == FT[:, 1]) & (FT[:, 1] == FT[:, 2])
    uvc[blank] = np.nan
    return V, F, uvc


def triangle_maps(V, F, uvc, image_wh):
    """Singular values (M,2) and orientation signs of the map from every triangle (in its own
    plane, isometrically) to its texture image (in pixels), plus the screen orientation sign of the
    triangle is left to the caller."""
    P = V[F]
    e1 = P[:, 1] - P[:, 0]
    e2 = P[:, 2] - P[:, 0]
    # local 2D coordinates of the triangle: x along e1, y in the plane
    l1 = np.linalg.norm(e1, axis=1)
    x_axis = e1 / np.maximum(l1[:, None], 1e-12)
    n = np.cross(e1, e2)
    y_axis = np.cross(n, x_axis)
    y_axis /= np.maximum(np.linalg.norm(y_axis, axis=1, keepdims=True), 1e-12)
    a = np.stack([l1, np.zeros_like(l1)], axis=1)  # p1 - p0 in local coords
    b = np.stack([(e2 * x_axis).sum(1), (e2 * y_axis).sum(1)], axis=1)  # p2 - p0
    Q = uvc * np.array(image_wh)  # pixels, v still up
    qa = Q[:, 1] - Q[:, 0]
    qb = Q[:, 2] - Q[:, 0]
    # J maps local coords to texture: J [a b] = [qa qb]  ->  J = [qa qb] [a b]^-1
    AB = np.stack([a, b], axis=2)  # (M,2,2) columns a, b
    QAB = np.stack([qa, qb], axis=2)
    det = AB[:, 0, 0] * AB[:, 1, 1] - AB[:, 0, 1] * AB[:, 1, 0]
    inv = np.stack(
        [
            np.stack([AB[:, 1, 1], -AB[:, 0, 1]], axis=1),
            np.stack([-AB[:, 1, 0], AB[:, 0, 0]], axis=1),
        ],
        axis=1,
    ) / np.where(np.abs(det) < 1e-18, np.nan, det)[:, None, None]
    J = QAB @ inv
    sv = np.linalg.svd(np.nan_to_num(J), compute_uv=False)
    return sv, np.sign(np.linalg.det(np.nan_to_num(J))), n


def uv_quality(V, F, uvc, view, image_wh=(1.0, 1.0)):
    """Dict with the number of textured triangles whose corners are not all on one texture point,
    the fraction whose three corners collapsed onto one point, the fraction (of the rest) that are
    degenerate (two corners on one point or three in a line: no patch at all), and over the
    triangles with a real patch the median conformal distortion, the fraction distorted more than
    tenfold and the flipped fraction (plus its screen-area-weighted version). A triangle is
    flipped when its winding on the texture is opposite to its winding on screen seen from
    `view`. image_wh scales u and v to pixels so a non-square image does not count as
    distortion."""
    textured = ~np.isnan(uvc).any(axis=(1, 2))
    span = uvc[textured]
    distinct = (np.abs(span - span[:, :1]) > 1e-9).any(axis=(1, 2))  # not all three on one point
    keep = np.nonzero(textured)[0][distinct]
    collapsed = 1.0 - distinct.mean() if textured.any() else 0.0
    sv, _, _ = triangle_maps(V, F[keep], uvc[keep], image_wh)
    R = view_frame(view)
    # screen winding: project the triangle onto (screen right, screen up)
    P = V[F[keep]] @ R.T
    s1 = P[:, 1, :2] - P[:, 0, :2]
    s2 = P[:, 2, :2] - P[:, 0, :2]
    screen_area2 = s1[:, 0] * s2[:, 1] - s1[:, 1] * s2[:, 0]
    sign_screen = np.sign(screen_area2)
    # texture winding, in the same orientation convention (u right, v up)
    Q = uvc[keep] * np.array(image_wh)
    qa, qb = Q[:, 1] - Q[:, 0], Q[:, 2] - Q[:, 0]
    tex2 = qa[:, 0] * qb[:, 1] - qa[:, 1] * qb[:, 0]
    # two corners on one point or three in a line: no patch at all, no orientation either
    degenerate = np.abs(tex2) <= 1e-12 * ((qa**2).sum(axis=1) + (qb**2).sum(axis=1))
    ok = ~degenerate & (sign_screen != 0)
    flipped = (np.sign(tex2) != sign_screen) & ok
    cd = sv[:, 0] / np.maximum(sv[:, 1], 1e-12)
    # a triangle seen edge-on has no reliable winding on screen; weighting the flips by screen
    # area discounts those silhouette slivers
    area = np.abs(screen_area2) * ok
    flipped_area = float((area * flipped).sum() / max(area.sum(), 1e-12))
    n_ok = max(int(ok.sum()), 1)
    return dict(
        triangles=int(len(keep)),
        collapsed=round(float(collapsed), 4),
        degenerate=round(float(degenerate.mean()) if len(keep) else 0.0, 4),
        conformal_median=round(float(np.median(cd[ok])) if ok.any() else float("nan"), 3),
        distortion_over_10=round(float((cd[ok] > 10).mean()) if ok.any() else 0.0, 4),
        flipped=round(float(flipped.sum() / n_ok), 4),
        flipped_area=round(flipped_area, 4),
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--obj", required=True, help="target_textured.obj of transfer_texture.py")
    ap.add_argument("--view", type=float, nargs=3, required=True, help="the texture's camera side")
    ap.add_argument("--label", default="")
    ap.add_argument("--csv", help="append a row here")
    args = ap.parse_args()
    V, F, uvc = read_textured_obj(args.obj)
    png = os.path.splitext(args.obj)[0] + ".png"
    image_wh = (1.0, 1.0)
    if os.path.exists(png):
        from PIL import Image

        image_wh = tuple(float(x) for x in Image.open(png).size)
    q = uv_quality(V, F, uvc, args.view, image_wh)
    print(dict(label=args.label, **q))
    if args.csv:
        new = not os.path.exists(args.csv)
        with open(args.csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["label"] + list(q.keys()))
            if new:
                w.writeheader()
            w.writerow(dict(label=args.label, **q))


if __name__ == "__main__":
    main()
