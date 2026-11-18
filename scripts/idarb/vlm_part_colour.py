"""
Colouring a full mesh from the photo with a vision-language model as the only matcher (the
owner's idea of 2026-11-18): the mesh is cut into small regions, the model looks at renders in
which every region carries a numbered mark and at the photo, decides which part of the object
each region is and where that part lies in the photo, and the region takes the colour found
there. No feature of the chain is involved.

  make     <out>/<pair>/: the full mesh cut into --regions regions (farthest-point seeds, every
           vertex to its nearest seed); target_<v>.png, renders from six directions (the pair's
           view, turned left and right by 60 degrees, and the same from behind) with a numbered
           mark at each region's seed, in the view that faces it best; photo.png, the photo
           with a pixel ruler (as vlm_keypoint_trial.py); regions.npz and meta.json.
  colour   reads <out>/<pair>/answer.json ({"12": [x, y], ...}: where region 12's part lies in
           photo.png) and writes two coloured meshes and their previews:
             flat      every region in the median photo colour around its answer (regions
                       without an answer take the nearest answered region's colour)
             smooth    the answers as anchors of a smooth map (affine plus radial basis
                       functions, fit_prior) from the mesh into the photo plane; every vertex
                       takes the photo colour at its own mapped pixel, so patterns inside a
                       part come along
           <out>/<pair>/{flat,smooth}.ply and --figure_dir/<pair>.png (photo, flat, smooth,
           each mesh from the pair's view and from the opposite one).

CPU only, a few seconds per pair.

Usage (repository root, mesh environment):
    python scripts/idarb/vlm_part_colour.py make --pairs cat cat_lion labubu_man
    python scripts/idarb/vlm_part_colour.py colour
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_features as ef  # noqa: E402
import match2d as m2  # noqa: E402
import transfer_texture as tt  # noqa: E402
import vlm_keypoint_trial as vk  # noqa: E402
from correspondence_maps import load_mesh  # noqa: E402
from refine_correspondence import farthest_points, fit_prior  # noqa: E402
from sweep_anchor_filter import git_commit  # noqa: E402


def six_views(view, turn):
    view = np.asarray(view, dtype=np.float64)
    return m2.turned_views(view, turn) + m2.turned_views(-view, turn)


def best_views(visible, facing):
    """visible, facing (views, points): the visible view that faces each point best (-1: no
    view sees it)."""
    score = np.where(np.asarray(visible, dtype=bool), np.asarray(facing, dtype=np.float64), -2.0)
    best = score.argmax(axis=0)
    best[score.max(axis=0) <= -2.0] = -1
    return best


def disc_colour(photo, mask, row_col, radius):
    """The median colour of the object pixels within `radius` pixels of (row, col); the
    nearest object pixel's colour when the disc holds none."""
    rows, cols = np.nonzero(mask)
    tree = cKDTree(np.stack([rows, cols], axis=1) + 0.5)
    near = tree.query_ball_point(np.asarray(row_col, dtype=np.float64), radius)
    if not near:
        near = [tree.query(np.asarray(row_col, dtype=np.float64))[1]]
    return np.median(photo[rows[near], cols[near]].astype(np.float64), axis=0)


def pixel_colours(photo, mask, row_col):
    """The colour of the object pixel nearest to every (row, col)."""
    rows, cols = np.nonzero(mask)
    i = cKDTree(np.stack([rows, cols], axis=1) + 0.5).query(np.asarray(row_col))[1]
    return photo[rows[i], cols[i]]


def draw_marks(img, marks):
    from PIL import Image, ImageDraw, ImageFont

    try:
        font = ImageFont.load_default(size=17)
    except TypeError:  # older Pillow
        font = ImageFont.load_default()
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    for number, (py, px) in marks:
        d.ellipse([px - 5, py - 5, px + 5, py + 5], outline=(220, 0, 0), width=2)
        text = str(number)
        box = d.textbbox((px + 8, py - 20), text, font=font)
        d.rectangle([box[0] - 2, box[1] - 1, box[2] + 2, box[3] + 1], fill=(255, 255, 255),
                    outline=(0, 0, 200))
        d.text((px + 8, py - 20), text, fill=(0, 0, 200), font=font)
    return im


def pair_rows(pairs):
    return [r for r in ef.pair_table(["same", "cross"]) if not pairs or r[1] in pairs]


def source_photo(src_mesh, margin):
    mesh = load_mesh(src_mesh)
    Vs = np.asarray(mesh.vertices, dtype=np.float64)
    photo, vert = m2.albedo_image(Vs, np.asarray(mesh.visual.vertex_colors)[:, :3])
    fg = vert >= 0
    photo = photo.copy()
    photo[~fg] = 255
    crop, box = m2.square_crop(photo, fg, margin)
    return photo, fg, crop, box


def make(args):
    for _group, name, _s, src_mesh, target, transform, extra in pair_rows(args.pairs):
        out = os.path.join(args.out, name)
        os.makedirs(out, exist_ok=True)
        _photo, _fg, crop, box = source_photo(src_mesh, args.margin)
        vk.draw_photo(crop, args.photo_size).save(os.path.join(out, "photo.png"))

        tmesh = load_mesh(ef.TARGET_MESH[target])
        Vd = np.asarray(tmesh.vertices, dtype=np.float64)
        Fd = np.asarray(tmesh.faces, dtype=np.int64)
        normals = np.asarray(tmesh.vertex_normals, dtype=np.float64)
        seeds = farthest_points(Vd, args.regions)
        region = cKDTree(Vd[seeds]).query(Vd)[1]
        views = six_views(ef.view_of(transform or extra), args.turn)
        visible, facing, pixels, images = [], [], [], []
        for v in views:
            buf = tt.raster_buffers(Vd, Fd, v, args.render_size)
            vis, py, px = m2.visible_vertices(Vd, Fd, buf, args.render_size)
            visible.append(vis[seeds])
            facing.append(normals[seeds] @ v)
            pixels.append(np.stack([py[seeds], px[seeds]], axis=1))
            images.append(m2.render_target(Vd, Fd, v, args.render_size)[0])
        best = best_views(visible, facing)
        renders = {}
        for i in sorted(set(best.tolist()) - {-1}):
            ks = [int(k) for k in np.nonzero(best == i)[0]]
            draw_marks(images[i], [(k + 1, pixels[i][k]) for k in ks]).save(
                os.path.join(out, f"target_{i}.png"))
            renders[f"target_{i}.png"] = [k + 1 for k in ks]
        np.savez_compressed(os.path.join(out, "regions.npz"), seeds=seeds, region=region,
                            views=np.asarray(views))
        meta = dict(pair=name, regions=int(len(seeds)), box=[int(b) for b in box],
                    photo_size=args.photo_size, renders=renders,
                    unseen=[int(k) + 1 for k in np.nonzero(best < 0)[0]],
                    commit=git_commit(), command=" ".join(sys.argv))
        with open(os.path.join(out, "meta.json"), "w") as f:
            json.dump(meta, f, indent=1)
        print(json.dumps(dict(pair=name, renders={k: len(v) for k, v in renders.items()},
                              unseen=meta["unseen"])), flush=True)


def colour(args):
    from PIL import Image, ImageDraw

    os.makedirs(args.figure_dir, exist_ok=True)
    for _group, name, _s, src_mesh, target, _tr, _ex in pair_rows(args.pairs):
        folder = os.path.join(args.out, name)
        answer = os.path.join(folder, "answer.json")
        if not os.path.exists(answer):
            continue
        with open(answer) as f:
            ans = json.load(f)
        with open(os.path.join(folder, "meta.json")) as f:
            meta = json.load(f)
        z = np.load(os.path.join(folder, "regions.npz"))
        seeds, region, views = z["seeds"], z["region"], z["views"]
        photo, fg, crop, box = source_photo(src_mesh, args.margin)
        tmesh = load_mesh(ef.TARGET_MESH[target])
        Vd = np.asarray(tmesh.vertices, dtype=np.float64)
        Fd = np.asarray(tmesh.faces, dtype=np.int64)

        has = np.array([ans.get(str(k + 1)) is not None for k in range(len(seeds))])
        rc = np.array([vk.ruler_to_image(ans[str(k + 1)], box, meta["photo_size"])
                       for k in np.nonzero(has)[0]], dtype=np.float64)
        radius = args.disc * box[2]
        seed_rgb = np.array([disc_colour(photo, fg, p, radius) for p in rc])
        # flat: unanswered regions take the nearest answered region's colour
        nearest = cKDTree(Vd[seeds[has]]).query(Vd[seeds])[1]
        flat = seed_rgb[nearest][region]
        # smooth: the answers as anchors of a map from the mesh into the photo plane
        diag = float(np.linalg.norm(Vd.max(axis=0) - Vd.min(axis=0)))
        smooth = pixel_colours(photo, fg, fit_prior(Vd[seeds[has]], rc, args.sigma * diag)(Vd))

        size = args.preview_size
        panels = [Image.fromarray(crop).resize((size, size), Image.LANCZOS).convert("RGB")]
        for label, rgb in (("flat", flat), ("smooth", smooth)):
            rgba = np.concatenate([np.clip(rgb, 0, 255),
                                   np.full((len(rgb), 1), 255.0)], axis=1).astype(np.uint8)
            out_mesh = tmesh.copy()
            out_mesh.visual.vertex_colors = rgba
            out_mesh.export(os.path.join(folder, label + ".ply"))
            for v in (views[0], views[3]):
                img = m2.render_target(Vd, Fd, v, size, colors=rgb * 1.25)[0]
                panels.append(Image.fromarray(img))
        sheet = Image.new("RGB", (size * len(panels), size + 50), (255, 255, 255))
        d = ImageDraw.Draw(sheet)
        titles = ["photo", "flat, the pair's view", "flat, opposite view",
                  "smooth, the pair's view", "smooth, opposite view"]
        for i, (im, title) in enumerate(zip(panels, titles)):
            sheet.paste(im, (i * size, 26))
            d.text((i * size + 8, 6), title, fill=(0, 0, 0))
        d.text((8, size + 8), f"{name}: {int(has.sum())} of {len(seeds)} regions answered",
               fill=(0, 0, 0))
        path = os.path.join(args.figure_dir, name + ".png")
        sheet.save(path)
        with open(os.path.join(args.figure_dir, name + "_run.json"), "w") as f:
            json.dump(dict(commit=git_commit(), command=" ".join(sys.argv),
                           regions=int(len(seeds)), answered=int(has.sum()),
                           model=ans.get("model", "")), f, indent=1)
        print(path, flush=True)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("mode", choices=["make", "colour"])
    ap.add_argument("--out", default="outputs/diff3f/28_vlm_part_colour")
    ap.add_argument("--figure_dir", default="results/diff3f/vlm_part_colour")
    ap.add_argument("--pairs", nargs="*", help="only these pair names")
    ap.add_argument("--regions", type=int, default=48)
    ap.add_argument("--photo_size", type=int, default=1000)
    ap.add_argument("--render_size", type=int, default=1000)
    ap.add_argument("--preview_size", type=int, default=520)
    ap.add_argument("--turn", type=float, default=60.0)
    ap.add_argument("--margin", type=float, default=0.08)
    ap.add_argument("--disc", type=float, default=0.012,
                    help="radius of the colour sample, of the photo crop's side")
    ap.add_argument("--sigma", type=float, default=0.25,
                    help="width of the smooth map's basis functions, of the mesh diagonal")
    args = ap.parse_args()
    if args.mode == "make":
        make(args)
    else:
        colour(args)


if __name__ == "__main__":
    main()
