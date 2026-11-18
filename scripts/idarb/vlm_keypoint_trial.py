"""
A trial of a vision-language model as the matcher at the annotated keypoints: the model sees a
grey render of the full mesh with numbered marks at the keypoints, and the photo behind the
partial mesh with a pixel ruler, and answers where each mark lies in the photo. No feature of
the chain is involved, and the keypoint names are not shown.

  make    per annotated pair, into <out>/<pair>/:
            target_<v>.png   the render from one viewing direction with the numbered marks of
                             the keypoints first visible from it (the pair's own view, turned
                             left and right, then the same three from behind)
            photo.png        the albedo image on white, cropped square to the object, scaled to
                             --photo_size pixels, with ruler lines every 100 pixels
            meta.json        the crop box, the scale and the keypoints of every render
  score   reads <out>/<pair>/answer.json ({"1": [x, y], ...}: ruler coordinates in photo.png,
          x to the right, y down), takes the vertex of the partial mesh at the nearest object
          pixel, and scores it like keypoint_table.py: the distance to the annotated source
          point over the partial mesh's bounding-box diagonal, in 3D and in the image plane.
          "blind": false in an answer file marks answers given or corrected after the
          annotation of that pair was seen; they are summarised apart.
          Writes per_keypoint.csv and summary.csv to --csv_dir.

The answers are produced by the model reading the images (in a session, by hand); the file
records which model in "model". CPU only, a few seconds per pair.

Usage (repository root, mesh environment):
    python scripts/idarb/vlm_keypoint_trial.py make --pairs cat labubu_camel
    python scripts/idarb/vlm_keypoint_trial.py score --csv_dir results/diff3f/vlm_keypoints
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_features as ef  # noqa: E402
import match2d as m2  # noqa: E402
import transfer_texture as tt  # noqa: E402
from correspondence_maps import load_mesh  # noqa: E402
from sweep_anchor_filter import git_commit  # noqa: E402

RULER = 100


def keypoint_file(name, keypoints, landmarks):
    for folder in (keypoints, landmarks):
        path = os.path.join(folder, name + ".json")
        if os.path.exists(path):
            return path
    return None


def candidate_views(view, turn):
    """The view, turned left and right, then the same three from behind."""
    view = np.asarray(view, dtype=np.float64)
    return m2.turned_views(view, turn) + m2.turned_views(-view, turn)


def assign_views(visible):
    """visible (views, keypoints) bool -> the first view that sees each keypoint (-1: none)."""
    visible = np.asarray(visible, dtype=bool)
    first = visible.argmax(axis=0)
    first[~visible.any(axis=0)] = -1
    return first


def ruler_to_image(xy, box, photo_size):
    """Ruler coordinates (x, y) of photo.png -> pixel (row, col) of the albedo image."""
    oy, ox, side = box
    x, y = xy
    return oy + y * side / photo_size, ox + x * side / photo_size


def nearest_object_vertex(row_col, vert):
    """The partial-mesh vertex at the object pixel nearest to (row, col)."""
    rows, cols = np.nonzero(vert >= 0)
    i = cKDTree(np.stack([rows, cols], axis=1) + 0.5).query(np.asarray(row_col))[1]
    return int(vert[rows[i], cols[i]])


def draw_marks(img, marks):
    from PIL import Image, ImageDraw

    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    for number, (py, px) in marks:
        d.ellipse([px - 7, py - 7, px + 7, py + 7], outline=(220, 0, 0), width=2)
        d.ellipse([px - 1, py - 1, px + 1, py + 1], fill=(220, 0, 0))
        d.text((px + 11, py - 16), str(number), fill=(0, 0, 220))
    return im


def draw_photo(crop, photo_size):
    from PIL import Image, ImageDraw

    # no border: a ruler coordinate is the pixel position in the file itself. A first version
    # had a 36-pixel band for the numbers and the model's answers came back shifted by it.
    out = Image.fromarray(crop).resize((photo_size, photo_size), Image.LANCZOS).convert("RGB")
    d = ImageDraw.Draw(out, "RGBA")
    for v in range(RULER // 2, photo_size, RULER // 2):
        strong = v % RULER == 0
        shade = (0, 0, 0, 70 if strong else 28)
        d.line([v, 0, v, photo_size], fill=shade)
        d.line([0, v, photo_size, v], fill=shade)
        if strong:
            for pos in ((v + 3, 2), (v + 3, photo_size - 13), (2, v + 2), (photo_size - 24, v + 2)):
                d.text(pos, str(v), fill=(0, 0, 0, 255))
    return out


def make(args):
    for _group, name, _s, src_mesh, target, transform, extra in ef.pair_table(["same", "cross"]):
        kp = keypoint_file(name, args.keypoints, args.landmarks)
        if kp is None or (args.pairs and name not in args.pairs):
            continue
        with open(kp) as f:
            lm_t = np.asarray(json.load(f)["target"], dtype=np.float64)
        out = os.path.join(args.out, name)
        os.makedirs(out, exist_ok=True)

        mesh = load_mesh(src_mesh)
        Vs = np.asarray(mesh.vertices, dtype=np.float64)
        photo, vert = m2.albedo_image(Vs, np.asarray(mesh.visual.vertex_colors)[:, :3])
        fg = vert >= 0
        photo = photo.copy()
        photo[~fg] = 255
        crop, box = m2.square_crop(photo, fg, args.margin)
        draw_photo(crop, args.photo_size).save(os.path.join(out, "photo.png"))

        tmesh = load_mesh(ef.TARGET_MESH[target])
        Vd = np.asarray(tmesh.vertices, dtype=np.float64)
        Fd = np.asarray(tmesh.faces, dtype=np.int64)
        t_idx = cKDTree(Vd).query(lm_t)[1]
        views = candidate_views(ef.view_of(transform or extra), args.turn)
        visible, pixels, images = [], [], []
        for v in views:
            buf = tt.raster_buffers(Vd, Fd, v, args.render_size)
            vis, py, px = m2.visible_vertices(Vd, Fd, buf, args.render_size)
            visible.append(vis[t_idx])
            pixels.append(np.stack([py[t_idx], px[t_idx]], axis=1))
            images.append(m2.render_target(Vd, Fd, v, args.render_size)[0])
        first = assign_views(visible)
        renders = {}
        for i in sorted(set(first.tolist()) - {-1}):
            ks = [int(k) for k in np.nonzero(first == i)[0]]
            draw_marks(images[i], [(k + 1, pixels[i][k]) for k in ks]).save(
                os.path.join(out, f"target_{i}.png"))
            renders[f"target_{i}.png"] = [k + 1 for k in ks]
        meta = dict(pair=name, keypoints=len(lm_t), box=[int(b) for b in box],
                    photo_size=args.photo_size, renders=renders,
                    unseen=[int(k) + 1 for k in np.nonzero(first < 0)[0]],
                    commit=git_commit(), command=" ".join(sys.argv))
        with open(os.path.join(out, "meta.json"), "w") as f:
            json.dump(meta, f, indent=1)
        print(json.dumps(dict(pair=name, renders=renders, unseen=meta["unseen"])), flush=True)


def score(args):
    rows = []
    for group, name, _s, src_mesh, _t, _tr, _ex in ef.pair_table(["same", "cross"]):
        folder = os.path.join(args.out, name)
        answer = os.path.join(folder, "answer.json")
        kp = keypoint_file(name, args.keypoints, args.landmarks)
        if kp is None or not os.path.exists(answer):
            continue
        with open(kp) as f:
            lm_s = np.asarray(json.load(f)["source"], dtype=np.float64)
        with open(answer) as f:
            ans = json.load(f)
        with open(os.path.join(folder, "meta.json")) as f:
            meta = json.load(f)
        mesh = load_mesh(src_mesh)
        Vs = np.asarray(mesh.vertices, dtype=np.float64)
        _, vert = m2.albedo_image(Vs, np.asarray(mesh.visual.vertex_colors)[:, :3])
        diag = float(np.linalg.norm(Vs.max(axis=0) - Vs.min(axis=0)))
        for k in range(len(lm_s)):
            xy = ans.get(str(k + 1))
            if xy is None:
                continue
            v = nearest_object_vertex(ruler_to_image(xy, meta["box"], meta["photo_size"]), vert)
            d = Vs[v] - lm_s[k]
            rows.append(dict(group=group, pair=name, keypoint=k + 1, model=ans.get("model", ""),
                             blind=int(bool(ans.get("blind", True))),
                             error=round(float(np.linalg.norm(d)) / diag, 4),
                             error_plane=round(float(np.linalg.norm(d[:2])) / diag, 4)))
    if not rows:
        sys.exit("no answer.json found under " + args.out)
    os.makedirs(args.csv_dir, exist_ok=True)
    summary = []
    for group, blind in (("same", 1), ("cross", 1), ("same", 0), ("cross", 0)):
        rs = [r for r in rows if r["group"] == group and r["blind"] == blind]
        if not rs:
            continue
        for key in ("error", "error_plane"):
            e = np.array([r[key] for r in rs])
            summary.append(dict(group=group, blind=blind, measure=key,
                                pairs=len({r["pair"] for r in rs}),
                                keypoints=len(e), hits_2=int((e < 0.02).sum()),
                                hits_5=int((e < 0.05).sum()), hits_10=int((e < 0.10).sum()),
                                pck_2=round(100.0 * float((e < 0.02).mean()), 1),
                                mean_error=round(float(e.mean()), 4),
                                median_error=round(float(np.median(e)), 4)))
    for fname, table in (("per_keypoint.csv", rows), ("summary.csv", summary)):
        with open(os.path.join(args.csv_dir, fname), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
            w.writeheader()
            w.writerows(table)
    with open(os.path.join(args.csv_dir, "run.json"), "w") as f:
        json.dump(dict(commit=git_commit(), command=" ".join(sys.argv)), f, indent=1)
    cols = list(summary[0].keys())
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for r in summary:
        print("| " + " | ".join(str(r[c]) for c in cols) + " |")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("mode", choices=["make", "score"])
    ap.add_argument("--out", default="outputs/diff3f/27_vlm_keypoints")
    ap.add_argument("--csv_dir", default="results/diff3f/vlm_keypoints")
    ap.add_argument("--keypoints", default="results/diff3f/keypoints")
    ap.add_argument("--landmarks", default="results/diff3f/landmarks")
    ap.add_argument("--pairs", nargs="*", help="only these pair names")
    ap.add_argument("--photo_size", type=int, default=1000)
    ap.add_argument("--render_size", type=int, default=900)
    ap.add_argument("--turn", type=float, default=40.0)
    ap.add_argument("--margin", type=float, default=0.08)
    args = ap.parse_args()
    if args.mode == "make":
        make(args)
    else:
        score(args)


if __name__ == "__main__":
    main()
