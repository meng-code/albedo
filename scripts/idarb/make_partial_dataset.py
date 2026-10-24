"""
Build a synthetic partial-to-full dataset from Objaverse 1.0: full meshes, partial meshes cut by one
camera (make_partial_mesh.py), and the ground-truth correspondence between them. Phase A of the
dataset plan: no features yet. Runs on the GPU server in the dataset environment (python 3.10,
objaverse 0.1.7, trimesh, open3d 0.20, scipy, pillow; open3d needs libegl and libgl from
conda-forge), but needs no GPU. Data goes under --root (default ~/semtex/dataset), outside the
repository.

Two steps, run in order with --step:

select    Reads the LVIS category annotations of Objaverse (about 46 k objects in 1156 categories)
          and the Sketchfab metadata of the objects in the categories of GROUPS below (license,
          name, author, face and vertex counts, size of the GLB file). Keeps the objects whose
          license is CC BY 4.0 ("by") or CC0 and whose Sketchfab face count is within
          --face_range widened by 25 % (the exact count is checked after conversion), and writes
          candidates.csv in a fixed random order (a hash of --seed and the uid, so adding
          categories does not reorder the others). Downloads are made with curl, which is
          much faster here than the objaverse package's own downloader.
meshes    Goes through the candidates of each group category by category; in every round only
          the categories with the fewest kept objects try their next candidate, so the kept
          counts stay balanced, until the group's quota is reached. Each GLB is downloaded, all
          its parts are joined into one triangle mesh (trimesh), texture colours are sampled to
          vertex colours (grey 180 when the model has none; column no_colour), vertices at the
          same position are merged, and degenerate, duplicate and unreferenced elements are
          dropped. The mesh is rejected unless its largest connected component holds at least
          --min_main of the faces (separate eyes, clothes or legs are fine; scattered pieces are
          not). The components with less than --min_piece of the faces (floating specks) are
          removed, the mesh is centred and scaled so its bounding box fits the unit cube
          [-0.5, 0.5]^3, and it must then have --face_range faces and a shortest bounding-box
          side of at least --min_aspect of the longest. It is saved as full/<uid>.ply and cut:
          --views camera directions in the front hemisphere (+z is taken as the front and +y as
          up, as in glTF; yaw uniform in --yaw, elevation uniform in --elev, drawn from a
          generator seeded by --seed and the uid), each cut with make_partial_mesh.py (--scale 1
          --shift 0 0 0, so the partial differs from the full mesh only by the camera rotation)
          into partial/<uid>_v<k>.ply and _gt.npz. Partials with fewer than --min_partial
          vertices are dropped, and an object is kept only when at least one partial is left.
          Writes objects.csv, rejected.csv (every rejection with its reason), manifest.json (one
          entry per pair) and split.json, a validation split of --val_share of each group's
          objects (both views of an object go to the same split; also a column of objects.csv).
          Stops if the downloaded GLBs exceed --max_gb.

Usage:
    python scripts/idarb/make_partial_dataset.py --step select
    python scripts/idarb/make_partial_dataset.py --step meshes
"""

import argparse
import csv
import gzip
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np

HF = "https://huggingface.co/datasets/allenai/objaverse/resolve/main"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# group: (quota, {category: [LVIS categories]})
GROUPS = {
    "animal": (200, {
        "cat": ["cat", "kitten"],
        "dog": ["dog", "bulldog", "pug-dog", "shepherd_dog", "puppy"],
        "horse": ["horse", "pony"],
        "cow": ["cow", "horned_cow"],
        "sheep": ["sheep", "lamb_(animal)"],
        "pig": ["hog"],
        "deer": ["deer"],
        "lion": ["lion"],
        "tiger": ["tiger"],
        "bear": ["bear", "polar_bear"],
        "elephant": ["elephant"],
        "giraffe": ["giraffe"],
        "camel": ["camel"],
        "bird": ["bird"],
        "duck": ["duck"],
        "fish": ["fish"],
        "teddy_bear": ["teddy_bear"],
    }),
    "person": (100, {
        "person": ["person"],
        "doll": ["doll", "rag_doll"],
        "statue": ["statue_(sculpture)"],
        "figurine": ["figurine"],
        "puppet": ["puppet"],
        "scarecrow": ["scarecrow"],
    }),
    "object": (100, {
        "chair": ["chair"],
        "armchair": ["armchair"],
        "sofa": ["sofa"],
        "table": ["table", "dining_table", "coffee_table"],
        "lamp": ["lamp", "table_lamp"],
        "vase": ["vase"],
        "car": ["car_(automobile)"],
        "airplane": ["airplane"],
    }),
}
LICENSES = {"by": "CC BY 4.0", "cc0": "CC0 1.0"}


def curl(url, path):
    """Download url to path (through a .part file), unless path exists."""
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    subprocess.run(["curl", "-sSLf", "--retry", "3", "-o", path + ".part", url], check=True)
    os.replace(path + ".part", path)


def read_gz_json(path):
    with gzip.open(path, "rt") as f:
        return json.load(f)


def step_select(args):
    cache = os.path.join(args.root, "objaverse_cache")
    curl(HF + "/lvis-annotations.json.gz", os.path.join(cache, "lvis-annotations.json.gz"))
    curl(HF + "/object-paths.json.gz", os.path.join(cache, "object-paths.json.gz"))
    lvis = read_gz_json(os.path.join(cache, "lvis-annotations.json.gz"))
    paths = read_gz_json(os.path.join(cache, "object-paths.json.gz"))

    rows, seen = [], set()
    for group, (_, cats) in GROUPS.items():
        for cat, lvis_names in cats.items():
            for ln in lvis_names:
                for uid in lvis[ln]:
                    if uid not in seen:  # an object in two categories keeps the first
                        seen.add(uid)
                        rows.append(dict(uid=uid, group=group, category=cat, lvis=ln))
    chunks = sorted({paths[r["uid"]].split("/")[1] for r in rows})
    print("objects in the chosen categories:", len(rows), "metadata chunks:", len(chunks))
    for i, c in enumerate(chunks):
        curl(f"{HF}/metadata/{c}.json.gz", os.path.join(cache, "metadata", c + ".json.gz"))
        if (i + 1) % 20 == 0:
            print("metadata", i + 1, "/", len(chunks), flush=True)
    meta = {}
    for c in chunks:
        meta.update(read_gz_json(os.path.join(cache, "metadata", c + ".json.gz")))

    lo, hi = args.face_range[0] / 1.25, args.face_range[1] * 1.25
    counts = dict(in_categories=len(rows), with_metadata=0, license_ok=0, faces_ok=0)
    licenses = {}
    keep = []
    for r in rows:
        m = meta.get(r["uid"])
        if m is None:
            continue
        counts["with_metadata"] += 1
        lic = m.get("license")
        licenses[lic] = licenses.get(lic, 0) + 1
        if lic not in LICENSES:
            continue
        counts["license_ok"] += 1
        faces = int(m.get("faceCount") or 0)
        if not lo <= faces <= hi:
            continue
        counts["faces_ok"] += 1
        glb = (m.get("archives") or {}).get("glb") or {}
        r.update(name=m.get("name", ""), license=LICENSES[lic],
                 author=(m.get("user") or {}).get("username", ""),
                 url=m.get("viewerUrl", ""), sketchfab_faces=faces,
                 sketchfab_vertices=int(m.get("vertexCount") or 0),
                 glb_bytes=int(glb.get("size") or 0), glb_path=paths[r["uid"]])
        keep.append(r)
    keep.sort(key=lambda r: hashlib.sha1(f"{args.seed}:{r['uid']}".encode()).hexdigest())
    out = os.path.join(args.root, "candidates.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(keep[0]))
        w.writeheader()
        w.writerows(keep)
    per_cat = {}
    for r in keep:
        k = r["group"] + "/" + r["category"]
        per_cat[k] = per_cat.get(k, 0) + 1
    report = dict(counts=counts, licenses=licenses, candidates_per_category=per_cat,
                  candidate_glb_gb=round(sum(r["glb_bytes"] for r in keep) / 1e9, 2))
    with open(os.path.join(args.root, "select_report.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1))
    print("saved", out)


def vertex_colors(g):
    """Per-vertex RGB(A) of one glTF primitive: the base-colour texture sampled at the vertex UVs
    (times the base-colour factor), else the factor alone, else stored vertex colours, else grey."""
    import trimesh

    n = len(g.vertices)
    vis = g.visual
    if getattr(vis, "kind", None) == "vertex":
        return np.asarray(vis.vertex_colors)
    mat = getattr(vis, "material", None)
    factor = getattr(mat, "baseColorFactor", None)
    factor = np.ones(4) if factor is None else np.asarray(factor, dtype=np.float64)
    if factor.max() > 1.0:  # trimesh may store the factor as 0-255
        factor = factor / 255.0
    tex = getattr(mat, "baseColorTexture", None)
    uv = getattr(vis, "uv", None)
    if tex is not None and uv is not None and len(uv) == n:
        col = trimesh.visual.uv_to_color(uv, tex).astype(np.float64)
        return np.clip(col * factor, 0, 255).astype(np.uint8)
    stored = getattr(vis, "vertex_attributes", {}).get("color")
    if stored is not None and len(stored) == n:
        stored = np.asarray(stored, dtype=np.float64)
        return (stored * 255 if stored.max() <= 1.0 else stored).astype(np.uint8)
    if getattr(mat, "baseColorFactor", None) is not None:
        return np.tile(np.round(factor * 255), (n, 1)).astype(np.uint8)
    return np.full((n, 4), 180, dtype=np.uint8)


def clean_mesh(glb_path):
    """One triangle mesh from a GLB, with vertex colours, merged and cleaned."""
    import trimesh

    scene = trimesh.load(glb_path, force="scene")
    parts = []
    for name in scene.graph.nodes_geometry:
        tf, geom_name = scene.graph[name]
        g = scene.geometry[geom_name]
        if not isinstance(g, trimesh.Trimesh) or len(g.faces) == 0:
            continue
        g = g.copy()
        colors = vertex_colors(g)
        g.apply_transform(tf)
        parts.append((np.asarray(g.vertices), np.asarray(g.faces), np.asarray(colors)[:, :3]))
    if not parts:
        raise ValueError("no triangle geometry")
    off = np.cumsum([0] + [len(p[0]) for p in parts])
    verts = np.vstack([p[0] for p in parts])
    faces = np.vstack([p[1] + o for p, o in zip(parts, off)])
    colors = np.vstack([p[2] for p in parts]).astype(np.float64)

    # merge vertices at the same position (texture seams split them), averaging their colours
    lo, hi = verts.min(0), verts.max(0)
    tol = 1e-6 * float(np.linalg.norm(hi - lo))
    key = np.round((verts - lo) / max(tol, 1e-12)).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    n = int(inv.max()) + 1
    cnt = np.bincount(inv, minlength=n).astype(np.float64)
    new_verts = np.stack([np.bincount(inv, verts[:, k], n) for k in range(3)], 1) / cnt[:, None]
    new_cols = np.stack([np.bincount(inv, colors[:, k], n) for k in range(3)], 1) / cnt[:, None]
    faces = inv[faces]

    # drop degenerate (repeated index or zero area) and duplicate faces, then unused vertices
    f = faces[(faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2])
              & (faces[:, 0] != faces[:, 2])]
    tri = new_verts[f]
    area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    f = f[area > 1e-12 * float(np.linalg.norm(hi - lo)) ** 2]
    _, first = np.unique(np.sort(f, axis=1), axis=0, return_index=True)
    f = f[np.sort(first)]
    used = np.unique(f)
    remap = -np.ones(n, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return new_verts[used], remap[f], np.clip(np.round(new_cols[used]), 0, 255).astype(np.uint8)


def components(n_verts, faces):
    """Connected-component id of every face (faces linked by shared vertices) and the number of
    faces in each component."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    e = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]]])
    g = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n_verts, n_verts))
    _, lab = connected_components(g, directed=False)
    face_lab = lab[faces[:, 0]]
    return face_lab, np.bincount(face_lab)


def drop_small_pieces(v, f, c, min_piece):
    """Remove the components with fewer than min_piece of the faces, and the unused vertices.
    Returns v, f, c and the share of the faces in the largest component (before removal)."""
    face_lab, sizes = components(len(v), f)
    share = float(sizes.max() / len(f))
    f = f[sizes[face_lab] >= min_piece * len(f)]
    used = np.unique(f)
    remap = -np.ones(len(v), dtype=np.int64)
    remap[used] = np.arange(len(used))
    return v[used], remap[f], c[used], share


def uid_rng(seed, uid):
    """A random generator that depends only on the seed and the object, not on the order."""
    return np.random.default_rng(int(hashlib.sha1(f"{seed}:{uid}".encode()).hexdigest()[:12], 16))


def cut_partials(uid, ply, args):
    """Cut --views front-hemisphere partials of one full mesh with make_partial_mesh.py; returns
    the ones with at least --min_partial vertices (the others are deleted) and the number cut."""
    rng = uid_rng(args.seed, uid)
    script = os.path.join(ROOT, "scripts", "idarb", "make_partial_mesh.py")
    out_list = []
    for k in range(args.views):
        yaw = np.deg2rad(rng.uniform(*args.yaw))
        elev = np.deg2rad(rng.uniform(*args.elev))
        view = [float(np.cos(elev) * np.sin(yaw)), float(np.sin(elev)),
                float(np.cos(elev) * np.cos(yaw))]
        out = os.path.join(args.root, "partial", f"{uid}_v{k}")
        cmd = [sys.executable, script, "--mesh", ply, "--out", out, "--view"]
        cmd += [f"{x:.6f}" for x in view] + ["--scale", "1", "--shift", "0", "0", "0"]
        res = subprocess.run(cmd, check=True, capture_output=True, text=True)
        info = json.loads(res.stdout.splitlines()[0])
        if info["n_partial_vertices"] < args.min_partial:
            for ext in (".ply", "_gt.npz"):
                os.remove(out + ext)
            continue
        out_list.append(dict(view=k, view_vector=[round(x, 6) for x in view],
                             yaw_deg=round(float(np.rad2deg(yaw)), 3),
                             elev_deg=round(float(np.rad2deg(elev)), 3),
                             n_partial_vertices=info["n_partial_vertices"],
                             n_partial_faces=info["n_partial_faces"],
                             partial=out + ".ply", gt=out + "_gt.npz"))
    return out_list


def process_object(r, args):
    """Download, clean, filter and cut one candidate. Returns (object row, partials) or raises
    Rejected with the reason."""
    import trimesh

    glb = os.path.join(args.root, "glb", r["uid"] + ".glb")
    curl(f"{HF}/{r['glb_path']}", glb)
    v, fc, col = clean_mesh(glb)
    n_faces_cleaned = len(fc)
    v, fc, col, share = drop_small_pieces(v, fc, col, args.min_piece)
    lo, hi = v.min(0), v.max(0)  # normalise to the unit cube
    v = (v - (lo + hi) / 2) / float((hi - lo).max())
    ext = np.sort(hi - lo)
    aspect = float(ext[0] / ext[2])
    if share < args.min_main:
        raise Rejected(f"main component {share:.3f}")
    if not args.face_range[0] <= len(fc) <= args.face_range[1]:
        raise Rejected(f"faces {len(fc)}")
    if aspect < args.min_aspect:
        raise Rejected(f"aspect {aspect:.3f}")
    ply = os.path.join(args.root, "full", r["uid"] + ".ply")
    trimesh.Trimesh(vertices=v, faces=fc, vertex_colors=col, process=False).export(ply)
    partials = cut_partials(r["uid"], ply, args)
    if not partials:
        os.remove(ply)
        raise Rejected(f"no partial with {args.min_partial} vertices ({len(v)} in the full mesh)")
    row = dict(r, n_vertices=len(v), n_faces_cleaned=n_faces_cleaned, n_faces=len(fc),
               main_component=round(share, 4), aspect=round(aspect, 4),
               no_colour=bool((col == 180).all()), n_partials=len(partials),
               glb_mb=round(os.path.getsize(glb) / 1e6, 2), ply=ply)
    return row, partials


class Rejected(Exception):
    pass


def step_meshes(args):
    with open(os.path.join(args.root, "candidates.csv")) as f:
        cands = list(csv.DictReader(f))
    for d in ("glb", "full", "partial"):
        os.makedirs(os.path.join(args.root, d), exist_ok=True)
    kept, rejected, manifest = [], [], []
    downloaded = 0
    for group, (quota, cats) in GROUPS.items():
        queues = {c: [r for r in cands if r["group"] == group and r["category"] == c] for c in cats}
        n_cat = {c: 0 for c in cats}
        while sum(n_cat.values()) < quota and downloaded <= args.max_gb * 1e9:
            open_cats = [c for c in cats if queues[c]]
            if not open_cats:
                break
            # balance the kept counts: only the categories with the fewest kept objects try next
            low = min(n_cat[c] for c in open_cats)
            for c in [c for c in open_cats if n_cat[c] == low]:
                if sum(n_cat.values()) >= quota:
                    break
                r = queues[c].pop(0)
                t0 = time.time()
                try:
                    row, partials = process_object(r, args)
                    reason = None
                except Rejected as e:
                    reason = str(e)
                except Exception as e:  # broken files: keep going, record why
                    reason = f"error {type(e).__name__}: {str(e)[:80]}"
                glb = os.path.join(args.root, "glb", r["uid"] + ".glb")
                if os.path.exists(glb):
                    downloaded += os.path.getsize(glb)
                if reason is not None:
                    rejected.append(dict(uid=r["uid"], group=group, category=c, reason=reason))
                else:
                    kept.append(row)
                    n_cat[c] += 1
                    for p in partials:
                        manifest.append(dict(uid=r["uid"], group=group, category=c, **p,
                                             n_full_vertices=row["n_vertices"],
                                             n_full_faces=row["n_faces"], full=row["ply"]))
                print(group, c, r["uid"], "kept" if reason is None else reason,
                      f"{time.time() - t0:.1f}s", f"downloaded {downloaded / 1e9:.2f} GB",
                      flush=True)
        if downloaded > args.max_gb * 1e9:
            print("stopping: downloads above --max_gb")
        print(group, "kept", sum(n_cat.values()), "of quota", quota, n_cat, flush=True)

    # validation split: --val_share of each group's objects, both views in the same split
    val = []
    for group in GROUPS:
        uids = sorted(o["uid"] for o in kept if o["group"] == group)
        uids.sort(key=lambda u: hashlib.sha1(f"val{args.seed}:{u}".encode()).hexdigest())
        val += uids[:int(round(args.val_share * len(uids)))]
    val = set(val)
    for o in kept:
        o["split"] = "val" if o["uid"] in val else "train"
    for m in manifest:
        m["split"] = "val" if m["uid"] in val else "train"
    for name, table in (("objects.csv", kept), ("rejected.csv", rejected)):
        if table:
            with open(os.path.join(args.root, name), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(table[0]))
                w.writeheader()
                w.writerows(table)
    command = " ".join([os.path.relpath(__file__, ROOT)] + sys.argv[1:])
    with open(os.path.join(args.root, "manifest.json"), "w") as f:
        json.dump(dict(command=command, git_commit=git_commit(), pairs=manifest), f, indent=1)
    with open(os.path.join(args.root, "split.json"), "w") as f:
        json.dump(dict(val=sorted(val), train=sorted(o["uid"] for o in kept if o["uid"] not in val),
                       val_share=args.val_share, seed=args.seed), f, indent=1)
    print("kept", len(kept), "rejected", len(rejected), "pairs", len(manifest), "val objects",
          len(val), f"downloaded {downloaded / 1e9:.2f} GB")


def git_commit():
    try:
        return subprocess.check_output(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step", required=True, choices=["select", "meshes"])
    ap.add_argument("--root", default=os.path.expanduser("~/semtex/dataset"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--face_range", type=int, nargs=2, default=(2000, 200000))
    ap.add_argument("--min_main", type=float, default=0.5)
    ap.add_argument("--min_piece", type=float, default=0.01)
    ap.add_argument("--min_aspect", type=float, default=0.1)
    ap.add_argument("--max_gb", type=float, default=15.0)
    ap.add_argument("--views", type=int, default=2)
    ap.add_argument("--yaw", type=float, nargs=2, default=(-60.0, 60.0))
    ap.add_argument("--elev", type=float, nargs=2, default=(-10.0, 30.0))
    ap.add_argument("--min_partial", type=int, default=1500)
    ap.add_argument("--val_share", type=float, default=0.1)
    args = ap.parse_args()
    dict(select=step_select, meshes=step_meshes)[args.step](args)


if __name__ == "__main__":
    main()
