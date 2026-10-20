"""
Functional maps (pyFM, as vendored in third_party/Diff3F) with Diff3F descriptors between a
partial source mesh and a full target mesh, compared with feature-only matching and with the
registration of register_partial.py.

A functional map is a smooth, global correspondence computed in the spectral basis of each mesh
(the first n_ev eigenfunctions of the Laplace-Beltrami operator): descriptors are projected on the
bases, a small matrix C mapping one basis to the other is fitted, and the point-to-point map comes
from nearest neighbours in the spectral embedding. Diff3F uses it on full shapes. Here the source is
one-sided, so the target is optionally cut to the part the registration says the source covers
("crop"), which makes the two shapes more alike.

For every target vertex (that the source covers) the map gives a source vertex; the ground truth of
the synthetic pairs scores it (fraction within 1 % / 5 % of the diagonal of the true vertex). Writes
results/diff3f/functional_map_eval.csv.

Needs potpourri3d and robust_laplacian (pip) in the mesh environment.

Usage (repository root):
    python scripts/idarb/eval_functional_map.py --pairs cow_synth cat_synth --n_ev 50 100
"""

import argparse
import csv
import os
import sys
import time

import numpy as np
import trimesh
from scipy.spatial import cKDTree

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "third_party", "Diff3F"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pyFM.functional import FunctionalMapping  # noqa: E402
from pyFM.mesh import TriMesh  # noqa: E402

import register_partial as rp  # noqa: E402

PAIRS = dict(
    cow_synth=(
        "results/diff3f/synthetic/cow_synth.ply",
        "cow_synth_partial.npz",
        "third_party/Diff3F/meshes/cow.obj",
        "cow_full.npz",
        "results/diff3f/synthetic/cow_synth_gt.npz",
    ),
    cat_synth=(
        "results/diff3f/synthetic/cat_synth.ply",
        "cat_synth_partial.npz",
        "third_party/Diff3F/meshes/cat.off",
        "cat_full.npz",
        "results/diff3f/synthetic/cat_synth_gt.npz",
    ),
    cow=(
        "outputs/mesh_depth/cow/relief.ply",
        "cow_partial.npz",
        "third_party/Diff3F/meshes/cow.obj",
        "cow_full.npz",
        None,
    ),
    cat=(
        "outputs/mesh_depth/cat/relief.ply",
        "cat_partial.npz",
        "third_party/Diff3F/meshes/cat.off",
        "cat_full.npz",
        None,
    ),
)

FIELDS = [
    "pair", "crop", "n_ev", "map", "seconds", "part_agreement", "dist_to_mapped_median",
    "gt_err_median", "gt_acc_1pct", "gt_acc_5pct",
]  # fmt: skip


def functional_map(Vs, Fs, fs, Vd, Fd, fd, n_ev):
    """Point-to-point map from every vertex of (Vd, Fd) to a vertex of (Vs, Fs), before and after
    ZoomOut refinement."""
    mesh1 = TriMesh(Vs, Fs)
    mesh2 = TriMesh(Vd, Fd)
    model = FunctionalMapping(mesh1, mesh2)
    model.preprocess(
        n_ev=(n_ev, n_ev),
        n_descr=fs.shape[1],
        landmarks=None,
        descr1=fs,
        descr2=fd,
        subsample_step=0,
        verbose=False,
    )
    model.fit(w_descr=1e0, w_lap=1e-2, w_dcomm=1e-1, w_orient=0, verbose=False)
    out = {"fm": model.get_p2p(n_jobs=1)}
    model.zoomout_refine(nit=10, step=1, verbose=False)
    out["fm+zoomout"] = model.get_p2p(n_jobs=1)
    return out


def score(ctx, name, m, crop, n_ev, seconds):
    """One row: part agreement and distance of the mapped source vertex for the covered target
    vertices, plus the ground-truth error when the pair has one."""
    Vd, diag, covered = ctx["Vd"], ctx["diag"], ctx["covered"]
    d = np.linalg.norm(ctx["aligned"][m] - Vd, axis=1) / diag
    row = dict(
        pair=ctx["pair"],
        crop=crop,
        n_ev=n_ev,
        map=name,
        seconds=seconds,
        part_agreement=round(
            float((ctx["src_parts"][m[covered]] == ctx["dst_parts"][covered]).mean()), 3
        ),
        dist_to_mapped_median=round(float(np.median(d[covered])), 4),
    )
    if ctx["gt"] is not None:
        orig = ctx["gt"]["orig_idx"]
        has = np.zeros(len(Vd), dtype=bool)
        has[orig] = True
        err = np.linalg.norm(Vd[orig[m]] - Vd, axis=1) / diag
        row.update(
            gt_err_median=round(float(np.median(err[has])), 4),
            gt_acc_1pct=round(float((err[has] < 0.01).mean()), 3),
            gt_acc_5pct=round(float((err[has] < 0.05).mean()), 3),
        )
    print(row)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", default=["cow_synth", "cat_synth"])
    ap.add_argument("--n_ev", type=int, nargs="+", default=[50, 100])
    ap.add_argument("--feature_dir", default="outputs/diff3f")
    ap.add_argument("--out", default="results/diff3f/functional_map_eval.csv")
    args = ap.parse_args()

    rows = []
    for pair in args.pairs:
        src_mesh, src_npz, dst_mesh, dst_npz, gt_file = PAIRS[pair]
        src = rp.load_features(os.path.join(args.feature_dir, src_npz))
        dst = rp.load_features(os.path.join(args.feature_dir, dst_npz))
        ms = trimesh.load(src_mesh, process=False, force="mesh")
        md = trimesh.load(dst_mesh, process=False, force="mesh")
        Vs, Fs = np.asarray(ms.vertices, float), np.asarray(ms.faces, int)
        Vd, Fd = np.asarray(md.vertices, float), np.asarray(md.faces, int)
        # descriptors on every vertex: the nearest sampled point's feature
        fs = src["feat"][cKDTree(src["xyz"]).query(Vs)[1]].astype(np.float64)
        fd = dst["feat"][cKDTree(dst["xyz"]).query(Vd)[1]].astype(np.float64)
        gt = dict(np.load(gt_file)) if gt_file else None

        res = rp.register(src, dst, gt)
        s, R, t, a = res["fine"]
        aligned = rp.transform(rp.apply_depth(Vs, a), s, R, t)
        d_reg, j_reg = cKDTree(aligned).query(Vd)
        diag = float(np.linalg.norm(Vd.max(0) - Vd.min(0)))
        covered = d_reg < 0.05 * diag
        src_parts, dst_parts = rp.part_labels(fs.astype(np.float32), fd.astype(np.float32), 6)
        sim_nn = np.empty(len(Vd), dtype=np.int64)
        for i in range(0, len(Vd), 2048):
            sim_nn[i : i + 2048] = (fd[i : i + 2048] @ fs.T).argmax(axis=1)

        ctx = dict(
            pair=pair,
            aligned=aligned,
            Vd=Vd,
            diag=diag,
            covered=covered,
            src_parts=src_parts,
            dst_parts=dst_parts,
            gt=gt,
        )
        rows.append(score(ctx, "feature_nn", sim_nn, "", "", ""))
        rows.append(score(ctx, "registration_nn", j_reg, "", "", ""))
        for crop in (False, True):
            if crop:
                keep_f = covered[Fd].all(axis=1)
                used = np.unique(Fd[keep_f])
                new_index = -np.ones(len(Vd), dtype=np.int64)
                new_index[used] = np.arange(len(used))
                Vd2, Fd2, fd2 = Vd[used], new_index[Fd[keep_f]], fd[used]
            else:
                Vd2, Fd2, fd2, used = Vd, Fd, fd, np.arange(len(Vd))
            for n_ev in args.n_ev:
                t0 = time.time()
                try:
                    maps = functional_map(Vs, Fs, fs, Vd2, Fd2, fd2, n_ev)
                except Exception as e:  # noqa: BLE001
                    print(f"{pair} crop={crop} n_ev={n_ev}: failed ({e})")
                    continue
                seconds = round(time.time() - t0, 1)
                for name, m2 in maps.items():
                    full = j_reg.copy()  # outside the crop: the registration's nearest point
                    full[used] = m2
                    rows.append(score(ctx, name, full, crop, n_ev, seconds))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print("saved", args.out)


if __name__ == "__main__":
    main()
