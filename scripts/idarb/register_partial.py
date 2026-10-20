"""
Register a partial mesh to a full mesh with Diff3F semantic features. CPU only (NumPy, SciPy,
trimesh); runs on the laptop or the server.

Input: two .npz files written by diff3f_features.py, the "source" (our partial relief mesh from
normal_to_mesh.py) and the "target" (a full mesh of the same kind of object). Registration means
finding the scale, rotation and translation that put the source onto the target.

Steps:
  1. match: cosine similarity between every source vector and every target vector; keep the pairs
     that are each other's best match (mutual nearest neighbours). Semantics, not shape, decides
     these matches, so they survive a different pose, style or coordinate system.
  2. coarse: RANSAC over the matches. Each hypothesis takes 3 matches, fits a similarity transform
     (scale, rotation, translation; Umeyama 1991) and is scored by how many matches it brings
     within --inlier_dist of each other. The best hypothesis is refitted on its inliers.
  3. fine: rigid ICP with the scale fixed, between all source points and the target points,
     trimmed to the source points that have a target point within --icp_dist.
  4. baseline: the same ICP started from centroid-and-extent alignment only, without features,
     to show what the features add.

Distances are given as fractions of the target's bounding-box diagonal. Outputs in --out_dir:
metrics.csv, transform.json (scale, rotation, translation and the 4x4 matrix), aligned_source.ply
(the full source mesh moved onto the target, if --source_mesh is given, else the source points),
target.ply, correspondence_source.ply / correspondence_target.ply (matched points share a colour)
and target_from_source_colors.ply (target points coloured with the source's vertex colours through
the feature matches: the albedo transfer the project aims at), plus preview.png when matplotlib is
installed.

Usage (repository root, mesh environment):
    python scripts/idarb/register_partial.py --source outputs/diff3f/cow_partial.npz \
        --target outputs/diff3f/cow_full.npz --source_mesh outputs/mesh_depth/cow/relief.ply \
        --target_mesh third_party/Diff3F/meshes/cow.obj --out_dir outputs/diff3f/cow_registration
"""

import argparse
import csv
import json
import os

import numpy as np
from scipy.spatial import cKDTree

# ------------------------------------------------------------------------------
# features and matches
# ------------------------------------------------------------------------------


def load_features(path):
    """xyz (N,3 float64), feat (N,D float32 unit length), colors (N,3 uint8 or None), meta dict."""
    z = np.load(path, allow_pickle=False)
    feat = z["feat"].astype(np.float32)
    feat /= np.maximum(np.linalg.norm(feat, axis=1, keepdims=True), 1e-8)
    colors = z["colors"] if len(z["colors"]) else None
    meta = json.loads(str(z["meta"])) if "meta" in z else {}
    return z["xyz"].astype(np.float64), feat, colors, meta


def nearest_in_feature_space(fa, fb, chunk=2048):
    """For every row of fa: the index of the most similar row of fb and that cosine similarity."""
    best = np.empty(len(fa), dtype=np.int64)
    best_sim = np.empty(len(fa), dtype=np.float32)
    for i in range(0, len(fa), chunk):
        s = fa[i : i + chunk] @ fb.T
        j = s.argmax(axis=1)
        best[i : i + chunk] = j
        best_sim[i : i + chunk] = s[np.arange(len(j)), j]
    return best, best_sim


def mutual_matches(fs, ft):
    """Pairs (i, k) where target k is source i's best match and source i is target k's best match.
    Returns source indices, target indices and the similarities."""
    s2t, sim = nearest_in_feature_space(fs, ft)
    t2s, _ = nearest_in_feature_space(ft, fs)
    i = np.nonzero(t2s[s2t] == np.arange(len(fs)))[0]
    return i, s2t[i], sim[i]


def collect_matches(fs, ft, mode="both", keep_frac=0.5):
    """Candidate matches for RANSAC. mode "mutual": only mutual nearest neighbours (few, clean).
    mode "both": every source point with its best target point plus every target point with its
    best source point, keeping the most similar keep_frac of them (many, with outliers that RANSAC
    has to reject). Returns source indices, target indices and similarities."""
    if mode == "mutual":
        return mutual_matches(fs, ft)
    s2t, sim_s = nearest_in_feature_space(fs, ft)
    t2s, sim_t = nearest_in_feature_space(ft, fs)
    i = np.concatenate([np.arange(len(fs)), t2s])
    k = np.concatenate([s2t, np.arange(len(ft))])
    sim = np.concatenate([sim_s, sim_t])
    _, first = np.unique(np.stack([i, k], axis=1), axis=0, return_index=True)
    i, k, sim = i[first], k[first], sim[first]
    keep = sim >= np.quantile(sim, 1.0 - keep_frac)
    return i[keep], k[keep], sim[keep]


def feature_agreement(src_xyz, src_feat, dst_tree, dst_feat, s, R, t, max_dist):
    """How well a transform agrees with the features: the mean cosine similarity between each
    source point's feature and the feature of the target point it lands next to. Source points
    farther than max_dist from the target count as 0. A transform that puts the body on the head
    scores low here even if it brings many feature matches close together."""
    p = transform(src_xyz, s, R, t)
    d, j = dst_tree.query(p)
    a = np.einsum("ij,ij->i", src_feat, dst_feat[j])
    a[d > max_dist] = 0.0
    return float(a.mean())


# ------------------------------------------------------------------------------
# similarity transforms
# ------------------------------------------------------------------------------


def transform(points, s, R, t):
    """s * R @ p + t for every row p."""
    return s * (points @ R.T) + t


def to_matrix(s, R, t):
    M = np.eye(4)
    M[:3, :3] = s * R
    M[:3, 3] = t
    return M


def rotation_angle_deg(R1, R2):
    """Angle of the rotation that takes R1 to R2."""
    c = (np.trace(R1.T @ R2) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def umeyama(src, dst, with_scale=True):
    """Least-squares similarity transform dst ~ s * R @ src + t (Umeyama 1991).
    Returns s, R (3x3, det +1), t."""
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    xs = src - mu_s
    xd = dst - mu_d
    cov = xd.T @ xs / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    if with_scale:
        var_s = (xs**2).sum() / len(src)
        s = float((D * np.diag(S)).sum() / var_s)
    else:
        s = 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def ransac_candidates(
    src,
    dst,
    inlier_dist,
    n_iter=5000,
    seed=0,
    scale_range=(0.1, 10.0),
    n_keep=8,
    min_angle_deg=15.0,
):
    """RANSAC over matched points with outliers (src[i] is supposed to correspond to dst[i]).
    Every hypothesis is a similarity transform from 3 matches, scored by the number of matches it
    brings within inlier_dist. Returns up to n_keep distinct hypotheses as (s, R, t, inlier mask),
    best first, each refitted on its inliers. Hypotheses whose rotations differ by less than
    min_angle_deg and whose scales differ by less than 20 % count as the same."""
    rng = np.random.default_rng(seed)
    n = len(src)
    if n < 3:
        raise ValueError("need at least 3 matches")
    hyps = []
    for _ in range(n_iter):
        k = rng.choice(n, 3, replace=False)
        a = src[k]
        # skip nearly collinear triples: they do not fix a rotation
        if np.linalg.norm(np.cross(a[1] - a[0], a[2] - a[0])) < 1e-12:
            continue
        s, R, t = umeyama(a, dst[k])
        if not (scale_range[0] <= s <= scale_range[1]):
            continue
        d = np.linalg.norm(transform(src, s, R, t) - dst, axis=1)
        count = int((d < inlier_dist).sum())
        if count >= 3:
            hyps.append((count, s, R, t))
    hyps.sort(key=lambda h: -h[0])
    kept = []
    for count, s, R, t in hyps:
        same = any(
            rotation_angle_deg(R, R2) < min_angle_deg and abs(np.log(s / s2)) < np.log(1.2)
            for _, s2, R2, _ in kept
        )
        if same:
            continue
        kept.append((count, s, R, t))
        if len(kept) == n_keep:
            break
    out = []
    for _, s, R, t in kept:
        d = np.linalg.norm(transform(src, s, R, t) - dst, axis=1)
        inliers = d < inlier_dist
        for _ in range(3):  # refit on the inliers, then recount
            if inliers.sum() < 3:
                break
            s, R, t = umeyama(src[inliers], dst[inliers])
            d = np.linalg.norm(transform(src, s, R, t) - dst, axis=1)
            inliers = d < inlier_dist
        out.append((s, R, t, inliers))
    return out


def ransac_similarity(src, dst, inlier_dist, n_iter=5000, seed=0, scale_range=(0.1, 10.0)):
    """The best RANSAC hypothesis only: s, R, t and the boolean inlier mask."""
    cands = ransac_candidates(src, dst, inlier_dist, n_iter, seed, scale_range, n_keep=1)
    if not cands:
        raise RuntimeError("RANSAC found no transform with 3 inliers")
    return cands[0]


def icp_rigid(src, dst, s, R, t, max_dist, iters=100, tol=1e-7):
    """Point-to-point ICP with the scale fixed. Every iteration pairs each transformed source point
    with its nearest target point, drops pairs farther apart than max_dist and refits R, t.
    Returns R, t, the RMS distance of the kept pairs and the fraction of source points kept."""
    tree = cKDTree(dst)
    rmse_prev = np.inf
    rmse = np.inf
    kept = 0.0
    for _ in range(iters):
        p = transform(src, s, R, t)
        d, j = tree.query(p)
        keep = d < max_dist
        if keep.sum() < 10:
            break
        rmse = float(np.sqrt(np.mean(d[keep] ** 2)))
        kept = float(keep.mean())
        _, R, t = umeyama(s * src[keep], dst[j[keep]], with_scale=False)
        if abs(rmse_prev - rmse) < tol:
            break
        rmse_prev = rmse
    return R, t, rmse, kept


# ------------------------------------------------------------------------------
# evaluation
# ------------------------------------------------------------------------------


def surface_distances(points, target_tree):
    d, _ = target_tree.query(points)
    return d


def stage_metrics(name, src, dst_tree, s, R, t, diag, matches=None, close=0.02):
    """Distances of the transformed source points to the nearest target point, as fractions of the
    target diagonal; and, if matches (i, k, dst) are given, the distance between matched points."""
    p = transform(src, s, R, t)
    d = surface_distances(p, dst_tree) / diag
    row = dict(
        stage=name,
        scale=round(float(s), 4),
        rms_to_target=round(float(np.sqrt(np.mean(d**2))), 4),
        median_to_target=round(float(np.median(d)), 4),
        frac_within_2pct=round(float((d < close).mean()), 3),
    )
    if matches is not None:
        i, k, dst = matches
        dm = np.linalg.norm(p[i] - dst[k], axis=1) / diag
        row["match_median"] = round(float(np.median(dm)), 4)
        row["match_within_5pct"] = round(float((dm < 0.05).mean()), 3)
    return row


# ------------------------------------------------------------------------------
# outputs
# ------------------------------------------------------------------------------


def position_colors(points):
    """Diff3F's get_colors: position inside the bounding box as an RGB colour."""
    lo = points.min(axis=0, keepdims=True)
    hi = points.max(axis=0, keepdims=True)
    return (np.clip((points - lo) / np.maximum(hi - lo, 1e-9), 0, 1) * 255).astype(np.uint8)


def save_points(path, points, colors):
    import trimesh

    trimesh.PointCloud(points, colors=colors).export(path)


def save_mesh_transformed(path, mesh_path, s, R, t):
    import trimesh

    m = trimesh.load(mesh_path, process=False, force="mesh")
    m.vertices = transform(np.asarray(m.vertices, dtype=np.float64), s, R, t)
    m.export(path)


def save_preview(path, panels):
    """panels: list of (title, [(points, colors_or_None, size), ...])."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, no preview.png")
        return
    n = len(panels)
    fig = plt.figure(figsize=(4 * n, 4))
    for col, (title, clouds) in enumerate(panels):
        ax = fig.add_subplot(1, n, col + 1, projection="3d")
        allp = np.concatenate([c[0] for c in clouds])
        for pts, col_rgb, size in clouds:
            c = None if col_rgb is None else col_rgb / 255.0
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=size, c=c, depthshade=False, linewidths=0)
        center = (allp.min(0) + allp.max(0)) / 2
        r = (allp.max(0) - allp.min(0)).max() / 2
        ax.set_xlim(center[0] - r, center[0] + r)
        ax.set_ylim(center[1] - r, center[1] + r)
        ax.set_zlim(center[2] - r, center[2] + r)
        ax.set_title(title, fontsize=9)
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ------------------------------------------------------------------------------
# main
# ------------------------------------------------------------------------------


def register(
    src_xyz,
    src_feat,
    dst_xyz,
    dst_feat,
    inlier_dist=0.05,
    icp_dist=0.05,
    n_iter=5000,
    seed=0,
    match_mode="both",
    n_candidates=8,
):
    """The whole pipeline on arrays. Up to n_candidates distinct RANSAC hypotheses are each refined
    by ICP and scored by feature_agreement; the best score wins ("selected"). The scale is limited
    to 0.4 to 2.5 times the ratio of the bounding-box diagonals. Returns a dict with matches,
    transforms and metric rows (one per stage and candidate)."""
    diag = float(np.linalg.norm(dst_xyz.max(0) - dst_xyz.min(0)))
    src_diag = float(np.linalg.norm(src_xyz.max(0) - src_xyz.min(0)))
    dst_tree = cKDTree(dst_xyz)
    i, k, sim = collect_matches(src_feat, dst_feat, match_mode)
    matches = (i, k, dst_xyz)
    n_matches = int(len(i))

    ratio = diag / src_diag
    cands = ransac_candidates(
        src_xyz[i],
        dst_xyz[k],
        inlier_dist * diag,
        n_iter=n_iter,
        seed=seed,
        scale_range=(0.4 * ratio, 2.5 * ratio),
        n_keep=n_candidates,
    )
    if not cands:
        raise RuntimeError("RANSAC found no transform with 3 inliers")

    def agreement(s, R, t):
        return feature_agreement(src_xyz, src_feat, dst_tree, dst_feat, s, R, t, icp_dist * diag)

    rows = []
    s, R, t, inliers = cands[0]
    row = stage_metrics("ransac", src_xyz, dst_tree, s, R, t, diag, matches)
    row.update(
        candidate=0,
        n_matches=n_matches,
        n_inliers=int(inliers.sum()),
        agreement=round(agreement(s, R, t), 4),
    )
    rows.append(row)

    refined = []
    for c, (s, R, t, inliers) in enumerate(cands):
        R2, t2, _, kept = icp_rigid(src_xyz, dst_xyz, s, R, t, icp_dist * diag)
        a = agreement(s, R2, t2)
        refined.append((a, c, s, R, t, R2, t2, inliers, kept))
        row = stage_metrics(f"candidate{c}+icp", src_xyz, dst_tree, s, R2, t2, diag, matches)
        row.update(
            candidate=c,
            n_matches=n_matches,
            n_inliers=int(inliers.sum()),
            icp_kept=round(kept, 3),
            agreement=round(a, 4),
        )
        rows.append(row)

    a, c, s, R, t, R2, t2, inliers, kept = max(refined, key=lambda r: r[0])
    row = stage_metrics("selected", src_xyz, dst_tree, s, R2, t2, diag, matches)
    row.update(
        candidate=c,
        n_matches=n_matches,
        n_inliers=int(inliers.sum()),
        icp_kept=round(kept, 3),
        agreement=round(a, 4),
    )
    rows.append(row)

    # baseline without features: same size and centre, no rotation
    s0 = ratio
    R0 = np.eye(3)
    t0 = dst_xyz.mean(0) - s0 * src_xyz.mean(0)
    R1, t1, _, kept1 = icp_rigid(src_xyz, dst_xyz, s0, R0, t0, icp_dist * diag)
    row = stage_metrics("baseline_icp_no_features", src_xyz, dst_tree, s0, R1, t1, diag, matches)
    row.update(icp_kept=round(kept1, 3), agreement=round(agreement(s0, R1, t1), 4))
    rows.append(row)

    return dict(
        diag=diag,
        matches=(i, k, sim),
        inliers=inliers,
        selected=c,
        coarse=(s, R, t),
        fine=(s, R2, t2),
        baseline=(s0, R1, t1),
        rows=rows,
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--source", required=True, help="features of the partial mesh (.npz)")
    ap.add_argument("--target", required=True, help="features of the full mesh (.npz)")
    ap.add_argument("--source_mesh", help="the partial mesh file, to write it transformed")
    ap.add_argument("--target_mesh", help="the full mesh file, copied next to the outputs")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument(
        "--inlier_dist",
        type=float,
        default=0.05,
        help="RANSAC inlier distance, fraction of the target diagonal",
    )
    ap.add_argument(
        "--icp_dist",
        type=float,
        default=0.05,
        help="ICP pairing distance, fraction of the target diagonal",
    )
    ap.add_argument("--ransac_iters", type=int, default=5000)
    ap.add_argument(
        "--matches",
        choices=["both", "mutual"],
        default="both",
        help="both: best match of every point in both directions; mutual: mutual best only",
    )
    ap.add_argument(
        "--candidates", type=int, default=8, help="RANSAC hypotheses refined and scored"
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    src_xyz, src_feat, src_col, src_meta = load_features(args.source)
    dst_xyz, dst_feat, dst_col, dst_meta = load_features(args.target)
    print(f"source {len(src_xyz)} points, target {len(dst_xyz)} points")

    res = register(
        src_xyz,
        src_feat,
        dst_xyz,
        dst_feat,
        inlier_dist=args.inlier_dist,
        icp_dist=args.icp_dist,
        n_iter=args.ransac_iters,
        seed=args.seed,
        match_mode=args.matches,
        n_candidates=args.candidates,
    )
    i, k, sim = res["matches"]
    s, R, t = res["fine"]
    os.makedirs(args.out_dir, exist_ok=True)

    fields = [
        "stage",
        "candidate",
        "scale",
        "n_matches",
        "n_inliers",
        "icp_kept",
        "agreement",
        "rms_to_target",
        "median_to_target",
        "frac_within_2pct",
        "match_median",
        "match_within_5pct",
    ]
    with open(os.path.join(args.out_dir, "metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(res["rows"])
    for row in res["rows"]:
        print(row)

    with open(os.path.join(args.out_dir, "transform.json"), "w") as f:
        json.dump(
            dict(
                source=args.source,
                target=args.target,
                scale=s,
                rotation=R.tolist(),
                translation=t.tolist(),
                matrix=to_matrix(s, R, t).tolist(),
                coarse_matrix=to_matrix(*res["coarse"]).tolist(),
                selected_candidate=res["selected"],
                source_meta=src_meta,
                target_meta=dst_meta,
                inlier_dist=args.inlier_dist,
                icp_dist=args.icp_dist,
                matches=args.matches,
                candidates=args.candidates,
                seed=args.seed,
            ),
            f,
            indent=1,
        )

    # point clouds and meshes for MeshLab
    aligned = transform(src_xyz, s, R, t)
    if args.source_mesh:
        save_mesh_transformed(
            os.path.join(args.out_dir, "aligned_source.ply"), args.source_mesh, s, R, t
        )
    else:
        save_points(os.path.join(args.out_dir, "aligned_source.ply"), aligned, src_col)
    if args.target_mesh:
        import trimesh

        trimesh.load(args.target_mesh, process=False, force="mesh").export(
            os.path.join(args.out_dir, "target.ply")
        )
    else:
        save_points(os.path.join(args.out_dir, "target.ply"), dst_xyz, dst_col)

    # correspondence colours: source by position, target by its best-matching source point
    src_pc = position_colors(src_xyz)
    t2s, _ = nearest_in_feature_space(dst_feat, src_feat)
    save_points(os.path.join(args.out_dir, "correspondence_source.ply"), src_xyz, src_pc)
    save_points(os.path.join(args.out_dir, "correspondence_target.ply"), dst_xyz, src_pc[t2s])
    if src_col is not None:
        save_points(
            os.path.join(args.out_dir, "target_from_source_colors.ply"), dst_xyz, src_col[t2s]
        )

    grey = np.full((len(dst_xyz), 3), 170, np.uint8)
    red = np.full((len(src_xyz), 3), (220, 60, 60), np.uint8)
    s0, R1, t1 = res["baseline"]
    sc, Rc, tc = res["coarse"]
    panels = [
        (
            "source (red) and target, centred",
            [
                (dst_xyz, grey, 1),
                (transform(src_xyz, s0, np.eye(3), dst_xyz.mean(0) - s0 * src_xyz.mean(0)), red, 1),
            ],
        ),
        (
            "after RANSAC on feature matches",
            [(dst_xyz, grey, 1), (transform(src_xyz, sc, Rc, tc), red, 1)],
        ),
        ("after ICP refinement", [(dst_xyz, grey, 1), (aligned, red, 1)]),
        (
            "baseline ICP without features",
            [(dst_xyz, grey, 1), (transform(src_xyz, s0, R1, t1), red, 1)],
        ),
        ("correspondence colours: source", [(src_xyz, src_pc, 2)]),
        ("correspondence colours: target", [(dst_xyz, src_pc[t2s], 2)]),
    ]
    if src_col is not None:
        panels.append(("target with source colours", [(dst_xyz, src_col[t2s], 2)]))
    save_preview(os.path.join(args.out_dir, "preview.png"), panels)
    print("saved", args.out_dir)


if __name__ == "__main__":
    main()
