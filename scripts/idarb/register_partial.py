"""
Register a partial mesh to a full mesh with Diff3F semantic features. CPU only (NumPy, SciPy,
trimesh); runs on the laptop or the server.

Input: two .npz files written by diff3f_features.py, the "source" (our partial relief mesh from
normal_to_mesh.py) and the "target" (a full mesh of the same kind of object). Registration means
finding the scale, rotation and translation that put the source onto the target, and the
point-to-point correspondence that comes with it.

Steps:
  1. match: cosine similarity between every source vector and every target vector. Semantics, not
     shape, decides these matches, so they survive a different pose, style or coordinate system.
     --matches both (default) takes the best match of every point in both directions and keeps the
     most similar half; --matches mutual keeps only mutual best matches. --consensus drops matches
     whose spatial neighbours on the source do not land near the same place on the target.
  2. coarse: RANSAC over the matches. Each hypothesis takes 3 matches, fits a similarity transform
     (scale, rotation, translation; Umeyama 1991) and is scored by how many matches it brings
     within --inlier_dist of each other. Up to --candidates distinct hypotheses are kept.
  3. fine: every candidate is refined by ICP. --refine icp pairs each source point with its nearest
     target point (geometry only); --refine joint pairs it with the target point that minimises
     feature distance plus spatial distance, with the spatial tolerance shrinking over the
     iterations (feature-guided ICP). --normals rejects pairs whose normals disagree (needs the
     meshes); --depth_scale also fits one factor on the source's z (a relief mesh's depth is only
     known up to the depth-prior scale).
  4. select: each refined candidate is scored by feature agreement, the mean cosine similarity
     between every source point's feature and the feature of the target point it lands on; the
     best one is "selected". A transform that puts the body on the head scores low here even if it
     brings many feature matches together.
  5. baseline: ICP from centroid-and-size alignment only, without features.

Distances are fractions of the target's bounding-box diagonal. Outputs in --out_dir: metrics.csv
(one row per stage and candidate), transform.json, aligned_source.ply (the source mesh moved onto
the target, or the source points), target.ply, correspondence_{source,target}.ply (matched points
share a colour), the albedo transfer onto the target mesh (subdivided --paint_subdiv times so that
small parts have vertices to carry a colour): target_from_source_colors.ply through feature matches
only, target_from_source_colors_registered.ply through the registration (each covered vertex takes
the colour of the aligned source point that is both close and of the same kind of part; uncovered
vertices stay grey), target_from_source_colors_combined.ply (registered where covered, features
elsewhere); correspondences.npz and preview.png (when matplotlib is installed). Without
--target_mesh the coloured target files are point clouds of the sampled points.

--gt <file>_gt.npz from make_partial_mesh.py adds ground-truth columns: rotation and scale error,
and the accuracy of the correspondences (fraction within 1 % and 5 % of the diagonal of the true
point), including the feature-only correspondence without any transform ("feature_nn").

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
from scipy.cluster.vq import kmeans2
from scipy.spatial import cKDTree

DIFFUSION_DIMS = 1280  # Diff3F feature = [1280 diffusion UNet dims | 768 DINOv2 dims]

# ------------------------------------------------------------------------------
# features and matches
# ------------------------------------------------------------------------------


def normalize_rows(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)


def load_features(path, feature="full", mesh=None):
    """A dict with xyz (N,3 float64), feat (N,D float32 unit length), colors (N,3 uint8 or None),
    idx (vertex index of every point), meta, and normals (N,3) when a mesh file is given."""
    z = np.load(path, allow_pickle=False)
    feat = z["feat"].astype(np.float32)
    if feature == "diffusion":
        feat = feat[:, :DIFFUSION_DIMS]
    elif feature == "dino":
        feat = feat[:, DIFFUSION_DIMS:]
    elif feature != "full":
        raise ValueError(feature)
    out = dict(
        xyz=z["xyz"].astype(np.float64),
        feat=normalize_rows(feat),
        colors=z["colors"] if len(z["colors"]) else None,
        idx=z["idx"] if "idx" in z.files else np.arange(len(feat)),
        meta=json.loads(str(z["meta"])) if "meta" in z.files else {},
        normals=None,
    )
    if mesh is not None:
        import trimesh

        m = trimesh.load(mesh, process=False, force="mesh")
        out["normals"] = np.asarray(m.vertex_normals, dtype=np.float64)[out["idx"]]
    return out


def similarity_matrix(fa, fb, chunk=4096):
    """Cosine similarity of every row of fa with every row of fb (both unit length), float16."""
    S = np.empty((len(fa), len(fb)), dtype=np.float16)
    for i in range(0, len(fa), chunk):
        S[i : i + chunk] = fa[i : i + chunk] @ fb.T
    return S


def matches_from_similarity(S, mode="both", keep_frac=0.5):
    """Candidate matches (source indices, target indices, similarities) from the similarity matrix.
    "mutual": pairs that are each other's best match (few, clean). "both": every source point with
    its best target point plus every target point with its best source point, keeping the most
    similar keep_frac of them (many, with outliers for RANSAC to reject)."""
    s2t = S.argmax(axis=1)
    t2s = S.argmax(axis=0)
    sim_s = S[np.arange(len(s2t)), s2t].astype(np.float32)
    if mode == "mutual":
        i = np.nonzero(t2s[s2t] == np.arange(len(s2t)))[0]
        return i, s2t[i], sim_s[i]
    sim_t = S[t2s, np.arange(len(t2s))].astype(np.float32)
    i = np.concatenate([np.arange(len(s2t)), t2s])
    k = np.concatenate([s2t, np.arange(len(t2s))])
    sim = np.concatenate([sim_s, sim_t])
    _, first = np.unique(np.stack([i, k], axis=1), axis=0, return_index=True)
    i, k, sim = i[first], k[first], sim[first]
    keep = sim >= np.quantile(sim, 1.0 - keep_frac)
    return i[keep], k[keep], sim[keep]


def consensus_filter(src_xyz, dst_xyz, i, k, diag, n_neigh=20, radius=0.1, min_frac=0.3):
    """Keep a match (i, k) only if at least min_frac of the n_neigh matches nearest to source point
    i (on the source) land within radius of target point k (on the target): wrong matches are
    scattered, right ones move together."""
    if len(i) <= n_neigh:
        return np.ones(len(i), dtype=bool)
    tree = cKDTree(src_xyz[i])
    _, nb = tree.query(src_xyz[i], k=n_neigh + 1)
    nb = nb[:, 1:]
    d = np.linalg.norm(dst_xyz[k[nb]] - dst_xyz[k][:, None, :], axis=2)
    return (d < radius * diag).mean(axis=1) >= min_frac


def feature_agreement(src_xyz, src_feat, dst_tree, dst_feat, s, R, t, max_dist, a=1.0):
    """How well a transform agrees with the features: the mean cosine similarity between each
    source point's feature and the feature of the target point it lands next to. Source points
    farther than max_dist from the target count as 0."""
    p = transform(apply_depth(src_xyz, a), s, R, t)
    d, j = dst_tree.query(p)
    agree = np.einsum("ij,ij->i", src_feat, dst_feat[j])
    agree[d > max_dist] = 0.0
    return float(agree.mean())


def part_labels(src_feat, dst_feat, n_parts=6, seed=0):
    """k-means parts on the target features, and the nearest centroid for every source point
    (Diff3F's zero-shot segmentation). Returns source labels, target labels."""
    centroids, dst_labels = kmeans2(
        dst_feat.astype(np.float64), n_parts, minit="++", seed=np.random.default_rng(seed)
    )
    d = (
        (src_feat**2).sum(1)[:, None]
        - 2.0 * src_feat @ centroids.T
        + (centroids**2).sum(1)[None, :]
    )
    return d.argmin(axis=1), dst_labels


# ------------------------------------------------------------------------------
# similarity transforms
# ------------------------------------------------------------------------------


def transform(points, s, R, t):
    """s * R @ p + t for every row p."""
    return s * (points @ R.T) + t


def apply_depth(points, a):
    """Multiply z (the depth of a relief mesh) by a."""
    if a == 1.0:
        return points
    q = points.copy()
    q[:, 2] *= a
    return q


def to_matrix(s, R, t, a=1.0):
    M = np.eye(4)
    M[:3, :3] = s * R @ np.diag([1.0, 1.0, a])
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


def solve_depth_factor(p, q, s, R, t, bounds=(0.3, 3.0)):
    """Least-squares a in q ~ s R diag(1,1,a) p + t, with s, R, t fixed."""
    u = s * p[:, 2:3] * R[:, 2][None, :]
    v = q - t - s * (np.column_stack([p[:, 0], p[:, 1], np.zeros(len(p))]) @ R.T)
    a = float((u * v).sum() / max((u * u).sum(), 1e-12))
    return float(np.clip(a, bounds[0], bounds[1]))


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
    return distinct_hypotheses(hyps, src, dst, inlier_dist, n_keep, min_angle_deg, scale_range)


def distinct_hypotheses(hyps, src, dst, inlier_dist, n_keep, min_angle_deg=15.0, scale_range=None):
    """Sort (count, s, R, t) hypotheses by count, drop near-duplicates, refit the first n_keep on
    their inliers (keeping the scale inside scale_range). Returns [(s, R, t, inliers), ...]."""
    hyps = sorted(hyps, key=lambda h: -h[0])
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
            if scale_range is not None:
                s_clip = float(np.clip(s, scale_range[0], scale_range[1]))
                if s_clip != s:  # the refit left the allowed range: keep the bound
                    _, R, t = umeyama(s_clip * src[inliers], dst[inliers], with_scale=False)
                    s = s_clip
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


# ------------------------------------------------------------------------------
# refinement (ICP)
# ------------------------------------------------------------------------------


def normals_compatible(src_normals, dst_normals, R, j, min_dot=-0.3):
    """Pairs whose source normal, rotated, points roughly like the target normal."""
    if src_normals is None or dst_normals is None:
        return None
    return np.einsum("ij,ij->i", src_normals @ R.T, dst_normals[j]) > min_dot


def icp_rigid(
    src,
    dst,
    s,
    R,
    t,
    max_dist,
    iters=100,
    tol=1e-7,
    src_normals=None,
    dst_normals=None,
    depth_scale=False,
    a=1.0,
):
    """Point-to-point ICP with the scale fixed. Every iteration pairs each transformed source point
    with its nearest target point, drops pairs farther apart than max_dist (and with incompatible
    normals, when given) and refits R, t (and the depth factor a when depth_scale).
    Returns R, t, a, the RMS distance of the kept pairs, the fraction of source points kept and
    the final pairing (target index per source point)."""
    tree = cKDTree(dst)
    rmse_prev = np.inf
    rmse = np.inf
    kept = 0.0
    j = np.zeros(len(src), dtype=np.int64)
    for _ in range(iters):
        q = apply_depth(src, a)
        p = transform(q, s, R, t)
        d, j = tree.query(p)
        keep = d < max_dist
        ok = normals_compatible(src_normals, dst_normals, R, j)
        if ok is not None:
            keep &= ok
        if keep.sum() < 10:
            break
        rmse = float(np.sqrt(np.mean(d[keep] ** 2)))
        kept = float(keep.mean())
        _, R, t = umeyama(s * q[keep], dst[j[keep]], with_scale=False)
        if depth_scale:
            a = solve_depth_factor(src[keep], dst[j[keep]], s, R, t)
        if abs(rmse_prev - rmse) < tol:
            break
        rmse_prev = rmse
    return R, t, a, rmse, kept, j


def icp_joint(
    src,
    dst,
    S,
    s,
    R,
    t,
    diag,
    deltas=(0.2, 0.1, 0.05, 0.03),
    iters=8,
    scale_range=None,
    refit_scale=False,
    src_normals=None,
    dst_normals=None,
    depth_scale=False,
    a=1.0,
    chunk=3000,
):
    """Feature-guided ICP. Each source point is paired with the target point that minimises
    (1 - cosine similarity) + (spatial distance / delta)^2, so the pairing respects the semantics
    and the geometry at the same time; delta shrinks over the stages, from "trust the features" to
    "trust the geometry". The scale stays as it comes from the feature matches unless refit_scale
    (refitting it lets the geometry shrink the source: a smaller source is closer to any surface),
    and the depth factor a is refitted when depth_scale. Returns s, R, t, a, the RMS distance of
    the kept pairs, the fraction kept and the pairing."""
    n_src = len(src)
    S32_rows = None  # (1 - S) converted chunk by chunk
    q2 = (dst**2).sum(axis=1)
    j = np.zeros(n_src, dtype=np.int64)
    rmse = np.inf
    kept = 0.0
    for delta in deltas:
        dd = (delta * diag) ** 2
        for _ in range(iters):
            q = apply_depth(src, a)
            p = transform(q, s, R, t)
            cost_min = np.empty(n_src)
            for i0 in range(0, n_src, chunk):
                pc = p[i0 : i0 + chunk]
                d2 = (pc**2).sum(axis=1)[:, None] - 2.0 * pc @ dst.T + q2[None, :]
                S32_rows = 1.0 - S[i0 : i0 + chunk].astype(np.float32)
                c = S32_rows + np.maximum(d2, 0.0) / dd
                jj = c.argmin(axis=1)
                j[i0 : i0 + chunk] = jj
                cost_min[i0 : i0 + chunk] = c[np.arange(len(jj)), jj]
            d = np.linalg.norm(p - dst[j], axis=1)
            keep = d < 2.0 * delta * diag
            ok = normals_compatible(src_normals, dst_normals, R, j)
            if ok is not None:
                keep &= ok
            if keep.sum() < 10:
                break
            rmse = float(np.sqrt(np.mean(d[keep] ** 2)))
            kept = float(keep.mean())
            if refit_scale:
                s_new, R, t = umeyama(q[keep], dst[j[keep]], with_scale=True)
                if scale_range is not None:
                    s_new = float(np.clip(s_new, scale_range[0], scale_range[1]))
                    if s_new != s:  # the clipped scale needs its own translation
                        _, R, t = umeyama(s_new * q[keep], dst[j[keep]], with_scale=False)
                s = s_new
            else:
                _, R, t = umeyama(s * q[keep], dst[j[keep]], with_scale=False)
            if depth_scale:
                a = solve_depth_factor(src[keep], dst[j[keep]], s, R, t)
    return s, R, t, a, rmse, kept, j


# ------------------------------------------------------------------------------
# evaluation
# ------------------------------------------------------------------------------


def stage_metrics(name, src_xyz, dst_tree, s, R, t, diag, a=1.0, matches=None, close=0.02):
    """Distances of the transformed source points to the nearest target point, as fractions of the
    target diagonal; and, if matches (i, k, dst) are given, the distance between matched points."""
    p = transform(apply_depth(src_xyz, a), s, R, t)
    d, _ = dst_tree.query(p)
    d = d / diag
    row = dict(
        stage=name,
        scale=round(float(s), 4),
        depth_factor=round(float(a), 3),
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


def gt_targets(src, dst, gt):
    """Row in dst of the true counterpart of every source point (-1 when the true vertex was not
    sampled in dst). gt: orig_idx (partial vertex -> full vertex), s, R, t."""
    inv = -np.ones(int(gt["orig_idx"].max()) + 1 + len(dst["xyz"]), dtype=np.int64)
    inv[dst["idx"]] = np.arange(len(dst["idx"]))
    full_vertex = gt["orig_idx"][src["idx"]]
    full_vertex = np.where(full_vertex < len(inv), full_vertex, -1)
    rows = np.where(full_vertex >= 0, inv[np.maximum(full_vertex, 0)], -1)
    return rows


def correspondence_metrics(assign, gt_rows, dst_xyz, diag):
    """Accuracy of a correspondence (target row per source point) against the ground truth."""
    valid = gt_rows >= 0
    err = np.linalg.norm(dst_xyz[assign[valid]] - dst_xyz[gt_rows[valid]], axis=1) / diag
    return dict(
        corr_acc_1pct=round(float((err < 0.01).mean()), 3),
        corr_acc_5pct=round(float((err < 0.05).mean()), 3),
        corr_median=round(float(np.median(err)), 4),
    )


def transform_metrics(src_xyz, gt_rows, dst_xyz, gt, s, R, t, a, diag):
    valid = gt_rows >= 0
    p = transform(apply_depth(src_xyz, a), s, R, t)
    pos_err = np.linalg.norm(p[valid] - dst_xyz[gt_rows[valid]], axis=1) / diag
    return dict(
        rot_err_deg=round(rotation_angle_deg(R, gt["R"]), 2),
        scale_ratio=round(float(s / float(gt["s"])), 4),
        pos_err_median=round(float(np.median(pos_err)), 4),
    )


def points_metrics(
    name,
    points,
    src_feat,
    dst_tree,
    dst_feat,
    src_parts,
    dst_parts,
    diag,
    max_dist,
    gt_rows=None,
    dst_xyz=None,
    **extra,
):
    """The metrics of stage_metrics / evaluate for an already-moved point set (piecewise
    alignment), where there is no single transform."""
    d, j = dst_tree.query(points)
    dn = d / diag
    agree = np.einsum("ij,ij->i", src_feat, dst_feat[j])
    agree[d > max_dist] = 0.0
    row = dict(
        stage=name,
        rms_to_target=round(float(np.sqrt(np.mean(dn**2))), 4),
        median_to_target=round(float(np.median(dn)), 4),
        frac_within_2pct=round(float((dn < 0.02).mean()), 3),
        agreement=round(float(agree.mean()), 4),
        part_agreement=round(float((src_parts == dst_parts[j]).mean()), 3),
    )
    if gt_rows is not None:
        valid = gt_rows >= 0
        pos_err = np.linalg.norm(points[valid] - dst_xyz[gt_rows[valid]], axis=1) / diag
        row["pos_err_median"] = round(float(np.median(pos_err)), 4)
        row.update(correspondence_metrics(j, gt_rows, dst_xyz, diag))
    row.update(extra)
    return row


# ------------------------------------------------------------------------------
# piecewise refinement
# ------------------------------------------------------------------------------


def piecewise_refine(
    base_xyz,
    dst_xyz,
    src_parts,
    dst_parts,
    diag,
    icp_dist=0.05,
    scale_range=(0.8, 1.25),
    max_rot_deg=15.0,
    min_src=150,
    min_dst=50,
    iters=50,
    blend_sigma=0.05,
):
    """Per-part refinement after the global registration. base_xyz are the globally aligned source
    points. For every part (k-means label on the features) with enough points on both sides, a
    similarity transform with its own scale (clipped to scale_range) is fitted by ICP between the
    source and the target points of that part only, so a cartoon head that is too big for the
    realistic head shrinks onto it while the body stays where it is. A part keeps the global
    alignment when its transform rotates more than max_rot_deg, does not bring the part closer, or
    uncovers target points it covered before (the way a shrinking part would). The transforms are
    then blended: every point moves with a weighted mix of all part transforms, the weights falling
    off with the distance to each part (blend_sigma, fraction of the diagonal), so there are no
    tears between parts. Returns the moved points and the part table used by blend_piecewise()."""
    cap = icp_dist * diag
    info = {}

    def capped_rms(d):
        return float(np.sqrt(np.mean(np.minimum(d, cap) ** 2))) / diag

    for p in np.unique(dst_parts):
        si = np.nonzero(src_parts == p)[0]
        di = np.nonzero(dst_parts == p)[0]
        if len(si) < min_src or len(di) < min_dst:
            continue
        sp, dp = base_xyz[si], dst_xyz[di]
        tree = cKDTree(dp)
        rms0 = capped_rms(tree.query(sp)[0])
        cover0 = float((cKDTree(sp).query(dp)[0] < cap).mean())
        s_p, R_p, t_p = 1.0, np.eye(3), np.zeros(3)
        for _ in range(iters):
            q = transform(sp, s_p, R_p, t_p)
            d, j = tree.query(q)
            keep = d < cap
            if keep.sum() < 20:
                break
            s_new, R_new, t_new = umeyama(sp[keep], dp[j[keep]], with_scale=True)
            s_clip = float(np.clip(s_new, scale_range[0], scale_range[1]))
            if s_clip != s_new:
                _, R_new, t_new = umeyama(s_clip * sp[keep], dp[j[keep]], with_scale=False)
            if (s_clip, np.abs(R_new - R_p).max(), np.abs(t_new - t_p).max()) == (s_p, 0.0, 0.0):
                break
            s_p, R_p, t_p = s_clip, R_new, t_new
        q = transform(sp, s_p, R_p, t_p)
        rms1 = capped_rms(tree.query(q)[0])
        cover1 = float((cKDTree(q).query(dp)[0] < cap).mean())
        accepted = (
            rms1 < rms0
            and rotation_angle_deg(R_p, np.eye(3)) <= max_rot_deg
            and cover1 >= 0.9 * cover0
        )
        info[int(p)] = dict(
            s=s_p if accepted else 1.0,
            R=R_p if accepted else np.eye(3),
            t=t_p if accepted else np.zeros(3),
            idx=si,
            accepted=bool(accepted),
            n_src=int(len(si)),
            n_dst=int(len(di)),
            rms_before=round(rms0, 4),
            rms_after=round(rms1 if accepted else rms0, 4),
            cover_before=round(cover0, 3),
            cover_after=round(cover1 if accepted else cover0, 3),
        )
    return blend_piecewise(base_xyz, base_xyz, info, blend_sigma * diag), info


def blend_piecewise(points, base_xyz, info, sigma):
    """Move points (in the globally aligned frame) with the part transforms of info, blended by
    exp(-d^2 / 2 sigma^2) where d is the distance from the point to the nearest sampled point of
    each part (base_xyz[info[p]["idx"]]). Parts without an accepted transform pull towards the
    identity, so the blend is the identity far from every refined part."""
    if not info:
        return points.copy()
    weights = []
    moved = []
    for tr in info.values():
        d, _ = cKDTree(base_xyz[tr["idx"]]).query(points)
        weights.append(np.exp(-(d**2) / (2.0 * sigma**2)) + 1e-12)
        moved.append(transform(points, tr["s"], tr["R"], tr["t"]))
    weights = np.stack(weights, axis=1)
    weights /= weights.sum(axis=1, keepdims=True)
    out = np.zeros_like(points)
    for w, m in zip(weights.T, moved):
        out += w[:, None] * m
    return out


def apply_piecewise(points, base_xyz, info, s, R, t, a=1.0, sigma=0.05):
    """Move arbitrary points of the source mesh with the global transform and then with the
    blended part transforms. base_xyz: the globally aligned sampled source points; sigma as a
    fraction of the target diagonal is converted by the caller (pass sigma * diag)."""
    base = transform(apply_depth(points, a), s, R, t)
    return blend_piecewise(base, base_xyz, info, sigma)


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


def save_target_colored(path, target_mesh, dst_xyz, colors):
    """The target as a mesh with one colour per vertex (every mesh vertex takes the colour of the
    nearest sampled point), so viewers show a continuous coloured surface; without the mesh file,
    a point cloud of the sampled points."""
    if target_mesh is None:
        save_points(path, dst_xyz, colors)
        return
    import trimesh

    m = trimesh.load(target_mesh, process=False, force="mesh")
    _, near = cKDTree(dst_xyz).query(np.asarray(m.vertices, dtype=np.float64))
    trimesh.Trimesh(
        vertices=m.vertices, faces=m.faces, vertex_colors=colors[near], process=False
    ).export(path)


def symmetry_planes(points):
    """Candidate mirror planes of a shape: the three principal planes through the centroid, each
    with the mean distance from the reflected points to the shape as a fraction of the bounding-box
    diagonal (small means the shape really is symmetric about that plane). Returns a list of
    (centre, unit normal, error), best first."""
    c = points.mean(axis=0)
    diag = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    _, _, vt = np.linalg.svd(points - c, full_matrices=False)
    tree = cKDTree(points)
    out = []
    for n in vt:
        refl = points - 2.0 * ((points - c) @ n)[:, None] * n[None, :]
        out.append((c, n, float(tree.query(refl)[0].mean()) / diag))
    return sorted(out, key=lambda p: p[2])


def symmetry_plane(points):
    """The best candidate of symmetry_planes()."""
    return symmetry_planes(points)[0]


def paint_target(
    target_mesh,
    dst_xyz,
    dst_feat,
    src_feat,
    aligned_src,
    src_colors,
    diag,
    subdiv=1,
    delta=0.03,
    cover_dist=0.05,
    chunk=1024,
    src_parts=None,
    dst_parts=None,
    part_penalty=0.5,
    feature_weight=1.0,
    mirror=True,
    mirror_tol=0.02,
    max_plane_err=0.02,
    full_src_xyz=None,
    full_src_colors=None,
    refine_radius=0.03,
):
    """Albedo transfer onto the target mesh, subdivided subdiv times so that small parts have
    vertices to carry a colour. Every target vertex takes its feature from the 3 nearest sampled
    target points. Returns a dict with V, F, the masks covered and mirrored, the plane, and four
    colour arrays:
      feature: the colour of the source point with the most similar feature, anywhere;
      registered: for vertices within cover_dist of the aligned source, the source point that
        minimises feature_weight (1 - similarity) + (distance / delta)^2 + part_penalty if the
        parts differ, so the colour comes from nearby AND from the same kind of part (an eye
        vertex takes an eye colour from the eye next to it); uncovered vertices stay grey;
      mirror: registered, plus the uncovered vertices whose mirror image across the target's
        symmetry plane lies on a covered vertex (within mirror_tol), which take that vertex's
        colour: the far side of a bilaterally symmetric object gets the near side's albedo. Only
        when the plane fits within max_plane_err;
      combined: mirror where available, feature elsewhere.
    With full_src_xyz / full_src_colors (every vertex of the aligned source mesh with its colour,
    not only the sampled points that carry features), the chosen sampled point only marks the
    region and the colour comes from the full-resolution vertex nearest to the target vertex
    within refine_radius of it, so the painting keeps the photo's detail."""
    import trimesh

    m = trimesh.load(target_mesh, process=False, force="mesh")
    V = np.asarray(m.vertices, dtype=np.float64)
    F = np.asarray(m.faces, dtype=np.int64)
    for _ in range(subdiv):
        V, F = trimesh.remesh.subdivide(V, F)
    _, nn = cKDTree(dst_xyz).query(V, k=3)
    feat_v = normalize_rows(dst_feat[nn].mean(axis=1))
    vparts = dst_parts[nn[:, 0]] if dst_parts is not None else None
    d_s, _ = cKDTree(aligned_src).query(V)
    covered = d_s < cover_dist * diag
    a2 = (aligned_src**2).sum(axis=1)
    dd = (delta * diag) ** 2
    best_joint = np.empty(len(V), dtype=np.int64)
    best_feat = np.empty(len(V), dtype=np.int64)
    for i in range(0, len(V), chunk):
        vc = V[i : i + chunk]
        sim = feat_v[i : i + chunk] @ src_feat.T
        best_feat[i : i + chunk] = sim.argmax(axis=1)
        d2 = (vc**2).sum(axis=1)[:, None] - 2.0 * vc @ aligned_src.T + a2[None, :]
        cost = feature_weight * (1.0 - sim) + np.maximum(d2, 0.0) / dd
        if vparts is not None and part_penalty > 0:  # colours from the same kind of part
            cost += part_penalty * (vparts[i : i + chunk][:, None] != src_parts[None, :])
        best_joint[i : i + chunk] = cost.argmin(axis=1)
    col_feat = src_colors[best_feat]
    col_reg = np.full((len(V), 3), 170, np.uint8)
    col_reg[covered] = src_colors[best_joint[covered]]
    if full_src_xyz is not None and covered.any():
        idx_c = np.nonzero(covered)[0]
        _, j_k = cKDTree(full_src_xyz).query(V[idx_c], k=8)
        p_sel = aligned_src[best_joint[idx_c]]
        inside = (
            np.linalg.norm(full_src_xyz[j_k] - p_sel[:, None, :], axis=2) < refine_radius * diag
        )
        has = inside.any(axis=1)
        first = inside.argmax(axis=1)  # the nearest full-resolution vertex inside the region
        col_reg[idx_c[has]] = full_src_colors[j_k[has, first[has]]]
    col_mirror = col_reg.copy()
    mirrored = np.zeros(len(V), dtype=bool)
    plane = None
    if mirror and covered.any() and not covered.all():
        # among the planes the shape is symmetric about, take the one whose reflection carries
        # the most uncovered vertices onto covered ones (a box has three mirror planes; only the
        # one between the near and the far side is useful)
        cov_idx = np.nonzero(covered)[0]
        unc_idx = np.nonzero(~covered)[0]
        cov_tree = cKDTree(V[cov_idx])
        best = None
        for c, n, err in symmetry_planes(V):
            if err > max_plane_err:
                continue
            refl = V - 2.0 * ((V - c) @ n)[:, None] * n[None, :]
            d_m, j_m = cov_tree.query(refl[unc_idx])
            ok = d_m < mirror_tol * diag
            if best is None or ok.sum() > best[0]:
                best = (int(ok.sum()), (c, n, err), ok, j_m)
        if best is not None:
            _, plane, ok, j_m = best
            col_mirror[unc_idx[ok]] = col_reg[cov_idx[j_m[ok]]]
            mirrored[unc_idx[ok]] = True
        else:
            plane = symmetry_planes(V)[0]
    col_comb = col_feat.copy()
    col_comb[covered] = col_reg[covered]
    col_comb[mirrored] = col_mirror[mirrored]
    return dict(
        V=V,
        F=F,
        covered=covered,
        mirrored=mirrored,
        plane=plane,
        feature=col_feat,
        registered=col_reg,
        mirror=col_mirror,
        combined=col_comb,
    )


def save_mesh_transformed(path, mesh_path, s, R, t, a=1.0):
    import trimesh

    m = trimesh.load(mesh_path, process=False, force="mesh")
    m.vertices = transform(apply_depth(np.asarray(m.vertices, dtype=np.float64), a), s, R, t)
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

DEFAULTS = dict(
    inlier_dist=0.05,
    icp_dist=0.05,
    n_iter=5000,
    seeds=(0,),
    match_mode="both",
    consensus=False,
    n_candidates=8,
    refine="icp",
    n_joint=3,
    joint_refit_scale=False,
    depth_grid=(0.5, 0.7, 1.0, 1.4, 2.0),
    n_depth=3,
    scale_low=0.5,
    scale_high=2.5,
    scale_bands=((0.5, 0.8), (0.8, 1.3), (1.3, 2.5)),
    min_coverage=0.2,
    piecewise=False,
    part_scale_range=(0.8, 1.25),
    part_max_rot=15.0,
    blend_sigma=0.05,
    normals=False,
    depth_scale=False,
    n_parts=6,
)


def register(src, dst, gt=None, **options):
    """The whole pipeline. src, dst: dicts from load_features (xyz, feat, normals, idx). gt: dict
    from make_partial_mesh.py's _gt.npz, optional. Options: see DEFAULTS and the module docstring.
    Returns a dict with the selected transform (s, R, t, a), the correspondences and metric rows."""
    o = dict(DEFAULTS)
    o.update(options)
    src_xyz, src_feat = src["xyz"], src["feat"]
    dst_xyz, dst_feat = dst["xyz"], dst["feat"]
    src_n = src.get("normals") if o["normals"] else None
    dst_n = dst.get("normals") if o["normals"] else None
    diag = float(np.linalg.norm(dst_xyz.max(0) - dst_xyz.min(0)))
    src_diag = float(np.linalg.norm(src_xyz.max(0) - src_xyz.min(0)))
    ratio = diag / src_diag
    # a front-view relief has about the extent of the full object: the selected scale was 0.55 to
    # 1.2 times the bounding-box ratio on every pair that registered, 0.38 on the one that collapsed
    scale_range = (o["scale_low"] * ratio, o["scale_high"] * ratio)
    dst_tree = cKDTree(dst_xyz)
    gt_rows = gt_targets(src, dst, gt) if gt is not None else None
    src_parts, dst_parts = part_labels(src_feat, dst_feat, o["n_parts"])

    def evaluate(name, s, R, t, a, assign=None, **extra):
        row = stage_metrics(name, src_xyz, dst_tree, s, R, t, diag, a, matches)
        p = transform(apply_depth(src_xyz, a), s, R, t)
        d, j = dst_tree.query(p)
        if assign is None:
            assign = j
        row["agreement"] = round(
            feature_agreement(
                src_xyz, src_feat, dst_tree, dst_feat, s, R, t, o["icp_dist"] * diag, a
            ),
            4,
        )
        row["part_agreement"] = round(float((src_parts == dst_parts[j]).mean()), 3)
        # the same agreement seen from the target: every target point near the source compares its
        # feature with that of the nearest source point, points not near the source count 0. A
        # source squeezed into one part of the target scores well from the source side only.
        d_t, j_t = cKDTree(p).query(dst_xyz)
        near = d_t < o["icp_dist"] * diag
        agree_t = np.einsum("ij,ij->i", dst_feat, src_feat[j_t])
        agree_t[~near] = 0.0
        row["coverage"] = round(float(near.mean()), 3)
        row["agreement_t"] = round(float(agree_t.mean()), 4)
        # selection score: the agreement, discounted only when the source covers less of the
        # target than any front view does (registered pairs cover 0.24 to 0.75 of the target, the
        # collapsed bird 0.16). Averaging in agreement_t instead rewards size, not correctness.
        row["score"] = round(row["agreement"] * min(1.0, row["coverage"] / o["min_coverage"]), 4)
        if gt_rows is not None:
            row.update(transform_metrics(src_xyz, gt_rows, dst_xyz, gt, s, R, t, a, diag))
            row.update(correspondence_metrics(assign, gt_rows, dst_xyz, diag))
        row.update(extra)
        return row

    # 1. matches
    S = similarity_matrix(src_feat, dst_feat)
    i, k, sim = matches_from_similarity(S, o["match_mode"])
    if o["consensus"]:
        keep = consensus_filter(src_xyz, dst_xyz, i, k, diag)
        i, k, sim = i[keep], k[keep], sim[keep]
    matches = (i, k, dst_xyz)
    n_matches = int(len(i))
    rows = []
    if gt_rows is not None:
        rows.append(
            dict(
                stage="feature_nn",
                **correspondence_metrics(S.argmax(axis=1), gt_rows, dst_xyz, diag),
            )
        )

    # 2. coarse
    hyps = []
    # RANSAC once per scale band, so the pool always holds hypotheses at every plausible scale;
    # the best-inlier solution alone can be a small source squeezed into a part of the target
    bands = [(lo * ratio, hi * ratio) for lo, hi in o["scale_bands"]] or [scale_range]
    for seed in o["seeds"]:
        for band in bands:
            band = (max(band[0], scale_range[0]), min(band[1], scale_range[1]))
            if band[0] >= band[1]:
                continue
            for s, R, t, inl in ransac_candidates(
                src_xyz[i],
                dst_xyz[k],
                o["inlier_dist"] * diag,
                n_iter=o["n_iter"],
                seed=seed,
                scale_range=band,
                n_keep=o["n_candidates"],
            ):
                hyps.append((int(inl.sum()), s, R, t))
    cands = distinct_hypotheses(
        hyps,
        src_xyz[i],
        dst_xyz[k],
        o["inlier_dist"] * diag,
        o["n_candidates"],
        scale_range=scale_range,
    )
    if not cands:
        raise RuntimeError("RANSAC found no transform with 3 inliers")
    s, R, t, inliers = cands[0]
    rows.append(
        evaluate(
            "ransac", s, R, t, 1.0, candidate=0, n_matches=n_matches, n_inliers=int(inliers.sum())
        )
    )

    # 3. fine: geometric ICP on every candidate
    refined = []
    for c, (s, R, t, inliers) in enumerate(cands):
        R2, t2, a, _, kept, j = icp_rigid(
            src_xyz,
            dst_xyz,
            s,
            R,
            t,
            o["icp_dist"] * diag,
            src_normals=src_n,
            dst_normals=dst_n,
        )
        row = evaluate(
            f"candidate{c}+icp",
            s,
            R2,
            t2,
            a,
            j,
            candidate=c,
            n_matches=n_matches,
            n_inliers=int(inliers.sum()),
            icp_kept=round(kept, 3),
        )
        rows.append(row)
        refined.append((row["score"], c, s, R2, t2, a, j, inliers, kept, (s, R, t)))
    refined.sort(key=lambda r: -r[0])

    # depth factor: restart the ICP of the best few candidates from several depth factors and let
    # it fit the factor; the pose and the factor are coupled, so a single start from 1.0 stops
    # short when the relief is far too deep or too shallow
    if o["depth_scale"]:
        with_depth = []
        for _agree, c, s, R, t, _a, _j, inliers, _kept, coarse in refined[: o["n_depth"]]:
            for a0 in o["depth_grid"]:
                R2, t2, a, _, kept, j = icp_rigid(
                    src_xyz,
                    dst_xyz,
                    s,
                    R,
                    t,
                    o["icp_dist"] * diag,
                    src_normals=src_n,
                    dst_normals=dst_n,
                    depth_scale=True,
                    a=a0,
                )
                row = evaluate(
                    f"candidate{c}+icp+depth{a0:g}",
                    s,
                    R2,
                    t2,
                    a,
                    j,
                    candidate=c,
                    n_matches=n_matches,
                    n_inliers=int(inliers.sum()),
                    icp_kept=round(kept, 3),
                )
                rows.append(row)
                with_depth.append((row["score"], c, s, R2, t2, a, j, inliers, kept, coarse))
        refined = sorted(with_depth + refined, key=lambda r: -r[0])

    # feature-guided ICP on the best few
    if o["refine"] == "joint":
        joint = []
        for _agree, c, s, R, t, a, _j, inliers, _kept, coarse in refined[: o["n_joint"]]:
            s2, R2, t2, a2, _, kept2, j2 = icp_joint(
                src_xyz,
                dst_xyz,
                S,
                s,
                R,
                t,
                diag,
                scale_range=scale_range,
                refit_scale=o["joint_refit_scale"],
                src_normals=src_n,
                dst_normals=dst_n,
                depth_scale=o["depth_scale"],
                a=a,
            )
            row = evaluate(
                f"candidate{c}+joint",
                s2,
                R2,
                t2,
                a2,
                j2,
                candidate=c,
                n_matches=n_matches,
                n_inliers=int(inliers.sum()),
                icp_kept=round(kept2, 3),
            )
            rows.append(row)
            joint.append((row["score"], c, s2, R2, t2, a2, j2, inliers, kept2, coarse))
        # the geometric results stay in the pool: the joint step must earn its place
        refined = sorted(joint + refined, key=lambda r: -r[0])

    # 4. select
    agree, c, s, R, t, a, j, inliers, kept, coarse = refined[0]
    rows.append(
        evaluate(
            "selected",
            s,
            R,
            t,
            a,
            j,
            candidate=c,
            n_matches=n_matches,
            n_inliers=int(inliers.sum()),
            icp_kept=round(kept, 3),
        )
    )

    # 5. baseline without features: same size and centre, no rotation
    s0 = ratio
    t0 = dst_xyz.mean(0) - s0 * src_xyz.mean(0)
    R1, t1, a1, _, kept1, j1 = icp_rigid(
        src_xyz,
        dst_xyz,
        s0,
        np.eye(3),
        t0,
        o["icp_dist"] * diag,
        src_normals=src_n,
        dst_normals=dst_n,
        depth_scale=o["depth_scale"],
    )
    rows.append(evaluate("baseline_icp_no_features", s0, R1, t1, a1, j1, icp_kept=round(kept1, 3)))

    # 6. optional per-part refinement of the selected alignment
    pw_xyz, pw_info = None, {}
    if o["piecewise"]:
        base = transform(apply_depth(src_xyz, a), s, R, t)
        pw_xyz, pw_info = piecewise_refine(
            base,
            dst_xyz,
            src_parts,
            dst_parts,
            diag,
            icp_dist=o["icp_dist"],
            scale_range=o["part_scale_range"],
            max_rot_deg=o["part_max_rot"],
            blend_sigma=o["blend_sigma"],
        )
        rows.append(
            points_metrics(
                "piecewise",
                pw_xyz,
                src_feat,
                dst_tree,
                dst_feat,
                src_parts,
                dst_parts,
                diag,
                o["icp_dist"] * diag,
                gt_rows,
                dst_xyz,
                candidate=c,
                n_matches=n_matches,
                n_inliers=int(inliers.sum()),
            )
        )

    return dict(
        diag=diag,
        aligned_piecewise=pw_xyz,
        piecewise_info=pw_info,
        src_parts=src_parts,
        dst_parts=dst_parts,
        matches=(i, k, sim),
        inliers=inliers,
        selected=c,
        coarse=coarse,
        fine=(s, R, t, a),
        assign=j,
        feature_nn=S.argmax(axis=1),
        baseline=(s0, R1, t1, a1),
        rows=rows,
        options=o,
    )


FIELDS = [
    "stage", "candidate", "scale", "depth_factor", "n_matches", "n_inliers", "icp_kept",
    "agreement", "agreement_t", "score", "part_agreement", "coverage", "rms_to_target",
    "median_to_target", "frac_within_2pct",
    "match_median", "match_within_5pct", "rot_err_deg", "scale_ratio", "pos_err_median",
    "corr_acc_1pct", "corr_acc_5pct", "corr_median",
]  # fmt: skip


def write_rows(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--source", required=True, help="features of the partial mesh (.npz)")
    ap.add_argument("--target", required=True, help="features of the full mesh (.npz)")
    ap.add_argument("--source_mesh", help="the partial mesh file (normals, transformed output)")
    ap.add_argument(
        "--target_mesh", help="the full mesh file (normals, copied next to the outputs)"
    )
    ap.add_argument("--gt", help="_gt.npz from make_partial_mesh.py for ground-truth columns")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--feature", choices=["full", "diffusion", "dino"], default="full")
    ap.add_argument("--matches", choices=["both", "mutual"], default=DEFAULTS["match_mode"])
    ap.add_argument("--consensus", action="store_true", help="local-consensus match filter")
    ap.add_argument("--inlier_dist", type=float, default=DEFAULTS["inlier_dist"])
    ap.add_argument("--icp_dist", type=float, default=DEFAULTS["icp_dist"])
    ap.add_argument("--ransac_iters", type=int, default=DEFAULTS["n_iter"])
    ap.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULTS["seeds"]))
    ap.add_argument("--candidates", type=int, default=DEFAULTS["n_candidates"])
    ap.add_argument(
        "--scale_low",
        type=float,
        default=DEFAULTS["scale_low"],
        help="lowest allowed scale, times the ratio of the bounding-box diagonals",
    )
    ap.add_argument("--scale_high", type=float, default=DEFAULTS["scale_high"])
    ap.add_argument("--refine", choices=["icp", "joint"], default=DEFAULTS["refine"])
    ap.add_argument("--normals", action="store_true", help="reject ICP pairs with opposite normals")
    ap.add_argument("--depth_scale", action="store_true", help="also fit a depth factor on z")
    ap.add_argument("--parts", type=int, default=DEFAULTS["n_parts"])
    ap.add_argument(
        "--paint_subdiv",
        type=int,
        default=2,
        help="subdivide the target mesh this many times before painting it (4x vertices each)",
    )
    ap.add_argument(
        "--paint_delta",
        type=float,
        default=0.03,
        help="spatial tolerance of the painting, fraction of the target diagonal",
    )
    ap.add_argument(
        "--piecewise",
        action="store_true",
        help="after the global alignment, align every feature part on its own (own scale)",
    )
    ap.add_argument(
        "--part_penalty",
        type=float,
        default=0.5,
        help="painting: cost added when the colour would come from a different part",
    )
    ap.add_argument(
        "--feature_weight",
        type=float,
        default=1.0,
        help="painting: weight of the feature term (0 = nearest aligned point only)",
    )
    ap.add_argument(
        "--no_mirror",
        action="store_true",
        help="painting: do not fill the far side through the target's symmetry plane",
    )
    ap.add_argument(
        "--no_full_colors",
        action="store_true",
        help="painting: use only the sampled points' colours, not every source vertex",
    )
    args = ap.parse_args()

    src = load_features(args.source, args.feature, args.source_mesh)
    dst = load_features(args.target, args.feature, args.target_mesh)
    gt = dict(np.load(args.gt, allow_pickle=False)) if args.gt else None
    print(f"source {len(src['xyz'])} points, target {len(dst['xyz'])} points")

    res = register(
        src,
        dst,
        gt,
        inlier_dist=args.inlier_dist,
        icp_dist=args.icp_dist,
        n_iter=args.ransac_iters,
        seeds=tuple(args.seeds),
        match_mode=args.matches,
        consensus=args.consensus,
        n_candidates=args.candidates,
        scale_low=args.scale_low,
        scale_high=args.scale_high,
        refine=args.refine,
        normals=args.normals,
        depth_scale=args.depth_scale,
        n_parts=args.parts,
        piecewise=args.piecewise,
    )
    s, R, t, a = res["fine"]
    os.makedirs(args.out_dir, exist_ok=True)
    write_rows(os.path.join(args.out_dir, "metrics.csv"), res["rows"])
    for row in res["rows"]:
        print({k: v for k, v in row.items() if v is not None})

    opts = {k: (list(v) if isinstance(v, tuple) else v) for k, v in res["options"].items()}
    with open(os.path.join(args.out_dir, "transform.json"), "w") as f:
        json.dump(
            dict(
                source=args.source,
                target=args.target,
                feature=args.feature,
                scale=s,
                rotation=R.tolist(),
                translation=t.tolist(),
                depth_factor=a,
                matrix=to_matrix(s, R, t, a).tolist(),
                coarse_matrix=to_matrix(*res["coarse"]).tolist(),
                selected_candidate=res["selected"],
                options=opts,
                source_meta=src["meta"],
                target_meta=dst["meta"],
            ),
            f,
            indent=1,
        )
    np.savez_compressed(
        os.path.join(args.out_dir, "correspondences.npz"),
        source_idx=src["idx"],
        target_idx=dst["idx"],
        registered=res["assign"],
        feature_nn=res["feature_nn"],
    )

    # point clouds and meshes for MeshLab
    src_xyz, dst_xyz = src["xyz"], dst["xyz"]
    aligned = transform(apply_depth(src_xyz, a), s, R, t)
    if args.source_mesh:
        save_mesh_transformed(
            os.path.join(args.out_dir, "aligned_source.ply"), args.source_mesh, s, R, t, a
        )
    else:
        save_points(os.path.join(args.out_dir, "aligned_source.ply"), aligned, src["colors"])
    if args.target_mesh:
        import trimesh

        trimesh.load(args.target_mesh, process=False, force="mesh").export(
            os.path.join(args.out_dir, "target.ply")
        )
    else:
        save_points(os.path.join(args.out_dir, "target.ply"), dst_xyz, dst["colors"])

    # correspondence colours: source by position, target by its best-matching source point
    src_pc = position_colors(src_xyz)
    t2s = similarity_matrix(dst["feat"], src["feat"]).argmax(axis=1)
    save_points(os.path.join(args.out_dir, "correspondence_source.ply"), src_xyz, src_pc)
    save_target_colored(
        os.path.join(args.out_dir, "correspondence_target.ply"),
        args.target_mesh,
        dst_xyz,
        src_pc[t2s],
    )
    # albedo transfer: through features only, and through the registration (nearest aligned source
    # point, only where the source covers the target)
    src_col = src["colors"]
    grey = np.full((len(dst_xyz), 3), 170, np.uint8)
    reg_col = grey.copy()
    d_t, near_src = cKDTree(aligned).query(dst_xyz)
    covered = d_t < args.icp_dist * res["diag"]
    if src_col is not None:
        reg_col[covered] = src_col[near_src[covered]]  # point-level version for the preview
        if args.target_mesh:
            import trimesh

            aligned_paint = (
                aligned if res["aligned_piecewise"] is None else res["aligned_piecewise"]
            )
            # every vertex of the source mesh with its photo colour, moved like the sampled points
            full_xyz, full_col = None, None
            if args.source_mesh and not args.no_full_colors:
                m_full = trimesh.load(args.source_mesh, process=False, force="mesh")
                if m_full.visual.kind == "vertex":
                    v_full = np.asarray(m_full.vertices, dtype=np.float64)
                    full_col = np.asarray(m_full.visual.vertex_colors[:, :3], dtype=np.uint8)
                    if res["aligned_piecewise"] is None:
                        full_xyz = transform(apply_depth(v_full, a), s, R, t)
                    else:
                        full_xyz = apply_piecewise(
                            v_full,
                            aligned,
                            res["piecewise_info"],
                            s,
                            R,
                            t,
                            a,
                            sigma=res["options"]["blend_sigma"] * res["diag"],
                        )
            paint = paint_target(
                args.target_mesh,
                dst_xyz,
                dst["feat"],
                src["feat"],
                aligned_paint,
                src_col,
                res["diag"],
                subdiv=args.paint_subdiv,
                delta=args.paint_delta,
                cover_dist=args.icp_dist,
                src_parts=res["src_parts"],
                dst_parts=res["dst_parts"],
                part_penalty=args.part_penalty,
                feature_weight=args.feature_weight,
                mirror=not args.no_mirror,
                full_src_xyz=full_xyz,
                full_src_colors=full_col,
            )
            V, F = paint["V"], paint["F"]
            for name, key in [
                ("target_from_source_colors.ply", "feature"),
                ("target_from_source_colors_registered.ply", "registered"),
                ("target_from_source_colors_mirrored.ply", "mirror"),
                ("target_from_source_colors_combined.ply", "combined"),
            ]:
                trimesh.Trimesh(
                    vertices=V, faces=F, vertex_colors=paint[key], process=False
                ).export(os.path.join(args.out_dir, name))
            plane_err = None if paint["plane"] is None else round(paint["plane"][2], 4)
            print(
                f"painted target: {len(V)} vertices, {paint['covered'].mean():.0%} covered, "
                f"{paint['mirrored'].mean():.0%} mirrored (symmetry plane error {plane_err})"
            )
        else:
            save_points(
                os.path.join(args.out_dir, "target_from_source_colors.ply"), dst_xyz, src_col[t2s]
            )
            save_points(
                os.path.join(args.out_dir, "target_from_source_colors_registered.ply"),
                dst_xyz,
                reg_col,
            )
    if res["aligned_piecewise"] is not None and args.source_mesh:
        import trimesh

        m = trimesh.load(args.source_mesh, process=False, force="mesh")
        m.vertices = apply_piecewise(
            np.asarray(m.vertices, dtype=np.float64),
            aligned,
            res["piecewise_info"],
            s,
            R,
            t,
            a,
            sigma=res["options"]["blend_sigma"] * res["diag"],
        )
        m.export(os.path.join(args.out_dir, "aligned_source_piecewise.ply"))
        for p, tr in res["piecewise_info"].items():
            print(
                f"part {p}: {tr['n_src']} source points, "
                f"{'accepted' if tr['accepted'] else 'kept global'}, scale {tr['s']:.3f}, "
                f"rotation {rotation_angle_deg(tr['R'], np.eye(3)):.1f} deg, "
                f"rms {tr['rms_before']} -> {tr['rms_after']}, "
                f"cover {tr['cover_before']} -> {tr['cover_after']}"
            )

    red = np.full((len(src_xyz), 3), (220, 60, 60), np.uint8)
    s0, R1, t1, a1 = res["baseline"]
    sc, Rc, tc = res["coarse"]
    centred = transform(src_xyz, s0, np.eye(3), dst_xyz.mean(0) - s0 * src_xyz.mean(0))
    coarse = transform(src_xyz, sc, Rc, tc)
    base = transform(apply_depth(src_xyz, a1), s0, R1, t1)
    panels = [
        ("source (red) and target, centred", [(dst_xyz, grey, 1), (centred, red, 1)]),
        ("after RANSAC on feature matches", [(dst_xyz, grey, 1), (coarse, red, 1)]),
        (f"selected after {args.refine} refinement", [(dst_xyz, grey, 1), (aligned, red, 1)]),
        ("baseline ICP without features", [(dst_xyz, grey, 1), (base, red, 1)]),
        ("correspondence colours: source", [(src_xyz, src_pc, 2)]),
        ("correspondence colours: target", [(dst_xyz, src_pc[t2s], 2)]),
    ]
    if src_col is not None:
        panels.append(("target with source colours (features)", [(dst_xyz, src_col[t2s], 2)]))
        panels.append(("target with source colours (registered)", [(dst_xyz, reg_col, 2)]))
    save_preview(os.path.join(args.out_dir, "preview.png"), panels)
    print("saved", args.out_dir)


if __name__ == "__main__":
    main()
