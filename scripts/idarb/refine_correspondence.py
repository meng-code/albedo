"""
Refine the nearest-neighbour correspondence from a full target mesh to a partial source (relief):
keep the semantics of the Diff3F features, make the match consistent along the surface.

Nearest neighbour alone matches every target vertex on its own, so neighbouring target vertices
jump between distant source points: streaks in the transferred texture, errors inside parts. Three
additions, each optional:

  anchors    reliable pairs. Either manual landmarks (two MeshLab PickPoints .pp files, or a JSON
             with "source" and "target" lists of points or vertex indices, same order), or
             automatically the mutual nearest neighbours (the target point's best source point has
             the target point as its best match), thinned to a spread subset.
  prior      a smooth map from target positions to source positions fitted on the anchors (affine
             plus Gaussian radial basis functions, ridge-regularised). Every target vertex then
             prefers source candidates near its predicted position: among its k most similar source
             points it minimises (1 - similarity) + w_prior * (distance / radius)^2.
  smoothing  a few passes in which every target vertex re-chooses among its candidates with an
             extra penalty on the distance to the mean matched position of its mesh neighbours, so
             the match varies smoothly along the surface.

Manual landmarks are kept exactly (their target vertices take their source vertices). Writes an npz
with t2s (a source mesh vertex per target mesh vertex: the table transfer_texture.py and
correspondence_maps.py read with --correspondence) and prints, with --gt on the synthetic pairs,
the fraction of target vertices matched within 1 % / 5 % of the diagonal of their true source
vertex, next to the plain nearest neighbour.

Usage (repository root, mesh environment):
    python scripts/idarb/refine_correspondence.py \
        --source outputs/diff3f/features/cat_partial.npz \
        --source_mesh outputs/mesh_depth/cat/relief.ply \
        --target outputs/diff3f/features/lion_full.npz \
        --target_mesh third_party/Diff3F/meshes/lion.off \
        --out outputs/diff3f/04_texture_refined/same_category/cat_lion/refined.npz
"""

import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import register_partial as rp  # noqa: E402
from correspondence_maps import mesh_features  # noqa: E402


def topk_candidates(fd, fs, k, chunk=2048):
    """For every row of fd, the k most similar rows of fs (unit features) and the similarities."""
    k = min(k, fs.shape[0])
    idx = np.empty((len(fd), k), dtype=np.int64)
    sim = np.empty((len(fd), k), dtype=np.float32)
    for i in range(0, len(fd), chunk):
        S = fd[i : i + chunk] @ fs.T
        part = np.argpartition(-S, k - 1, axis=1)[:, :k]
        idx[i : i + chunk] = part
        sim[i : i + chunk] = np.take_along_axis(S, part, axis=1)
    return idx, sim


def mutual_pairs(fd, fs):
    """Pairs (target row, source row) that are each other's nearest neighbour, with similarity."""
    t2s, s_t = topk_candidates(fd, fs, 1)
    s2t, _ = topk_candidates(fs, fd, 1)
    t2s, s2t = t2s[:, 0], s2t[:, 0]
    t = np.nonzero(s2t[t2s] == np.arange(len(fd)))[0]
    return t, t2s[t], s_t[t, 0]


def farthest_points(X, n, seed=0):
    """Indices of n points of X spread out by farthest-point sampling."""
    if n >= len(X):
        return np.arange(len(X))
    rng = np.random.default_rng(seed)
    chosen = [int(rng.integers(len(X)))]
    d = np.linalg.norm(X - X[chosen[0]], axis=1)
    for _ in range(n - 1):
        j = int(d.argmax())
        chosen.append(j)
        d = np.minimum(d, np.linalg.norm(X - X[j], axis=1))
    return np.array(chosen)


def fit_prior(X, Y, sigma, ridge=1e-3):
    """A smooth map X -> Y through anchor pairs: an affine part (least squares, or a similarity
    transform when there are fewer than 8 anchors) plus Gaussian radial basis functions on the
    residuals, with ridge regularisation. Returns predict(Q)."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    n = len(X)
    if n >= 8:
        Xh = np.concatenate([X, np.ones((n, 1))], axis=1)
        A = np.linalg.solve(Xh.T @ Xh + ridge * np.eye(4), Xh.T @ Y)

        def base(Q):
            return np.concatenate([Q, np.ones((len(Q), 1))], axis=1) @ A

    else:
        s, R, t = rp.umeyama(X, Y, with_scale=n >= 3)

        def base(Q):
            return rp.transform(Q, s, R, t)

    res = Y - base(X)
    D = np.linalg.norm(X[:, None] - X[None], axis=2)
    K = np.exp(-(D**2) / (2 * sigma**2))
    W = np.linalg.solve(K + ridge * np.eye(n), res)

    def predict(Q):
        Dq = np.linalg.norm(Q[:, None] - X[None], axis=2)
        return base(Q) + np.exp(-(Dq**2) / (2 * sigma**2)) @ W

    return predict


def vertex_adjacency(F, n):
    """Row-normalised vertex adjacency (mean over the one-ring) as a sparse matrix."""
    i = np.concatenate([F[:, 0], F[:, 1], F[:, 2], F[:, 1], F[:, 2], F[:, 0]])
    j = np.concatenate([F[:, 1], F[:, 2], F[:, 0], F[:, 0], F[:, 1], F[:, 2]])
    A = sp.coo_matrix((np.ones(len(i)), (i, j)), shape=(n, n)).tocsr()
    A.data[:] = 1.0
    deg = np.asarray(A.sum(axis=1)).ravel()
    return sp.diags(1.0 / np.maximum(deg, 1)) @ A


def refine(
    Vd,
    Fd,
    fd,
    src_xyz,
    fs,
    anchors=None,
    k=64,
    radius=0.05,
    w_prior=1.0,
    iters=2,
    radius_s=0.05,
    w_smooth=0.3,
    sigma=0.25,
):
    """Target vertex -> source sample. fd: a feature per target vertex; fs, src_xyz: the source
    samples. anchors: (target positions, source positions) or None. radius, radius_s and sigma are
    fractions of the source / target bounding-box diagonals."""
    diag_s = float(np.linalg.norm(src_xyz.max(axis=0) - src_xyz.min(axis=0)))
    diag_d = float(np.linalg.norm(Vd.max(axis=0) - Vd.min(axis=0)))
    cand, sim = topk_candidates(fd, fs, k)
    C = src_xyz[cand]  # (n, k, 3)
    cost = 1.0 - sim.astype(np.float64)
    if anchors is not None and len(anchors[0]) >= 3:
        if len(anchors[0]) == len(Vd) and anchors[0] is Vd:  # a dense prior, one per vertex
            pred = anchors[1]
        else:
            pred = fit_prior(anchors[0], anchors[1], sigma * diag_d)(Vd)
        d = np.linalg.norm(C - pred[:, None], axis=2) / (radius * diag_s)
        cost = cost + w_prior * d**2
    rows = np.arange(len(fd))
    choice = cost.argmin(axis=1)
    if iters:
        A = vertex_adjacency(Fd, len(Vd))
        for _ in range(iters):
            M = A @ C[rows, choice]
            d = np.linalg.norm(C - M[:, None], axis=2) / (radius_s * diag_s)
            choice = (cost + w_smooth * d**2).argmin(axis=1)
    return cand[rows, choice], sim[rows, choice]


def smooth_field(Vd, Fd, S, w, lam=1.0, order=1):
    """A source-space position per target vertex that follows the matched positions S (n,3) where
    the confidence w (n,) is high and varies smoothly along the target surface elsewhere. Minimises
    sum_i w_i |P_i - S_i|^2 + lam |L P|^2, L the graph Laplacian of the mesh (order 2: the
    bi-Laplacian, smoother). The result is continuous, so the texture coordinates derived from it
    have no jumps inside a triangle."""
    n = len(Vd)
    i = np.concatenate([Fd[:, 0], Fd[:, 1], Fd[:, 2], Fd[:, 1], Fd[:, 2], Fd[:, 0]])
    j = np.concatenate([Fd[:, 1], Fd[:, 2], Fd[:, 0], Fd[:, 0], Fd[:, 1], Fd[:, 2]])
    A = sp.coo_matrix((np.ones(len(i)), (i, j)), shape=(n, n)).tocsr()
    A.data[:] = 1.0
    L = sp.diags(np.asarray(A.sum(axis=1)).ravel()) - A
    if order == 2:
        L = L.T @ L
    W = sp.diags(np.asarray(w, dtype=np.float64))
    P = spsolve((W + lam * L).tocsc(), W @ np.asarray(S, dtype=np.float64))
    return np.asarray(P)


def arap_field(Vd, Fd, S, w, P0, lam=1.0, iters=5, scale_range=(0.3, 3.0)):
    """As-rigid-as-possible field: like smooth_field, but the smoothness term asks every vertex's
    one-ring to move by a similarity transform (rotation and scale, never a reflection) instead of
    merely varying smoothly. Alternates a per-vertex Procrustes fit of that transform with a sparse
    solve for the positions. Minimises
        sum_i w_i |P_i - S_i|^2 + lam * sum_edges |(P_j - P_i) - M_ij (x_j - x_i)|^2,
    M_ij the mean of the two end vertices' similarity transforms. Starts from P0."""
    n = len(Vd)
    i = np.concatenate([Fd[:, 0], Fd[:, 1], Fd[:, 2], Fd[:, 1], Fd[:, 2], Fd[:, 0]])
    j = np.concatenate([Fd[:, 1], Fd[:, 2], Fd[:, 0], Fd[:, 0], Fd[:, 1], Fd[:, 2]])
    A = sp.coo_matrix((np.ones(len(i)), (i, j)), shape=(n, n)).tocsr()
    A.data[:] = 1.0
    A = A.tocoo()
    ei, ej = A.row, A.col  # every directed edge once
    L = sp.diags(np.asarray(A.sum(axis=1)).ravel()) - A.tocsr()
    W = sp.diags(np.asarray(w, dtype=np.float64))
    M = (W + lam * L).tocsc()
    dx = Vd[ej] - Vd[ei]
    norm2 = np.zeros(n)
    np.add.at(norm2, ei, (dx**2).sum(axis=1))
    # the scale range is relative to the global size ratio of the two shapes (a target in
    # different units must not be forced to keep its own edge lengths)
    ratio = float(np.linalg.norm(S.max(axis=0) - S.min(axis=0)))
    ratio /= max(float(np.linalg.norm(Vd.max(axis=0) - Vd.min(axis=0))), 1e-12)
    scale_range = (scale_range[0] * ratio, scale_range[1] * ratio)
    P = P0.copy()
    for _ in range(iters):
        dp = P[ej] - P[ei]
        # per-vertex covariance of its one-ring: C_i = sum_j dx dp^T
        C = np.zeros((n, 3, 3))
        np.add.at(C, ei, dx[:, :, None] * dp[:, None, :])
        U, sig, Vt = np.linalg.svd(C)
        R = np.transpose(Vt, (0, 2, 1)) @ np.transpose(U, (0, 2, 1))
        flip = np.linalg.det(R) < 0
        if flip.any():  # a reflection fitted best: take the nearest proper rotation
            Vt_f = Vt[flip].copy()
            Vt_f[:, 2, :] *= -1
            R[flip] = np.transpose(Vt_f, (0, 2, 1)) @ np.transpose(U[flip], (0, 2, 1))
            sig[flip, 2] *= -1
        s = np.clip(sig.sum(axis=1) / np.maximum(norm2, 1e-12), *scale_range)
        T = s[:, None, None] * R  # (n,3,3) similarity of every vertex
        d = np.einsum("eab,eb->ea", 0.5 * (T[ei] + T[ej]), dx)  # M_ij (x_j - x_i)
        B = np.zeros((n, 3))
        np.add.at(B, ei, -d)
        P = np.asarray(spsolve(M, W @ S + lam * B))
    return P


def bad_triangles(Vd, Fd, P, view, max_distortion=10.0):
    """Triangles whose image under the field P (projected on the relief plane x, y) is mirrored
    with respect to their winding on screen seen from `view`, or stretched more than
    `max_distortion`-fold: the triangles that would show as reversed patches or streaks."""
    from transfer_texture import view_frame
    from uv_metrics import triangle_maps

    R = view_frame(view)
    S = Vd[Fd] @ R.T
    s1, s2 = S[:, 1, :2] - S[:, 0, :2], S[:, 2, :2] - S[:, 0, :2]
    sign_screen = np.sign(s1[:, 0] * s2[:, 1] - s1[:, 1] * s2[:, 0])
    Q = P[Fd][:, :, :2]
    q1, q2 = Q[:, 1] - Q[:, 0], Q[:, 2] - Q[:, 0]
    sign_field = np.sign(q1[:, 0] * q2[:, 1] - q1[:, 1] * q2[:, 0])
    flipped = (sign_field != sign_screen) & (sign_screen != 0)
    uvc = np.concatenate([Q, np.zeros_like(Q[:, :, :1])], axis=2)[:, :, :2]
    sv, _, _ = triangle_maps(Vd, Fd, uvc, (1.0, 1.0))
    distorted = sv[:, 0] / np.maximum(sv[:, 1], 1e-12) > max_distortion
    return flipped | distorted, flipped, distorted


def confidence(sim, Vd, anchors=None, landmarks_t=None):
    """Data weight per target vertex: the similarity of its match scaled between its 10th and
    90th percentiles, raised for the vertices at anchors and fixed high at landmarks."""
    lo, hi = np.percentile(sim, 10), np.percentile(sim, 90)
    w = np.clip((sim - lo) / max(hi - lo, 1e-6), 0.05, 1.0)
    if anchors is not None and len(anchors[0]):
        w[cKDTree(Vd).query(anchors[0])[1]] = 3.0
    if landmarks_t is not None and len(landmarks_t):
        w[landmarks_t] = 10.0
    return w


def read_pp(path):
    """Points of a MeshLab PickPoints (.pp) file, in file order."""
    pts = []
    for p in ET.parse(path).getroot().iter("point"):
        if p.get("active", "1") == "1":
            pts.append([float(p.get(a)) for a in ("x", "y", "z")])
    return np.array(pts, dtype=np.float64)


def load_landmarks(args, Vs, Vd):
    """Landmark pairs as (source vertex indices, target vertex indices), or None."""
    if args.landmarks_source and args.landmarks_target:
        ps, pt = read_pp(args.landmarks_source), read_pp(args.landmarks_target)
    elif args.landmarks:
        with open(args.landmarks) as f:
            lm = json.load(f)
        ps, pt = np.asarray(lm["source"]), np.asarray(lm["target"])
        if ps.ndim == 1:  # vertex indices
            return ps.astype(np.int64), pt.astype(np.int64)
    else:
        return None
    if len(ps) != len(pt):
        raise ValueError("source and target landmarks must come in equal numbers, same order")
    return cKDTree(Vs).query(ps)[1], cKDTree(Vd).query(pt)[1]


def gt_landmarks(gt, Vd, n, seed=0):
    """n well-spread source vertices with their true target vertex, from a synthetic pair's GT:
    (source vertex indices, target vertex indices)."""
    orig = gt["orig_idx"]  # source vertex -> target vertex
    s_sel = farthest_points(Vd[orig], n, seed)
    return s_sel.astype(np.int64), orig[s_sel].astype(np.int64)


def evaluate(t2s, Vd, gt):
    orig = gt["orig_idx"]
    diag = float(np.linalg.norm(Vd.max(axis=0) - Vd.min(axis=0)))
    has = np.zeros(len(Vd), dtype=bool)
    has[orig] = True
    err = np.linalg.norm(Vd[orig[t2s]] - Vd, axis=1) / diag
    return dict(
        acc_1pct=round(float((err[has] < 0.01).mean()), 4),
        acc_5pct=round(float((err[has] < 0.05).mean()), 4),
        err_median=round(float(np.median(err[has])), 4),
    )


def evaluate_positions(P, Vs, Vd, gt):
    """Like evaluate, for continuous source-space positions P (n,3): the distance to the true
    source vertex of each target vertex, in units of the target's diagonal (full = s R partial + t,
    so source-space distances are target distances divided by gt['s'])."""
    orig = gt["orig_idx"]
    s_of_t = -np.ones(len(Vd), dtype=np.int64)
    s_of_t[orig] = np.arange(len(orig))
    has = s_of_t >= 0
    diag = float(np.linalg.norm(Vd.max(axis=0) - Vd.min(axis=0))) / float(gt["s"])
    err = np.linalg.norm(P[has] - Vs[s_of_t[has]], axis=1) / diag
    return dict(
        acc_1pct=round(float((err < 0.01).mean()), 4),
        acc_5pct=round(float((err < 0.05).mean()), 4),
        err_median=round(float(np.median(err)), 4),
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--source", required=True, help="features of the partial mesh (.npz)")
    ap.add_argument("--source_mesh", required=True)
    ap.add_argument("--target", required=True, help="features of the full mesh (.npz)")
    ap.add_argument("--target_mesh", required=True)
    ap.add_argument("--feature", choices=["full", "diffusion", "dino"], default="full")
    ap.add_argument("--landmarks", help="JSON with source and target points or vertex indices")
    ap.add_argument("--landmarks_source", help="MeshLab .pp file picked on the source mesh")
    ap.add_argument("--landmarks_target", help="MeshLab .pp file picked on the target mesh")
    ap.add_argument("--gt", help="_gt.npz of a synthetic pair: evaluation (and --gt_landmarks)")
    ap.add_argument("--gt_landmarks", type=int, default=0, help="simulate n manual landmarks")
    ap.add_argument(
        "--anchors",
        choices=["mutual", "none"],
        default="mutual",
        help="automatic anchors when no landmarks are given",
    )
    ap.add_argument(
        "--add_mutual",
        action="store_true",
        help="with landmarks: also use the mutual nearest neighbours as anchors",
    )
    ap.add_argument(
        "--prior_transform",
        help="transform.json of register_partial.py: use the rigid registration (inverted) as the "
        "position prior instead of a map fitted on anchors; keeps left and right apart",
    )
    ap.add_argument("--max_anchors", type=int, default=400)
    ap.add_argument("--k", type=int, default=64, help="candidates per target vertex")
    ap.add_argument(
        "--radius", type=float, default=0.05, help="prior window, of the source diagonal"
    )
    ap.add_argument("--w_prior", type=float, default=1.0)
    ap.add_argument("--iters", type=int, default=2, help="smoothing passes (0: none)")
    ap.add_argument("--radius_s", type=float, default=0.05, help="smoothing scale, of the diagonal")
    ap.add_argument("--w_smooth", type=float, default=0.3)
    ap.add_argument(
        "--field",
        type=float,
        default=0.0,
        help="smoothness weight of a continuous source-position field fitted to the matches "
        "(0: off); the npz then also holds src_pos, which transfer_texture.py prefers",
    )
    ap.add_argument("--field_order", type=int, choices=[1, 2], default=1)
    ap.add_argument(
        "--arap",
        type=int,
        default=0,
        help="iterations of the as-rigid-as-possible field after the smooth field (0: off): the "
        "smoothness term then asks every one-ring to move by a rotation and a scale, never a "
        "reflection, which is what a flipped triangle is",
    )
    ap.add_argument("--arap_weight", type=float, default=None, help="default: --field")
    ap.add_argument(
        "--field_iters",
        type=int,
        default=0,
        help="re-solve the field this many times, each time lowering the confidence of the "
        "vertices of triangles that came out mirrored or over-stretched (needs --target_view)",
    )
    ap.add_argument(
        "--target_view",
        type=float,
        nargs=3,
        default=None,
        help="direction of the relief's camera in target space (the third column of the "
        "registration's rotation), for the orientation test of --field_iters",
    )
    ap.add_argument("--out", required=True, help="npz with t2s")
    args = ap.parse_args()

    src = rp.load_features(args.source, args.feature)
    Vs, Fs, _, _ = mesh_features(args.source, args.source_mesh, args.feature)
    Vd, Fd, fd, _ = mesh_features(args.target, args.target_mesh, args.feature)
    dst = rp.load_features(args.target, args.feature)
    gt = dict(np.load(args.gt)) if args.gt else None

    lm = load_landmarks(args, Vs, Vd)
    if lm is None and gt is not None and args.gt_landmarks:
        lm = gt_landmarks(gt, Vd, args.gt_landmarks)
    anchors, kind = None, "no anchors"
    if args.prior_transform:
        # the registration maps source to target: y = s R (depth a applied to x) + t; invert it to
        # predict a source position for every target vertex and hand those over as dense anchors
        with open(args.prior_transform) as f:
            tr = json.load(f)
        R = np.array(tr["rotation"], dtype=np.float64)
        x = ((Vd - np.array(tr["translation"])) / tr["scale"]) @ R
        x[:, 2] /= tr.get("depth_factor", 1.0)
        anchors = (Vd, x)
        kind = "rigid registration prior"
    if lm is not None:
        s_lm, t_lm = lm
        anchors = (Vd[t_lm], Vs[s_lm])
        kind = f"{len(t_lm)} landmarks"
    if (lm is None and args.anchors == "mutual") or (lm is not None and args.add_mutual):
        t, s, w = mutual_pairs(dst["feat"], src["feat"])
        keep = farthest_points(dst["xyz"][t], args.max_anchors)
        mutual = (dst["xyz"][t[keep]], src["xyz"][s[keep]])
        if anchors is None:
            anchors = mutual
        else:
            anchors = (
                np.concatenate([anchors[0], mutual[0]]),
                np.concatenate([anchors[1], mutual[1]]),
            )
        kind += f" + {len(keep)} mutual anchors (of {len(t)} mutual pairs)"

    t2s_sample, sim = refine(
        Vd,
        Fd,
        fd,
        src["xyz"],
        src["feat"],
        anchors,
        k=args.k,
        radius=args.radius,
        w_prior=args.w_prior,
        iters=args.iters,
        radius_s=args.radius_s,
        w_smooth=args.w_smooth,
    )
    t2s = src["idx"][t2s_sample].astype(np.int64)
    if lm is not None:
        t2s[t_lm] = s_lm
    stats = dict(anchors=kind, k=args.k, iters=args.iters, field=args.field)
    extra = {}
    if args.field > 0:
        S = Vs[t2s]
        w = confidence(sim, Vd, anchors, t_lm if lm is not None else None)
        P = smooth_field(Vd, Fd, S, w, args.field, args.field_order)
        if args.arap:
            lam = args.arap_weight if args.arap_weight is not None else args.field
            P = arap_field(Vd, Fd, S, w, P, lam, args.arap)
        if args.field_iters and args.target_view is not None:
            # flipped or over-stretched triangles: take the data weight away from their vertices
            # (and the ring around them) and let the field, rigid if --arap, fill them in from
            # their neighbours
            protected = np.zeros(len(Vd), dtype=bool)
            if lm is not None:
                protected[t_lm] = True
            A1 = vertex_adjacency(Fd, len(Vd))
            history = []
            for _ in range(args.field_iters):
                bad, flipped, distorted = bad_triangles(Vd, Fd, P, args.target_view)
                history.append([int(flipped.sum()), int(distorted.sum())])
                mark = np.zeros(len(Vd))
                mark[np.unique(Fd[bad])] = 1.0
                mark = np.maximum(mark, A1 @ mark > 0)  # one ring further
                w = np.where((mark > 0) & ~protected, 0.02, w)
                P = smooth_field(Vd, Fd, S, w, args.field, args.field_order)
                if args.arap:
                    P = arap_field(Vd, Fd, S, w, P, lam, args.arap)
            bad, flipped, distorted = bad_triangles(Vd, Fd, P, args.target_view)
            history.append([int(flipped.sum()), int(distorted.sum())])
            stats["field_iters_flipped_distorted"] = history
        extra["src_pos"] = P.astype(np.float32)
        t2s_field = src["idx"][cKDTree(src["xyz"]).query(P)[1]].astype(np.int64)
    if gt is not None:
        nn = src["idx"][topk_candidates(fd, src["feat"], 1)[0][:, 0]]
        stats["nearest_neighbour"] = evaluate(nn, Vd, gt)
        stats["refined"] = evaluate(t2s, Vd, gt)
        if args.field > 0:
            stats["field_positions"] = evaluate_positions(P, Vs, Vd, gt)
            stats["field_snapped"] = evaluate(t2s_field, Vd, gt)
    print(json.dumps(stats))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(
        args.out, t2s=t2s, options=json.dumps(vars(args)), stats=json.dumps(stats), **extra
    )


if __name__ == "__main__":
    main()
