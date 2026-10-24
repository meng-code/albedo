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


def similarities(fd, fs, cand, chunk=2048):
    """Similarity of every row of fd to its own candidate rows of fs (cand: (n, m) indices)."""
    out = np.empty(cand.shape, dtype=np.float32)
    for i in range(0, len(fd), chunk):
        out[i : i + chunk] = np.einsum("nd,nmd->nm", fd[i : i + chunk], fs[cand[i : i + chunk]])
    return out


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


def filter_anchors(T, S, mode, thr, seed=0):
    """Keep the anchors that agree with the others. mode "rigid": a similarity transform fitted
    by RANSAC on 4-anchor samples, anchors farther than thr (fraction of the source diagonal)
    from it are dropped; mode "loo": every anchor is predicted from the others by the smooth
    prior (fit_prior), and anchors predicted farther than thr away are dropped. Returns the
    mask of kept anchors. Left-right swapped mutual pairs (the features cannot tell a left ear
    from a right one) are the anchors this removes."""
    n = len(T)
    if n < 6:
        return np.ones(n, dtype=bool)
    diag_s = float(np.linalg.norm(S.max(axis=0) - S.min(axis=0)))
    if mode == "rigid":
        rng = np.random.default_rng(seed)
        best = None
        for _ in range(1500):
            i = rng.choice(n, 4, replace=False)
            s, R, t = rp.umeyama(T[i], S[i])
            res = np.linalg.norm(rp.transform(T, s, R, t) - S, axis=1)
            inl = res < thr * diag_s
            if best is None or inl.sum() > best.sum():
                best = inl
        if best.sum() >= 4:
            s, R, t = rp.umeyama(T[best], S[best])
            best = np.linalg.norm(rp.transform(T, s, R, t) - S, axis=1) < thr * diag_s
        return best
    if mode == "loo":
        diag_t = float(np.linalg.norm(T.max(axis=0) - T.min(axis=0)))
        keep = np.ones(n, dtype=bool)
        res = np.empty(n)
        for i in range(n):
            m = np.arange(n) != i
            res[i] = np.linalg.norm(fit_prior(T[m], S[m], 0.25 * diag_t)(T[i : i + 1])[0] - S[i])
        keep = res < thr * diag_s
        return keep
    raise ValueError(mode)


def confident_matches(fd, fs, Vd, src_xyz, quantile=0.7, mutual_dist=0.05):
    """Target vertices whose nearest neighbour can be trusted as it is: similarity in the top
    (1 - quantile) of the mesh and the source sample's own best target vertex within mutual_dist
    of the target's diagonal (a soft mutual test). On the keypoint benchmark these matches put
    65 % of the keypoints within 2 % of the diagonal (all matches 54 %, the refined map 40 %),
    so they are kept fixed and only the rest is refined and smoothed. Returns the mask and the
    nearest-neighbour sample index per vertex."""
    cand, sim = topk_candidates(fd, fs, 1)
    nn = cand[:, 0]
    back = topk_candidates(fs, fd, 1)[0][:, 0]  # every source sample's best target vertex
    diag_d = float(np.linalg.norm(Vd.max(axis=0) - Vd.min(axis=0)))
    d_back = np.linalg.norm(Vd[back[nn]] - Vd, axis=1) / diag_d
    keep = (sim[:, 0] >= np.quantile(sim[:, 0], quantile)) & (d_back < mutual_dist)
    return keep, nn


def anchor_view(T, S, thr=0.1, seed=0):
    """The relief camera's direction in target space, from the anchors: a RANSAC similarity
    transform y = s R x + t from target to source positions, then the source axis +z (the relief
    camera) pulled back by R. Replaces a texture camera borrowed from a registration (which, for
    a cross-category pair, came from another pair, and for one same-category pair looked at the
    target from behind)."""
    n = len(T)
    diag_s = float(np.linalg.norm(S.max(axis=0) - S.min(axis=0)))
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(3000 if n >= 6 else 1):
        i = rng.choice(n, min(4, n), replace=False)
        s, R, t = rp.umeyama(T[i], S[i])
        res = np.linalg.norm(rp.transform(T, s, R, t) - S, axis=1)
        inl = res < thr * diag_s
        if best is None or inl.sum() > best.sum():
            best = inl
    if best.sum() >= 4:
        s, R, t = rp.umeyama(T[best], S[best])
    else:
        s, R, t = rp.umeyama(T, S)
    v = R.T @ np.array([0.0, 0.0, 1.0])
    return v / np.linalg.norm(v), int(best.sum())


def vertex_adjacency(F, n):
    """Row-normalised vertex adjacency (mean over the one-ring) as a sparse matrix."""
    i = np.concatenate([F[:, 0], F[:, 1], F[:, 2], F[:, 1], F[:, 2], F[:, 0]])
    j = np.concatenate([F[:, 1], F[:, 2], F[:, 0], F[:, 0], F[:, 1], F[:, 2]])
    A = sp.coo_matrix((np.ones(len(i)), (i, j)), shape=(n, n)).tocsr()
    A.data[:] = 1.0
    deg = np.asarray(A.sum(axis=1)).ravel()
    return sp.diags(1.0 / np.maximum(deg, 1)) @ A


def face_fans(Vd, Fd, tview, min_cos=0.3, max_faces=10):
    """For every target vertex, the faces around it that face the texture camera (normal from
    the face winding, cosine to the view above min_cos): the other two corners in the face's
    cyclic order (J, K, shape (n, m), -1 padded) and the sign of the face's winding on screen
    (SGN, (n, m), 0 padded). The texture of such a face must wind the same way on the relief
    image, so a candidate source point that reverses the winding of a vertex's fan mirrors the
    texture there."""
    from transfer_texture import view_frame

    tview = np.asarray(tview, dtype=np.float64)
    tview = tview / np.linalg.norm(tview)
    R = view_frame(tview)
    S = Vd @ R.T
    nrm = np.cross(Vd[Fd[:, 1]] - Vd[Fd[:, 0]], Vd[Fd[:, 2]] - Vd[Fd[:, 0]])
    nrm = nrm / np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    front = np.nonzero(nrm @ tview > min_cos)[0]
    s1 = S[Fd[front, 1], :2] - S[Fd[front, 0], :2]
    s2 = S[Fd[front, 2], :2] - S[Fd[front, 0], :2]
    sign = np.sign(s1[:, 0] * s2[:, 1] - s1[:, 1] * s2[:, 0]).astype(np.int8)
    n = len(Vd)
    J = -np.ones((n, max_faces), dtype=np.int64)
    K = -np.ones((n, max_faces), dtype=np.int64)
    SGN = np.zeros((n, max_faces), dtype=np.int8)
    count = np.zeros(n, dtype=np.int64)
    for r in range(3):
        i = Fd[front, r]
        j = Fd[front, (r + 1) % 3]
        k = Fd[front, (r + 2) % 3]
        for f in range(len(front)):
            c = count[i[f]]
            if c < max_faces:
                J[i[f], c], K[i[f], c], SGN[i[f], c] = j[f], k[f], sign[f]
                count[i[f]] = c + 1
    return J, K, SGN


def mirrored_fraction(C, P, J, K, SGN, chunk=1024):
    """For every vertex i and candidate c, the fraction of i's fan faces whose winding on the
    relief image (x, y of the source positions: the two other corners at their current
    positions P, the vertex at the candidate C[i, c]) disagrees with their winding on screen."""
    n, k = C.shape[:2]
    out = np.zeros((n, k), dtype=np.float32)
    valid = J >= 0
    cnt = np.maximum(valid.sum(axis=1), 1)
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        Pj = P[np.maximum(J[a:b], 0)][:, :, :2]  # (c, m, 2)
        Pk = P[np.maximum(K[a:b], 0)][:, :, :2]
        Cc = C[a:b, :, None, :2]  # (c, k, 1, 2)
        u = Pj[:, None] - Cc  # (c, k, m, 2)
        v = Pk[:, None] - Cc
        tex = u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]
        bad = (np.sign(tex).astype(np.int8) != SGN[a:b, None, :]) & valid[a:b, None, :]
        out[a:b] = bad.sum(axis=2) / cnt[a:b, None]
    return out


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
    pred=None,
    k_prior=0,
    parts=None,
    w_part=0.5,
    tview=None,
    w_flip=0.0,
    fixed=None,
):
    """Target vertex -> source sample. fd: a feature per target vertex; fs, src_xyz: the source
    samples. anchors: (target positions, source positions) or None; pred: a predicted source
    position per target vertex that replaces the map fitted on the anchors. radius, radius_s and
    sigma are fractions of the source / target bounding-box diagonals. k_prior > 0 adds the
    k_prior source samples nearest to the predicted position to the candidates, so a target part
    whose feature matches all sit far away (the camel's hump, which looks like a face to the
    features) can settle on a less similar point where the map around it says it belongs.
    parts: (source part per source sample, matched source part per target vertex or -1) adds
    w_part to candidates outside the part the vertex's own part is matched with. tview and
    w_flip > 0 add, in the smoothing passes, w_flip times the fraction of the vertex's camera-
    facing fan that a candidate would mirror on the relief image (the two other corners at their
    current matches), so a left-right symmetric choice that reverses the texture is avoided.
    fixed: (mask, source sample per vertex) of matches kept as they are, outside the window and
    the smoothing."""
    diag_s = float(np.linalg.norm(src_xyz.max(axis=0) - src_xyz.min(axis=0)))
    diag_d = float(np.linalg.norm(Vd.max(axis=0) - Vd.min(axis=0)))
    cand, sim = topk_candidates(fd, fs, k)
    if pred is None and anchors is not None and len(anchors[0]) >= 3:
        pred = fit_prior(anchors[0], anchors[1], sigma * diag_d)(Vd)
    if pred is not None and k_prior > 0:
        near = cKDTree(src_xyz).query(pred, min(k_prior, len(src_xyz)))[1].reshape(len(fd), -1)
        cand = np.concatenate([cand, near], axis=1)
        sim = np.concatenate([sim, similarities(fd, fs, near)], axis=1)
    C = src_xyz[cand]  # (n, k, 3)
    cost = 1.0 - sim.astype(np.float64)
    if parts is not None:
        part_s, want = parts
        cost = cost + w_part * ((part_s[cand] != want[:, None]) & (want[:, None] >= 0))
    if pred is not None:
        d = np.linalg.norm(C - pred[:, None], axis=2) / (radius * diag_s)
        cost = cost + w_prior * d**2
    rows = np.arange(len(fd))
    if fixed is not None:
        # the fixed vertices' own match replaces their first candidate and costs nothing
        mask, nn = fixed
        cand[mask, 0] = nn[mask]
        sim[mask, 0] = 1.0
        C = src_xyz[cand]
        cost = 1.0 - sim.astype(np.float64)
        if parts is not None:
            cost = cost + w_part * ((part_s[cand] != want[:, None]) & (want[:, None] >= 0))
        if pred is not None:
            d = np.linalg.norm(C - pred[:, None], axis=2) / (radius * diag_s)
            cost = cost + w_prior * d**2
        cost[mask] = 1e9
        cost[mask, 0] = 0.0
    choice = cost.argmin(axis=1)
    fans = face_fans(Vd, Fd, tview) if (w_flip > 0 and tview is not None) else None
    if iters:
        A = vertex_adjacency(Fd, len(Vd))
        for _ in range(iters):
            P = C[rows, choice]
            M = A @ P
            d = np.linalg.norm(C - M[:, None], axis=2) / (radius_s * diag_s)
            total = cost + w_smooth * d**2
            if fans is not None:
                total = total + w_flip * mirrored_fraction(C, P, *fans)
            choice = total.argmin(axis=1)
            if fixed is not None:
                choice[fixed[0]] = 0
    chosen = C[rows, choice]
    # how far the chosen point sits from where the prior expected it, in units of the window
    off_prior = None
    if pred is not None:
        off_prior = np.linalg.norm(chosen - pred, axis=1) / (radius * diag_s)
    return cand[rows, choice], sim[rows, choice], off_prior


def gate(w, sim, off_prior, Vd, anchors, sim_quantile=0.2, prior_mult=2.0, anchor_frac=0.35):
    """Take the data weight away from vertices whose match is not to be trusted: similarity in
    the lowest `sim_quantile` of the mesh, or the chosen point more than `prior_mult` windows
    from where the prior expected it, or no anchor within `anchor_frac` of the target's diagonal.
    The field (rigid with --arap) then continues the neighbours' map across them, so a part the
    source does not have (the camel's hump) follows the body around it instead of borrowing the
    most similar patch from elsewhere. Returns the new weights and the mask of gated vertices."""
    bad = sim < np.quantile(sim, sim_quantile)
    if off_prior is not None:
        bad |= off_prior > prior_mult
    if anchors is not None and len(anchors[0]):
        diag_d = float(np.linalg.norm(Vd.max(axis=0) - Vd.min(axis=0)))
        bad |= cKDTree(anchors[0]).query(Vd)[0] > anchor_frac * diag_d
    w = w.copy()
    w[bad & (w < 3.0)] = 1e-3  # anchors and landmarks keep their weight
    return w, bad


def solve_fixed(M, rhs, hard=None):
    """Solve M P = rhs (M sparse, rhs (n,3)); with hard = (mask, values) the masked rows take the
    given values exactly (Dirichlet constraints) and only the free rows are solved, with the
    fixed ones moved to the right-hand side."""
    if hard is None or not hard[0].any():
        return np.asarray(spsolve(M.tocsc(), rhs))
    mask, values = hard
    free = ~mask
    M = M.tocsr()
    Mff = M[free][:, free]
    Mfh = M[free][:, mask]
    P = np.zeros_like(rhs, dtype=np.float64)
    P[mask] = values[mask]
    P[free] = np.asarray(spsolve(Mff.tocsc(), rhs[free] - Mfh @ values[mask]))
    return P


def rematch_within(P, cand_xyz, sims, radius):
    """For every vertex, the most similar candidate (cand_xyz (n,k,3), sims (n,k)) within
    `radius` of its field position P (n,3); -1 where no candidate is that close."""
    d = np.linalg.norm(cand_xyz - P[:, None, :], axis=2)
    allowed = d < radius
    masked = np.where(allowed, sims, -np.inf)
    best = masked.argmax(axis=1)
    best[~allowed.any(axis=1)] = -1
    return best


def smooth_field(Vd, Fd, S, w, lam=1.0, order=1, hard=None):
    """A source-space position per target vertex that follows the matched positions S (n,3) where
    the confidence w (n,) is high and varies smoothly along the target surface elsewhere. Minimises
    sum_i w_i |P_i - S_i|^2 + lam * sum_edges |P_i - P_j|^2 (order 1, the Dirichlet energy, solved
    through the graph Laplacian L) or + lam |L P|^2 (order 2, the bi-Laplacian, smoother). The
    result is continuous, so the texture coordinates derived from it have no jumps inside a
    triangle."""
    n = len(Vd)
    i = np.concatenate([Fd[:, 0], Fd[:, 1], Fd[:, 2], Fd[:, 1], Fd[:, 2], Fd[:, 0]])
    j = np.concatenate([Fd[:, 1], Fd[:, 2], Fd[:, 0], Fd[:, 0], Fd[:, 1], Fd[:, 2]])
    A = sp.coo_matrix((np.ones(len(i)), (i, j)), shape=(n, n)).tocsr()
    A.data[:] = 1.0
    L = sp.diags(np.asarray(A.sum(axis=1)).ravel()) - A
    if order == 2:
        L = L.T @ L
    W = sp.diags(np.asarray(w, dtype=np.float64))
    return solve_fixed(W + lam * L, W @ np.asarray(S, dtype=np.float64), hard)


def arap_field(
    Vd, Fd, S, w, P0, lam=1.0, iters=5, scale_range=(0.3, 3.0), view=None, w_view=0.0,
    hard=None,
):
    """As-rigid-as-possible field: like smooth_field, but the smoothness term asks every vertex's
    one-ring to move by a similarity transform (rotation and scale, never a reflection) instead of
    merely varying smoothly. Alternates a per-vertex Procrustes fit of that transform with a sparse
    solve for the positions. Minimises
        sum_i w_i |P_i - S_i|^2 + lam * sum_edges |(P_j - P_i) - M_ij (x_j - x_i)|^2,
    M_ij the mean of the two end vertices' similarity transforms. Starts from P0.
    view (the relief camera's direction in target space) with w_view > 0 adds to every vertex's
    Procrustes fit a virtual edge pair that asks its rotation to carry that direction onto the
    relief's own viewing axis (+z): the side of the target the texture camera sees must land on
    the side of the relief its camera saw, which rules out the rotations that mirror the
    texture (a limb turned about its own axis). The pair is weighted by w_view times the
    one-ring's squared edge length, so it is comparable to the edges. hard = (mask, values)
    pins the masked vertices to the given positions (Dirichlet constraints)."""
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
    view_term = None
    if view is not None and w_view > 0:
        v = np.asarray(view, dtype=np.float64)
        v = v / np.linalg.norm(v)
        view_term = (w_view * ratio * norm2)[:, None, None] * np.outer(v, [0.0, 0.0, 1.0])[None]
    P = P0.copy()
    for _ in range(iters):
        dp = P[ej] - P[ei]
        # per-vertex covariance of its one-ring: C_i = sum_j dx dp^T
        C = np.zeros((n, 3, 3))
        np.add.at(C, ei, dx[:, :, None] * dp[:, None, :])
        U, sig, Vt = np.linalg.svd(C if view_term is None else C + view_term)
        R = np.transpose(Vt, (0, 2, 1)) @ np.transpose(U, (0, 2, 1))
        flip = np.linalg.det(R) < 0
        if flip.any():  # a reflection fitted best: take the nearest proper rotation
            Vt_f = Vt[flip].copy()
            Vt_f[:, 2, :] *= -1
            R[flip] = np.transpose(Vt_f, (0, 2, 1)) @ np.transpose(U[flip], (0, 2, 1))
        # the scale from the edges alone, given the rotation: sum_j dp_j . R dx_j / sum |dx_j|^2
        s = np.einsum("nab,nba->n", R, C) / np.maximum(norm2, 1e-12)
        s = np.clip(s, *scale_range)
        T = s[:, None, None] * R  # (n,3,3) similarity of every vertex
        d = np.einsum("eab,eb->ea", 0.5 * (T[ei] + T[ej]), dx)  # M_ij (x_j - x_i)
        B = np.zeros((n, 3))
        np.add.at(B, ei, -d)
        P = solve_fixed(M, W @ S + lam * B, hard)
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
    front = sign_screen > 0  # only the side the camera sees carries texture
    flipped = (sign_field != sign_screen) & front
    uvc = np.concatenate([Q, np.zeros_like(Q[:, :, :1])], axis=2)[:, :, :2]
    sv, _, _ = triangle_maps(Vd, Fd, uvc, (1.0, 1.0))
    distorted = (sv[:, 0] / np.maximum(sv[:, 1], 1e-12) > max_distortion) & front
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
    ap.add_argument(
        "--anchor_filter",
        default=None,
        help="drop automatic anchors that disagree with the others: rigid:<thr> (RANSAC "
        "similarity transform) or loo:<thr> (leave-one-out smooth prior); thr is a fraction of "
        "the source diagonal, for example rigid:0.1",
    )
    ap.add_argument("--k", type=int, default=64, help="candidates per target vertex")
    ap.add_argument(
        "--radius", type=float, default=0.05, help="prior window, of the source diagonal"
    )
    ap.add_argument("--w_prior", type=float, default=1.0)
    ap.add_argument(
        "--parts",
        help="part_anchors.npz of part_anchors.py: candidates outside the matched part are "
        "penalised, and vertices of a target part with no matched source part lose their data "
        "weight in the field, so they follow their neighbours (the hump follows the body)",
    )
    ap.add_argument("--w_part", type=float, default=0.5, help="penalty outside the matched part")
    ap.add_argument(
        "--keep_confident",
        type=float,
        default=0.0,
        help="keep the nearest-neighbour match of vertices whose similarity is above this "
        "quantile and that pass the soft mutual test (0: off; 0.7 keeps the top 30 %%)",
    )
    ap.add_argument("--keep_mutual_dist", type=float, default=0.05, help="soft mutual test")
    ap.add_argument("--keep_weight", type=float, default=3.0, help="field weight of kept matches")
    ap.add_argument(
        "--hard_confident",
        type=float,
        default=0.0,
        help="pin the nearest-neighbour match of the confident vertices (similarity above this "
        "quantile, soft mutual test) as hard constraints of the final field (0: off)",
    )
    ap.add_argument(
        "--hard_radius",
        type=float,
        default=0.0,
        help="with --hard_confident: pin only the confident matches within this fraction of the "
        "source diagonal of a first, soft solution of the field (0: all confident matches)",
    )
    ap.add_argument(
        "--final_rematch",
        type=float,
        default=0.0,
        help="after the final field, give every vertex its most similar candidate within this "
        "fraction of the source diagonal of its field position (the field position where none is "
        "that close) and solve the field once more (0: off)",
    )
    ap.add_argument(
        "--view_from_anchors",
        action="store_true",
        help="take the texture camera (--target_view) from the mutual anchors' similarity "
        "transform instead of a stored registration; written to the npz as view",
    )
    ap.add_argument(
        "--w_flip",
        type=float,
        default=0.0,
        help="penalty for mirroring the texture around a vertex (needs --target_view)",
    )
    ap.add_argument(
        "--coarse_to_fine",
        type=int,
        default=0,
        help="levels of field-then-rematch iterations before the final field, the smoothness "
        "weight falling by --coarse_factor per level (non-rigid ICP style); 0: off",
    )
    ap.add_argument("--coarse_factor", type=float, default=4.0)
    ap.add_argument(
        "--coarse_k_prior",
        type=int,
        default=16,
        help="source samples nearest to the field's prediction offered as candidates per level",
    )
    ap.add_argument(
        "--w_view",
        type=float,
        default=0.0,
        help="ARAP: weight of the view-consistency pair in every vertex's rotation fit "
        "(needs --target_view)",
    )
    ap.add_argument("--parts_min_conf", type=float, default=0.0, help="part pairs below: ignored")
    ap.add_argument(
        "--k_prior",
        type=int,
        default=0,
        help="also offer the k source samples nearest to the prior's prediction as candidates",
    )
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
        "--gate",
        action="store_true",
        help="drop the data weight of vertices whose match is not trustworthy and let the field "
        "continue the neighbours' map across them (parts the source does not have)",
    )
    ap.add_argument("--gate_sim", type=float, default=0.2, help="similarity quantile gated")
    ap.add_argument("--gate_prior", type=float, default=2.0, help="windows away from the prior")
    ap.add_argument("--gate_anchor", type=float, default=0.35, help="no anchor within this diag")
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
    anchors, kind, dense_pred, mutual = None, "no anchors", None, None
    if args.prior_transform:
        # the registration maps source to target: y = s R (depth a applied to x) + t; invert it to
        # predict a source position for every target vertex (replaces the map fitted on anchors)
        with open(args.prior_transform) as f:
            tr = json.load(f)
        R = np.array(tr["rotation"], dtype=np.float64)
        dense_pred = ((Vd - np.array(tr["translation"])) / tr["scale"]) @ R
        dense_pred[:, 2] /= tr.get("depth_factor", 1.0)
        kind = "rigid registration prior"
    if lm is not None:
        s_lm, t_lm = lm
        anchors = (Vd[t_lm], Vs[s_lm])
        kind = f"{len(t_lm)} landmarks"
    if (lm is None and args.anchors == "mutual") or (lm is not None and args.add_mutual):
        t, s, w = mutual_pairs(dst["feat"], src["feat"])
        keep = farthest_points(dst["xyz"][t], args.max_anchors)
        mutual = (dst["xyz"][t[keep]], src["xyz"][s[keep]])
        dropped = ""
        if args.anchor_filter:
            mode, thr = args.anchor_filter.split(":")
            ok = filter_anchors(mutual[0], mutual[1], mode, float(thr))
            dropped = f", {int((~ok).sum())} dropped by {args.anchor_filter}"
            mutual = (mutual[0][ok], mutual[1][ok])
        if args.view_from_anchors:
            view, n_in = anchor_view(mutual[0], mutual[1])
            args.target_view = [float(x) for x in view]
            kind += f", view from {n_in} anchors {np.round(view, 3).tolist()}"
        if anchors is None:
            anchors = mutual
        else:
            anchors = (
                np.concatenate([anchors[0], mutual[0]]),
                np.concatenate([anchors[1], mutual[1]]),
            )
        kind += f" + {len(mutual[0])} mutual anchors (of {len(t)} mutual pairs{dropped})"

    fixed = None
    if args.keep_confident > 0:
        keep_mask, nn_all = confident_matches(
            fd, src["feat"], Vd, src["xyz"], args.keep_confident, args.keep_mutual_dist
        )
        fixed = (keep_mask, nn_all)
    parts, unmatched = None, None
    if args.parts:
        z = np.load(args.parts)
        pairs = [(int(a), int(b)) for a, b, c in z["pairs"] if c >= args.parts_min_conf]
        matched = dict(pairs)
        lab_t = z["labels_t"][cKDTree(dst["xyz"]).query(Vd)[1]]
        want = np.array([matched.get(int(j), -1) for j in lab_t], dtype=np.int64)
        unmatched = want < 0
        parts = (z["labels_s"].astype(np.int64), want)
        kind += f" + parts {pairs}"
    t2s_sample, sim, off_prior = refine(
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
        pred=dense_pred,
        k_prior=args.k_prior,
        parts=parts,
        w_part=args.w_part,
        tview=args.target_view,
        w_flip=args.w_flip,
        fixed=fixed,
    )
    t2s = src["idx"][t2s_sample].astype(np.int64)
    if lm is not None:
        t2s[t_lm] = s_lm
    stats = dict(anchors=kind, k=args.k, iters=args.iters, field=args.field)
    if fixed is not None:
        stats["confident_fraction"] = round(float(fixed[0].mean()), 4)
    extra = {}
    if args.coarse_to_fine and args.field > 0:
        # non-rigid ICP in the manner of Jung et al. 2025: alternate the field (the deformation)
        # with the choice of matches, the smoothness falling from coarse_factor^levels times the
        # final weight to the final weight; every level predicts each vertex's source position
        # from the current field instead of from the anchor prior, so the whole map converges
        # together instead of each vertex trusting its own best feature match
        for level in range(args.coarse_to_fine, 0, -1):
            lam = args.field * args.coarse_factor**level
            S = Vs[t2s]
            if mutual is not None:
                S[cKDTree(Vd).query(mutual[0])[1]] = mutual[1]
            w = confidence(sim, Vd, anchors, t_lm if lm is not None else None)
            if fixed is not None:
                w[fixed[0]] = np.maximum(w[fixed[0]], args.keep_weight)
            P = smooth_field(Vd, Fd, S, w, lam, args.field_order)
            if args.arap:
                P = arap_field(
                    Vd, Fd, S, w, P, lam, max(args.arap // 2, 1),
                    view=args.target_view, w_view=args.w_view,
                )
            t2s_sample, sim, off_prior = refine(
                Vd, Fd, fd, src["xyz"], src["feat"], anchors,
                k=args.k, radius=args.radius, w_prior=args.w_prior, iters=args.iters,
                radius_s=args.radius_s, w_smooth=args.w_smooth, pred=P,
                k_prior=max(args.k_prior, args.coarse_k_prior), parts=parts, w_part=args.w_part,
                tview=args.target_view, w_flip=args.w_flip, fixed=fixed,
            )
            t2s = src["idx"][t2s_sample].astype(np.int64)
            if lm is not None:
                t2s[t_lm] = s_lm
        stats["coarse_to_fine"] = [args.coarse_to_fine, args.coarse_factor]
    if args.field > 0:
        S = Vs[t2s]
        if mutual is not None:  # an anchor's vertex follows the anchor, not its refined match
            S[cKDTree(Vd).query(mutual[0])[1]] = mutual[1]
        w = confidence(sim, Vd, anchors, t_lm if lm is not None else None)
        if fixed is not None:
            w[fixed[0]] = np.maximum(w[fixed[0]], args.keep_weight)
        if unmatched is not None and unmatched.any():
            w[unmatched & (w < 3.0)] = 1e-3  # a part the source does not have: follow the rest
            stats["unmatched_part_fraction"] = round(float(unmatched.mean()), 4)
        if args.gate:
            w, gated = gate(
                w, sim, off_prior, Vd, anchors, args.gate_sim, args.gate_prior, args.gate_anchor
            )
            stats["gated_fraction"] = round(float(gated.mean()), 4)
            extra["gated"] = gated
        P = smooth_field(Vd, Fd, S, w, args.field, args.field_order)
        lam = args.arap_weight if args.arap_weight is not None else args.field
        if args.arap:
            P = arap_field(
                Vd, Fd, S, w, P, lam, args.arap, view=args.target_view, w_view=args.w_view
            )
        diag_src = float(np.linalg.norm(src["xyz"].max(axis=0) - src["xyz"].min(axis=0)))
        hard = None
        if args.hard_confident > 0:
            # the trusted nearest neighbours become Dirichlet constraints of the field: the
            # map passes through them exactly instead of averaging them away
            h_mask, h_nn = confident_matches(
                fd, src["feat"], Vd, src["xyz"], args.hard_confident, args.keep_mutual_dist
            )
            h_val = src["xyz"][h_nn].astype(np.float64)
            if args.hard_radius > 0:
                h_mask &= np.linalg.norm(h_val - P, axis=1) < args.hard_radius * diag_src
            if lm is not None:
                h_mask[t_lm] = False  # landmarks keep their own (high) data weight
            hard = (h_mask, h_val)
            stats["hard_fraction"] = round(float(h_mask.mean()), 4)
            P = smooth_field(Vd, Fd, S, w, args.field, args.field_order, hard=hard)
            if args.arap:
                P = arap_field(
                    Vd, Fd, S, w, P, lam, args.arap, view=args.target_view,
                    w_view=args.w_view, hard=hard,
                )
        if args.final_rematch > 0:
            # a last discrete choice near the field: the most similar candidate within the
            # radius replaces the matched position (the field itself where there is none), the
            # replaced vertices get the kept-match weight, and the field is solved once more
            cand, sims = topk_candidates(fd, src["feat"], args.k)
            best = rematch_within(P, src["xyz"][cand], sims, args.final_rematch * diag_src)
            has = best >= 0
            S2 = P.copy()
            S2[has] = src["xyz"][cand[np.arange(len(Vd)), np.maximum(best, 0)]][has]
            w2 = w.copy()
            w2[has] = np.maximum(w2[has], args.keep_weight)
            if mutual is not None:
                S2[cKDTree(Vd).query(mutual[0])[1]] = mutual[1]
            stats["rematch_fraction"] = round(float(has.mean()), 4)
            P = smooth_field(Vd, Fd, S2, w2, args.field, args.field_order, hard=hard)
            if args.arap:
                P = arap_field(
                    Vd, Fd, S2, w2, P, lam, args.arap, view=args.target_view,
                    w_view=args.w_view, hard=hard,
                )
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
    if args.target_view is not None:
        extra["view"] = np.asarray(args.target_view, dtype=np.float64)
    np.savez_compressed(
        args.out, t2s=t2s, options=json.dumps(vars(args)), stats=json.dumps(stats), **extra
    )


if __name__ == "__main__":
    main()
