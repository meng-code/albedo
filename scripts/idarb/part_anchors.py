"""
Part-level anchors between two feature-sampled meshes, after SemAnCorr (Dong et al., RA-L 2026,
arXiv 2607.28382): instead of trusting single points that happen to be mutual nearest neighbours,
match whole parts first, and take anchors only inside parts that are matched with confidence.

  1. parts        each mesh is clustered into K parts: k-means on [PCA-reduced feature, w_p *
                  normalised position]; clusters split into connected pieces and small pieces
                  merged into their neighbours.
  2. relative     the similarity of two parts is the cosine of their mean features after the
                  mean feature of each object has been subtracted, so what is common to the whole
                  object (it is a cat) drops out and only what tells the parts apart remains.
  3. alignment    a similarity transform between the two meshes and a soft part assignment are
                  optimised together: a part pair scores by its relative feature similarity plus
                  the geometric agreement of the two point clouds after the transform (chamfer
                  distance); the schedule starts semantic and ends geometric.
  4. selection    bilateral margin confidence: a pair is confident when each side prefers the
                  other over every alternative; the alpha most confident pairs are taken, each
                  part used once. A part with no confident counterpart (a hump the source does
                  not have) gets no anchor, so the field later carries its neighbours' map across.
  5. anchors      inside every anchor pair, points are matched by nearest neighbour after each
                  region is centred and scaled to its own bounding box (the global alignment
                  applied first), then thinned by farthest-point sampling.

Returns anchors as (target positions, source positions), the input refine_correspondence.py uses
for its prior; the part labels and the chosen pairs are returned for inspection.
"""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

import register_partial as rp


def pca_reduce(feat, d):
    X = np.asarray(feat, dtype=np.float64)
    X = X - X.mean(axis=0)
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    P = X @ vt[:d].T
    return P / max(np.linalg.norm(P, axis=1).mean(), 1e-9)


def kmeans(X, k, seed=0, iters=50):
    rng = np.random.default_rng(seed)
    # k-means++ start
    centres = [X[rng.integers(len(X))]]
    for _ in range(k - 1):
        d = np.min([((X - c) ** 2).sum(axis=1) for c in centres], axis=0)
        centres.append(X[rng.choice(len(X), p=d / d.sum())])
    C = np.array(centres)
    labels = None
    for _ in range(iters):
        new = ((X[:, None] - C[None]) ** 2).sum(axis=2).argmin(axis=1)
        if labels is not None and np.array_equal(new, labels):
            break
        labels = new
        for j in range(k):
            if (labels == j).any():
                C[j] = X[labels == j].mean(axis=0)
    return labels


def clean_parts(xyz, labels, knn=8, min_frac=0.02):
    """Split every label into connected pieces (on a k-nearest-neighbour graph) and merge pieces
    smaller than min_frac of the points into the neighbouring piece they touch most."""
    n = len(xyz)
    nb = cKDTree(xyz).query(xyz, knn + 1)[1][:, 1:]
    i = np.repeat(np.arange(n), knn)
    j = nb.ravel()
    same = labels[i] == labels[j]
    A = coo_matrix((np.ones(same.sum()), (i[same], j[same])), shape=(n, n))
    _, piece = connected_components(A, directed=False)
    while True:
        sizes = np.bincount(piece)
        small = np.nonzero((sizes < min_frac * n) & (sizes > 0))[0]
        if len(small) == 0:
            break
        p = small[sizes[small].argmin()]
        members = piece == p
        # the neighbouring piece with the most edges into this one
        touch = piece[nb[members].ravel()]
        touch = touch[touch != p]
        if len(touch) == 0:
            touch = piece[cKDTree(xyz[~members]).query(xyz[members])[1]]
            touch = np.nonzero(~members)[0][touch]
            touch = piece[touch]
        piece[members] = np.bincount(touch).argmax()
    _, piece = np.unique(piece, return_inverse=True)
    return piece


def find_parts(xyz, feat, k=6, d=32, w_p=0.3, seed=0):
    """Part label per point."""
    P = pca_reduce(feat, d)
    Q = xyz - xyz.mean(axis=0)
    Q = Q / max(np.linalg.norm(Q, axis=1).mean(), 1e-9)
    labels = kmeans(np.concatenate([P, w_p * Q], axis=1), k, seed)
    return clean_parts(xyz, labels)


def part_embeddings(feat, labels):
    E = np.stack([feat[labels == j].mean(axis=0) for j in range(labels.max() + 1)])
    return E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-9)


def relative_similarity(E1, E2):
    """Cosine between part embeddings after each object's mean is removed."""
    A = E1 - E1.mean(axis=0)
    B = E2 - E2.mean(axis=0)
    A = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-9)
    B = B / np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-9)
    return A @ B.T


def unit_sphere(xyz):
    c = xyz.mean(axis=0)
    r = max(np.linalg.norm(xyz - c, axis=1).max(), 1e-9)
    return (xyz - c) / r


def chamfer(P, Q):
    return 0.5 * (cKDTree(Q).query(P)[0].mean() + cKDTree(P).query(Q)[0].mean())


def softmax(x, axis=1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def joint_alignment(parts1, parts2, S, iters=12, sigma=0.15, seed=0, n_init=4, n_pts=300):
    """Similarity transform (s, R, t) from mesh 1 to mesh 2 and the final part scores, from a
    few initialisations: alternate a soft assignment of parts (softmax of the joint score) with a
    weighted fit of the transform on the matched part centroids. The weight of the semantic term
    falls from 1 to 0.2 over the iterations while the geometric term rises."""
    rng = np.random.default_rng(seed)
    sub1 = [p[rng.choice(len(p), min(n_pts, len(p)), replace=False)] for p in parts1]
    sub2 = [p[rng.choice(len(p), min(n_pts, len(p)), replace=False)] for p in parts2]
    c1 = np.stack([p.mean(axis=0) for p in parts1])
    c2 = np.stack([p.mean(axis=0) for p in parts2])
    K1, K2 = len(parts1), len(parts2)

    def geo(s, R, t):
        G = np.empty((K1, K2))
        for a in range(K1):
            moved = rp.transform(sub1[a], s, R, t)
            for b in range(K2):
                G[a, b] = np.exp(-chamfer(moved, sub2[b]) / (2 * sigma**2))
        return G

    def fit(W):
        i, j = np.nonzero(W > 0.05)
        return weighted_umeyama(c1[i], c2[j], W[i, j])

    best = None
    for init in range(n_init):
        # initial transform from the semantic assignment, with a random rotation about the
        # vertical axis for the later initialisations
        W = softmax(S / 0.1)
        if init == 0:
            s, R, t = fit(W)
        else:
            ang = rng.uniform(0, 2 * np.pi)
            R0 = np.array(
                [[np.cos(ang), 0, np.sin(ang)], [0, 1, 0], [-np.sin(ang), 0, np.cos(ang)]]
            )
            s, R, t = 1.0, R0, c2.mean(axis=0) - c1.mean(axis=0) @ R0.T
        for it in range(iters):
            lam_s = 1.0 - 0.8 * it / max(iters - 1, 1)
            score = lam_s * S + (1.0 - lam_s) * geo(s, R, t)
            W = softmax(score / 0.1)
            s, R, t = fit(W)
        score = 0.2 * S + 0.8 * geo(s, R, t)
        total = float((softmax(score / 0.1) * score).sum())
        if best is None or total > best[0]:
            best = (total, s, R, t, score)
    return best[1], best[2], best[3], best[4]


def weighted_umeyama(X, Y, w):
    """Similarity transform Y ~ s R X + t minimising the weighted squared error (Umeyama 1991
    with weights). Returns s, R (det +1), t."""
    w = np.asarray(w, dtype=np.float64) / max(np.sum(w), 1e-12)
    mx, my = w @ X, w @ Y
    Xc, Yc = X - mx, Y - my
    cov = (Yc * w[:, None]).T @ Xc
    U, D, Vt = np.linalg.svd(cov)
    sign = np.ones(3)
    if np.linalg.det(U @ Vt) < 0:
        sign[-1] = -1
    R = U @ np.diag(sign) @ Vt
    var = float((w * (Xc**2).sum(axis=1)).sum())
    s = float((D * sign).sum() / max(var, 1e-12))
    t = my - s * R @ mx
    return s, R, t


def margin_confidence(score):
    """min over both sides of (score of the pair minus the best alternative on that side)."""
    K1, K2 = score.shape
    conf = np.empty_like(score)
    for a in range(K1):
        for b in range(K2):
            row = np.delete(score[a], b).max() if K2 > 1 else -np.inf
            col = np.delete(score[:, b], a).max() if K1 > 1 else -np.inf
            conf[a, b] = min(score[a, b] - row, score[a, b] - col)
    return conf


def select_pairs(conf, alpha, min_conf=0.0):
    """Greedy: the most confident pairs, each part used once, confidence above min_conf."""
    pairs = []
    used1, used2 = set(), set()
    order = np.dstack(np.unravel_index(np.argsort(-conf, axis=None), conf.shape))[0]
    for a, b in order:
        if len(pairs) >= alpha:
            break
        if a in used1 or b in used2 or conf[a, b] < min_conf:
            continue
        pairs.append((int(a), int(b), float(conf[a, b])))
        used1.add(a)
        used2.add(b)
    return pairs


def region_normalise(P):
    lo, hi = P.min(axis=0), P.max(axis=0)
    return (P - (lo + hi) / 2) / np.maximum(hi - lo, 1e-9)


def anchor_points(parts1, parts2, pairs, s, R, t, n_kp=40, seed=0):
    """Nearest-neighbour point pairs inside every anchor pair, in each region's own normalised
    frame after the global alignment, thinned to n_kp per region: (points of mesh 1, of mesh 2)."""
    out1, out2 = [], []
    for a, b, _ in pairs:
        P1 = parts1[a]
        P2 = parts2[b]
        N1 = region_normalise(rp.transform(P1, s, R, t))
        N2 = region_normalise(P2)
        j = cKDTree(N2).query(N1)[1]
        keep = farthest(P1, min(n_kp, len(P1)), seed)
        out1.append(P1[keep])
        out2.append(P2[j[keep]])
    if not out1:
        return np.zeros((0, 3)), np.zeros((0, 3))
    return np.concatenate(out1), np.concatenate(out2)


def farthest(X, n, seed=0):
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


def part_anchors(xyz_t, feat_t, xyz_s, feat_s, k=6, alpha=3, n_kp=40, min_conf=0.0, seed=0):
    """Anchors (target positions, source positions) from part matching, plus a dict with the
    part labels of both point sets, the chosen (target part, source part, confidence) triples,
    the part score matrix and the transform target -> source."""
    lab_t = find_parts(xyz_t, feat_t, k, seed=seed)
    lab_s = find_parts(xyz_s, feat_s, k, seed=seed)
    E_t = part_embeddings(feat_t, lab_t)
    E_s = part_embeddings(feat_s, lab_s)
    S = relative_similarity(E_t, E_s)
    # both meshes in the unit sphere for the geometric term; the anchors go back to mesh units
    ct, rt = xyz_t.mean(axis=0), max(np.linalg.norm(xyz_t - xyz_t.mean(axis=0), axis=1).max(), 1e-9)
    cs, rs = xyz_s.mean(axis=0), max(np.linalg.norm(xyz_s - xyz_s.mean(axis=0), axis=1).max(), 1e-9)
    nt, ns = (xyz_t - ct) / rt, (xyz_s - cs) / rs
    parts_t = [nt[lab_t == j] for j in range(lab_t.max() + 1)]
    parts_s = [ns[lab_s == j] for j in range(lab_s.max() + 1)]
    s, R, t, score = joint_alignment(parts_t, parts_s, S, seed=seed)
    pairs = select_pairs(margin_confidence(score), alpha, min_conf)
    a_t, a_s = anchor_points(parts_t, parts_s, pairs, s, R, t, n_kp, seed)
    a_t = a_t * rt + ct
    a_s = a_s * rs + cs
    info = dict(labels_t=lab_t, labels_s=lab_s, pairs=pairs, score=score, similarity=S)
    return (a_t, a_s), info


PALETTE = np.array(
    [
        [230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48],
        [145, 30, 180], [70, 240, 240], [240, 50, 230], [210, 245, 60], [250, 190, 212],
        [0, 128, 128], [220, 190, 255], [170, 110, 40], [128, 128, 0], [0, 0, 128],
    ],
    dtype=np.uint8,
)


def part_colours(labels, pairs, side):
    """A colour per point: matched part pairs share a palette colour (by their rank), every other
    part gets a grey of its own."""
    col = np.full((len(labels), 3), 160, dtype=np.uint8)
    matched = {}
    for rank, (a, b, _) in enumerate(pairs):
        matched[a if side == "t" else b] = PALETTE[rank % len(PALETTE)]
    for j in range(labels.max() + 1):
        if j in matched:
            col[labels == j] = matched[j]
        else:
            col[labels == j] = 90 + (j * 37) % 120
    return col


def main():
    import argparse
    import json
    import os

    import trimesh

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--source", required=True, help="features of the partial mesh (.npz)")
    ap.add_argument("--target", required=True, help="features of the full mesh (.npz)")
    ap.add_argument("--k", type=int, default=6, help="parts per mesh")
    ap.add_argument("--alpha", type=int, default=3, help="anchor part pairs")
    ap.add_argument("--min_conf", type=float, default=0.0)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    src = rp.load_features(args.source)
    dst = rp.load_features(args.target)
    (a_t, a_s), info = part_anchors(
        dst["xyz"], dst["feat"], src["xyz"], src["feat"], args.k, args.alpha, min_conf=args.min_conf
    )
    os.makedirs(args.out_dir, exist_ok=True)
    sides = (("target", dst, info["labels_t"], "t"), ("source", src, info["labels_s"], "s"))
    for name, z, lab, side in sides:
        trimesh.PointCloud(z["xyz"], colors=part_colours(lab, info["pairs"], side)).export(
            os.path.join(args.out_dir, f"parts_{name}.ply")
        )
    summary = dict(
        pairs=info["pairs"],
        anchors=int(len(a_t)),
        similarity=np.round(info["similarity"], 3).tolist(),
        score=np.round(info["score"], 3).tolist(),
        max_similarity=round(float(np.abs(info["similarity"]).max()), 3),
    )
    with open(os.path.join(args.out_dir, "parts.json"), "w") as f:
        json.dump(summary, f, indent=1)
    np.savez_compressed(
        os.path.join(args.out_dir, "part_anchors.npz"), target=a_t, source=a_s,
        labels_t=info["labels_t"], labels_s=info["labels_s"],
    )
    print(json.dumps(dict(pairs=summary["pairs"], anchors=summary["anchors"],
                          max_similarity=summary["max_similarity"])))


if __name__ == "__main__":
    main()
