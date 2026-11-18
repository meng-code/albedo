"""
Functional maps between two FULL meshes on Diff3F features: the "generate, then match with a
3D-to-3D method" baseline. The source is the full mesh generated from the photo (Hunyuan3D-2.1
or TRELLIS.2, with the relief's keypoints carried over by align_generated.py), the target the
usual full mesh; the keypoint accuracy is measured as in eval_generated.py.

Functional maps (Ovsjanikov et al. 2012) match two shapes through functions instead of points:
each shape gets the first k eigenfunctions of its Laplace-Beltrami operator (a smooth,
shape-wide basis, like low Fourier modes), descriptor functions (here the Diff3F features,
reduced by PCA) are written in that basis, and a small k x k matrix C is solved so that it carries
every source descriptor's coefficients onto the target's, while nearly commuting with the two
Laplacians (an isometry would commute exactly). C is then turned into a point-to-point map by a
nearest neighbour between the target's spectral embedding (Phi_t C) and the source's (Phi_s),
and refined by ZoomOut (Melzi et al. 2019): alternately re-estimate C from the point map with a
few more eigenfunctions and the point map from C, which denoises and sharpens the map. The
whole mesh (both sides, including the back) takes part, which the partial-source chain cannot
use; that is what the supervisor's question is about.

Features exist on a sample of vertices (diff3f_features.py); every vertex takes the feature of
its nearest sampled vertex. The Laplacian is robust_laplacian's intrinsic-Delaunay mesh
Laplacian; a generated mesh in several pieces keeps its largest component when that holds
--min_component of the vertices, else the point-cloud Laplacian connects the pieces.

Usage (repository root, mesh environment; CPU, one to three minutes per pair):
    python scripts/idarb/functional_map_match.py --model hunyuan3d --groups same \
        --out_dir outputs/diff3f/22_functional_maps/hunyuan3d \
        --csv results/diff3f/fmap_hunyuan3d.csv
    python scripts/idarb/functional_map_match.py --source_mesh A.ply --source A.npz \
        --target_mesh B.off --target B.npz --keypoints kp.json --out map.npz
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_features as ef  # noqa: E402
from correspondence_maps import load_mesh  # noqa: E402
from eval_keypoints import landmark_errors, summary  # noqa: E402


def largest_component(V, F, min_share):
    """Keep the largest connected component when it holds at least min_share of the vertices.
    Returns V, F, the index map old -> new (-1 for dropped vertices) and the share."""
    from scipy.sparse.csgraph import connected_components

    rows = np.concatenate([F[:, 0], F[:, 1], F[:, 2]])
    cols = np.concatenate([F[:, 1], F[:, 2], F[:, 0]])
    g = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(V), len(V)))
    n_comp, lab = connected_components(g, directed=False)
    if n_comp <= 1:
        return V, F, np.arange(len(V)), 1.0
    big = np.bincount(lab).argmax()
    keep = lab == big
    share = float(keep.mean())
    if share < min_share:
        return V, F, np.arange(len(V)), share
    new = np.full(len(V), -1, dtype=np.int64)
    new[keep] = np.arange(int(keep.sum()))
    Fk = F[keep[F].all(axis=1)]
    return V[keep], new[Fk], new, share


def laplacian_basis(V, F, k, cloud=False):
    """First k eigenpairs of the (generalised) Laplace-Beltrami problem L phi = lambda M phi,
    M-orthonormal: Phi^T M Phi = I. Returns eigenvalues, Phi (n x k) and M."""
    import robust_laplacian as rl

    if cloud or F is None or len(F) == 0:
        L, M = rl.point_cloud_laplacian(np.ascontiguousarray(V, dtype=np.float64))
    else:
        L, M = rl.mesh_laplacian(np.ascontiguousarray(V, dtype=np.float64),
                                 np.ascontiguousarray(F, dtype=np.int64))
    L = L.tocsc()
    M = M.tocsc()
    # unit total area on both shapes, so that the eigenvalues are comparable between them
    area = float(M.diagonal().sum())
    L = L * 1.0  # the stiffness matrix is scale-free; the mass scales with the area
    M = M / area
    evals, evecs = spla.eigsh(L, k=k, M=M, sigma=-1e-6, which="LM")
    order = np.argsort(evals)
    evals = np.clip(evals[order], 0, None)
    evecs = evecs[:, order]
    # M-normalise
    nrm = np.sqrt(np.einsum("ik,ik->k", evecs, M @ evecs))
    evecs /= nrm
    return evals, evecs, M


def vertex_features(npz_path, V):
    """2048-d Diff3F feature of every vertex: that of the nearest sampled vertex."""
    z = np.load(npz_path, allow_pickle=True)
    xyz = z["xyz"].astype(np.float64)
    feat = z["feat"].astype(np.float32)
    feat /= np.maximum(np.linalg.norm(feat, axis=1, keepdims=True), 1e-8)
    j = cKDTree(xyz).query(V)[1]
    return feat[j]


def descriptors(Fs, Ft, dim, rng):
    """A joint PCA of the two feature sets to dim numbers per vertex (fit on a subsample)."""
    from sklearn.decomposition import PCA

    n = min(20000, len(Fs) + len(Ft))
    both = np.concatenate([Fs, Ft])
    sub = rng.choice(len(both), size=min(n, len(both)), replace=False)
    pca = PCA(n_components=dim, random_state=0).fit(both[sub])
    Ds = pca.transform(Fs)
    Dt = pca.transform(Ft)
    scale = np.sqrt(pca.explained_variance_)
    return Ds / scale, Dt / scale  # whitened, so every descriptor counts the same


def solve_fmap(A, B, ev_s, ev_t, w_lap):
    """C (k_t x k_s) minimising ||C A - B||^2 + w_lap ||C diag(ev_s) - diag(ev_t) C||^2.
    A, B: spectral coefficients of the descriptors (k_s x d, k_t x d). Row-wise closed form."""
    k_t = len(ev_t)
    k_s = len(ev_s)
    AAt = A @ A.T
    C = np.zeros((k_t, k_s))
    for i in range(k_t):
        D = np.diag((ev_s - ev_t[i]) ** 2) * w_lap
        C[i] = np.linalg.solve(AAt + D + 1e-9 * np.eye(k_s), A @ B[i])
    return C


def nearest_rows(A, B, chunk=2048):
    """For every row of B, the index of its nearest row of A (Euclidean). Brute force through
    BLAS in chunks: in the 50- to 2048-dimensional spaces used here a k-d tree degenerates to a
    scan with far more overhead (a 2048-d query of 20 000 rows against 90 000 took over an
    hour), and the matrix product takes seconds."""
    A = np.ascontiguousarray(A, dtype=np.float64)
    B = np.ascontiguousarray(B, dtype=np.float64)
    a2 = np.einsum("ij,ij->i", A, A)
    out = np.empty(len(B), dtype=np.int64)
    for i in range(0, len(B), chunk):
        b = B[i:i + chunk]
        d = a2[None, :] - 2.0 * (b @ A.T)  # + |b|^2, constant per row
        out[i:i + chunk] = np.argmin(d, axis=1)
    return out


def fmap_to_p2p(C, Phi_s, Phi_t):
    """Point map target -> source: row i of Phi_t C is matched to its nearest row of Phi_s."""
    k_t, k_s = C.shape
    emb_t = Phi_t[:, :k_t] @ C
    emb_s = Phi_s[:, :k_s]
    return nearest_rows(emb_s, emb_t)


def p2p_to_fmap(t2s, Phi_s, Phi_t, M_t, k):
    """C from a point map: the mass-weighted projection of the pulled-back source basis."""
    return Phi_t[:, :k].T @ (M_t @ Phi_s[t2s, :k])


def zoomout(C, Phi_s, Phi_t, M_t, k_max, step):
    """ZoomOut: grow the map from C's size to k_max, re-estimating C and the point map."""
    k = C.shape[0]
    t2s = fmap_to_p2p(C, Phi_s, Phi_t)
    while k < k_max:
        k = min(k + step, k_max)
        C = p2p_to_fmap(t2s, Phi_s, Phi_t, M_t, k)
        t2s = fmap_to_p2p(C, Phi_s, Phi_t)
    return C, t2s


def match_pair(src_mesh, src_npz, dst_mesh, dst_npz, kp_path, out_path, args, rng):
    t0 = time.time()
    ms = load_mesh(src_mesh)
    md = load_mesh(dst_mesh)
    Vs_all = np.asarray(ms.vertices, dtype=np.float64)
    Fs_all = np.asarray(ms.faces, dtype=np.int64)
    Vd = np.asarray(md.vertices, dtype=np.float64)
    Fd = np.asarray(md.faces, dtype=np.int64)
    Vs, Fs, remap, share = largest_component(Vs_all, Fs_all, args.min_component)
    cloud_s = share < args.min_component
    Vd, Fd, remap_d, share_d = largest_component(Vd, Fd, args.min_component)
    cloud_d = share_d < args.min_component
    feat_s = vertex_features(src_npz, Vs)
    feat_d = vertex_features(dst_npz, Vd)
    Ds, Dt = descriptors(feat_s, feat_d, args.descr_dim, rng)
    k = args.k
    ev_s, Phi_s, M_s = laplacian_basis(Vs, Fs, max(k, args.k_zoomout), cloud_s)
    ev_t, Phi_t, M_t = laplacian_basis(Vd, Fd, max(k, args.k_zoomout), cloud_d)
    if args.verbose:
        print(json.dumps(dict(ev_s=np.round(ev_s[:6], 3).tolist(),
                              ev_t=np.round(ev_t[:6], 3).tolist())), flush=True)
    # spectral coefficients of the descriptors, first k eigenfunctions
    A = Phi_s[:, :k].T @ (M_s @ Ds)
    B = Phi_t[:, :k].T @ (M_t @ Dt)
    C0 = solve_fmap(A, B, ev_s[:k], ev_t[:k], args.w_lap)
    t2s_fm = fmap_to_p2p(C0, Phi_s, Phi_t)
    C1, t2s_zo = zoomout(C0, Phi_s, Phi_t, M_t, args.k_zoomout, args.zoomout_step)
    # reference: nearest neighbour in the descriptor space at the vertices (no spectral part)
    t2s_nn = nearest_rows(feat_s, feat_d)
    # ZoomOut started from the nearest-neighbour map instead of the solved C (the usual
    # spectral post-processing of a feature matching)
    C_nn = p2p_to_fmap(t2s_nn, Phi_s, Phi_t, M_t, k)
    _, t2s_nnzo = zoomout(C_nn, Phi_s, Phi_t, M_t, args.k_zoomout, args.zoomout_step)
    # back to the original target indexing (dropped target vertices take the nearest kept one)
    maps = {"fmap": t2s_fm, "zoomout": t2s_zo, "nn": t2s_nn, "nn_zoomout": t2s_nnzo}
    if len(Vd) != len(remap_d):
        kept = cKDTree(Vd).query(np.asarray(md.vertices, dtype=np.float64))[1]
        maps = {n: m[kept] for n, m in maps.items()}
    # source indices back to the original source mesh
    if len(Vs) != len(remap):
        back = np.nonzero(remap >= 0)[0]
        maps = {n: back[m] for n, m in maps.items()}
    res = dict(n_source=int(len(Vs_all)), n_target=int(len(remap_d)),
               source_component_share=round(share, 3), source_cloud=bool(cloud_s),
               target_cloud=bool(cloud_d), k=k, k_zoomout=args.k_zoomout)
    if kp_path:
        with open(kp_path) as f:
            lm = json.load(f)
        lm_s = np.asarray(lm["source"], dtype=np.float64)
        lm_t = np.asarray(lm["target"], dtype=np.float64)
        diag_s = float(np.linalg.norm(Vs_all.max(axis=0) - Vs_all.min(axis=0)))
        Vd_all = np.asarray(md.vertices, dtype=np.float64)
        for name, m in maps.items():
            s = summary(landmark_errors(Vs_all[m], Vd_all, lm_s, lm_t, diag_s))
            res[name + "_pck_2"] = s["pck_2"]
            res[name + "_pck_5"] = s["pck_5"]
            res[name + "_mean"] = s["mean"]
        res["kp_n"] = int(len(lm_s))
    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
        np.savez(out_path, t2s=maps["zoomout"], src_pos=Vs_all[maps["zoomout"]].astype(np.float32),
                 t2s_fmap=maps["fmap"], t2s_nn=maps["nn"], t2s_nn_zoomout=maps["nn_zoomout"],
                 C=C1,
                 meta=json.dumps(dict(res, source=src_mesh, target=dst_mesh)))
    res["seconds"] = round(time.time() - t0, 1)
    return res


FIELDS = ["group", "pair", "model", "n_source", "n_target", "source_component_share",
          "source_cloud", "target_cloud", "k", "k_zoomout", "kp_n", "fmap_pck_2", "zoomout_pck_2",
          "nn_pck_2", "nn_zoomout_pck_2", "fmap_pck_5", "zoomout_pck_5", "nn_pck_5",
          "nn_zoomout_pck_5", "fmap_mean", "zoomout_mean", "nn_mean", "nn_zoomout_mean", "seconds"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["hunyuan3d", "trellis2", "relief"],
                    help="batch over the annotated pairs with this source")
    ap.add_argument("--groups", nargs="*", default=["same"])
    ap.add_argument("--pairs", nargs="*")
    ap.add_argument("--gen_features", default="outputs/diff3f/features_zoom2_gen")
    ap.add_argument("--feature_dir", default="outputs/diff3f/features_zoom2")
    ap.add_argument("--keypoints_gen", default="results/diff3f/keypoints_gen")
    ap.add_argument("--keypoints_relief", default="results/diff3f/keypoints")
    ap.add_argument("--landmarks", default="results/diff3f/landmarks")
    ap.add_argument("--out_dir")
    ap.add_argument("--csv")
    ap.add_argument("--source_mesh")
    ap.add_argument("--source")
    ap.add_argument("--target_mesh")
    ap.add_argument("--target")
    ap.add_argument("--keypoints")
    ap.add_argument("--out")
    ap.add_argument("--k", type=int, default=50, help="eigenfunctions for the initial map")
    ap.add_argument("--k_zoomout", type=int, default=120, help="eigenfunctions after ZoomOut")
    ap.add_argument("--zoomout_step", type=int, default=5)
    ap.add_argument("--descr_dim", type=int, default=40)
    ap.add_argument("--w_lap", type=float, default=1e-2,
                    help="weight of the Laplacian commutativity term")
    ap.add_argument("--min_component", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    if not args.model:
        res = match_pair(args.source_mesh, args.source, args.target_mesh, args.target,
                         args.keypoints, args.out, args, rng)
        print(json.dumps(res))
        return

    rows = []
    for group, name, src, relief, target, _tr, _e in ef.pair_table(args.groups):
        if args.pairs and name not in args.pairs:
            continue
        if args.model == "relief":
            src_mesh = relief
            src_npz = os.path.join(args.feature_dir, src + "_partial.npz")
            kp_path = os.path.join(args.keypoints_relief, name + ".json")
            if not os.path.exists(kp_path):  # the older annotations (keypoint_table.py's fallback)
                kp_path = os.path.join(args.landmarks, name + ".json")
        else:
            src_mesh = os.path.join(args.gen_features, args.model, src + "_mesh.ply")
            src_npz = os.path.join(args.gen_features, args.model, src + ".npz")
            kp_path = os.path.join(args.keypoints_gen, args.model, name + ".json")
        dst_npz = os.path.join(args.feature_dir, target + "_full.npz")
        dst_mesh = ef.TARGET_MESH[target]
        if not all(os.path.exists(p) for p in (src_mesh, src_npz, kp_path, dst_npz, dst_mesh)):
            print(json.dumps(dict(pair=name, skipped="missing input")), flush=True)
            continue
        out = os.path.join(args.out_dir, group, name, "fmap.npz") if args.out_dir else None
        res = match_pair(src_mesh, src_npz, dst_mesh, dst_npz, kp_path, out, args, rng)
        row = dict(group=group, pair=name, model=args.model, **res)
        rows.append(row)
        print(json.dumps(row), flush=True)
    if args.csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)) or ".", exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
    for group in args.groups:
        rs = [r for r in rows if r["group"] == group]
        if not rs:
            continue
        n = sum(r["kp_n"] for r in rs)
        pooled = {m: round(100 * sum(r[m + "_pck_2"] * r["kp_n"] for r in rs) / n, 1)
                  for m in ("fmap", "zoomout", "nn", "nn_zoomout")}
        print(f"{group}: pairs {len(rs)} keypoints {n} pooled PCK@2 % {pooled}")


if __name__ == "__main__":
    main()
