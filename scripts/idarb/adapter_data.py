"""
Training pairs for the feature adapter (numpy only, shared by train_adapter.py and the tests).

A pair of the partial-to-full dataset (make_partial_dataset.py) gives a partial mesh, its full
mesh and the ground truth orig_idx (the full-mesh vertex of every partial vertex). Diff3F
features exist for a sample of each mesh's vertices (idx, xyz, feat in the .npz). For every
sampled partial vertex the target is the feature of the sampled full vertex nearest to its
true full vertex, kept when that vertex lies within `tol` of the full diagonal; the pair also
carries the full-mesh positions so that negatives can be chosen by distance.
"""

import json
import os

import numpy as np
from scipy.spatial import cKDTree


def load_features(path):
    z = np.load(path)
    return z["idx"].astype(np.int64), z["xyz"].astype(np.float64), z["feat"].astype(np.float32)


def pair_arrays(partial_npz, full_npz, gt_npz, full_vertices, tol=0.005):
    """Returns dict(fp, ff, pos, ff_all, pos_all, keep_fraction): the partial features fp (n,D),
    their target full features ff (n,D), the targets' full-mesh positions pos (n,3), every sampled
    full feature ff_all (m,D) with positions pos_all (m,3), and the fraction of partial samples
    whose true vertex had a sampled full vertex within tol of the diagonal."""
    idx_p, _, fp = load_features(partial_npz)
    idx_f, xyz_f, ff = load_features(full_npz)
    orig = np.load(gt_npz)["orig_idx"].astype(np.int64)
    true_pos = np.asarray(full_vertices, dtype=np.float64)[orig[idx_p]]
    d, j = cKDTree(xyz_f).query(true_pos)
    diag = float(np.linalg.norm(xyz_f.max(axis=0) - xyz_f.min(axis=0)))
    ok = d <= tol * diag
    return dict(fp=fp[ok], ff=ff[j[ok]], pos=xyz_f[j[ok]], ff_all=ff, pos_all=xyz_f,
                keep_fraction=float(ok.mean()) if len(ok) else 0.0)


def dataset_pairs(root, feature_dir, split=None):
    """The dataset's pairs (manifest.json) that have both feature files, as dicts with the paths
    needed by pair_arrays, optionally one split ("train" or "val")."""
    with open(os.path.join(root, "manifest.json")) as f:
        pairs = json.load(f)["pairs"]
    out = []
    for m in pairs:
        if split and m.get("split") != split:
            continue
        stem = os.path.splitext(os.path.basename(m["partial"]))[0]  # <uid>_v<k>
        p_npz = os.path.join(feature_dir, stem + ".npz")
        f_npz = os.path.join(feature_dir, m["uid"] + "_full.npz")
        if os.path.exists(p_npz) and os.path.exists(f_npz):
            out.append(dict(uid=m["uid"], stem=stem, category=m.get("category"),
                            partial_npz=p_npz, full_npz=f_npz,
                            gt_npz=os.path.join(root, m["gt"]) if not os.path.isabs(m["gt"])
                            else m["gt"],
                            full_ply=os.path.join(root, m["full"]) if not os.path.isabs(m["full"])
                            else m["full"]))
    return out


def mlp_forward(feat, weights):
    """The adapter in numpy, for applying saved weights without torch: a residual two-layer
    perceptron on unit features, output renormalised. weights: dict with ln_w, ln_b, w1, b1,
    w2, b2 (as saved by train_adapter.py); the output keeps the input's dimension."""
    x = np.asarray(feat, dtype=np.float32)
    mu = x.mean(axis=1, keepdims=True)
    var = x.var(axis=1, keepdims=True)
    h = (x - mu) / np.sqrt(var + 1e-5) * weights["ln_w"] + weights["ln_b"]
    h = h @ weights["w1"].T + weights["b1"]
    h = 0.5 * h * (1.0 + np.tanh(0.7978845608 * (h + 0.044715 * h**3)))  # GELU (tanh form)
    h = h @ weights["w2"].T + weights["b2"]
    out = x + h
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)
