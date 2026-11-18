"""Part-level anchors: clustering, relative similarity, confident bijective selection."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "idarb"))
import part_anchors as pa  # noqa: E402


def two_blobs(rng, n=200):
    xyz = np.concatenate([rng.normal(size=(n, 3)) * 0.1, rng.normal(size=(n, 3)) * 0.1 + [1, 0, 0]])
    f = np.zeros((2 * n, 8))
    f[:n, 0] = 1.0
    f[n:, 1] = 1.0
    f += 0.05 * rng.normal(size=f.shape)
    return xyz, f / np.linalg.norm(f, axis=1, keepdims=True)


def test_find_parts_separates_two_blobs():
    rng = np.random.default_rng(0)
    xyz, f = two_blobs(rng)
    lab = pa.find_parts(xyz, f, k=2)
    assert lab.max() == 1
    assert len(np.unique(lab[:200])) == 1 and len(np.unique(lab[200:])) == 1
    assert lab[0] != lab[200]


def test_weighted_umeyama_recovers_a_similarity():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(12, 3))
    ang = 0.7
    R = np.array([[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1]])
    Y = 1.7 * X @ R.T + [0.3, -0.2, 0.5]
    s, R2, t = pa.weighted_umeyama(X, Y, np.ones(12))
    assert abs(s - 1.7) < 1e-6 and np.allclose(R2, R, atol=1e-6)
    assert np.allclose(t, [0.3, -0.2, 0.5], atol=1e-6)


def test_margin_confidence_and_bijective_selection():
    score = np.array([[0.9, 0.1, 0.0], [0.2, 0.8, 0.7], [0.0, 0.75, 0.1]])
    conf = pa.margin_confidence(score)
    assert conf[0, 0] > 0.6  # a clear winner on both sides
    assert conf[1, 2] < 0  # row 1 prefers column 1
    pairs = pa.select_pairs(conf, alpha=3, min_conf=0.0)
    used_rows = [a for a, _, _ in pairs]
    used_cols = [b for _, b, _ in pairs]
    assert len(set(used_rows)) == len(used_rows) and len(set(used_cols)) == len(used_cols)
    assert pairs[0][:2] == (0, 0)


def test_part_anchors_match_blob_to_blob():
    rng = np.random.default_rng(2)
    xyz_t, f_t = two_blobs(rng)
    xyz_s, f_s = two_blobs(rng)
    (a_t, a_s), info = pa.part_anchors(xyz_t, f_t, xyz_s, f_s, k=2, alpha=2, n_kp=10)
    assert len(a_t) == len(a_s) == 20
    # every anchor pair joins the same blob on both sides (blob 0 around x=0, blob 1 at x=1)
    assert np.all((a_t[:, 0] > 0.5) == (a_s[:, 0] > 0.5))
