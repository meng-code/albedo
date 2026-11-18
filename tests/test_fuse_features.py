"""Joining two feature sets: the joined cosine similarity is the weighted mean of the two."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "idarb"))
import fuse_features as ff  # noqa: E402


def test_fused_similarity_is_the_weighted_mean():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(6, 5)).astype(np.float16)
    b = rng.normal(size=(6, 3)).astype(np.float16)
    f = ff.fuse(a, b, 0.25)
    assert f.shape == (6, 8)
    ua = a.astype(np.float32) / np.linalg.norm(a.astype(np.float32), axis=1, keepdims=True)
    ub = b.astype(np.float32) / np.linalg.norm(b.astype(np.float32), axis=1, keepdims=True)
    expect = 0.25 * (ua @ ua.T) + 0.75 * (ub @ ub.T)
    assert np.allclose(f @ f.T, expect, atol=1e-5)
    assert np.allclose(np.linalg.norm(f, axis=1), 1.0, atol=1e-5)
