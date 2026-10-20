"""Synthetic checks of the registration maths in scripts/idarb/register_partial.py: a known
similarity transform must be recovered from matches with outliers, and ICP must remove a small
remaining misalignment. No Diff3F features are needed."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import register_partial as rp


def cube_surface_points(n, rng):
    """Points on the surface of a box with three different side lengths, so no rotation maps it
    onto itself."""
    p = rng.uniform(-1, 1, size=(n, 3))
    axis = rng.integers(0, 3, size=n)
    sign = rng.choice([-1.0, 1.0], size=n)
    p[np.arange(n), axis] = sign
    p[:, 0] *= 1.5  # make the box non-cubic so no symmetry remains
    p[:, 2] *= 0.6
    return p


def random_similarity(rng):
    R = Rotation.random(random_state=int(rng.integers(0, 2**31))).as_matrix()
    s = float(rng.uniform(0.5, 3.0))
    t = rng.normal(size=3) * 2
    return s, R, t


def test_umeyama_recovers_exact_similarity():
    rng = np.random.default_rng(0)
    src = cube_surface_points(200, rng)
    s, R, t = random_similarity(rng)
    s2, R2, t2 = rp.umeyama(src, rp.transform(src, s, R, t))
    assert abs(s2 - s) < 1e-9
    assert rp.rotation_angle_deg(R, R2) < 1e-4  # arccos near 1 is numerically coarse
    assert np.allclose(t2, t, atol=1e-9)


def test_ransac_recovers_similarity_with_outliers():
    rng = np.random.default_rng(1)
    src = cube_surface_points(400, rng)
    s, R, t = random_similarity(rng)
    dst = rp.transform(src, s, R, t) + rng.normal(scale=0.005, size=src.shape)
    diag = float(np.linalg.norm(dst.max(0) - dst.min(0)))
    wrong = rng.random(len(src)) < 0.4  # 40 % wrong matches
    dst[wrong] = dst[rng.permutation(len(src))[wrong]]
    s2, R2, t2, inliers = rp.ransac_similarity(src, dst, 0.02 * diag, n_iter=2000, seed=0)
    assert abs(s2 / s - 1) < 0.01
    assert rp.rotation_angle_deg(R, R2) < 1.0
    assert inliers.sum() >= 0.5 * len(src)


def test_icp_refines_small_misalignment():
    rng = np.random.default_rng(2)
    src = cube_surface_points(3000, rng)
    s, R, t = random_similarity(rng)
    dst = rp.transform(src, s, R, t)
    diag = float(np.linalg.norm(dst.max(0) - dst.min(0)))
    wobble = Rotation.from_rotvec(np.radians([4.0, -3.0, 2.0])).as_matrix()
    R_start = wobble @ R
    t_start = t + 0.02 * diag
    R2, t2, rmse, kept = rp.icp_rigid(src, dst, s, R_start, t_start, max_dist=0.2 * diag)
    assert rp.rotation_angle_deg(R, R2) < 0.2
    assert rmse < 1e-3 * diag
    assert kept > 0.99


def test_register_end_to_end_on_synthetic_features():
    """A partial copy of the target with its features: half the points, transformed, with noisy
    features. The full pipeline must land on the target and beat the feature-free baseline."""
    rng = np.random.default_rng(3)
    dst = cube_surface_points(2000, rng)
    dst_feat = rng.normal(size=(len(dst), 32)).astype(np.float32)
    keep = dst[:, 2] > 0  # one side only, like a relief mesh
    s, R, t = random_similarity(rng)
    src = rp.transform(dst[keep], 1 / s, R.T, -R.T @ t / s)  # inverse, so src -> dst is (s, R, t)
    src_feat = dst_feat[keep] + rng.normal(scale=0.3, size=(keep.sum(), 32)).astype(np.float32)
    res = rp.register(src, src_feat, dst, dst_feat, n_iter=1000, seed=0)
    s2, R2, _ = res["fine"]
    assert abs(s2 / s - 1) < 0.02
    assert rp.rotation_angle_deg(R, R2) < 2.0
    rows = {r["stage"]: r for r in res["rows"]}
    assert rows["ransac+icp"]["rms_to_target"] < 0.01
    assert rows["ransac+icp"]["rms_to_target"] <= rows["baseline_icp_no_features"]["rms_to_target"]


def test_ransac_rejects_too_few_matches():
    with pytest.raises(ValueError):
        rp.ransac_similarity(np.zeros((2, 3)), np.zeros((2, 3)), 0.1)
