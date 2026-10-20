"""Synthetic checks of the registration maths in scripts/idarb/register_partial.py: a known
similarity transform must be recovered from matches with outliers, ICP must remove a small
remaining misalignment, the depth factor must be found, and the whole pipeline must beat the
feature-free baseline on synthetic features. No Diff3F features are needed."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import register_partial as rp


def box_surface_points(n, rng):
    """Points on the surface of a box with three different side lengths, so no rotation maps it
    onto itself."""
    p = rng.uniform(-1, 1, size=(n, 3))
    axis = rng.integers(0, 3, size=n)
    sign = rng.choice([-1.0, 1.0], size=n)
    p[np.arange(n), axis] = sign
    p[:, 0] *= 1.5
    p[:, 2] *= 0.6
    return p


def random_similarity(rng):
    R = Rotation.random(random_state=int(rng.integers(0, 2**31))).as_matrix()
    s = float(rng.uniform(0.5, 3.0))
    t = rng.normal(size=3) * 2
    return s, R, t


def test_umeyama_recovers_exact_similarity():
    rng = np.random.default_rng(0)
    src = box_surface_points(200, rng)
    s, R, t = random_similarity(rng)
    s2, R2, t2 = rp.umeyama(src, rp.transform(src, s, R, t))
    assert abs(s2 - s) < 1e-9
    assert rp.rotation_angle_deg(R, R2) < 1e-4  # arccos near 1 is numerically coarse
    assert np.allclose(t2, t, atol=1e-9)


def test_ransac_recovers_similarity_with_outliers():
    rng = np.random.default_rng(1)
    src = box_surface_points(400, rng)
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
    src = box_surface_points(3000, rng)
    s, R, t = random_similarity(rng)
    dst = rp.transform(src, s, R, t)
    diag = float(np.linalg.norm(dst.max(0) - dst.min(0)))
    wobble = Rotation.from_rotvec(np.radians([4.0, -3.0, 2.0])).as_matrix()
    R2, t2, a, rmse, kept, _ = rp.icp_rigid(
        src, dst, s, wobble @ R, t + 0.02 * diag, max_dist=0.2 * diag
    )
    assert rp.rotation_angle_deg(R, R2) < 0.2
    assert rmse < 1e-3 * diag
    assert kept > 0.99
    assert a == 1.0


def test_depth_factor_is_recovered():
    rng = np.random.default_rng(4)
    src = box_surface_points(3000, rng)
    s, R, t = random_similarity(rng)
    dst = rp.transform(rp.apply_depth(src, 0.6), s, R, t)  # the source's depth is 1/0.6 too deep
    a = rp.solve_depth_factor(src, dst, s, R, t)
    assert abs(a - 0.6) < 1e-6
    diag = float(np.linalg.norm(dst.max(0) - dst.min(0)))
    R2, t2, a2, rmse, _, _ = rp.icp_rigid(src, dst, s, R, t, max_dist=0.3 * diag, depth_scale=True)
    assert abs(a2 - 0.6) < 0.02
    assert rmse < 5e-3 * diag


def synthetic_pair(rng, n=2000, dims=32, noise=0.3):
    """A target box with random features and a one-sided, transformed copy as the source, with
    noisy features and the ground truth as make_partial_mesh.py would write it."""
    dst = box_surface_points(n, rng)
    dst_feat = rng.normal(size=(n, dims)).astype(np.float32)
    keep = np.nonzero(dst[:, 2] > 0)[0]
    s, R, t = random_similarity(rng)
    src = rp.transform(dst[keep], 1 / s, R.T, -R.T @ t / s)  # inverse, so src -> dst is (s, R, t)
    src_feat = dst_feat[keep] + rng.normal(scale=noise, size=(len(keep), dims)).astype(np.float32)
    source = dict(xyz=src, feat=rp.normalize_rows(src_feat), idx=np.arange(len(keep)), normals=None)
    target = dict(xyz=dst, feat=rp.normalize_rows(dst_feat), idx=np.arange(n), normals=None)
    gt = dict(orig_idx=keep, s=s, R=R, t=t)
    return source, target, gt, (s, R, t)


@pytest.mark.parametrize("refine", ["icp", "joint"])
def test_register_end_to_end_on_synthetic_features(refine):
    rng = np.random.default_rng(3)
    source, target, gt, (s, R, t) = synthetic_pair(rng)
    res = rp.register(source, target, gt, n_iter=1000, seed=0, refine=refine, n_candidates=4)
    s2, R2, _, a = res["fine"]
    assert abs(s2 / s - 1) < 0.02
    assert rp.rotation_angle_deg(R, R2) < 2.0
    assert a == 1.0
    rows = {r["stage"]: r for r in res["rows"]}
    sel, base = rows["selected"], rows["baseline_icp_no_features"]
    assert sel["rms_to_target"] < 0.01
    assert sel["rms_to_target"] <= base["rms_to_target"]
    assert sel["agreement"] > base["agreement"]
    assert sel["rot_err_deg"] < 2.0
    assert sel["corr_acc_5pct"] > 0.9
    assert rows["feature_nn"]["corr_acc_1pct"] > 0.9  # the synthetic features are clean


def test_consensus_filter_keeps_consistent_matches():
    rng = np.random.default_rng(5)
    dst = box_surface_points(1000, rng)
    src = dst.copy()
    i = np.arange(1000)
    k = i.copy()
    wrong = rng.random(1000) < 0.3
    k[wrong] = rng.permutation(1000)[wrong]
    diag = float(np.linalg.norm(dst.max(0) - dst.min(0)))
    keep = rp.consensus_filter(src, dst, i, k, diag)
    assert keep[~wrong].mean() > 0.95
    assert keep[wrong].mean() < 0.2


def test_ransac_rejects_too_few_matches():
    with pytest.raises(ValueError):
        rp.ransac_similarity(np.zeros((2, 3)), np.zeros((2, 3)), 0.1)
