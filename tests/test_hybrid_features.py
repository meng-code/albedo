import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "idarb"))
from hybrid_features import hybrid, relief_frame  # noqa: E402
from keypoint_rank import true_rank  # noqa: E402


def _transform():
    th = 0.3
    R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1.0]])
    return dict(camera_rotation=np.eye(3).tolist(), scale=1.5, depth_factor=0.8,
                rotation=R.tolist(), translation=[0.2, -0.1, 0.05])


def test_relief_frame_inverts_the_fit():
    tr = _transform()
    q = np.random.default_rng(0).normal(size=(20, 3))
    s, a = tr["scale"], tr["depth_factor"]
    R, t = np.array(tr["rotation"]), np.array(tr["translation"])
    forward = s * (q * [1, 1, a]) @ R.T + t
    assert np.allclose(relief_frame(forward, tr), q)


def test_hybrid_replaces_only_the_visible_front():
    # a box: the camera looks down -z, so only the z = +1 face is visible
    import pytest

    pytest.importorskip("open3d")  # the ray casting of align_generated.front_depth
    import trimesh

    box = trimesh.creation.box(extents=(2, 2, 2))
    V, F = np.asarray(box.vertices, dtype=np.float64), np.asarray(box.faces)
    tr = dict(camera_rotation=np.eye(3).tolist(), scale=1.0, depth_factor=1.0,
              rotation=np.eye(3).tolist(), translation=[0.0, 0.0, 0.0])
    S_xyz = V.copy()
    S_feat = np.zeros((len(V), 4), dtype=np.float32)
    # relief: the front face of the box, with a distinctive feature
    R_xyz = V[V[:, 2] > 0]
    R_feat = np.ones((len(R_xyz), 4), dtype=np.float32)
    feat, take = hybrid(S_xyz, S_feat, V, F, tr, R_xyz, R_feat, max_dist=0.05)
    assert take.sum() == 4 and (V[take, 2] > 0).all()
    assert (feat[take] == 1).all() and (feat[~take] == 0).all()


def test_true_rank_counts_higher_scores():
    sim = np.array([[0.9, 0.5, 0.7], [0.1, 0.8, 0.3]])
    rank, own = true_rank(sim, np.array([2, 1]))
    assert rank.tolist() == [1, 0]
    assert np.allclose(own, [0.7, 0.8])
