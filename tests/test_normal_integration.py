"""Tests for the normal integration in scripts/idarb/normal_to_mesh.py.

They use synthetic scenes whose true depth is known, the same ones as
`normal_to_mesh.py --selftest`: an ideal hemisphere, and a small sphere floating in front of a
large one.
"""

import pytest

import normal_to_mesh as ntm


@pytest.fixture(scope="module")
def hemisphere():
    return ntm.hemisphere_check()


def test_hemisphere_is_recovered(hemisphere):
    # Normals encoded like IDArb's output and integrated back give the true shape.
    assert hemisphere["mean_err"] < 0.05
    assert hemisphere["max_err"] < 0.1
    assert hemisphere["recovered_height"] == pytest.approx(hemisphere["true_height"], abs=0.5)


def test_hemisphere_sign_convention(hemisphere):
    # The centre of the hemisphere must come out closer to the camera than its rim.
    assert hemisphere["center_higher"]


def test_single_sphere_without_occlusion_keeps_full_depth():
    r = ntm.occlusion_check(with_front=False)
    assert r["ratio_normals"] == pytest.approx(1.0, abs=0.02)


@pytest.mark.parametrize("gap_center", [100.0, 150.0, 200.0])
def test_normals_alone_lose_the_occlusion_gap(gap_center):
    r = ntm.occlusion_check(gap_center)
    # The true gap grows with the distance between the spheres, but normals only see ~5 px.
    assert r["gap_true"] > 40
    assert r["gap_normals"] < 10
    assert r["ratio_normals"] < 0.8


@pytest.mark.parametrize("gap_center", [100.0, 150.0, 200.0])
def test_depth_prior_restores_the_occlusion_gap(gap_center):
    r = ntm.occlusion_check(gap_center)
    assert r["gap_depth"] == pytest.approx(r["gap_true"], abs=1.0)
    assert r["ratio_depth"] == pytest.approx(1.0, abs=0.02)
