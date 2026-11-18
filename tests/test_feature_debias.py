import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "idarb"))
import debias_check as dc  # noqa: E402
import feature_debias as fdb  # noqa: E402


def unit_rows(rng, n, d):
    x = rng.normal(size=(n, d))
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


def test_none_returns_inputs():
    rng = np.random.default_rng(0)
    fs, fd = unit_rows(rng, 20, 8), unit_rows(rng, 30, 8)
    a, b = fdb.debias(fs, fd, "none")
    assert a is fs and b is fd


def test_center_removes_a_shared_offset():
    # both shapes hold the same points, but each has its own large offset added: the nearest
    # neighbour is wrong before centering and right after it
    rng = np.random.default_rng(1)
    base = rng.normal(size=(40, 16))
    off_s, off_d = 5.0 * rng.normal(size=16), 5.0 * rng.normal(size=16)
    fs, fd = fdb.unit(base + off_s), fdb.unit(base + off_d)
    cs, cd = fdb.debias(fs, fd, "center")
    assert np.allclose(np.linalg.norm(cs, axis=1), 1.0, atol=1e-5)
    assert ((cd @ cs.T).argmax(axis=1) == np.arange(40)).mean() > 0.95


def test_whiten_shapes_and_unit_norm():
    rng = np.random.default_rng(2)
    fs, fd = unit_rows(rng, 50, 12), unit_rows(rng, 60, 12)
    ws, wd = fdb.debias(fs, fd, "whiten:5:0.5")
    assert ws.shape == (50, 5) and wd.shape == (60, 5)
    assert np.allclose(np.linalg.norm(wd, axis=1), 1.0, atol=1e-5)


def test_csls_dot_product_is_the_corrected_similarity():
    rng = np.random.default_rng(3)
    fs, fd = unit_rows(rng, 25, 6), unit_rows(rng, 35, 6)
    k = 4
    s2, d2 = fdb.debias(fs, fd, f"csls:{k}")
    S = fd @ fs.T
    r_d = -np.sort(-S, axis=1)[:, :k].mean(axis=1)
    r_s = -np.sort(-S.T, axis=1)[:, :k].mean(axis=1)
    want = S - 0.5 * r_d[:, None] - 0.5 * r_s[None, :]
    assert np.allclose(d2 @ s2.T, want, atol=1e-5)


def test_csls_demotes_a_hub():
    # one source point is close to the mean of everything, so plain similarity sends many
    # target points to it; after CSLS it is chosen less often
    rng = np.random.default_rng(4)
    pts = unit_rows(rng, 60, 10)
    hub = fdb.unit(pts.mean(axis=0, keepdims=True) + 0.4 * pts[:1])
    fs = np.vstack([pts + 0.0, hub]).astype(np.float32)
    fd = fdb.unit(pts + 0.9 * rng.normal(size=pts.shape) / np.sqrt(10))
    before = ((fd @ fs.T).argmax(axis=1) == 60).sum()
    s2, d2 = fdb.debias(fs, fd, "csls:5")
    after = ((d2 @ s2.T).argmax(axis=1) == 60).sum()
    assert after <= before


def test_tag_of():
    assert fdb.tag_of("whiten:256:0.5+csls:10") == "whiten-256-0p5_csls-10"


def test_best_within_takes_the_most_similar_allowed_sample():
    sim = np.array([[0.9, 0.5, 0.7], [0.9, 0.5, 0.7]])
    dist = np.array([[0.20, 0.01, 0.04], [0.20, 0.30, 0.40]])
    err, has = dc.best_within(sim, dist, 0.05)
    assert has.tolist() == [True, False]
    assert np.allclose(err, [0.04, 0.0])


def test_check_pair_on_identical_shapes():
    rng = np.random.default_rng(5)
    f = unit_rows(rng, 80, 16)
    xs = rng.uniform(size=(80, 3))
    t_idx = np.array([3, 10, 42])
    r = dc.check_pair(f, f, xs, t_idx, xs[t_idx], 1.0, 16, 0.02)
    assert r["hits_nn"] == 3 and r["bound"] == 3 and r["hits_rematch"] == 3
    assert r["hits_peak5"] == 3 and r["wrong"] == 0
