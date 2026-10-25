"""Pair assembly and the numpy adapter forward pass (no torch needed)."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "idarb"))
import adapter_data as ad  # noqa: E402


def test_pair_arrays_match_true_vertices(tmp_path):
    rng = np.random.default_rng(0)
    full_v = rng.random((50, 3))
    idx_f = np.arange(0, 50, 2)  # the full mesh is sampled at even vertices
    ff = rng.normal(size=(25, 4)).astype(np.float32)
    orig = np.array([4, 6, 7, 10])  # partial vertex -> full vertex; vertex 7 is not sampled
    idx_p = np.arange(4)
    fp = rng.normal(size=(4, 4)).astype(np.float32)
    np.savez(tmp_path / "p.npz", idx=idx_p, xyz=full_v[orig], feat=fp)
    np.savez(tmp_path / "f.npz", idx=idx_f, xyz=full_v[idx_f], feat=ff)
    np.savez(tmp_path / "gt.npz", orig_idx=orig)
    a = ad.pair_arrays(tmp_path / "p.npz", tmp_path / "f.npz", tmp_path / "gt.npz", full_v,
                       tol=1e-9)
    assert a["keep_fraction"] == 0.75  # vertex 7 has no sampled full vertex at its position
    assert np.allclose(a["ff"], ff[[2, 3, 5]])  # full samples 4, 6, 10 are rows 2, 3, 5
    assert np.allclose(a["pos"], full_v[[4, 6, 10]])


def test_mlp_forward_is_identity_with_zero_second_layer_and_unit_length():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(5, 8)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    w = dict(ln_w=np.ones(8, np.float32), ln_b=np.zeros(8, np.float32),
             w1=rng.normal(size=(6, 8)).astype(np.float32), b1=np.zeros(6, np.float32),
             w2=np.zeros((8, 6), np.float32), b2=np.zeros(8, np.float32))
    assert np.allclose(ad.mlp_forward(x, w), x, atol=1e-6)
    w["w2"] = rng.normal(size=(8, 6)).astype(np.float32) * 0.1
    y = ad.mlp_forward(x, w)
    assert np.allclose(np.linalg.norm(y, axis=1), 1.0, atol=1e-5)
    assert not np.allclose(y, x)


def test_torch_adapter_matches_numpy_forward():
    pytest = __import__("pytest")
    torch = pytest.importorskip("torch")
    import train_adapter as ta

    torch.manual_seed(0)
    net = ta.Adapter(8, 6, "cpu")
    with torch.no_grad():
        net.l2.weight.normal_(0, 0.1)
        net.l2.bias.normal_(0, 0.1)
    x = torch.nn.functional.normalize(torch.randn(5, 8), dim=1)
    with torch.no_grad():
        y_torch = net(x).numpy()
    y_np = ad.mlp_forward(x.numpy(), net.weights())
    assert np.allclose(y_torch, y_np, atol=1e-5)
