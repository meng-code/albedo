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
    assert a["target_row"].tolist() == [2, 3, 5]
    assert np.allclose(a["pos"], full_v[[4, 6, 10]])  # the true positions
    # a targets sidecar replaces the nearest-sample search
    np.save(tmp_path / "p_targets.npy", np.array([2, -1, 3, 5]))
    b = ad.pair_arrays(tmp_path / "p.npz", tmp_path / "f.npz", tmp_path / "gt.npz", full_v,
                       targets_npy=str(tmp_path / "p_targets.npy"))
    assert b["target_row"].tolist() == [2, 3, 5] and b["keep_fraction"] == 0.75


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


def test_train_adapter_runs_on_a_tiny_fake_dataset(tmp_path):
    """One epoch on the CPU over a fabricated dataset: exercises the whole training loop
    (streaming pairs, every loss term, both evaluation directions, saving)."""
    pytest = __import__("pytest")
    pytest.importorskip("torch")
    import json
    import subprocess

    rng = np.random.default_rng(0)
    root, feat = tmp_path / "ds", tmp_path / "feat"
    (root / "full").mkdir(parents=True)
    (root / "partial").mkdir()
    feat.mkdir()
    pairs = []
    for u, uid in enumerate(["aaa", "bbb", "ccc", "ddd"]):
        n_full = 60
        V = rng.random((n_full, 3))
        # a PLY the trimesh loader reads
        with open(root / "full" / f"{uid}.ply", "w") as f:
            f.write(f"ply\nformat ascii 1.0\nelement vertex {n_full}\nproperty float x\n"
                    "property float y\nproperty float z\nelement face 1\n"
                    "property list uchar int vertex_indices\nend_header\n")
            f.writelines(f"{v[0]:f} {v[1]:f} {v[2]:f}\n" for v in V)
            f.write("3 0 1 2\n")
        idx_f = np.arange(n_full)
        ff = rng.normal(size=(n_full, 16)).astype(np.float32)
        ff /= np.linalg.norm(ff, axis=1, keepdims=True)
        np.savez(feat / f"{uid}_full.npz", idx=idx_f, xyz=V.astype(np.float32), feat=ff,
                 meta="{}")
        for k in range(2):
            orig = rng.choice(n_full, 30, replace=False)
            np.savez(root / "partial" / f"{uid}_v{k}_gt.npz", orig_idx=orig, s=1.0,
                     R=np.eye(3), t=np.zeros(3), meta="{}")
            fp = ff[orig] + 0.1 * rng.normal(size=(30, 16)).astype(np.float32)
            np.savez(feat / f"{uid}_v{k}.npz", idx=np.arange(30), xyz=V[orig].astype(np.float32),
                     feat=fp / np.linalg.norm(fp, axis=1, keepdims=True), meta="{}")
            pairs.append(dict(uid=uid, group="g", category="c", partial=f"partial/{uid}_v{k}.ply",
                              gt=f"partial/{uid}_v{k}_gt.npz", full=f"full/{uid}.ply",
                              split="val" if u >= 3 else "train"))
    json.dump(dict(pairs=pairs), open(root / "manifest.json", "w"))
    out = tmp_path / "adapter.npz"
    script = os.path.join(os.path.dirname(__file__), "..", "scripts", "idarb", "train_adapter.py")
    res = subprocess.run([sys.executable, script, "--root", str(root), "--feature_dir", str(feat),
                          "--out", str(out), "--epochs", "2", "--device", "cpu", "--hidden", "8",
                          "--batch_vertices", "16", "--negatives", "32", "--val_samples", "16",
                          "--w_soft", "0.5", "--w_resid", "0.01"],
                         capture_output=True, text=True)
    assert res.returncode == 0, res.stderr[-2000:]
    lines = [json.loads(line) for line in res.stdout.splitlines() if line.startswith("{")]
    assert lines[-1]["saved"] == str(out) and out.exists()
    epoch_lines = [ln for ln in lines if "val" in ln and "epoch" in ln and ln["epoch"] > 0]
    assert len(epoch_lines) == 2 and "f2p" in epoch_lines[0]["val"]
    w = dict(np.load(out))
    assert w["w1"].shape == (8, 16) and w["w2"].shape == (16, 8)
