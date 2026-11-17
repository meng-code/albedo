"""
Statistical corrections of a pair's features before matching ("debiasing"). They change no
model and need no training: every step works on the two feature sets of one pair.

  center          subtract each shape's own mean feature and renormalise: removes what all points
                  of one shape share (the partial mesh against the full mesh, one individual
                  against another), which pushes every match the same way
  whiten:D:A      center, project on the D leading principal directions of both shapes together
                  (each shape counting equally) and divide direction i by its variance to the
                  power A / 2 (A = 1: full whitening, A = 0.5: half), then renormalise: a few
                  dominant directions no longer decide every similarity
  csls:K          cross-domain similarity local scaling (Conneau et al. 2018): the similarity of
                  a target point t and a source point s becomes
                  cos(t, s) - r(t) / 2 - r(s) / 2, r being a point's mean similarity to its K
                  most similar points of the other shape. A hub (a point similar to everything)
                  has a large r and is chosen less often. Implemented as two extra feature
                  columns, so every dot product downstream is the corrected similarity.

Steps are joined with "+", for example "center+csls:10". "none" returns the inputs.
"""

import numpy as np


def unit(x):
    x = np.asarray(x, dtype=np.float64)
    return (x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)).astype(np.float32)


def center(fs, fd):
    fs, fd = np.asarray(fs, dtype=np.float64), np.asarray(fd, dtype=np.float64)
    return unit(fs - fs.mean(axis=0)), unit(fd - fd.mean(axis=0))


def whiten(fs, fd, dims=256, alpha=1.0):
    a = np.asarray(fs, dtype=np.float64)
    b = np.asarray(fd, dtype=np.float64)
    a, b = a - a.mean(axis=0), b - b.mean(axis=0)
    cov = a.T @ a / len(a) + b.T @ b / len(b)
    lam, U = np.linalg.eigh(cov)
    order = np.argsort(-lam)[: min(int(dims), len(lam))]
    lam = np.maximum(lam[order], 0.0)
    W = U[:, order] * (lam + 1e-6 * lam.max()) ** (-0.5 * float(alpha))
    return unit(a @ W), unit(b @ W)


def mean_topk(a, b, k, chunk=2048):
    """For every row of a, the mean of its k largest dot products with the rows of b."""
    k = min(int(k), len(b))
    out = np.empty(len(a), dtype=np.float32)
    for i in range(0, len(a), chunk):
        S = a[i : i + chunk] @ b.T
        out[i : i + chunk] = -np.partition(-S, k - 1, axis=1)[:, :k].mean(axis=1)
    return out


def csls(fs, fd, k=10):
    fs, fd = np.asarray(fs, dtype=np.float32), np.asarray(fd, dtype=np.float32)
    r_s, r_d = mean_topk(fs, fd, k), mean_topk(fd, fs, k)
    one_s, one_d = np.ones((len(fs), 1), np.float32), np.ones((len(fd), 1), np.float32)
    # fd2 . fs2 = fd . fs - r_d / 2 - r_s / 2
    fs2 = np.hstack([fs, one_s, -0.5 * r_s[:, None]])
    fd2 = np.hstack([fd, -0.5 * r_d[:, None], one_d])
    return fs2, fd2


def debias(fs, fd, spec):
    """Apply the steps of spec ("none", "center", "whiten:256:1", "center+csls:10", ...) to the
    source features fs and the target features fd; returns the two new arrays."""
    if not spec or spec == "none":
        return fs, fd
    for step in spec.split("+"):
        name, _, rest = step.partition(":")
        par = [p for p in rest.split(":") if p]
        if name == "center":
            fs, fd = center(fs, fd)
        elif name == "whiten":
            fs, fd = whiten(fs, fd, int(par[0]) if par else 256,
                            float(par[1]) if len(par) > 1 else 1.0)
        elif name == "csls":
            fs, fd = csls(fs, fd, int(par[0]) if par else 10)
        else:
            raise ValueError("unknown debias step: " + step)
    return fs, fd


def tag_of(spec):
    return spec.replace("+", "_").replace(":", "-").replace(".", "p")
