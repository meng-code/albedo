"""The GPU lifting path of diff3f_features.py must keep Diff3F's semantics: one pixel per vertex
and view (the last pair in row-major order of the ball-query result), the view counted once."""

import os
import sys

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "idarb"))


def _cpu_lift(q, per_pixel, n_vertices):
    """Diff3F's own lifting, as in diff3f_features.py: fancy-index writes with duplicates."""
    acc = torch.zeros((n_vertices, per_pixel.shape[0]))
    count = torch.zeros((n_vertices, 1))
    mask = q != -1
    count[q[mask]] += 1
    acc[q[mask]] += per_pixel.repeat_interleave(mask.sum(dim=1), dim=1).T
    return acc, count


def test_last_pair_per_vertex_matches_diff3f_lifting():
    from diff3f_features import last_pair_per_vertex

    g = torch.Generator().manual_seed(0)
    n_pixels, k, n_vertices, dim = 40, 5, 12, 3
    q = torch.randint(-1, n_vertices, (n_pixels, k), generator=g)
    q[3] = -1  # a pixel whose ball is empty
    per_pixel = torch.randn((dim, n_pixels), generator=g)

    acc_cpu, count_cpu = _cpu_lift(q, per_pixel, n_vertices)
    vsel, psel = last_pair_per_vertex(q)
    acc = torch.zeros((n_vertices, dim))
    count = torch.zeros((n_vertices, 1))
    count[vsel] += 1
    acc[vsel] += per_pixel[:, psel].T

    assert torch.equal(vsel, torch.unique(q[q != -1]))  # every vertex hit, once each
    assert torch.equal(count, count_cpu)  # the view is counted once per vertex
    assert torch.allclose(acc, acc_cpu)  # the same pixel's feature as the CPU write
