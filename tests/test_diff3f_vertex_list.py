"""diff3f_features.select_vertices: the --vertex_list sample and the default random sample."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "idarb"))


def test_select_vertices():
    pytest.importorskip("torch")
    pytest.importorskip("trimesh")
    import diff3f_features as d3

    # an explicit list is used exactly, in its own order
    lst = np.array([7, 2, 9])
    assert np.array_equal(d3.select_vertices(10, 5, 0, lst), lst)
    with pytest.raises(ValueError):
        d3.select_vertices(10, 5, 0, np.array([1, 1]))
    with pytest.raises(ValueError):
        d3.select_vertices(10, 5, 0, np.array([10]))
    # without a list: all vertices, or a sorted seeded subset (unchanged behaviour)
    assert np.array_equal(d3.select_vertices(4, 5, 0), np.arange(4))
    a = d3.select_vertices(100, 10, 0)
    expected = np.sort(np.random.default_rng(0).choice(100, 10, replace=False))
    assert np.array_equal(a, expected)
