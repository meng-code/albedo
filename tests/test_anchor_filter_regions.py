import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "idarb"))
import anchor_filter_regions as afr  # noqa: E402
import sweep_anchor_filter as saf  # noqa: E402


def test_region_counts_reports_emptied_regions():
    region = np.array([0, 0, 1, 1, 3])
    keep = np.array([True, False, False, False, True])
    before, kept, with_any, emptied = afr.region_counts(region, keep, 4)
    assert before.tolist() == [2, 2, 0, 1] and kept.tolist() == [1, 0, 0, 1]
    assert with_any == 3 and emptied == 1


def test_regions_assigns_every_point_to_its_nearest_seed():
    rng = np.random.default_rng(0)
    xyz = np.concatenate([rng.normal(size=(50, 3)) * 0.05,
                          rng.normal(size=(50, 3)) * 0.05 + [10.0, 0.0, 0.0]])
    reg = afr.regions(xyz, 2)
    assert len(set(reg[:50])) == 1 and len(set(reg[50:])) == 1 and reg[0] != reg[99]


def test_largest_gap():
    xyz = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    assert np.isclose(afr.largest_gap(xyz, xyz[:1]), 1.0)


def test_filter_tags():
    assert saf.tag_of("none") == "none"
    assert saf.tag_of("0.1") == "rigid0p1"
    assert saf.tag_of("loo:0.2") == "loo0p2"
    assert saf.spec_of("0.3") == "rigid:0.3"
