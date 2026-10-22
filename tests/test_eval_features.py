"""The feature-set evaluation driver: pair table, stored viewing directions, output parsing."""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "idarb"))
import eval_features as ef  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")


def test_pair_table_covers_every_group_and_points_at_existing_files():
    rows = ef.pair_table(["same", "cross", "synthetic"])
    assert len(rows) == len(ef.SAME) + len(ef.CROSS) + len(ef.SYNTH)
    assert len({r[1] for r in rows}) == len(rows)  # pair names are unique
    for group, name, src, src_mesh, target, transform, extra in rows:
        assert target in ef.TARGET_MESH
        assert os.path.exists(os.path.join(ROOT, ef.TARGET_MESH[target])), target
        # every pair has a stored registration to take the view from
        assert os.path.exists(os.path.join(ROOT, transform or extra)), name
        if group == "synthetic":
            assert extra.endswith("_gt.npz") and os.path.exists(os.path.join(ROOT, extra))
        assert f"{src}" in src_mesh


def test_view_of_is_the_rotations_third_column(tmp_path):
    R = np.eye(3)[[1, 2, 0]]  # a permutation matrix: a valid rotation
    p = tmp_path / "transform.json"
    p.write_text(json.dumps(dict(rotation=R.tolist())))
    assert np.allclose(ef.view_of(str(p)), R[:, 2])


def test_last_json_reads_the_final_dict_line():
    text = "warning\n{'a': 1}\n{\"b\": 2, \"c\": [1, 2]}\n"
    assert ef.last_json(text) == {"b": 2, "c": [1, 2]}
    assert ef.last_json("{'textured_faces': 3}\n") == {"textured_faces": 3}
    assert ef.last_json("nothing") == {}


def test_summarise_averages_per_group_and_variant():
    rows = [
        dict(group="same", pair="a", variant="nn", within_5pct=0.8, mutual_anchors=10),
        dict(group="same", pair="b", variant="nn", within_5pct=0.6, mutual_anchors=30),
        dict(group="same", pair="a", variant="arap", within_5pct=1.0, mutual_anchors=10),
    ]
    for r in rows:
        for col in ef.CSV_FIELDS:
            r.setdefault(col, None)
    s = ef.summarise(rows)
    assert s["same/nn"]["pairs"] == 2 and s["same/nn"]["within_5pct"] == 0.7
    assert s["same/nn"]["mutual_anchors"] == 20
    assert s["same/arap"]["within_5pct"] == 1.0
    assert "gt_acc_1pct" not in s["same/nn"]  # columns without values are left out
