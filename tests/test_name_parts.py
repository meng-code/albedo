"""Naming parts and matching them by name: a toy and a camel with made-up part vectors and
the server's naming format; the hump has no counterpart and must stay unmatched."""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "idarb"))
import name_parts as npx  # noqa: E402


def test_name_similarity_groups():
    assert npx.name_similarity([("face", 1)], [("face", 1)]) == 0.25
    assert npx.name_similarity([("face", 1)], [("head", 1)]) == 0.7 * 0.25
    assert npx.name_similarity([("hump", 1)], [("face", 1)]) == 0.0
    assert npx.word_of("a photo of the front leg of a camel") == "front leg"
    assert npx.word_of("a photo of the hump") == "hump"


def test_matching_leaves_the_hump_unmatched(tmp_path):
    rng = np.random.default_rng(0)
    camel = [["hump", "snout", "head"], ["front leg", "leg", "foot"], ["torso", "body", "back"],
             ["head", "face", "ear"]]
    toy = [["face", "head", "snout"], ["shirt", "trousers", "body"], ["leg", "shoe", "foot"]]
    for name, parts, lab in (("camel", camel, [0, 1, 2, 3, 3, 2, 1]), ("toy", toy, [0, 1, 2, 2])):
        js = dict(parts=[dict(part=i, size=50, top_free=[[f"a photo of the {w}", 0.05 - 0.01 * k]
                                                        for k, w in enumerate(ws)],
                              embedding=rng.normal(size=768).tolist())
                         for i, ws in enumerate(parts)])
        with open(tmp_path / f"{name}.json", "w") as f:
            json.dump(js, f)
        np.savez(tmp_path / f"{name}_parts.npz", labels=np.array(lab))
    sys.argv = ["name_parts", "--target", str(tmp_path / "camel.json"), "--target_labels",
                str(tmp_path / "camel_parts.npz"), "--source", str(tmp_path / "toy.json"),
                "--source_labels", str(tmp_path / "toy_parts.npz"),
                "--out", str(tmp_path / "m" / "part_match.npz")]
    npx.main()
    report = json.load(open(tmp_path / "m" / "part_match.json"))
    matched = {p["target"]: p["source"] for p in report["pairs"]}
    assert matched[3] == 0 and matched[1] == 2 and matched[2] == 1  # head, leg, body
    assert 0 not in matched and report["unmatched_target"][0]["name"] == "hump"
    z = np.load(tmp_path / "m" / "part_match.npz")
    assert z["pairs"].shape == (3, 3) and len(z["labels_t"]) == 7
