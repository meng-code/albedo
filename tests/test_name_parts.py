"""Naming parts from a vocabulary and matching them by name: a toy and a camel in a made-up
joint space where every part word is a direction and an object adds a shared offset."""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "idarb"))
import name_parts as npx  # noqa: E402

WORDS = ["head", "face", "leg", "body", "hump", "shirt"]


def word_vec(w, rng_dim=16):
    v = np.zeros(rng_dim)
    v[WORDS.index(w)] = 1.0
    return v


def test_names_and_matching_leave_the_hump_unmatched(tmp_path):
    rng = np.random.default_rng(0)
    offset_t = 0.6 * rng.normal(size=16)  # "camel-ness", shared by every camel part
    offset_s = 0.6 * rng.normal(size=16)
    camel = ["head", "leg", "body", "hump"]
    toy = ["face", "leg", "shirt", "head"]
    emb_t = np.stack([word_vec(w) + offset_t + 0.05 * rng.normal(size=16) for w in camel])
    emb_s = np.stack([word_vec(w) + offset_s + 0.05 * rng.normal(size=16) for w in toy])
    text = [f"a photo of the {w}" for w in WORDS]
    text += [f"a photo of the {w} of a camel" for w in WORDS]
    vemb = np.stack([word_vec(w) for w in WORDS] * 2)
    np.savez(tmp_path / "vocab.npz", text=np.array(text), emb=vemb)
    _, words, ve = npx.load_vocabulary(str(tmp_path / "vocab.npz"))
    assert words[:6] == WORDS and words[6:] == WORDS
    names_t = npx.name_parts(npx.unit(emb_t), ve, words)
    assert [n[0][0] for n in names_t] == camel
    names_s = npx.name_parts(npx.unit(emb_s), ve, words)
    S = npx.part_scores(npx.unit(emb_t), npx.unit(emb_s), names_t, names_s)
    pairs = npx.match_parts(S, 10, 0.02)
    matched = {a: b for a, b, _ in pairs}
    assert matched[0] == 3 and matched[1] == 1  # head-head, leg-leg
    assert 3 not in matched  # the hump has no counterpart on the toy

    # the command line writes the labels and pairs for refine --parts
    for name, emb, lab in (("camel", emb_t, [0, 1, 2, 3, 3, 2]), ("toy", emb_s, [0, 1, 2, 3, 1])):
        with open(tmp_path / f"{name}.json", "w") as f:
            parts = [dict(id=i, size=10, embedding=e.tolist()) for i, e in enumerate(emb)]
            json.dump(dict(parts=parts), f)
        np.savez(tmp_path / f"{name}_parts.npz", labels=np.array(lab))
    sys.argv = ["name_parts", "--target", str(tmp_path / "camel.json"), "--target_labels",
                str(tmp_path / "camel_parts.npz"), "--source", str(tmp_path / "toy.json"),
                "--source_labels", str(tmp_path / "toy_parts.npz"), "--vocabulary",
                str(tmp_path / "vocab.npz"), "--out", str(tmp_path / "m" / "part_match.npz")]
    npx.main()
    z = np.load(tmp_path / "m" / "part_match.npz")
    assert z["pairs"].shape[1] == 3 and len(z["labels_t"]) == 6
    report = json.load(open(tmp_path / "m" / "part_match.json"))
    assert any(u["name"] == "hump" for u in report["unmatched_target"])
