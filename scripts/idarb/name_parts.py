"""
Match the parts of two meshes by what they are called, not by how their points look: the first
version of our own method for pairs of different kinds (a toy and a camel).

Each mesh was clustered into parts (part_anchors.py; labels in <name>_parts.npz) and, on the
server, every part was pooled through SigLIP2's attention head over the rendered views, so each
part has one vector in SigLIP2's joint image-text space, next to the text vectors of a part
vocabulary ("a photo of the hump of a camel", ...). This script

  1. names every part: the vocabulary entries closest to its vector (object-free prompts);
  2. scores every target-source part pair by the cosine of the two part vectors after each
     object's mean part vector is removed (what is common to the whole object drops out), plus a
     bonus when the two parts' best names agree;
  3. keeps the confident one-to-one pairs (bilateral margin, as in part_anchors.py). A target part
     whose name has no counterpart on the source (the hump) stays unmatched, so with
     refine_correspondence.py --parts its vertices follow their neighbours instead of borrowing the
     most similar patch from elsewhere.

Writes the same part_match.npz layout as part_anchors.py (labels_t, labels_s, pairs) and a JSON
with the names and the score matrix.

Usage (repository root, mesh environment):
    python scripts/idarb/name_parts.py \
        --target results/diff3f/part_naming/camel_full.json \
        --target_labels results/diff3f/part_labels_tmp/camel_full_parts.npz \
        --source results/diff3f/part_naming/labubu_partial.json \
        --source_labels results/diff3f/part_labels_tmp/labubu_partial_parts.npz \
        --vocabulary results/diff3f/part_naming/vocabulary.npz --out <dir>/part_match.npz
"""

import argparse
import json
import os

import numpy as np

from part_anchors import margin_confidence, select_pairs


def unit(x):
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def load_part_file(path):
    """Part vectors (K, D) and sizes (K,) of one mesh, from the server's JSON (a list of parts with
    an "embedding", under "parts" or at the top level) or an npz with "embedding" and "size"."""
    if path.endswith(".npz"):
        z = np.load(path)
        emb = z["embedding"]
        size = z["size"] if "size" in z.files else np.ones(len(emb))
        return unit(emb), np.asarray(size, dtype=np.float64)
    with open(path) as f:
        d = json.load(f)
    parts = d["parts"] if isinstance(d, dict) else d
    parts = sorted(parts, key=lambda p: int(p.get("id", 0)))
    emb = np.stack([np.asarray(p["embedding"], dtype=np.float64) for p in parts])
    size = np.array([float(p.get("size", 1.0)) for p in parts])
    return unit(emb), size


def load_vocabulary(path):
    """Prompt strings and their unit text vectors; the part word of each prompt is the text
    between "the " and " of" (or the end), so prompts of both forms give the same word."""
    z = np.load(path, allow_pickle=False)
    text = [str(t) for t in z["text"]]
    emb = unit(z["emb"])
    words = []
    for t in text:
        w = t
        if "the " in w:
            w = w.split("the ", 1)[1]
        if " of " in w:
            w = w.split(" of ", 1)[0]
        words.append(w.strip())
    return text, words, emb


def name_parts(emb, vocab_emb, vocab_words, top=5):
    """For every part, the `top` vocabulary words by cosine (merging duplicates of a word)."""
    S = emb @ vocab_emb.T
    out = []
    for row in S:
        best = {}
        for j in np.argsort(-row):
            w = vocab_words[j]
            if w not in best:
                best[w] = float(row[j])
            if len(best) >= top:
                break
        out.append(list(best.items()))
    return out


def part_scores(emb_t, emb_s, names_t, names_s, name_bonus=0.2):
    """Relative cosine between target and source parts plus a bonus when the best names agree."""
    A = unit(emb_t - emb_t.mean(axis=0))
    B = unit(emb_s - emb_s.mean(axis=0))
    S = A @ B.T
    for a, na in enumerate(names_t):
        for b, nb in enumerate(names_s):
            if na and nb and na[0][0] == nb[0][0]:
                S[a, b] += name_bonus
    return S


def match_parts(S, alpha=20, min_conf=0.02, min_score=0.3):
    """Confident one-to-one part pairs: bilateral margin confidence, pairs scoring below
    min_score never qualify (a part with no counterpart must stay unmatched)."""
    conf = margin_confidence(S)
    conf = np.where(S >= min_score, conf, -np.inf)
    return select_pairs(conf, alpha, min_conf)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--target", required=True, help="part vectors of the full mesh (json/npz)")
    ap.add_argument("--target_labels", required=True, help="<name>_parts.npz of the full mesh")
    ap.add_argument("--source", required=True, help="part vectors of the relief (json/npz)")
    ap.add_argument("--source_labels", required=True)
    ap.add_argument("--vocabulary", required=True, help="npz with text and emb")
    ap.add_argument("--alpha", type=int, default=20, help="at most this many part pairs")
    ap.add_argument("--min_conf", type=float, default=0.02)
    ap.add_argument("--min_score", type=float, default=0.3, help="pairs below never match")
    ap.add_argument("--name_bonus", type=float, default=0.2)
    ap.add_argument("--out", required=True, help="part_match.npz (for refine --parts)")
    args = ap.parse_args()

    emb_t, size_t = load_part_file(args.target)
    emb_s, size_s = load_part_file(args.source)
    _, words, vemb = load_vocabulary(args.vocabulary)
    names_t = name_parts(emb_t, vemb, words)
    names_s = name_parts(emb_s, vemb, words)
    S = part_scores(emb_t, emb_s, names_t, names_s, args.name_bonus)
    pairs = match_parts(S, args.alpha, args.min_conf, args.min_score)
    lt = np.load(args.target_labels)["labels"].astype(np.int64)
    ls = np.load(args.source_labels)["labels"].astype(np.int64)
    if lt.max() + 1 != len(emb_t) or ls.max() + 1 != len(emb_s):
        raise ValueError("the part files and the label files have different part counts")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(
        args.out, labels_t=lt, labels_s=ls,
        pairs=np.array([[a, b, c] for a, b, c in pairs], dtype=np.float64).reshape(-1, 3),
    )
    report = dict(
        target_names=[[n for n, _ in nm[:3]] for nm in names_t],
        source_names=[[n for n, _ in nm[:3]] for nm in names_s],
        pairs=[dict(target=a, source=b, confidence=round(c, 4),
                    target_name=names_t[a][0][0], source_name=names_s[b][0][0])
               for a, b, c in pairs],
        unmatched_target=[dict(part=j, name=names_t[j][0][0], size=float(size_t[j]))
                          for j in range(len(emb_t)) if j not in {a for a, _, _ in pairs}],
        scores=np.round(S, 3).tolist(),
    )
    with open(os.path.splitext(args.out)[0] + ".json", "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(dict(pairs=report["pairs"], unmatched_target=report["unmatched_target"])))


if __name__ == "__main__":
    main()
